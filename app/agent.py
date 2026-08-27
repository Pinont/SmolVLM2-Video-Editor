"""Iterative refine loop.

Each iteration:
  1. Run the pipeline with current Config.
  2. Run deterministic critic on the rendered output.
  3. Apply knob hints to the config.
  4. Repeat until convergence (no improvement / max iters / all checks pass).

State is persisted under output/iterations/<NNN>/ and output/state.json.
"""
from __future__ import annotations

import json
import time
import traceback
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import Config
from .critic import CriticReport, apply_knob_hints, review
from .pipeline import Pipeline
from .model import SmolVLM


class AgentLoop:
    def __init__(
        self,
        cfg: Config,
        model: SmolVLM,
        max_iterations: int = 5,
        wall_clock_minutes: int = 30,
        patience: int = 2,
    ):
        self.cfg = cfg
        self.model = model
        self.max_iterations = max_iterations
        self.wall_clock_seconds = wall_clock_minutes * 60
        self.patience = patience

        self.iter_dir = self.cfg.output_dir / "iterations"
        self.iter_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.cfg.output_dir / "state.json"

    # ------------------------------------------------------------------
    # Persistence

    def _iter_path(self, n: int) -> Path:
        return self.iter_dir / f"{n:03d}"

    def _save_iter(
        self,
        n: int,
        cfg_snapshot: Config,
        edl: list[dict],
        report: CriticReport,
        elapsed: float,
    ) -> None:
        d = self._iter_path(n)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(
            json.dumps(asdict(cfg_snapshot), indent=2, default=str),
            encoding="utf-8",
        )
        (d / "edit_decision_list.json").write_text(
            json.dumps(edl, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (d / "critic.json").write_text(
            json.dumps(report.to_json(), indent=2),
            encoding="utf-8",
        )
        (d / "meta.json").write_text(
            json.dumps(
                {"iteration": n, "elapsed_seconds": round(elapsed, 1),
                 "timestamp": datetime.utcnow().isoformat() + "Z"},
                indent=2,
            ),
            encoding="utf-8",
        )

    def _save_state(
        self,
        history: list[dict],
        winner_n: int | None,
    ) -> None:
        self.state_path.write_text(
            json.dumps(
                {
                    "history": history,
                    "winner": winner_n,
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Main loop

    def run(self, log=print) -> dict | None:
        history: list[dict] = []
        best_score = -1.0
        best_iter = 0
        no_improve_rounds = 0
        cfg = deepcopy(self.cfg)
        deadline = time.monotonic() + self.wall_clock_seconds
        winner_edl: list[dict] | None = None

        for n in range(1, self.max_iterations + 1):
            if time.monotonic() > deadline:
                log(f"[agent] wall-clock limit hit ({self.wall_clock_minutes}m), stopping.")
                break

            log(f"\n========== ITERATION {n}/{self.max_iterations} ==========")
            log(
                f"  segment={cfg.segment_seconds}s  overlap={cfg.overlap_seconds}s  "
                f"min_score={cfg.min_score}  max_clips={cfg.max_clips}  "
                f"target={cfg.target_seconds}s  two_pass={cfg.two_pass}"
            )
            t0 = time.monotonic()

            try:
                # Wipe per-iteration outputs but keep Stage-1/2 meta.
                cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
                cfg.output_dir.mkdir(parents=True, exist_ok=True)
                cfg.work_dir.mkdir(parents=True, exist_ok=True)

                pipe = Pipeline(cfg, self.model)
                edl = pipe.run() or []
                elapsed = time.monotonic() - t0

                highlight = cfg.output_dir / "highlight.mp4"
                report = review(highlight, edl, cfg.target_seconds)

                self._save_iter(n, cfg, edl, report, elapsed)

                history.append(
                    {
                        "iteration": n,
                        "score": report.score,
                        "issues": report.issues,
                        "knob_hints": report.knob_hints,
                        "n_clips": len(edl),
                        "elapsed": round(elapsed, 1),
                        "config": asdict(cfg),
                    }
                )

                log(
                    f"[iter {n}] score={report.score:.3f}  "
                    f"clips={len(edl)}  "
                    f"issues={report.issues or 'none'}  "
                    f"elapsed={elapsed:.1f}s"
                )

                # Winner tracking
                if report.score > best_score:
                    best_score = report.score
                    best_iter = n
                    winner_edl = edl
                    no_improve_rounds = 0
                    # Promote current render to canonical.
                    if highlight.exists():
                        canonical = cfg.output_dir / "highlight.mp4"
                        canonical.write_bytes(highlight.read_bytes())
                else:
                    no_improve_rounds += 1

                # Stop conditions
                if not report.issues and report.score >= 0.85:
                    log(f"[agent] all checks pass and score>=0.85, stopping.")
                    self._save_state(history, best_iter)
                    return self._finalize(cfg, winner_edl, history, best_iter)

                if no_improve_rounds >= self.patience:
                    log(
                        f"[agent] no improvement for {self.patience} rounds, stopping."
                    )
                    self._save_state(history, best_iter)
                    return self._finalize(cfg, winner_edl, history, best_iter)

                # Tune for next round.
                cfg_dict = asdict(cfg)
                tuned = apply_knob_hints(cfg_dict, report.knob_hints)
                cfg = Config(**{
                    **{k: v for k, v in asdict(self.cfg).items()},
                    **{
                        "segment_seconds": tuned["segment_seconds"],
                        "overlap_seconds": tuned["overlap_seconds"],
                        "min_score": tuned["min_score"],
                        "max_clips": int(tuned["max_clips"]),
                        "target_seconds": tuned["target_seconds"],
                        "context_before": tuned["context_before"],
                        "context_after": tuned["context_after"],
                    },
                })

            except Exception as e:
                log(f"[iter {n}] FAILED: {e}")
                traceback.print_exc()
                history.append(
                    {
                        "iteration": n,
                        "error": str(e),
                        "score": -1.0,
                    }
                )
                break

        self._save_state(history, best_iter if winner_edl else None)
        return self._finalize(cfg, winner_edl, history, best_iter)

    def _finalize(
        self,
        cfg: Config,
        edl: list[dict] | None,
        history: list[dict],
        winner: int,
    ) -> dict | None:
        if edl is None:
            print("[agent] no winning iteration produced an EDL.")
            return None
        # Persist the winner's EDL as canonical.
        (cfg.output_dir / "edit_decision_list.json").write_text(
            json.dumps(edl, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (cfg.output_dir / "highlights.json").write_text(
            json.dumps(edl, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[agent] winner: iteration {winner}  (history: {len(history)} iters)")
        return {
            "winner_iteration": winner,
            "history": history,
            "edl": edl,
        }