"""Textual TUI for the SmolVLM2 highlight agent loop.

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Header: iteration N/M  |  overall: 0.73  |  best: 0.78         │
  ├───────────────┬─────────────────────────┬──────────────────────┤
  │ Queue         │ Current iteration       │ Knobs                │
  │ video1 ✓      │ config: seg=6 min=7 …   │ segment:  6          │
  │ video2 …      │ critic:                 │ overlap:  2          │
  │               │   ✓ audio               │ min_score: 7.0       │
  │               │   ✗ sequential_clips    │ max_clips: 6         │
  │               │   …                     │ target: 45           │
  ├───────────────┴─────────────────────────┴──────────────────────┤
  │ Scores: iter1 ▆▆▆▆▆▆▆ 0.62  iter2 ▆▆▆▆▆▆▆▆▆▆▆ 0.73            │
  ├──────────────────────────────────────────────────────────────-─┤
  │ Live log (streaming pipeline stdout)                           │
  ├────────────────────────────────────────────────────────────────┤
  │ Footer: [s] start [p] pause [r] rerun [d] deep [q] quit        │
  └────────────────────────────────────────────────────────────────┘

Bindings:
  s — start the agent loop (one-shot per session)
  p — pause/resume log streaming
  r — rerun last iteration with same config
  d — toggle "deep review" (VLM critic; stubbed in v1)
  q — quit
"""

from __future__ import annotations

import threading
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Log,
    Static,
)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class QueuePanel(Static):
    """Shows the list of input videos + their per-video state."""

    def __init__(self, videos: list[Path]):
        super().__init__(id="queue")
        self.videos = videos

    def compose(self) -> ComposeResult:
        yield Static("[b]Queue[/b]", classes="panel-title")
        for v in self.videos:
            yield Static(f"  {v.name}", classes="queue-item")


class CurrentPanel(Static):
    """Shows the current iteration's config + critic verdict."""

    def __init__(self):
        super().__init__(id="current")
        self._config_lines: list[str] = []
        self._critic_lines: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("[b]Current iteration[/b]", classes="panel-title")
        yield Static("(idle)", id="current-config")
        yield Static("", classes="spacer")
        yield Static("[b]Critic[/b]", classes="panel-title")
        yield Static("(no verdict yet)", id="current-critic")

    def update_config(self, config: dict) -> None:
        keys = [
            "segment_seconds", "overlap_seconds", "min_score",
            "max_clips", "target_seconds", "context_before", "context_after",
        ]
        lines = [f"  {k}: {config.get(k)}" for k in keys if k in config]
        try:
            self.query_one("#current-config", Static).update("\n".join(lines) or "  (empty)")
        except Exception:
            pass

    def update_critic(self, critic: dict) -> None:
        axes = critic.get("axes", {})
        issues = critic.get("issues", [])
        lines = []
        for k, v in axes.items():
            lines.append(f"  {k}: {v:.2f}")
        if issues:
            lines.append("")
            lines.append("  [b]issues[/b]:")
            for tag in issues:
                lines.append(f"    ✗ {tag}")
        else:
            lines.append("")
            lines.append("  [green]✓ all checks pass[/green]")
        try:
            self.query_one("#current-critic", Static).update("\n".join(lines))
        except Exception:
            pass


class KnobsPanel(Static):
    """Live mirror of the running config knobs."""

    def __init__(self):
        super().__init__(id="knobs")

    def compose(self) -> ComposeResult:
        yield Static("[b]Knobs[/b]", classes="panel-title")
        yield Static("(idle)", id="knobs-body")

    def update_knobs(self, config: dict) -> None:
        keys = [
            "segment_seconds", "overlap_seconds", "min_score",
            "max_clips", "target_seconds", "context_before", "context_after",
        ]
        body = "\n".join(f"  {k}: {config.get(k)}" for k in keys if k in config)
        try:
            self.query_one("#knobs-body", Static).update(body or "  (empty)")
        except Exception:
            pass


def _sparkline(values: list[float], width: int = 20) -> str:
    bars = "▁▂▃▄▅▆▇█"
    if not values:
        return "(no scores yet)"
    if len(values) > width:
        values = values[-width:]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-6:
        return "".join(bars[0] for _ in values)
    out = []
    for v in values:
        idx = int((v - lo) / (hi - lo) * (len(bars) - 1))
        out.append(bars[idx])
    return "".join(out)


class ScoresPanel(Static):
    """Sparkline + numeric list of iteration scores."""

    def __init__(self):
        super().__init__(id="scores")

    def compose(self) -> ComposeResult:
        yield Static("[b]Scores[/b]", classes="panel-title")
        yield Static("(no iterations yet)", id="scores-body")

    def update_scores(self, scores: list[float]) -> None:
        if not scores:
            return
        spark = _sparkline(scores)
        lines = [f"  spark: {spark}"]
        for i, s in enumerate(scores, 1):
            lines.append(f"  iter {i}: {s:.3f}")
        try:
            self.query_one("#scores-body", Static).update("\n".join(lines))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class HighlighterApp(App):
    """Textual app that drives the agent loop and renders the dashboard."""

    CSS = """
    Screen { layout: vertical; }

    #top { height: auto; }
    #panels { height: 12; }
    #scores-row { height: 5; border: round $accent; }
    #log { height: 1fr; border: round $accent; }

    .panel-title { color: $accent; text-style: bold; }
    .spacer { height: 1; }

    #queue, #current, #knobs {
        border: round $accent;
        width: 1fr;
        height: 100%;
        padding: 0 1;
    }

    Log { background: $surface; }
    """

    BINDINGS = [
        Binding("s", "start", "Start"),
        Binding("p", "toggle_pause", "Pause"),
        Binding("r", "rerun", "Rerun"),
        Binding("d", "toggle_deep", "Deep review"),
        Binding("q", "quit", "Quit"),
    ]

    iteration = reactive(0)
    overall = reactive(0.0)
    best = reactive(0.0)
    paused = reactive(False)
    deep_review = reactive(False)

    def __init__(
        self,
        cfg,
        max_iterations: int = 5,
        wall_clock_seconds: float = 30 * 60,
    ):
        super().__init__()
        self.cfg = cfg
        self.max_iterations = max_iterations
        self.wall_clock_seconds = wall_clock_seconds
        self._agent_thread: threading.Thread | None = None
        self._scores: list[float] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Discover videos (mirrors pipeline.discover_videos but kept inline
        # so we don't pull tqdm into the TUI import path).
        from .pipeline import discover_videos

        videos = discover_videos(self.cfg.input_dir, self.cfg.extensions)
        if not videos:
            videos = [Path("(no videos found)")]

        with Horizontal(id="panels"):
            yield QueuePanel(videos)
            yield CurrentPanel()
            yield KnobsPanel()

        yield ScoresPanel(id="scores-row")
        yield Log(highlight=False, id="log")
        yield Footer()

    # --- reactivity ---------------------------------------------------------

    def watch_iteration(self, _old, _new) -> None:
        self.sub_title = (
            f"iteration {self.iteration}/{self.max_iterations}  "
            f"overall: {self.overall:.3f}  best: {self.best:.3f}"
        )

    def watch_overall(self, _old, new) -> None:
        self.sub_title = (
            f"iteration {self.iteration}/{self.max_iterations}  "
            f"overall: {new:.3f}  best: {self.best:.3f}"
        )

    def watch_best(self, _old, new) -> None:
        self.sub_title = (
            f"iteration {self.iteration}/{self.max_iterations}  "
            f"overall: {self.overall:.3f}  best: {new:.3f}"
        )

    # --- callbacks from agent loop (run on the UI thread) -------------------

    def cb_iteration_start(self, idx: int, overrides: dict) -> None:
        self.iteration = idx
        # Mirror the live config into the KnobsPanel.
        merged = {**self.cfg.__dict__, **(overrides or {})}
        for k in ("input_dir", "output_dir", "analysis_dir", "work_dir"):
            merged.pop(k, None)
        try:
            self.query_one(KnobsPanel).update_knobs(merged)
        except Exception:
            pass

    def cb_log(self, line: str) -> None:
        if self.paused:
            return
        try:
            self.query_one(Log).write_line(line)
        except Exception:
            pass

    def cb_critic(self, critic: dict) -> None:
        try:
            self.query_one(CurrentPanel).update_critic(critic)
        except Exception:
            pass

    def cb_iteration_end(self, record: dict) -> None:
        self.overall = float(record.get("overall_score", 0.0))
        self.best = max(self.best, self.overall)
        self._scores.append(self.overall)
        try:
            self.query_one(ScoresPanel).update_scores(self._scores)
            self.query_one(CurrentPanel).update_config(record.get("config", {}))
        except Exception:
            pass

    # --- actions ------------------------------------------------------------

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        self.cb_log(f"[tui] log streaming {'paused' if self.paused else 'resumed'}")

    def action_toggle_deep(self) -> None:
        self.deep_review = not self.deep_review
        self.cb_log(
            f"[tui] deep review {'enabled (VLM critic)' if self.deep_review else 'disabled (deterministic only)'}"
        )

    def action_rerun(self) -> None:
        self.cb_log("[tui] rerun requested — press 's' to relaunch the loop")

    def action_start(self) -> None:
        if self._agent_thread and self._agent_thread.is_alive():
            self.cb_log("[tui] agent loop already running")
            return
        self.cb_log("[tui] starting agent loop...")

        def _run() -> None:
            # Import inside the thread to avoid loading torch in the UI process.
            from . import agent as agent_mod

            try:
                agent_mod.run_agent_loop(
                    cfg=self.cfg,
                    max_iterations=self.max_iterations,
                    wall_clock_seconds=self.wall_clock_seconds,
                    on_iteration_start=lambda i, ov: self.call_from_thread(
                        self.cb_iteration_start, i, ov
                    ),
                    on_log=lambda line: self.call_from_thread(self.cb_log, line),
                    on_critic=lambda c: self.call_from_thread(self.cb_critic, c),
                    on_iteration_end=lambda r: self.call_from_thread(
                        self.cb_iteration_end, r
                    ),
                )
                self.call_from_thread(self.cb_log, "[tui] agent loop finished.")
            except Exception as e:  # noqa: BLE001
                self.call_from_thread(self.cb_log, f"[tui] agent loop crashed: {e}")

        self._agent_thread = threading.Thread(target=_run, daemon=True)
        self._agent_thread.start()


def run_tui(cfg, max_iterations: int = 5, wall_clock_seconds: float = 30 * 60):
    """Entry point: `python -m app.tui`."""
    app = HighlighterApp(
        cfg=cfg,
        max_iterations=max_iterations,
        wall_clock_seconds=wall_clock_seconds,
    )
    app.run()


if __name__ == "__main__":
    # Minimal CLI: parse enough args to find videos/output, then launch.
    import argparse
    from .config import Config

    p = argparse.ArgumentParser(prog="python -m app.tui")
    p.add_argument("--input", default="videos")
    p.add_argument("--output", default="output")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--wall-clock-seconds", type=float, default=30 * 60)
    args = p.parse_args()

    cfg = Config.from_args(  # type: ignore[attr-defined]
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        analysis_dir=Path(args.output) / "analysis",
        work_dir=Path(args.output) / "work",
    )
    run_tui(cfg, args.max_iterations, args.wall_clock_seconds)