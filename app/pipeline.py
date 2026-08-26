import json
from pathlib import Path

from tqdm import tqdm

from .ffmpeg import duration, make_chunk, render_clip, concat_files
from .model import SmolVLM
from .prompts import HIGHLIGHT_PROMPT
from .selector import Candidate, select_candidates, expand_candidate


def discover_videos(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def segment_ranges(total: float, segment: float, overlap: float):
    if overlap >= segment:
        raise ValueError("overlap must be smaller than segment length")

    step = segment - overlap
    start = 0.0

    while start < total:
        end = min(total, start + segment)
        yield start, end

        if end >= total:
            break

        start += step


class Pipeline:
    def __init__(self, config, model: SmolVLM):
        self.cfg = config
        self.model = model

    def analyze_one(self, video: Path) -> list[dict]:
        rel = video.relative_to(self.cfg.input_dir)
        safe_name = "__".join(rel.with_suffix("").parts)

        result_path = self.cfg.analysis_dir / f"{safe_name}.json"
        chunk_dir = self.cfg.work_dir / "chunks" / safe_name
        chunk_dir.mkdir(parents=True, exist_ok=True)

        if result_path.exists():
            try:
                existing = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        else:
            existing = []

        completed = {
            x["segment_id"]
            for x in existing
            if "segment_id" in x
        }

        total = duration(video)

        ranges = list(
            segment_ranges(
                total,
                self.cfg.segment_seconds,
                self.cfg.overlap_seconds,
            )
        )

        for start, end in tqdm(
            ranges,
            desc=f"Analyzing {video.name}",
        ):
            segment_id = f"{start:.3f}-{end:.3f}"

            if segment_id in completed:
                continue

            chunk = (
                chunk_dir
                / f"{start:012.3f}_{end:012.3f}.mp4"
            )

            if not chunk.exists():
                make_chunk(video, start, end, chunk)

            try:
                raw = self.model.analyze(
                    str(chunk),
                    HIGHLIGHT_PROMPT,
                )

                parsed = self.model.parse_json(raw)

                if parsed is None:
                    parsed = {
                        "keep": False,
                        "score": 0,
                        "category": "other",
                        "description": "",
                        "reason": "Invalid JSON from model",
                    }

                result = {
                    "segment_id": segment_id,
                    "file": str(video),
                    "start": start,
                    "end": end,
                    "score": float(parsed.get("score", 0)),
                    "keep": bool(parsed.get("keep", False)),
                    "category": str(
                        parsed.get("category", "other")
                    ),
                    "description": str(
                        parsed.get("description", "")
                    ),
                    "reason": str(
                        parsed.get("reason", "")
                    ),
                    "raw": raw,
                }

            except Exception as e:
                result = {
                    "segment_id": segment_id,
                    "file": str(video),
                    "start": start,
                    "end": end,
                    "score": 0,
                    "keep": False,
                    "category": "error",
                    "description": "",
                    "reason": str(e),
                    "raw": "",
                }

            existing.append(result)

            result_path.write_text(
                json.dumps(
                    existing,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            if not self.cfg.keep_chunks:
                try:
                    chunk.unlink()
                except OSError:
                    pass

        return existing

    def run(self):
        videos = discover_videos(
            self.cfg.input_dir,
            self.cfg.extensions,
        )

        if not videos:
            raise RuntimeError(
                f"No supported videos found in {self.cfg.input_dir}"
            )

        print(f"Found {len(videos)} source videos.")

        all_rows = []
        video_durations = {}

        for video in videos:
            video_durations[str(video)] = duration(video)
            all_rows.extend(self.analyze_one(video))

        candidates = []

        for row in all_rows:
            if not row.get("keep"):
                continue

            candidates.append(
                Candidate(
                    file=row["file"],
                    start=float(row["start"]),
                    end=float(row["end"]),
                    score=float(row["score"]),
                    category=row.get(
                        "category", "other"
                    ),
                    description=row.get(
                        "description", ""
                    ),
                    reason=row.get(
                        "reason", ""
                    ),
                )
            )

        selected = select_candidates(
            candidates,
            target_seconds=self.cfg.target_seconds,
            min_score=self.cfg.min_score,
            max_clips=self.cfg.max_clips,
        )

        expanded = [
            expand_candidate(
                c,
                video_durations[c.file],
                self.cfg.context_before,
                self.cfg.context_after,
            )
            for c in selected
        ]

        # Suppress overlaps again after context padding.
        final = []

        for c in sorted(
            expanded,
            key=lambda x: x.score,
            reverse=True,
        ):
            duplicate = False

            for x in final:
                if x.file != c.file:
                    continue

                overlap = max(
                    0.0,
                    min(x.end, c.end)
                    - max(x.start, c.start),
                )

                shortest = min(
                    x.duration,
                    c.duration,
                )

                if (
                    shortest > 0
                    and overlap / shortest >= 0.50
                ):
                    duplicate = True
                    break

            if not duplicate:
                final.append(c)

        # Strongest moments first for a highlight reel.
        final.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        edl = [
            {
                "index": i + 1,
                "file": c.file,
                "start": round(c.start, 3),
                "end": round(c.end, 3),
                "duration": round(c.duration, 3),
                "score": c.score,
                "category": c.category,
                "description": c.description,
                "reason": c.reason,
            }
            for i, c in enumerate(final)
        ]

        self.cfg.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        edl_path = (
            self.cfg.output_dir
            / "edit_decision_list.json"
        )

        edl_path.write_text(
            json.dumps(
                edl,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if not self.cfg.render:
            print("Rendering disabled.")
            return edl

        if not final:
            print(
                "No highlights passed the selection threshold."
            )
            return edl

        clip_dir = (
            self.cfg.work_dir
            / "rendered_clips"
        )
        clip_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        rendered = []

        for i, c in enumerate(final, 1):
            dst = clip_dir / f"{i:03d}.mp4"

            render_clip(
                Path(c.file),
                c.start,
                c.end,
                dst,
            )

            rendered.append(dst)

        output = (
            self.cfg.output_dir
            / "highlight.mp4"
        )

        concat_files(
            rendered,
            output,
        )

        highlights_path = (
            self.cfg.output_dir
            / "highlights.json"
        )

        highlights_path.write_text(
            json.dumps(
                edl,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\nFinished:")
        print(output)
        print(f"Selected clips: {len(final)}")
        print(
            "Approximate duration: "
            f"{sum(c.duration for c in final):.1f}s"
        )

        return edl
