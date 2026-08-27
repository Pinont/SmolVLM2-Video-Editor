import json
from pathlib import Path

from tqdm import tqdm

from .ffmpeg import duration, make_chunk, render_clip, concat_files, concat_scenes
from .model import SmolVLM
from .selector import Candidate, select_candidates, expand_candidate
from .stages import ensure_meta, chunk_scoring_prompt
from .audio_asr import transcribe_all_tracks, transcript_for_chunk


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
            (x["segment_id"], x.get("pass", 1))
            for x in existing
            if "segment_id" in x
        }

        total = duration(video)

        # Extract ALL audio tracks transcript once per video (cached).
        transcript_path = self.cfg.analysis_dir / f"{safe_name}._transcript.json"
        if transcript_path.exists():
            from .audio_asr import load_transcript
            transcript = load_transcript(transcript_path)
        else:
            try:
                transcript = transcribe_all_tracks(
                    video,
                    model_size="medium",
                    language="th",
                    max_tracks=2,
                )
                from .audio_asr import save_transcript
                save_transcript(transcript, transcript_path)
            except Exception as e:
                print(f"[warn] audio transcription failed: {e}")
                transcript = None

        # Optional: PANNs audio tagging for typed sound events (gunfire, scream, laugh, etc.)
        audio_tags = None
        if self.cfg.audio_tagging:
            tags_path = self.cfg.analysis_dir / f"{safe_name}._tags.json"
            if tags_path.exists():
                try:
                    from .audio_tagger import load_tags
                    audio_tags = load_tags(tags_path)
                except Exception:
                    pass
            else:
                try:
                    from .audio_tagger import PANNSTagger, merge_tags_across_tracks
                    tagger = PANNSTagger(device="cuda" if self.model.device == "cuda" else "cpu")
                    track_tags = tagger.tag_video_tracks(
                        video,
                        track_indices=[0, 1],
                        window_sec=10.0,
                        hop_sec=5.0,
                        top_k=10,
                        threshold=0.15,
                    )
                    audio_tags = merge_tags_across_tracks(track_tags)
                    from .audio_tagger import save_tags
                    save_tags(audio_tags, tags_path)
                except Exception as e:
                    print(f"[warn] audio tagging failed: {e}")
                    audio_tags = None

        # Stage 1 + Stage 2: ensure per-video meta is cached.
        meta = ensure_meta(
            self.model, video, self.cfg.analysis_dir,
            two_pass=self.cfg.two_pass,
        )

        ranges = list(
            segment_ranges(
                total,
                self.cfg.segment_seconds,
                self.cfg.overlap_seconds,
            )
        )

        # Build list of (pass_num, prompt) to run per chunk.
        passes = [(1, meta.highlights.get(1, ""))]
        if self.cfg.two_pass and meta.highlights.get(2):
            passes.append((2, meta.highlights.get(2, "")))

        for start, end in tqdm(
            ranges,
            desc=f"Analyzing {video.name}",
        ):
            for pass_num, highlight_types in passes:
                segment_id = f"{start:.3f}-{end:.3f}"
                key = (segment_id, pass_num)
                if key in completed:
                    continue

                chunk = (
                    chunk_dir
                    / f"{start:012.3f}_{end:012.3f}.mp4"
                )

                if not chunk.exists():
                    make_chunk(video, start, end, chunk)

                # Get transcript snippet for this time window.
                transcript_snippet = ""
                audio_events_snippet = ""
                audio_tags_snippet = ""
                if transcript:
                    transcript_snippet = transcript_for_chunk(
                        transcript, start, end,
                        context_before=2.0, context_after=2.0,
                    )
                    from .audio_asr import audio_events_for_chunk
                    audio_events_snippet = audio_events_for_chunk(
                        transcript, start, end,
                        context_before=2.0, context_after=2.0,
                        threshold=0.6,
                    )

                # PANNs audio tags for this window
                if audio_tags:
                    from .audio_tagger import tags_for_chunk
                    audio_tags_snippet = tags_for_chunk(
                        audio_tags, start, end,
                        context_before=2.0, context_after=2.0,
                        min_confidence=0.2,
                    )

                prompt = chunk_scoring_prompt(
                    highlight_types,
                    transcript=transcript_snippet,
                    audio_events=audio_events_snippet,
                    audio_tags=audio_tags_snippet,
                )

                try:
                    raw = self.model.analyze(
                        str(chunk),
                        prompt,
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
                        "pass": pass_num,
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
                        "pass": pass_num,
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

        # Two-pass reconciliation: per reference Space, prefer the
        # sparser of the two passes (set 2 if 0 < pct2 <= pct1 or pct1==0,
        # else set 1). Each segment is keyed by (file, segment_id).
        rows_by_key: dict[tuple[str, str], dict] = {}
        for row in all_rows:
            key = (row["file"], row["segment_id"])
            rows_by_key.setdefault(key, {})[row.get("pass", 1)] = row

        # Per the reference: compute kept-duration per pass, pick sparser.
        pass1_dur = sum(
            r[1]["end"] - r[1]["start"]
            for r in rows_by_key.values()
            if r.get(1, {}).get("keep")
        )
        pass2_dur = sum(
            r[2]["end"] - r[2]["start"]
            for r in rows_by_key.values()
            if r.get(2, {}).get("keep")
        )

        candidates: list[Candidate] = []
        if self.cfg.two_pass and any(2 in r for r in rows_by_key.values()):
            chosen_pass = (
                2 if (0 < pass2_dur <= pass1_dur or pass1_dur == 0) else 1
            )
        else:
            chosen_pass = 1

        for r in rows_by_key.values():
            row = r.get(chosen_pass)
            if not row or not row.get("keep"):
                continue
            candidates.append(
                Candidate(
                    file=row["file"],
                    start=float(row["start"]),
                    end=float(row["end"]),
                    score=float(row["score"]),
                    category=row.get("category", "other"),
                    description=row.get("description", ""),
                    reason=row.get("reason", ""),
                )
            )

        print(
            f"Pass1 kept {pass1_dur:.1f}s | "
            f"Pass2 kept {pass2_dur:.1f}s | "
            f"Using pass {chosen_pass} ({len(candidates)} candidates)"
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

        # Chronological playback: highlights should tell a story in the
        # order they happened, not be ordered by raw score.
        final.sort(
            key=lambda x: (x.file, x.start),
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

        # Single-pass concat via filter_complex: re-encodes once from the
        # source videos, preserves all audio tracks (mixed if more than
        # one). Much faster than render-each-clip + concat, and keeps the
        # mic track from multi-track recordings.
        per_file_scenes: dict[str, list[tuple[float, float]]] = {}
        for c in final:
            per_file_scenes.setdefault(c.file, []).append(
                (c.start, c.end)
            )

        # Per-file scenes are already chronological after the final sort,
        # but re-sort defensively in case the order changed.
        for f in per_file_scenes:
            per_file_scenes[f].sort()

        output = (
            self.cfg.output_dir
            / "highlight.mp4"
        )

        if len(per_file_scenes) == 1:
            src, scenes = next(iter(per_file_scenes.items()))
            concat_scenes(
                Path(src),
                scenes,
                output,
            )
        else:
            # Multiple source files: render per-file intermediates first,
            # then concatenate them with concat_files.
            clip_dir = (
                self.cfg.work_dir
                / "rendered_clips"
            )
            clip_dir.mkdir(parents=True, exist_ok=True)

            rendered: list[Path] = []
            for i, (src, scenes) in enumerate(per_file_scenes.items(), 1):
                dst = clip_dir / f"src_{i:03d}.mp4"
                concat_scenes(Path(src), scenes, dst)
                rendered.append(dst)

            concat_files(rendered, output)

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
