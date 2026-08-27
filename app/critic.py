"""Deterministic critic for the highlight reel.

Runs cheap, objective checks on the rendered output and the EDL,
emits a JSON report with `issues` (tag list) and `knob_hints`
(suggested config tweaks for the auto-tuner).

No VLM calls here — this layer is free and instant. The agent loop
optionally runs a stronger VLM critic on top.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from statistics import pstdev

from .ffmpeg import ffprobe_json


@dataclass
class CriticReport:
    issues: list[str] = field(default_factory=list)
    knob_hints: dict[str, float] = field(default_factory=dict)
    axes: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_json(self) -> dict:
        return {
            "score": self.score,
            "axes": self.axes,
            "issues": self.issues,
            "knob_hints": self.knob_hints,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: dict) -> "CriticReport":
        return cls(
            issues=list(data.get("issues", [])),
            knob_hints=dict(data.get("knob_hints", {})),
            axes=dict(data.get("axes", {})),
            notes=list(data.get("notes", [])),
            score=float(data.get("score", 0.0)),
        )


def _audio_rms(path: Path) -> list[float]:
    """1-second-window RMS in dB via ffmpeg's astats filter."""
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(path),
        "-af", "astats=metadata=1:reset=1,ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return []

    levels: list[float] = []
    for line in result.stderr.splitlines():
        if "RMS_level" in line and "=" in line:
            try:
                # Value is in dB, e.g. "-23.45"
                levels.append(float(line.split("=")[-1].strip()))
            except ValueError:
                pass
    return levels


def _clips_from_edl(edl: list[dict]) -> list[tuple[Path, float, float, float]]:
    """Flatten EDL into (file, start, end, score) tuples."""
    out: list[tuple[Path, float, float, float]] = []
    for row in edl:
        out.append((
            Path(row["file"]),
            float(row["start"]),
            float(row["end"]),
            float(row.get("score", 0.0)),
        ))
    return out


def review(
    output_mp4: Path,
    edl: list[dict],
    target_seconds: float,
) -> CriticReport:
    report = CriticReport()

    # (a) audio_present
    if not output_mp4.exists():
        report.issues.append("missing_output")
        report.notes.append(f"highlight.mp4 not found at {output_mp4}")
        return report

    try:
        probe = ffprobe_json(output_mp4)
    except Exception as e:
        report.issues.append("probe_failed")
        report.notes.append(str(e))
        return report

    audio_streams = [
        s for s in probe.get("streams", [])
        if s.get("codec_type") == "audio"
    ]
    video_streams = [
        s for s in probe.get("streams", [])
        if s.get("codec_type") == "video"
    ]

    report.axes["audio_present"] = 1.0 if audio_streams else 0.0
    if not audio_streams:
        report.issues.append("no_audio")

    # (b) tracks_preserved — check whether source had >1 audio track and
    # output dropped them.
    src_track_counts: dict[str, int] = {}
    clips = _clips_from_edl(edl)
    for f, *_ in clips:
        if str(f) in src_track_counts:
            continue
        try:
            src_track_counts[str(f)] = sum(
                1 for s in ffprobe_json(f).get("streams", [])
                if s.get("codec_type") == "audio"
            )
        except Exception:
            src_track_counts[str(f)] = 1

    src_max = max(src_track_counts.values()) if src_track_counts else 1
    out_tracks = len(audio_streams)
    report.axes["tracks_preserved"] = (
        1.0 if src_max <= out_tracks else 0.5 if out_tracks == 1 else 0.0
    )
    if src_max > out_tracks:
        report.issues.append("tracks_dropped")

    # (d) clip_count_in_range
    n_clips = len(clips)
    report.axes["clip_count"] = (
        1.0 if 3 <= n_clips <= 12 else 0.5 if n_clips > 0 else 0.0
    )
    if n_clips < 3:
        report.issues.append("too_few_clips")
        report.knob_hints["min_score"] = -0.5
        report.knob_hints["max_clips"] = +2.0
    elif n_clips > 12:
        report.issues.append("too_many_clips")
        report.knob_hints["max_clips"] = -2.0

    # (c) clips_sequential — flag if >70% of consecutive clips are adjacent.
    if clips:
        per_file: dict[str, list[tuple[float, float]]] = {}
        for f, s, e, _ in clips:
            per_file.setdefault(str(f), []).append((s, e))
        total_pairs = 0
        sequential_pairs = 0
        for scenes in per_file.values():
            scenes.sort()
            for a, b in zip(scenes, scenes[1:]):
                total_pairs += 1
                # Adjacent if next scene's start is within 1s of prev end.
                if abs(b[0] - a[1]) < 1.0:
                    sequential_pairs += 1
        if total_pairs:
            ratio = sequential_pairs / total_pairs
            report.axes["spread"] = 1.0 - ratio
            if ratio > 0.70:
                report.issues.append("sequential_clips")
                report.knob_hints["max_clips"] = -2.0
                report.knob_hints["min_score"] = +0.5

    # (e) duration_in_range
    total_dur = sum(e - s for _, s, e, _ in clips)
    report.axes["duration_ok"] = (
        1.0 if abs(total_dur - target_seconds) <= 0.20 * target_seconds
        else 0.5
    )
    if total_dur > 1.20 * target_seconds:
        report.issues.append("over_budget")
        report.knob_hints["target_seconds"] = 0.9
    elif total_dur < 0.50 * target_seconds and n_clips > 0:
        report.issues.append("under_budget")
        report.knob_hints["min_score"] = -0.5
        report.knob_hints["max_clips"] = +1.0

    # (f) score_variance
    scores = [sc for _, _, _, sc in clips]
    if len(scores) >= 2:
        sd = pstdev(scores)
        report.axes["score_variance"] = min(1.0, sd / 2.0)
        if sd < 0.5:
            report.issues.append("low_score_variance")
            report.knob_hints["min_score"] = +0.5

    # (h) clip_length_variance
    durs = [e - s for _, s, e, _ in clips]
    if len(durs) >= 2:
        dsd = pstdev(durs)
        report.axes["clip_length_variance"] = min(1.0, dsd / 3.0)
        if dsd < 0.5:
            report.issues.append("uniform_clip_lengths")

    # (g) audio dead air
    levels = _audio_rms(output_mp4)
    if levels:
        quiet = sum(1 for l in levels if l < -40.0)
        quiet_ratio = quiet / len(levels)
        report.axes["audio_energy"] = 1.0 - quiet_ratio
        if quiet_ratio > 0.30:
            report.issues.append("dead_air")
            report.knob_hints["min_score"] = +1.0

    # Overall score: simple average of axes (no VLM involvement).
    if report.axes:
        report.score = sum(report.axes.values()) / len(report.axes)

    return report


def apply_knob_hints(
    base: dict,
    hints: dict[str, float],
) -> dict:
    """Adjust a config-dict using knob hints, with safe bounds."""
    bounds = {
        "min_score": (3.0, 9.0),
        "max_clips": (2, 12),
        "context_before": (1.0, 10.0),
        "context_after": (1.0, 10.0),
        "target_seconds": (20.0, 300.0),
    }
    out = dict(base)
    for k, delta in hints.items():
        if k not in out:
            continue
        cur = float(out[k])
        new = cur + delta
        lo, hi = bounds.get(k, (cur, cur))
        out[k] = max(lo, min(hi, new))
    return out