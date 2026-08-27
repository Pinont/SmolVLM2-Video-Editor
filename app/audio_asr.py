"""Audio transcription for mic track using faster-whisper.

Extracts audio track 1 (index 1 = second audio stream) from source video,
runs Whisper-medium multilingual, returns timestamped Thai/English transcript.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None  # type: ignore


@dataclass
class Segment:
    start: float
    end: float
    text: str
    language: str
    probability: float
    no_speech_prob: float = 1.0  # 1.0 = definitely non-speech, 0.0 = definitely speech


@dataclass
class Transcript:
    segments: list[Segment]
    language: str
    full_text: str


def extract_audio_track(
    video: Path,
    track_index: int = 1,
    dst: Optional[Path] = None,
) -> Path:
    """Extract a specific audio track from video to a wav file.
    
    track_index=0 → first audio stream (usually game+Discord)
    track_index=1 → second audio stream (usually mic)
    """
    if dst is None:
        dst = Path(tempfile.mktemp(suffix=".wav"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-map", f"0:a:{track_index}",
        "-ac", "1",              # mono for Whisper
        "-ar", "16000",          # 16kHz for Whisper
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def transcribe_audio(
    audio_path: Path,
    model_size: str = "medium",
    language: str = "th",
    device: str = "cuda",
    compute_type: str = "float16",
) -> Transcript:
    """Run faster-whisper on extracted audio track.
    
    model_size: tiny, base, small, medium, large-v3
    language: "th" for Thai, "en" for English, None for auto-detect
    """
    if WhisperModel is None:
        raise RuntimeError("faster-whisper not installed. pip install faster-whisper")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    segments, info = model.transcribe(
        str(audio_path),
        language=language if language else None,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segs: list[Segment] = []
    for s in segments:
        # faster-whisper exposes no_speech_prob on each segment
        nsp = getattr(s, "no_speech_prob", 1.0)
        segs.append(Segment(
            start=s.start,
            end=s.end,
            text=s.text.strip(),
            language=info.language,
            probability=s.avg_logprob,
            no_speech_prob=nsp,
        ))

    full = " ".join(s.text for s in segs)
    return Transcript(segments=segs, language=info.language, full_text=full)


def transcribe_mic_track(
    video: Path,
    model_size: str = "medium",
    language: str = "th",
) -> Transcript:
    """Convenience: extract track 1 (mic) and transcribe in one call."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = Path(f.name)
    try:
        extract_audio_track(video, track_index=1, dst=wav)
        return transcribe_audio(wav, model_size=model_size, language=language)
    finally:
        try:
            wav.unlink()
        except OSError:
            pass


def transcribe_all_tracks(
    video: Path,
    model_size: str = "medium",
    language: str = "th",
    max_tracks: int = 2,
) -> Transcript:
    """Extract and transcribe multiple audio tracks, merge by timestamp.
    
    Useful when track 0 (game+Discord) and track 1 (mic) both contain speech.
    Returns a single Transcript with all segments chronologically merged.
    """
    from .ffmpeg import ffprobe_json
    
    probe = ffprobe_json(video)
    audio_streams = [
        s for s in probe.get("streams", [])
        if s.get("codec_type") == "audio"
    ]
    
    if not audio_streams:
        return Transcript(segments=[], language=language, full_text="")
    
    all_segments: list[Segment] = []
    track_count = min(len(audio_streams), max_tracks)
    
    for track_idx in range(track_count):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav = Path(f.name)
        try:
            extract_audio_track(video, track_index=track_idx, dst=wav)
            track_transcript = transcribe_audio(
                wav, model_size=model_size, language=language
            )
            # Tag segments with track source for debugging
            for seg in track_transcript.segments:
                # We'll merge by timestamp, so just collect all
                all_segments.append(seg)
        finally:
            try:
                wav.unlink()
            except OSError:
                pass
    
    # Sort all segments by start time
    all_segments.sort(key=lambda s: s.start)
    
    # Merge overlapping/adjacent segments (within 0.5s gap)
    merged: list[Segment] = []
    for seg in all_segments:
        if merged and seg.start - merged[-1].end < 0.5:
            # Extend previous segment
            merged[-1].end = max(merged[-1].end, seg.end)
            merged[-1].text += " " + seg.text
        else:
            merged.append(seg)
    
    full = " ".join(s.text for s in merged)
    return Transcript(
        segments=merged,
        language=language,
        full_text=full,
    )


def transcript_for_chunk(
    transcript: Transcript,
    chunk_start: float,
    chunk_end: float,
    context_before: float = 2.0,
    context_after: float = 2.0,
) -> str:
    """Return concatenated transcript text for a specific time window.
    
    Includes segments that overlap [chunk_start - context_before, chunk_end + context_after].
    """
    lo = chunk_start - context_before
    hi = chunk_end + context_after
    parts = [
        s.text
        for s in transcript.segments
        if not (s.end <= lo or s.start >= hi)
    ]
    return " ".join(parts).strip()


def audio_events_for_chunk(
    transcript: Transcript,
    chunk_start: float,
    chunk_end: float,
    context_before: float = 2.0,
    context_after: float = 2.0,
    threshold: float = 0.6,
) -> str:
    """Return description of likely non-speech audio events in the window.
    
    Segments with no_speech_prob >= threshold are likely screams, laughs,
    gasps, explosions, etc. We summarize them by time and energy.
    """
    lo = chunk_start - context_before
    hi = chunk_end + context_after
    events = []
    for s in transcript.segments:
        if s.end <= lo or s.start >= hi:
            continue
        if s.no_speech_prob >= threshold:
            # Heuristic: short + high prob = sharp event (scream/gasp)
            # long + high prob = sustained non-speech (music, engine, ambient)
            dur = s.end - s.start
            if dur < 1.5:
                events.append(f"  {s.start:.1f}s: sharp non-speech ({s.no_speech_prob:.0%})")
            else:
                events.append(f"  {s.start:.1f}s: sustained non-speech ({dur:.1f}s, {s.no_speech_prob:.0%})")
    return "\n".join(events) if events else "  (no strong non-speech events detected)"


def save_transcript(transcript: Transcript, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "language": transcript.language,
                "full_text": transcript.full_text,
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "language": s.language,
                        "probability": s.probability,
                        "no_speech_prob": s.no_speech_prob,
                    }
                    for s in transcript.segments
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_transcript(path: Path) -> Transcript:
    data = json.loads(path.read_text("utf-8"))
    return Transcript(
        language=data["language"],
        full_text=data["full_text"],
        segments=[
            Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                language=s["language"],
                probability=s["probability"],
                no_speech_prob=s.get("no_speech_prob", 1.0),
            )
            for s in data["segments"]
        ],
    )