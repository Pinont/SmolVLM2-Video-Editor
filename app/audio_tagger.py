"""Audio tagging using PANNs CNN14 (AudioSet 527 classes).

Provides multi-label sound event detection: Gunshot, Scream, Explosion,
Laughter, Cheering, Applause, etc. per time window.

Optional dependency: pip install panns-inference
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torchaudio
    from panns_inference import AudioTagging, labels
    PANNS_AVAILABLE = True
except Exception:
    PANNS_AVAILABLE = False
    AudioTagging = None
    labels = None
    torch = None

from .audio_asr import extract_audio_track


@dataclass
class AudioTag:
    """Single audio event detection."""
    label: str
    confidence: float
    start: float
    end: float


@dataclass
class TaggedWindow:
    """Tags for a specific time window."""
    start: float
    end: float
    tags: list[AudioTag] = field(default_factory=list)


class PANNSTagger:
    """Wrapper around PANNs CNN14 for clip-level tagging."""
    
    # Highlight-relevant AudioSet classes (subset of 527)
    HIGHLIGHT_CLASSES = {
        # Combat / action
        "Gunshot": ["Gunshot", "Machine gun", "Fusillade", "Artillery fire", "Explosion"],
        "Explosion": ["Explosion", "Burst", "Detonation"],
        # Vocal reactions
        "Scream": ["Scream", "Shriek", "Yell"],
        "Laughter": ["Laughter", "Laugh", "Giggle", "Chuckle", "Belly laugh"],
        "Cheering": ["Cheering", "Cheer", "Applause", "Clapping"],
        "Gasp": ["Gasp", "Gasp for air"],
        # Game sounds
        "Engine": ["Engine", "Vehicle", "Car", "Truck", "Motorcycle"],
        "Footsteps": ["Footsteps", "Walk", "Run"],
        "Impact": ["Impact", "Hit", "Punch", "Slap", "Smack"],
        # UI / meta
        "Music": ["Music", "Background music", "Theme music"],
        "Silence": ["Silence"],
    }
    
    # Flatten for quick lookup
    _CLASS_GROUPS = {v: k for k, vals in HIGHLIGHT_CLASSES.items() for v in vals}
    
    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
    ):
        if not PANNS_AVAILABLE:
            raise RuntimeError(
                "panns-inference not installed. pip install panns-inference"
            )
        self.device = device
        self.model = AudioTagging(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.sample_rate = 32000  # PANNs expects 32kHz
        self._labels = labels
    
    def tag_audio(
        self,
        audio_path: Path,
        window_sec: float = 10.0,
        hop_sec: float = 5.0,
        top_k: int = 10,
        threshold: float = 0.15,
    ) -> list[TaggedWindow]:
        """Tag audio file with sliding window.
        
        Returns TaggedWindow per window with tags above threshold.
        """
        # Load audio at 32kHz mono
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        audio = waveform.squeeze(0).numpy()  # (samples,)
        total_dur = len(audio) / self.sample_rate
        
        windows: list[TaggedWindow] = []
        start = 0.0
        
        while start < total_dur:
            end = min(start + window_sec, total_dur)
            if end - start < 1.0:  # skip too-short tail
                break
            
            # Extract window
            s_idx = int(start * self.sample_rate)
            e_idx = int(end * self.sample_rate)
            clip = audio[s_idx:e_idx]
            
            # PANNs expects (batch, samples)
            clip_batch = clip[None, :].astype(np.float32)
            
            # Inference
            with torch.no_grad():
                clip_tensor = torch.from_numpy(clip_batch).to(self.device)
                probs = self.model.inference(clip_tensor)  # (1, 527)
            
            probs = probs[0].cpu().numpy()
            
            # Top-k above threshold
            tags = []
            top_indices = np.argsort(probs)[::-1][:top_k]
            for idx in top_indices:
                prob = float(probs[idx])
                if prob < threshold:
                    break
                label = self._labels[idx]
                group = self._CLASS_GROUPS.get(label, "Other")
                tags.append(AudioTag(
                    label=f"{group}:{label}",
                    confidence=prob,
                    start=start,
                    end=end,
                ))
            
            windows.append(TaggedWindow(start=start, end=end, tags=tags))
            start += hop_sec
        
        return windows
    
    def tag_video_tracks(
        self,
        video: Path,
        track_indices: list[int] = [0, 1],
        window_sec: float = 10.0,
        hop_sec: float = 5.0,
        top_k: int = 10,
        threshold: float = 0.15,
    ) -> dict[int, list[TaggedWindow]]:
        """Extract and tag multiple audio tracks from video."""
        results: dict[int, list[TaggedWindow]] = {}
        
        for track_idx in track_indices:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav = Path(f.name)
            try:
                extract_audio_track(video, track_index=track_idx, dst=wav)
                # Resample to 32kHz for PANNs
                results[track_idx] = self.tag_audio(
                    wav, window_sec, hop_sec, top_k, threshold
                )
            finally:
                try:
                    wav.unlink()
                except OSError:
                    pass
        
        return results


def merge_tags_across_tracks(
    track_tags: dict[int, list[TaggedWindow]],
) -> list[TaggedWindow]:
    """Merge tags from multiple tracks by time window (max confidence per label)."""
    # Collect all unique window boundaries
    all_windows: dict[tuple[float, float], list[AudioTag]] = {}
    
    for track_idx, windows in track_tags.items():
        for w in windows:
            key = (round(w.start, 2), round(w.end, 2))
            if key not in all_windows:
                all_windows[key] = []
            # Tag with track source
            for tag in w.tags:
                tag.label = f"track{track_idx}:{tag.label}"
                all_windows[key].append(tag)
    
    # Merge: for each window, keep highest confidence per base label
    merged: list[TaggedWindow] = []
    for (start, end), tags in sorted(all_windows.items()):
        best: dict[str, AudioTag] = {}
        for tag in tags:
            base = tag.label.split(":", 1)[1] if ":" in tag.label else tag.label
            if base not in best or tag.confidence > best[base].confidence:
                best[base] = tag
        merged.append(TaggedWindow(start=start, end=end, tags=list(best.values())))
    
    return merged


def tags_for_chunk(
    merged_windows: list[TaggedWindow],
    chunk_start: float,
    chunk_end: float,
    context_before: float = 2.0,
    context_after: float = 2.0,
    min_confidence: float = 0.2,
) -> str:
    """Format tags overlapping a chunk window for prompt injection."""
    lo = chunk_start - context_before
    hi = chunk_end + context_after
    
    # Aggregate tags across overlapping windows
    tag_scores: dict[str, float] = {}
    for w in merged_windows:
        if w.end <= lo or w.start >= hi:
            continue
        for tag in w.tags:
            if tag.confidence < min_confidence:
                continue
            base = tag.label.split(":", 1)[1] if ":" in tag.label else tag.label
            tag_scores[base] = max(tag_scores.get(base, 0.0), tag.confidence)
    
    if not tag_scores:
        return "  (no significant audio events detected)"
    
    # Sort by confidence
    sorted_tags = sorted(tag_scores.items(), key=lambda x: -x[1])
    
    lines = []
    for label, conf in sorted_tags:
        lines.append(f"  {label}: {conf:.0%}")
    
    return "\n".join(lines)


def save_tags(windows: list[TaggedWindow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "start": w.start,
                        "end": w.end,
                        "tags": [
                            {
                                "label": t.label,
                                "confidence": t.confidence,
                                "start": t.start,
                                "end": t.end,
                            }
                            for t in w.tags
                        ],
                    }
                    for w in windows
                ]
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_tags(path: Path) -> list[TaggedWindow]:
    data = json.loads(path.read_text("utf-8"))
    return [
        TaggedWindow(
            start=w["start"],
            end=w["end"],
            tags=[
                AudioTag(
                    label=t["label"],
                    confidence=t["confidence"],
                    start=t["start"],
                    end=t["end"],
                )
                for t in w["tags"]
            ],
        )
        for w in data["windows"]
    ]