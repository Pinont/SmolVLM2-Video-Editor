"""Two-stage pre-analysis from the SmolVLM2-HighlightGenerator reference.

Stage 1: feed the *whole* video, get a free-form description of what's in it.
Stage 2: feed that description (no video) and ask the model to list the
         archetypal highlight moments worth clipping out.
Stage 3 (in pipeline.py): for each chunk, ask "does this match any of
         those highlight types?" + return a score.

Caching: results are persisted per-video so re-running with different
selector knobs doesn't re-describe the source.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .model import SmolVLM


SYSTEM_DESCRIBE = (
    "You are a helpful assistant that can understand videos. "
    "Describe what type of video this is and what's happening in it."
)

USER_DESCRIBE = (
    "What type of video is this and what's happening in it? "
    "Be specific about the content type and general activities you observe."
)

SYSTEM_HIGHLIGHTS_PROMPTS = {
    1: (
        "You are a highlight editor. List archetypal dramatic moments "
        "that would make compelling highlights if they appear in the "
        "video. Each moment should be specific enough to be recognizable "
        "but generic enough to potentially exist in other videos of this type."
    ),
    2: (
        "You are a helpful visual-language assistant that can understand "
        "videos and edit. You are tasked helping the user to create "
        "highlight reels for videos. Highlights should be rare and "
        "important events in the video in question."
    ),
}

USER_HIGHLIGHTS_PROMPTS = {
    1: "List potential highlight moments to look for in this video:",
    2: (
        "List dramatic moments that would make compelling highlights if "
        "they appear in the video. Each moment should be specific enough "
        "to be recognizable but generic enough to potentially exist in "
        "any video of this type:"
    ),
}


@dataclass
class VideoMeta:
    """Cached Stage 1 + Stage 2 output for a single source video."""
    description: str = ""
    highlights: dict[int, str] = field(default_factory=dict)  # prompt_num -> text
    schema_version: int = 1

    def to_json(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "description": self.description,
            "highlights": {str(k): v for k, v in self.highlights.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> "VideoMeta":
        return cls(
            description=data.get("description", ""),
            highlights={
                int(k): v for k, v in data.get("highlights", {}).items()
            },
            schema_version=data.get("schema_version", 1),
        )


def meta_path_for(video: Path, analysis_dir: Path) -> Path:
    rel = video.relative_to(video.parents[len(video.parents) - 2])
    safe_name = "__".join(rel.with_suffix("").parts)
    return analysis_dir / f"{safe_name}._meta.json"


def load_meta(video: Path, analysis_dir: Path) -> VideoMeta | None:
    p = meta_path_for(video, analysis_dir)
    if not p.exists():
        return None
    try:
        return VideoMeta.from_json(json.loads(p.read_text("utf-8")))
    except Exception:
        return None


def save_meta(video: Path, analysis_dir: Path, meta: VideoMeta) -> None:
    p = meta_path_for(video, analysis_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(meta.to_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def describe_video(model: SmolVLM, video_path: str) -> str:
    """Stage 1: free-form description of the whole video."""
    return model.analyze(video_path, USER_DESCRIBE).strip()


def derive_highlight_types(
    model: SmolVLM,
    description: str,
    prompt_num: int = 1,
) -> str:
    """Stage 2: given the description, list archetypal highlights."""
    system = SYSTEM_HIGHLIGHTS_PROMPTS[prompt_num]
    user = (
        f"Here is a description of a video:\n\n"
        f"{description}\n\n"
        f"{USER_HIGHLIGHTS_PROMPTS[prompt_num]}"
    )
    return model.generate_text(system, user, max_new_tokens=256).strip()


def ensure_meta(
    model: SmolVLM,
    video: Path,
    analysis_dir: Path,
    two_pass: bool = True,
) -> VideoMeta:
    """Return cached Stage 1+Stage 2 results, computing if missing."""
    cached = load_meta(video, analysis_dir)
    if cached and cached.description:
        have_set1 = bool(cached.highlights.get(1))
        have_set2 = bool(cached.highlights.get(2)) if two_pass else True
        if have_set1 and have_set2:
            return cached

    meta = cached or VideoMeta()
    if not meta.description:
        meta.description = describe_video(model, str(video))

    if not meta.highlights.get(1):
        meta.highlights[1] = derive_highlight_types(
            model, meta.description, prompt_num=1
        )
    if two_pass and not meta.highlights.get(2):
        meta.highlights[2] = derive_highlight_types(
            model, meta.description, prompt_num=2
        )

    save_meta(video, analysis_dir, meta)
    return meta


def chunk_scoring_prompt(
    highlight_types: str,
    transcript: str = "",
    audio_events: str = "",
    audio_tags: str = "",
) -> str:
    """Stage 3 prompt: per-chunk yes/no justification + score.
    
    If transcript is provided, it contains the player's Thai mic audio
    (transcribed via Whisper) for this time window. Use it as additional
    context — emotional reactions, callouts, etc. are strong highlight signals.
    
    If audio_events is provided, it describes non-speech audio segments
    (high no_speech_prob from Whisper) — likely screams, laughs, gasps,
    explosions, etc. These are strong highlight indicators even without words.
    
    If audio_tags is provided, it contains typed audio events from PANNs
    CNN14 (AudioSet) — e.g. Gunshot, Explosion, Laughter, Scream.
    These are high-confidence, specific event detections.
    """
    transcript_block = ""
    if transcript:
        transcript_block = f"""

Mic audio transcript (Thai, from player's microphone):
{transcript}

Use this to gauge emotional intensity, surprise, achievement, etc."""
    
    audio_events_block = ""
    if audio_events:
        audio_events_block = f"""

Non-speech audio events (likely screams, laughs, gasps, explosions):
{audio_events}

These indicate high-arousal moments even without spoken words."""
    
    audio_tags_block = ""
    if audio_tags:
        audio_tags_block = f"""

Typed audio events (PANNs CNN14, AudioSet classes):
{audio_tags}

These are specific detected events: Gunshot, Explosion, Laughter, Scream,
Cheering, etc. Strong highlight signals."""
    
    return f"""You are a video highlight analyzer. Your role is to identify
moments that have high dramatic value, focusing on displays of skill,
emotion, personality, or tension. Compare video segments against provided
example highlights to find moments with similar emotional impact and
visual interest, even if the specific actions differ.

Given these highlight examples:
{highlight_types}
{transcript_block}
{audio_events_block}
{audio_tags_block}

Does this video contain a moment that matches the core action of one of
the highlights? Answer with strict JSON only (no prose):

{{
  "keep": true,
  "score": 0,
  "category": "action|funny|dramatic|emotional|achievement|unexpected|informative|other",
  "description": "one short sentence",
  "reason": "one short sentence explaining the match (or lack thereof)"
}}

Scoring rules:
- 0-2: no match, dead air, clearly off-topic
- 3-4: weak match, skip (keep=false)
- 5-6: mild interest but not a highlight (keep=false)
- 7-8: solid highlight (keep=true)
- 9: excellent highlight (keep=true)
- 10: exceptional must-keep moment (keep=true)

Default to keep=false unless score >= 7.""".strip()