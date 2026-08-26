from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    analysis_dir: Path
    work_dir: Path

    model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    segment_seconds: float = 15.0
    overlap_seconds: float = 3.0

    target_seconds: float = 180.0
    context_before: float = 5.0
    context_after: float = 5.0

    min_score: float = 3.0
    max_clips: int = 25

    max_new_tokens: int = 160
    keep_chunks: bool = False
    render: bool = True

    extensions: tuple[str, ...] = (
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"
    )
