import argparse
from pathlib import Path

from .config import Config
from .model import SmolVLM
from .pipeline import Pipeline


def main():
    p = argparse.ArgumentParser(
        description="SmolVLM2 + FFmpeg automatic video highlight editor"
    )

    p.add_argument("--input", default="videos")
    p.add_argument("--output", default="output")
    p.add_argument("--analysis-dir", default="output/analysis")
    p.add_argument("--work-dir", default="work")

    p.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    )

    p.add_argument("--segment", type=float, default=15.0)
    p.add_argument("--overlap", type=float, default=3.0)

    p.add_argument("--target", type=float, default=180.0)
    p.add_argument("--context-before", type=float, default=5.0)
    p.add_argument("--context-after", type=float, default=5.0)

    p.add_argument("--min-score", type=float, default=6.5)
    p.add_argument("--max-clips", type=int, default=25)
    p.add_argument("--max-new-tokens", type=int, default=160)

    p.add_argument("--keep-chunks", action="store_true")
    p.add_argument("--no-render", action="store_true")

    args = p.parse_args()

    cfg = Config(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        analysis_dir=Path(args.analysis_dir),
        work_dir=Path(args.work_dir),
        model_name=args.model,
        segment_seconds=args.segment,
        overlap_seconds=args.overlap,
        target_seconds=args.target,
        context_before=args.context_before,
        context_after=args.context_after,
        min_score=args.min_score,
        max_clips=args.max_clips,
        max_new_tokens=args.max_new_tokens,
        keep_chunks=args.keep_chunks,
        render=not args.no_render,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    model = SmolVLM(
        cfg.model_name,
        cfg.max_new_tokens,
    )

    Pipeline(cfg, model).run()


if __name__ == "__main__":
    main()
