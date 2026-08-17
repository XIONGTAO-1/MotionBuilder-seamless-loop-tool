from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PREPROCESSING_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPROCESSING_ROOT))

from amass_babel_router.pipeline import PipelineConfig, build_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare audited AMASS/BABEL routing clips and classifier windows."
    )
    parser.add_argument("--amass-root", type=Path, default=Path("HDM05"))
    parser.add_argument("--babel-root", type=Path, default=Path("babel_v1.0_release"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("processed_amass_babel_router")
    )
    parser.add_argument("--min-clip-seconds", type=float, default=0.5)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--window-frames", type=int, default=45)
    parser.add_argument("--stride-frames", type=int, default=15)
    parser.add_argument("--val-subjects", nargs="+", default=["tr"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PipelineConfig(
        min_clip_seconds=args.min_clip_seconds,
        target_fps=args.target_fps,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        val_subjects=tuple(args.val_subjects),
        overwrite=args.overwrite,
    )
    summary = build_dataset(args.amass_root, args.babel_root, args.output_root, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
