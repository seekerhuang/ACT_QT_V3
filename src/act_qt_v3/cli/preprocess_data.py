from __future__ import annotations

import argparse
import json
from pathlib import Path

from act_qt_v3.config import (
    AdjustmentMethod,
    DatasetKind,
    PreprocessJobConfig,
    ProcessingMode,
    ReadOptions,
)
from act_qt_v3.preprocess.pipeline import run_preprocess_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行数据清洗与预处理任务。")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--dataset-kind", required=True, choices=[item.value for item in DatasetKind])
    parser.add_argument("--mode", default=ProcessingMode.PIT_SAFE.value, choices=[item.value for item in ProcessingMode])
    parser.add_argument(
        "--adjustment",
        default=AdjustmentMethod.RAW.value,
        choices=[item.value for item in AdjustmentMethod],
    )
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--symbols")
    parser.add_argument("--file-limit", type=int)
    parser.add_argument("--write-output", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rounding", type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    symbols = tuple(filter(None, (args.symbols or "").split(",")))

    config = PreprocessJobConfig(
        data_root=args.data_root,
        dataset_kind=DatasetKind(args.dataset_kind),
        mode=ProcessingMode(args.mode),
        adjustment=AdjustmentMethod(args.adjustment),
        read_options=ReadOptions(
            date_from=args.date_from,
            date_to=args.date_to,
            instrument_whitelist=symbols,
            file_limit=args.file_limit,
        ),
        write_output=args.write_output,
        output_root=args.output_root,
        rounding=args.rounding,
    )
    result = run_preprocess_job(config)
    print(json.dumps(result.metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
