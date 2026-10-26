from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from act_qt_v3.config import AdjustmentMethod, DatasetKind, PreprocessJobConfig, ProcessingMode
from act_qt_v3.dataio.discovery import build_data_version, collect_source_paths, resolve_dataset_path
from act_qt_v3.dataio.parquet_reader import load_dataset_frames
from act_qt_v3.preprocess.adjustment import (
    apply_backward_ratio_adjustment,
    attach_adj_factor_for_auction,
)
from act_qt_v3.preprocess.quality import build_quality_report
from act_qt_v3.preprocess.schema import normalize_frame, validate_schema_contract


@dataclass
class PipelineResult:
    data: pd.DataFrame
    metadata: dict[str, Any]
    quality_report: dict[str, Any]


def _load_frame(config: PreprocessJobConfig) -> tuple[pd.DataFrame, list[Path], Path]:
    dataset_path = resolve_dataset_path(config.data_root, config.dataset_kind, config.dataset_path)
    source_paths = collect_source_paths(dataset_path, config.dataset_kind, config.read_options)
    frame = load_dataset_frames(config.dataset_kind, source_paths, config.read_options)
    return frame, source_paths, dataset_path


def _load_stock_daily_adj_factor(config: PreprocessJobConfig) -> tuple[pd.DataFrame, Path]:
    dataset_path = resolve_dataset_path(config.data_root, DatasetKind.STOCK_DAILY)
    read_options = config.read_options
    limited_read_options = type(read_options)(
        date_from=read_options.date_from,
        date_to=read_options.date_to,
        instrument_whitelist=read_options.instrument_whitelist,
        file_limit=None,
        columns=("trade_date", "ts_code", "adj_factor"),
    )
    frame = load_dataset_frames(DatasetKind.STOCK_DAILY, [dataset_path], limited_read_options)
    frame = normalize_frame(frame, DatasetKind.STOCK_DAILY)
    if "adj_factor" not in frame.columns:
        raise ValueError("stock_daily adj_factor support frame missing adj_factor")
    return frame[["adj_factor"]], dataset_path


def _require_non_empty_frame(frame: pd.DataFrame, message: str) -> None:
    if frame.empty:
        raise ValueError(message)


def _empty_event_ts(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")


def _add_event_columns(frame: pd.DataFrame, dataset_kind: DatasetKind) -> pd.DataFrame:
    enriched = frame.copy()
    if dataset_kind == DatasetKind.STOCK_DAILY:
        trade_date = enriched.index.get_level_values("trade_date")
        enriched["event_date"] = trade_date
        enriched["event_ts"] = _empty_event_ts(enriched)
        return enriched

    if dataset_kind == DatasetKind.STOCK_1MIN:
        trade_time = enriched.index.get_level_values("trade_time")
        if trade_time.tz is None:
            event_ts = trade_time.tz_localize("Asia/Shanghai")
        else:
            event_ts = trade_time.tz_convert("Asia/Shanghai")
        enriched["event_ts"] = event_ts.as_unit("ns")
        return enriched

    if dataset_kind == DatasetKind.STOCK_AUCTION_OPEN:
        trade_date = enriched.index.get_level_values("trade_date")
        enriched["event_date"] = trade_date
        enriched["auction_session"] = "open_call_auction"
        enriched["event_ts"] = _empty_event_ts(enriched)
        return enriched

    if dataset_kind == DatasetKind.STOCK_AUCTION_CLOSE:
        trade_date = enriched.index.get_level_values("trade_date")
        enriched["event_date"] = trade_date
        enriched["auction_session"] = "close_call_auction"
        enriched["event_ts"] = _empty_event_ts(enriched)
        return enriched

    return enriched


def _event_ts_semantics(dataset_kind: DatasetKind) -> str:
    if dataset_kind == DatasetKind.STOCK_1MIN:
        return "trade_time"
    return "trade_date_only"


def _time_precision(dataset_kind: DatasetKind) -> str:
    if dataset_kind == DatasetKind.STOCK_1MIN:
        return "minute"
    return "date"


def _evaluate_join_coverage(join_audit: dict[str, Any], config: PreprocessJobConfig) -> bool:
    global_audit = join_audit["global"]
    coverage_ratio = float(global_audit["coverage_ratio"])
    drop_ratio = float(join_audit["dropped"]["dropped_ratio"])
    passed = (
        coverage_ratio >= config.min_join_coverage_ratio
        and drop_ratio <= config.max_join_drop_ratio
    )
    if not passed:
        raise ValueError(
            "join coverage below configured threshold: "
            f"coverage_ratio={coverage_ratio:.6f}, "
            f"drop_ratio={drop_ratio:.6f}, "
            f"min_join_coverage_ratio={config.min_join_coverage_ratio:.6f}, "
            f"max_join_drop_ratio={config.max_join_drop_ratio:.6f}"
        )
    return passed


def _output_leaf(config: PreprocessJobConfig) -> Path:
    return (
        config.effective_output_root
        / config.dataset_kind.value
        / config.mode.value
        / config.adjustment.value
        / f"date_from={config.read_options.output_date_from}__date_to={config.read_options.output_date_to}"
    )


def _write_output(result: PipelineResult, config: PreprocessJobConfig) -> Path:
    output_dir = _output_leaf(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_tmp = output_dir / "part-00000.tmp.parquet"
    parquet_path = output_dir / "part-00000.parquet"
    metadata_tmp = output_dir / "metadata.tmp.json"
    metadata_path = output_dir / "metadata.json"

    result.data.to_parquet(parquet_tmp)
    with metadata_tmp.open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, ensure_ascii=False, indent=2)

    os.replace(parquet_tmp, parquet_path)
    os.replace(metadata_tmp, metadata_path)
    return output_dir


def run_preprocess_job(config: PreprocessJobConfig) -> PipelineResult:
    raw_frame, source_paths, _ = _load_frame(config)
    normalized = normalize_frame(raw_frame, config.dataset_kind)
    _require_non_empty_frame(normalized, "empty dataset after applying filters")
    validate_schema_contract(normalized, config.dataset_kind)
    normalized = _add_event_columns(normalized, config.dataset_kind)

    join_audit = None
    join_coverage_passed = None
    adjustment_stats = {"pre_close_invalid_count": 0}
    adjusted = normalized
    metadata_source_paths = list(source_paths)

    if (
        config.mode == ProcessingMode.RESEARCH_STANDARDIZED
        and config.adjustment == AdjustmentMethod.BACKWARD_RATIO
    ):
        if config.dataset_kind in (DatasetKind.STOCK_AUCTION_OPEN, DatasetKind.STOCK_AUCTION_CLOSE):
            stock_daily_adj, stock_daily_path = _load_stock_daily_adj_factor(config)
            adjusted, join_audit = attach_adj_factor_for_auction(normalized, stock_daily_adj)
            _require_non_empty_frame(adjusted, "empty dataset after joining adj_factor")
            join_coverage_passed = _evaluate_join_coverage(join_audit, config)
            adjusted = _add_event_columns(adjusted, config.dataset_kind)
            metadata_source_paths.append(stock_daily_path)
        adjusted, adjustment_stats = apply_backward_ratio_adjustment(
            adjusted,
            config.dataset_kind,
            rounding=config.rounding,
        )

    metadata_source_paths = list(dict.fromkeys(Path(path) for path in metadata_source_paths))
    data_version = build_data_version(metadata_source_paths)
    quality_report = build_quality_report(
        adjusted,
        config.dataset_kind,
        data_version,
        pre_close_invalid_count=adjustment_stats["pre_close_invalid_count"],
        join_audit=join_audit,
        join_coverage_passed=join_coverage_passed,
    )

    metadata: dict[str, Any] = {
        "dataset_name": config.dataset_kind.value,
        "mode": config.mode.value,
        "adjustment": config.adjustment.value,
        "adjustment_applied": config.adjustment != AdjustmentMethod.RAW,
        "event_ts_semantics": _event_ts_semantics(config.dataset_kind),
        "time_precision": _time_precision(config.dataset_kind),
        "data_version": data_version,
        "source_paths": [str(path) for path in metadata_source_paths],
        "row_count_in": int(len(normalized)),
        "row_count_out": int(len(adjusted)),
        "quality_summary": quality_report,
    }
    if join_coverage_passed is not None:
        metadata["join_coverage_passed"] = join_coverage_passed

    result = PipelineResult(data=adjusted, metadata=metadata, quality_report=quality_report)
    if config.write_output:
        output_dir = _write_output(result, config)
        result.metadata["output_path"] = str(output_dir)
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(result.metadata, handle, ensure_ascii=False, indent=2)

    return result
