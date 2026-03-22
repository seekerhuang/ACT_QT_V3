from __future__ import annotations

from typing import Any

import pandas as pd

from act_qt_v3.config import DatasetKind
from act_qt_v3.preprocess.schema import get_schema_contract


def compute_duplicate_key_count(frame: pd.DataFrame, dataset_kind: DatasetKind) -> int:
    if dataset_kind == DatasetKind.STOCK_1MIN:
        keys = frame.reset_index()[["trade_date", "trade_time", "ts_code"]]
        return int(keys.duplicated().sum())
    return int(frame.index.duplicated().sum())


def _count_non_positive_price_rows(frame: pd.DataFrame, price_columns: tuple[str, ...]) -> int:
    available = [column for column in price_columns if column in frame.columns]
    if not available:
        return 0
    return int((frame[available] <= 0).any(axis=1).sum())


def build_quality_report(
    frame: pd.DataFrame,
    dataset_kind: DatasetKind,
    data_version: dict[str, Any],
    *,
    pre_close_invalid_count: int = 0,
    join_audit: dict[str, Any] | None = None,
    join_coverage_passed: bool | None = None,
) -> dict[str, Any]:
    contract = get_schema_contract(dataset_kind)
    missing_columns = [column for column in contract.required_columns if column not in frame.columns]
    required_null_counts = {
        column: int(frame[column].isna().sum())
        for column in contract.required_columns
        if column in frame.columns
    }
    report: dict[str, Any] = {
        "row_count": int(len(frame)),
        "duplicate_key_count": compute_duplicate_key_count(frame, dataset_kind),
        "missing_required_column_count": len(missing_columns),
        "missing_required_columns": missing_columns,
        "required_null_counts": required_null_counts,
        "non_positive_price_count": _count_non_positive_price_rows(frame, contract.price_columns),
        "negative_volume_count": int((frame[list(contract.volume_columns)] < 0).any(axis=1).sum())
        if contract.volume_columns
        else 0,
        "negative_amount_count": int((frame[list(contract.amount_columns)] < 0).any(axis=1).sum())
        if contract.amount_columns
        else 0,
        "pre_close_invalid_count": int(pre_close_invalid_count),
        "index_is_monotonic_increasing": bool(frame.index.is_monotonic_increasing),
        "data_version": data_version,
    }
    if join_audit is not None:
        report["join_audit"] = join_audit
    if join_coverage_passed is not None:
        report["join_coverage_passed"] = join_coverage_passed
    return report
