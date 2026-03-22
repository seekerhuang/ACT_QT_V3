from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from act_qt_v3.config import DatasetKind, ReadOptions


def _build_filter(read_options: ReadOptions):
    filters = []
    if read_options.normalized_date_from is not None:
        filters.append(ds.field("trade_date") >= read_options.normalized_date_from.to_pydatetime())
    if read_options.normalized_date_to is not None:
        filters.append(ds.field("trade_date") <= read_options.normalized_date_to.to_pydatetime())
    if read_options.instrument_whitelist:
        filters.append(ds.field("ts_code").isin(list(read_options.instrument_whitelist)))

    if not filters:
        return None

    expression = filters[0]
    for item in filters[1:]:
        expression = expression & item
    return expression


def _ensure_read_columns(columns: tuple[str, ...], dataset_kind: DatasetKind) -> list[str] | None:
    required = {"trade_date"}
    if dataset_kind == DatasetKind.STOCK_1MIN:
        required.update({"trade_time", "ts_code"})
    else:
        required.update({"ts_code"})

    if not columns:
        return None
    return sorted(set(columns).union(required))


def _read_single_parquet(path: Path, dataset_kind: DatasetKind, read_options: ReadOptions) -> pd.DataFrame:
    dataset = ds.dataset(str(path), format="parquet")
    columns = _ensure_read_columns(read_options.columns, dataset_kind)
    filter_expression = _build_filter(read_options)
    try:
        table = dataset.to_table(columns=columns, filter=filter_expression)
        frame = table.to_pandas()
    except pa.ArrowNotImplementedError:
        table = dataset.to_table(columns=columns)
        frame = table.to_pandas()
        if read_options.normalized_date_from is not None and "trade_date" in frame.columns:
            trade_date = pd.to_datetime(frame["trade_date"]).dt.normalize()
            frame = frame.loc[trade_date >= read_options.normalized_date_from]
        if read_options.normalized_date_to is not None and "trade_date" in frame.columns:
            trade_date = pd.to_datetime(frame["trade_date"]).dt.normalize()
            frame = frame.loc[trade_date <= read_options.normalized_date_to]
        if read_options.instrument_whitelist and "ts_code" in frame.columns:
            frame = frame.loc[frame["ts_code"].isin(read_options.instrument_whitelist)]
    return frame


def load_dataset_frames(
    dataset_kind: DatasetKind,
    source_paths: list[Path],
    read_options: ReadOptions,
) -> pd.DataFrame:
    frames = [_read_single_parquet(path, dataset_kind, read_options) for path in source_paths]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)
