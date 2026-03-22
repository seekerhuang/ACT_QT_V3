from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from act_qt_v3.config import DatasetKind, ReadOptions


DEFAULT_DATASET_PATHS = {
    DatasetKind.STOCK_DAILY: Path("stock_daily.parquet"),
    DatasetKind.STOCK_1MIN: Path("stock_1min"),
    DatasetKind.STOCK_AUCTION_OPEN: Path("stock_auction_o_daily.parquet"),
    DatasetKind.STOCK_AUCTION_CLOSE: Path("stock_auction_c_daily.parquet"),
}

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def resolve_dataset_path(data_root: Path, dataset_kind: DatasetKind, dataset_path: Path | None = None) -> Path:
    if dataset_path is not None:
        return dataset_path
    return data_root / DEFAULT_DATASET_PATHS[dataset_kind]


def collect_source_paths(dataset_path: Path, dataset_kind: DatasetKind, read_options: ReadOptions) -> list[Path]:
    if dataset_kind != DatasetKind.STOCK_1MIN:
        return [dataset_path]

    files = sorted(dataset_path.glob("*.parquet"))
    if read_options.instrument_whitelist:
        allow = set(read_options.instrument_whitelist)
        files = [path for path in files if path.stem in allow]
    if read_options.file_limit is not None:
        files = files[: read_options.file_limit]
    if not files:
        raise FileNotFoundError(f"no parquet files selected from {dataset_path}")
    return files


def build_data_version(source_paths: Iterable[Path]) -> dict[str, object]:
    source_paths = list(dict.fromkeys(Path(path) for path in source_paths))
    if not source_paths:
        raise ValueError("source_paths must not be empty")

    source_file_info = []
    for path in source_paths:
        stat = path.stat()
        source_file_info.append(
            {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "last_write_time": datetime.fromtimestamp(stat.st_mtime, tz=SHANGHAI_TZ).isoformat(),
            }
        )

    timestamps = [item["last_write_time"] for item in source_file_info]
    payload: dict[str, object] = {
        "file_count": len(source_paths),
        "source_paths": [str(path) for path in source_paths],
        "file_size_bytes_total": int(sum(item["size_bytes"] for item in source_file_info)),
        "source_file_info": source_file_info,
        "last_write_time_min": min(timestamps),
        "last_write_time_max": max(timestamps),
    }
    if len(source_paths) == 1:
        payload["last_write_time"] = timestamps[0]
    return payload
