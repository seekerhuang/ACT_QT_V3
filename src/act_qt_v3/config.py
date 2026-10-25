from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class DatasetKind(_StrEnum):
    STOCK_DAILY = "stock_daily"
    STOCK_1MIN = "stock_1min"
    STOCK_AUCTION_OPEN = "stock_auction_o_daily"
    STOCK_AUCTION_CLOSE = "stock_auction_c_daily"


class ProcessingMode(_StrEnum):
    PIT_SAFE = "pit_safe"
    RESEARCH_STANDARDIZED = "research_standardized"


class AdjustmentMethod(_StrEnum):
    RAW = "raw"
    BACKWARD_RATIO = "backward_ratio"


def _coerce_date(value: object) -> Optional[pd.Timestamp]:
    if value in (None, "", pd.NaT):
        return None
    return pd.Timestamp(value).normalize()


@dataclass(frozen=True)
class ReadOptions:
    date_from: object = None
    date_to: object = None
    instrument_whitelist: tuple[str, ...] = ()
    file_limit: Optional[int] = None
    columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.file_limit is not None and self.file_limit <= 0:
            raise ValueError("file_limit must be positive")
        object.__setattr__(self, "instrument_whitelist", tuple(self.instrument_whitelist))
        object.__setattr__(self, "columns", tuple(self.columns))

    @property
    def normalized_date_from(self) -> Optional[pd.Timestamp]:
        return _coerce_date(self.date_from)

    @property
    def normalized_date_to(self) -> Optional[pd.Timestamp]:
        return _coerce_date(self.date_to)

    @property
    def output_date_from(self) -> str:
        value = self.normalized_date_from
        return value.strftime("%Y-%m-%d") if value is not None else "all"

    @property
    def output_date_to(self) -> str:
        value = self.normalized_date_to
        return value.strftime("%Y-%m-%d") if value is not None else "all"


@dataclass(frozen=True)
class PreprocessJobConfig:
    data_root: Path
    dataset_kind: DatasetKind
    dataset_path: Optional[Path] = None
    mode: ProcessingMode = ProcessingMode.PIT_SAFE
    adjustment: AdjustmentMethod = AdjustmentMethod.RAW
    read_options: ReadOptions = field(default_factory=ReadOptions)
    write_output: bool = False
    output_root: Optional[Path] = None
    rounding: Optional[int] = None
    min_join_coverage_ratio: float = 0.93
    max_join_drop_ratio: float = 0.07

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        if self.dataset_path is not None:
            object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root))
        if self.mode == ProcessingMode.PIT_SAFE and self.adjustment != AdjustmentMethod.RAW:
            raise ValueError("pit_safe mode only supports raw adjustment in the first release")
        if not 0.0 <= self.min_join_coverage_ratio <= 1.0:
            raise ValueError("min_join_coverage_ratio must be within [0, 1]")
        if not 0.0 <= self.max_join_drop_ratio <= 1.0:
            raise ValueError("max_join_drop_ratio must be within [0, 1]")

    @property
    def effective_output_root(self) -> Path:
        if self.output_root is not None:
            return self.output_root
        return self.data_root.parent / "output"
