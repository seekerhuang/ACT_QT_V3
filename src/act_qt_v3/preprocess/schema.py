from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from act_qt_v3.config import DatasetKind


@dataclass(frozen=True)
class DatasetSchemaContract:
    dataset_kind: DatasetKind
    required_index_names: tuple[str, ...]
    required_columns: tuple[str, ...]
    price_columns: tuple[str, ...]
    volume_columns: tuple[str, ...]
    amount_columns: tuple[str, ...]


SCHEMA_CONTRACTS = {
    DatasetKind.STOCK_DAILY: DatasetSchemaContract(
        dataset_kind=DatasetKind.STOCK_DAILY,
        required_index_names=("trade_date", "ts_code"),
        required_columns=(
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
            "up_limit",
            "down_limit",
            "adj_factor",
        ),
        price_columns=("open", "high", "low", "close", "pre_close", "up_limit", "down_limit"),
        volume_columns=("vol",),
        amount_columns=("amount",),
    ),
    DatasetKind.STOCK_1MIN: DatasetSchemaContract(
        dataset_kind=DatasetKind.STOCK_1MIN,
        required_index_names=("trade_date", "trade_time"),
        required_columns=("ts_code", "open", "high", "low", "close", "vol", "amount", "adj_factor"),
        price_columns=("open", "high", "low", "close"),
        volume_columns=("vol",),
        amount_columns=("amount",),
    ),
    DatasetKind.STOCK_AUCTION_OPEN: DatasetSchemaContract(
        dataset_kind=DatasetKind.STOCK_AUCTION_OPEN,
        required_index_names=("trade_date", "ts_code"),
        required_columns=("open", "high", "low", "close", "vol", "amount", "vwap"),
        price_columns=("open", "high", "low", "close", "vwap"),
        volume_columns=("vol",),
        amount_columns=("amount",),
    ),
    DatasetKind.STOCK_AUCTION_CLOSE: DatasetSchemaContract(
        dataset_kind=DatasetKind.STOCK_AUCTION_CLOSE,
        required_index_names=("trade_date", "ts_code"),
        required_columns=("open", "high", "low", "close", "vol", "amount", "vwap"),
        price_columns=("open", "high", "low", "close", "vwap"),
        volume_columns=("vol",),
        amount_columns=("amount",),
    ),
}


def get_schema_contract(dataset_kind: DatasetKind) -> DatasetSchemaContract:
    return SCHEMA_CONTRACTS[dataset_kind]


def _normalize_datetime_columns(frame: pd.DataFrame, dataset_kind: DatasetKind) -> pd.DataFrame:
    normalized = frame.copy()
    if "trade_date" in normalized.columns:
        normalized["trade_date"] = pd.to_datetime(normalized["trade_date"]).dt.normalize()
    if dataset_kind == DatasetKind.STOCK_1MIN and "trade_time" in normalized.columns:
        normalized["trade_time"] = pd.to_datetime(normalized["trade_time"])
    return normalized


def normalize_frame(frame: pd.DataFrame, dataset_kind: DatasetKind) -> pd.DataFrame:
    contract = get_schema_contract(dataset_kind)
    normalized = _normalize_datetime_columns(frame, dataset_kind)

    if list(normalized.index.names) == list(contract.required_index_names):
        return normalized.sort_index()

    if all(name in normalized.columns for name in contract.required_index_names):
        normalized = normalized.set_index(list(contract.required_index_names)).sort_index()
        return normalized

    return normalized


def _duplicate_key_count(frame: pd.DataFrame, dataset_kind: DatasetKind) -> int:
    if dataset_kind == DatasetKind.STOCK_1MIN:
        keys = frame.reset_index()[["trade_date", "trade_time", "ts_code"]]
        return int(keys.duplicated().sum())
    return int(frame.index.duplicated().sum())


def _validate_stock_minute_time_alignment(frame: pd.DataFrame) -> None:
    trade_date = pd.DatetimeIndex(frame.index.get_level_values("trade_date")).normalize()
    trade_time = pd.DatetimeIndex(frame.index.get_level_values("trade_time"))
    if not trade_time.normalize().equals(trade_date):
        raise ValueError("trade_time must normalize to trade_date")


def _validate_stock_minute_session_time(frame: pd.DataFrame) -> None:
    trade_time = pd.DatetimeIndex(frame.index.get_level_values("trade_time"))
    minute_of_day = (trade_time.hour * 60) + trade_time.minute
    valid_morning = (minute_of_day >= 9 * 60 + 30) & (minute_of_day <= 11 * 60 + 30)
    valid_afternoon = (minute_of_day >= 13 * 60) & (minute_of_day <= 15 * 60)
    valid_session = valid_morning | valid_afternoon
    if not bool(valid_session.all()):
        raise ValueError("trade_time must fall within supported trading sessions")


def validate_schema_contract(frame: pd.DataFrame, dataset_kind: DatasetKind) -> None:
    contract = get_schema_contract(dataset_kind)
    if list(frame.index.names) != list(contract.required_index_names):
        raise ValueError(
            f"index names must be {contract.required_index_names}, got {tuple(frame.index.names)}"
        )

    missing_columns = [column for column in contract.required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    if _duplicate_key_count(frame, dataset_kind) > 0:
        raise ValueError("duplicate primary key rows detected")

    if not frame.index.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing")

    if dataset_kind == DatasetKind.STOCK_1MIN:
        _validate_stock_minute_time_alignment(frame)
        _validate_stock_minute_session_time(frame)
