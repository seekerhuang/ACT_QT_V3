import pandas as pd
import pytest

from act_qt_v3.config import DatasetKind
from act_qt_v3.preprocess.schema import normalize_frame, validate_schema_contract


def make_stock_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.5],
            "high": [10.2, 10.8],
            "low": [9.9, 10.2],
            "close": [10.1, 10.7],
            "pre_close": [9.8, 10.1],
            "change": [0.3, 0.6],
            "pct_chg": [3.0612, 5.9406],
            "vol": [1000.0, 1200.0],
            "amount": [10000.0, 13000.0],
            "up_limit": [11.0, 11.5],
            "down_limit": [8.8, 9.0],
            "adj_factor": [2.0, 2.1],
        }
    )


def make_stock_minute_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "trade_time": ["2024-01-02 09:30:00", "2024-01-02 09:31:00"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.1],
            "high": [10.0, 10.2],
            "low": [9.9, 10.0],
            "close": [10.0, 10.2],
            "vol": [100.0, 120.0],
            "amount": [1000.0, 1224.0],
            "adj_factor": [2.0, 2.0],
        }
    )


def test_normalize_and_validate_stock_daily_contract() -> None:
    normalized = normalize_frame(make_stock_daily_frame(), DatasetKind.STOCK_DAILY)
    validate_schema_contract(normalized, DatasetKind.STOCK_DAILY)

    assert normalized.index.names == ["trade_date", "ts_code"]
    assert normalized.index.is_monotonic_increasing


def test_validate_schema_contract_rejects_missing_columns() -> None:
    frame = make_stock_daily_frame().drop(columns=["adj_factor"])
    normalized = normalize_frame(frame, DatasetKind.STOCK_DAILY)

    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema_contract(normalized, DatasetKind.STOCK_DAILY)


def test_validate_schema_contract_rejects_wrong_index_names() -> None:
    frame = make_stock_daily_frame().set_index(["trade_date", "ts_code"])
    frame.index = frame.index.rename(["date", "symbol"])

    with pytest.raises(ValueError, match="index names"):
        validate_schema_contract(frame, DatasetKind.STOCK_DAILY)


def test_normalize_and_validate_stock_minute_contract() -> None:
    normalized = normalize_frame(make_stock_minute_frame(), DatasetKind.STOCK_1MIN)
    validate_schema_contract(normalized, DatasetKind.STOCK_1MIN)

    assert normalized.index.names == ["trade_date", "trade_time"]
    assert normalized["ts_code"].tolist() == ["000001.SZ", "000001.SZ"]


def test_validate_schema_contract_rejects_minute_trade_time_date_mismatch() -> None:
    frame = make_stock_minute_frame()
    frame.loc[1, "trade_time"] = "2024-01-03 09:31:00"
    normalized = normalize_frame(frame, DatasetKind.STOCK_1MIN)

    with pytest.raises(ValueError, match="trade_time must normalize to trade_date"):
        validate_schema_contract(normalized, DatasetKind.STOCK_1MIN)


def test_validate_schema_contract_rejects_invalid_minute_session_time() -> None:
    frame = make_stock_minute_frame()
    frame.loc[1, "trade_time"] = "2024-01-02 12:00:00"
    normalized = normalize_frame(frame, DatasetKind.STOCK_1MIN)

    with pytest.raises(ValueError, match="trade_time must fall within supported trading sessions"):
        validate_schema_contract(normalized, DatasetKind.STOCK_1MIN)
