import math

import pandas as pd

from act_qt_v3.config import DatasetKind
from act_qt_v3.preprocess.adjustment import (
    apply_backward_ratio_adjustment,
    attach_adj_factor_for_auction,
)
from act_qt_v3.preprocess.schema import normalize_frame


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


def make_auction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "open": [10.0, 20.0],
            "high": [10.1, 20.1],
            "low": [9.9, 19.9],
            "close": [10.0, 20.0],
            "vol": [100.0, 200.0],
            "amount": [1000.0, 4000.0],
            "vwap": [10.0, 20.0],
        }
    )


def test_apply_backward_ratio_adjustment_for_stock_daily() -> None:
    frame = normalize_frame(make_stock_daily_frame(), DatasetKind.STOCK_DAILY)

    adjusted, stats = apply_backward_ratio_adjustment(frame, DatasetKind.STOCK_DAILY)

    assert adjusted.loc[(pd.Timestamp("2024-01-02"), "000001.SZ"), "open"] == 20.0
    assert adjusted.loc[(pd.Timestamp("2024-01-03"), "000001.SZ"), "close"] == 22.47
    assert math.isclose(
        adjusted.loc[(pd.Timestamp("2024-01-02"), "000001.SZ"), "change"],
        0.6,
    )
    assert math.isclose(
        adjusted.loc[(pd.Timestamp("2024-01-02"), "000001.SZ"), "pct_chg"],
        (0.6 / 19.6) * 100,
    )
    assert stats["pre_close_invalid_count"] == 0


def test_apply_backward_ratio_adjustment_marks_invalid_pre_close() -> None:
    frame = make_stock_daily_frame()
    frame.loc[0, "pre_close"] = 0.0
    normalized = normalize_frame(frame, DatasetKind.STOCK_DAILY)

    adjusted, stats = apply_backward_ratio_adjustment(normalized, DatasetKind.STOCK_DAILY)

    assert pd.isna(adjusted.loc[(pd.Timestamp("2024-01-02"), "000001.SZ"), "pct_chg"])
    assert stats["pre_close_invalid_count"] == 1


def test_attach_adj_factor_for_auction_reports_join_coverage() -> None:
    auction = normalize_frame(make_auction_frame(), DatasetKind.STOCK_AUCTION_OPEN)
    stock_daily = normalize_frame(make_stock_daily_frame(), DatasetKind.STOCK_DAILY)

    joined, audit = attach_adj_factor_for_auction(auction, stock_daily[["adj_factor"]])

    assert len(joined) == 1
    assert audit["global"]["coverage_ratio"] == 0.5
    assert audit["global"]["rows_dropped"] == 1
    assert audit["dropped"]["reason_code"] == "missing_adj_factor_in_stock_daily"


def test_apply_backward_ratio_adjustment_for_auction_adjusts_vwap_only_after_join() -> None:
    auction = normalize_frame(make_auction_frame(), DatasetKind.STOCK_AUCTION_OPEN)
    stock_daily = normalize_frame(make_stock_daily_frame(), DatasetKind.STOCK_DAILY)
    joined, _ = attach_adj_factor_for_auction(auction, stock_daily[["adj_factor"]])

    adjusted, stats = apply_backward_ratio_adjustment(joined, DatasetKind.STOCK_AUCTION_OPEN)

    row = adjusted.iloc[0]
    assert row["open"] == 20.0
    assert row["vwap"] == 20.0
    assert row["vol"] == 100.0
    assert stats["pre_close_invalid_count"] == 0
