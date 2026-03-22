from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from act_qt_v3.config import DatasetKind
from act_qt_v3.preprocess.schema import get_schema_contract


def _maybe_round(frame: pd.DataFrame, columns: tuple[str, ...], rounding: int | None) -> None:
    if rounding is None:
        return
    frame.loc[:, list(columns)] = frame.loc[:, list(columns)].round(rounding)


def attach_adj_factor_for_auction(
    auction_frame: pd.DataFrame,
    stock_daily_adj_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "adj_factor" not in stock_daily_adj_frame.columns:
        raise ValueError("stock_daily_adj_frame must contain adj_factor")

    before_rows = int(len(auction_frame))
    joined = auction_frame.join(stock_daily_adj_frame[["adj_factor"]], how="inner")
    after_rows = int(len(joined))
    dropped_index = auction_frame.index.difference(stock_daily_adj_frame.index)

    before_by_date = auction_frame.groupby(level="trade_date").size()
    after_by_date = joined.groupby(level="trade_date").size()
    before_by_symbol = auction_frame.groupby(level="ts_code").size()
    after_by_symbol = joined.groupby(level="ts_code").size()

    def _build_coverage(before: pd.Series, after: pd.Series) -> dict[str, dict[str, float | int]]:
        payload: dict[str, dict[str, float | int]] = {}
        for key, before_count in before.items():
            after_count = int(after.get(key, 0))
            label = key.strftime("%Y-%m-%d") if hasattr(key, "strftime") else str(key)
            payload[label] = {
                "before": int(before_count),
                "after": after_count,
                "coverage_ratio": after_count / int(before_count) if before_count else 0.0,
            }
        return payload

    audit = {
        "global": {
            "before_rows": before_rows,
            "after_rows": after_rows,
            "rows_dropped": before_rows - after_rows,
            "coverage_ratio": after_rows / before_rows if before_rows else 0.0,
        },
        "by_trade_date": _build_coverage(before_by_date, after_by_date),
        "by_ts_code": _build_coverage(before_by_symbol, after_by_symbol),
        "dropped": {
            "reason_code": "missing_adj_factor_in_stock_daily",
            "rows_dropped": before_rows - after_rows,
            "dropped_ratio": (before_rows - after_rows) / before_rows if before_rows else 0.0,
            "dropped_keys_sample": [
                (index[0].strftime("%Y-%m-%d"), index[1]) for index in list(dropped_index[:10])
            ],
        },
    }
    return joined, audit


def apply_backward_ratio_adjustment(
    frame: pd.DataFrame,
    dataset_kind: DatasetKind,
    *,
    rounding: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = get_schema_contract(dataset_kind)
    if "adj_factor" not in frame.columns:
        raise ValueError("adj_factor is required for backward adjustment")

    adjusted = frame.copy()
    price_columns = [column for column in contract.price_columns if column in adjusted.columns]
    adjusted.loc[:, price_columns] = adjusted.loc[:, price_columns].mul(adjusted["adj_factor"], axis=0)
    _maybe_round(adjusted, tuple(price_columns), rounding)

    stats: dict[str, Any] = {"pre_close_invalid_count": 0}
    if dataset_kind == DatasetKind.STOCK_DAILY:
        adjusted["change"] = adjusted["close"] - adjusted["pre_close"]
        invalid = adjusted["pre_close"].isna() | (adjusted["pre_close"] <= 0)
        stats["pre_close_invalid_count"] = int(invalid.sum())
        adjusted["pct_chg"] = np.where(
            invalid,
            np.nan,
            adjusted["change"] / adjusted["pre_close"] * 100,
        )
        if rounding is not None:
            adjusted["change"] = adjusted["change"].round(rounding)
            adjusted["pct_chg"] = adjusted["pct_chg"].round(rounding)

    return adjusted, stats

