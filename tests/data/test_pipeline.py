from pathlib import Path

import pandas as pd
import pytest

from act_qt_v3.config import (
    AdjustmentMethod,
    DatasetKind,
    PreprocessJobConfig,
    ProcessingMode,
    ReadOptions,
)
from act_qt_v3.preprocess.pipeline import _add_event_columns, run_preprocess_job


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


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


def make_stock_minute_frame(ts_code: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "trade_time": ["2024-01-02 09:30:00", "2024-01-02 09:31:00"],
            "ts_code": [ts_code, ts_code],
            "open": [10.0, 10.1],
            "high": [10.0, 10.2],
            "low": [9.9, 10.0],
            "close": [10.0, 10.2],
            "vol": [100.0, 120.0],
            "amount": [1000.0, 1224.0],
            "adj_factor": [2.0, 2.0],
        }
    )


def make_auction_frame(ts_code: str = "000001.SZ") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02"],
            "ts_code": [ts_code],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            "close": [10.0],
            "vol": [100.0],
            "amount": [1000.0],
            "vwap": [10.0],
        }
    )


def make_partial_auction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "ts_code": ["000001.SZ", "999999.SZ"],
            "open": [10.0, 20.0],
            "high": [10.1, 20.1],
            "low": [9.9, 19.9],
            "close": [10.0, 20.0],
            "vol": [100.0, 200.0],
            "amount": [1000.0, 4000.0],
            "vwap": [10.0, 20.0],
        }
    )


def make_near_threshold_stock_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [f"2024-01-{day:02d}" for day in range(1, 16)],
            "ts_code": ["000001.SZ"] * 15,
            "open": [10.0 + day for day in range(15)],
            "high": [10.2 + day for day in range(15)],
            "low": [9.8 + day for day in range(15)],
            "close": [10.1 + day for day in range(15)],
            "pre_close": [9.9 + day for day in range(15)],
            "change": [0.2] * 15,
            "pct_chg": [2.0] * 15,
            "vol": [1000.0] * 15,
            "amount": [10000.0] * 15,
            "up_limit": [11.0 + day for day in range(15)],
            "down_limit": [8.8 + day for day in range(15)],
            "adj_factor": [2.0] * 15,
        }
    )


def make_near_threshold_auction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [f"2024-01-{day:02d}" for day in range(1, 17)],
            "ts_code": ["000001.SZ"] * 16,
            "open": [10.0 + day for day in range(16)],
            "high": [10.2 + day for day in range(16)],
            "low": [9.8 + day for day in range(16)],
            "close": [10.1 + day for day in range(16)],
            "vol": [100.0] * 16,
            "amount": [1000.0] * 16,
            "vwap": [10.0 + day for day in range(16)],
        }
    )


def test_pipeline_returns_raw_default_view(tmp_path: Path) -> None:
    data_root = tmp_path / "DATA"
    write_parquet(make_stock_daily_frame(), data_root / "stock_daily.parquet")

    config = PreprocessJobConfig(
        data_root=data_root,
        dataset_kind=DatasetKind.STOCK_DAILY,
    )

    result = run_preprocess_job(config)

    assert result.metadata["mode"] == ProcessingMode.PIT_SAFE.value
    assert result.metadata["adjustment"] == AdjustmentMethod.RAW.value
    assert result.metadata["adjustment_applied"] is False
    assert result.data.loc[(pd.Timestamp("2024-01-02"), "000001.SZ"), "open"] == 10.0


def test_pipeline_populates_daily_event_ts(tmp_path: Path) -> None:
    data_root = tmp_path / "DATA"
    write_parquet(make_stock_daily_frame(), data_root / "stock_daily.parquet")

    config = PreprocessJobConfig(
        data_root=data_root,
        dataset_kind=DatasetKind.STOCK_DAILY,
    )

    result = run_preprocess_job(config)

    assert str(result.data["event_ts"].dtype) == "datetime64[ns]"
    assert result.data["event_ts"].isna().all()
    assert result.metadata["event_ts_semantics"] == "trade_date_only"
    assert result.metadata["time_precision"] == "date"
    assert result.data["event_date"].notna().all()


def test_pipeline_adjusts_stock_minute_and_builds_event_ts(tmp_path: Path) -> None:
    data_root = tmp_path / "DATA"
    write_parquet(make_stock_minute_frame("000001.SZ"), data_root / "stock_1min" / "000001.SZ.parquet")
    write_parquet(make_stock_minute_frame("000002.SZ"), data_root / "stock_1min" / "000002.SZ.parquet")

    config = PreprocessJobConfig(
        data_root=data_root,
        dataset_kind=DatasetKind.STOCK_1MIN,
        mode=ProcessingMode.RESEARCH_STANDARDIZED,
        adjustment=AdjustmentMethod.BACKWARD_RATIO,
        read_options=ReadOptions(
            instrument_whitelist=("000001.SZ", "000002.SZ"),
            date_from="2024-01-02",
            date_to="2024-01-02",
        ),
    )

    result = run_preprocess_job(config)

    assert len(result.data) == 4
    assert str(result.data["event_ts"].dtype) == "datetime64[ns, Asia/Shanghai]"
    assert result.metadata["time_precision"] == "minute"
    assert result.data.iloc[0]["open"] == 20.0
    assert result.quality_report["duplicate_key_count"] == 0


def test_pipeline_rejects_empty_filtered_result(tmp_path: Path) -> None:
    data_root = tmp_path / "DATA"
    write_parquet(make_stock_daily_frame(), data_root / "stock_daily.parquet")

    config = PreprocessJobConfig(
        data_root=data_root,
        dataset_kind=DatasetKind.STOCK_DAILY,
        read_options=ReadOptions(instrument_whitelist=("999999.SZ",)),
    )

    with pytest.raises(ValueError, match="empty dataset after applying filters"):
        run_preprocess_job(config)


def test_pipeline_rejects_empty_auction_join_result(tmp_path: Path) -> None:
    data_root = tmp_path / "DATA"
    write_parquet(make_stock_daily_frame(), data_root / "stock_daily.parquet")
    write_parquet(make_auction_frame("999999.SZ"), data_root / "stock_auction_o_daily.parquet")

    config = PreprocessJobConfig(
        data_root=data_root,
        dataset_kind=DatasetKind.STOCK_AUCTION_OPEN,
        mode=ProcessingMode.RESEARCH_STANDARDIZED,
        adjustment=AdjustmentMethod.BACKWARD_RATIO,
    )

    with pytest.raises(ValueError, match="empty dataset after joining adj_factor"):
        run_preprocess_job(config)


def test_pipeline_auction_metadata_tracks_stock_daily_dependency(tmp_path: Path) -> None:
    data_root = tmp_path / "DATA"
    write_parquet(make_stock_daily_frame(), data_root / "stock_daily.parquet")
    write_parquet(make_auction_frame(), data_root / "stock_auction_o_daily.parquet")

    config = PreprocessJobConfig(
        data_root=data_root,
        dataset_kind=DatasetKind.STOCK_AUCTION_OPEN,
        mode=ProcessingMode.RESEARCH_STANDARDIZED,
        adjustment=AdjustmentMethod.BACKWARD_RATIO,
    )

    result = run_preprocess_job(config)

    assert any(path.endswith("stock_daily.parquet") for path in result.metadata["source_paths"])
    assert any(path.endswith("stock_daily.parquet") for path in result.metadata["data_version"]["source_paths"])
    assert result.data["event_ts"].isna().all()
    assert result.metadata["event_ts_semantics"] == "trade_date_only"
    assert result.metadata["time_precision"] == "date"
    assert result.metadata["join_coverage_passed"] is True
    assert result.quality_report["join_audit"]["global"]["coverage_ratio"] == 1.0


def test_pipeline_rejects_partial_auction_join_coverage_below_threshold(tmp_path: Path) -> None:
    data_root = tmp_path / "DATA"
    write_parquet(make_stock_daily_frame(), data_root / "stock_daily.parquet")
    write_parquet(make_partial_auction_frame(), data_root / "stock_auction_o_daily.parquet")

    config = PreprocessJobConfig(
        data_root=data_root,
        dataset_kind=DatasetKind.STOCK_AUCTION_OPEN,
        mode=ProcessingMode.RESEARCH_STANDARDIZED,
        adjustment=AdjustmentMethod.BACKWARD_RATIO,
    )

    with pytest.raises(ValueError, match="join coverage below configured threshold"):
        run_preprocess_job(config)


def test_pipeline_accepts_default_join_coverage_near_real_close_ratio(tmp_path: Path) -> None:
    data_root = tmp_path / "DATA"
    write_parquet(make_near_threshold_stock_daily_frame(), data_root / "stock_daily.parquet")
    write_parquet(make_near_threshold_auction_frame(), data_root / "stock_auction_c_daily.parquet")

    config = PreprocessJobConfig(
        data_root=data_root,
        dataset_kind=DatasetKind.STOCK_AUCTION_CLOSE,
        mode=ProcessingMode.RESEARCH_STANDARDIZED,
        adjustment=AdjustmentMethod.BACKWARD_RATIO,
    )

    result = run_preprocess_job(config)

    assert result.metadata["join_coverage_passed"] is True
    assert result.quality_report["join_audit"]["global"]["coverage_ratio"] == 15 / 16


def test_pipeline_writes_idempotent_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "DATA"
    output_root = tmp_path / "output"
    write_parquet(make_stock_daily_frame(), data_root / "stock_daily.parquet")

    config = PreprocessJobConfig(
        data_root=data_root,
        dataset_kind=DatasetKind.STOCK_DAILY,
        write_output=True,
        output_root=output_root,
        read_options=ReadOptions(date_from="2024-01-02", date_to="2024-01-03"),
    )

    result = run_preprocess_job(config)
    expected_dir = (
        output_root
        / "stock_daily"
        / ProcessingMode.PIT_SAFE.value
        / AdjustmentMethod.RAW.value
        / "date_from=2024-01-02__date_to=2024-01-03"
    )

    assert (expected_dir / "part-00000.parquet").exists()
    assert (expected_dir / "metadata.json").exists()
    assert result.metadata["output_path"] == str(expected_dir)


@pytest.mark.parametrize("unit", ["s", "us", "ns"])
def test_minute_event_timestamp_preserves_time_and_uses_nanoseconds(unit):
    times = pd.DatetimeIndex(["2024-01-02 09:31:00"]).as_unit(unit)
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(["2024-01-02"]), times],
        names=["trade_date", "trade_time"],
    )
    source = pd.DataFrame({"ts_code": ["000001.SZ"]}, index=index)
    original = source.copy(deep=True)

    result = _add_event_columns(source, DatasetKind.STOCK_1MIN)

    assert str(result["event_ts"].dtype) == "datetime64[ns, Asia/Shanghai]"
    assert result["event_ts"].iloc[0] == pd.Timestamp("2024-01-02 09:31:00", tz="Asia/Shanghai")
    pd.testing.assert_frame_equal(source, original)
