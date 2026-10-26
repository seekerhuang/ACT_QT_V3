import pytest

from act_qt_v3.config import ReadOptions


@pytest.mark.parametrize("file_limit", [0, -1, 1.5, "2", True])
def test_read_options_rejects_non_positive_file_limit(file_limit: int) -> None:
    with pytest.raises(ValueError, match="file_limit must be a positive integer"):
        ReadOptions(file_limit=file_limit)


@pytest.mark.parametrize("file_limit", [None, 1, 10])
def test_read_options_accepts_positive_or_unlimited_file_limit(file_limit: int | None) -> None:
    assert ReadOptions(file_limit=file_limit).file_limit == file_limit


def test_read_options_rejects_reversed_date_range() -> None:
    with pytest.raises(ValueError, match="date_from must be on or before date_to"):
        ReadOptions(date_from="2024-01-03", date_to="2024-01-02")


@pytest.mark.parametrize("date_from,date_to", [
    ("2024-01-02", "2024-01-02"),
    ("2024-01-02", "2024-01-03"),
    (None, "2024-01-02"),
    ("2024-01-02", None),
])
def test_read_options_accepts_ordered_or_open_date_range(date_from, date_to) -> None:
    ReadOptions(date_from=date_from, date_to=date_to)
