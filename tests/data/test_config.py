import pytest

from act_qt_v3.config import ReadOptions


@pytest.mark.parametrize("file_limit", [0, -1, 1.5, "2", True])
def test_read_options_rejects_non_positive_file_limit(file_limit: int) -> None:
    with pytest.raises(ValueError, match="file_limit must be a positive integer"):
        ReadOptions(file_limit=file_limit)


@pytest.mark.parametrize("file_limit", [None, 1, 10])
def test_read_options_accepts_positive_or_unlimited_file_limit(file_limit: int | None) -> None:
    assert ReadOptions(file_limit=file_limit).file_limit == file_limit
