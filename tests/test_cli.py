import sys
from types import SimpleNamespace

import pytest

from act_qt_v3.cli import preprocess_data


@pytest.mark.parametrize("symbols,expected", [
    ("000001.SZ, 000002.SZ", ("000001.SZ", "000002.SZ")),
    (" ,000001.SZ,, ", ("000001.SZ",)),
    ("", ()),
    (None, ()),
])
def test_cli_trims_symbols_and_ignores_empty_entries(symbols, expected, tmp_path, monkeypatch, capsys):
    captured = []

    def run_job(config):
        captured.append(config)
        return SimpleNamespace(metadata={})

    args = ["act-qt-preprocess", "--data-root", str(tmp_path), "--dataset-kind", "stock_daily"]
    if symbols is not None:
        args.extend(["--symbols", symbols])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(preprocess_data, "run_preprocess_job", run_job)

    preprocess_data.main()

    assert captured[0].read_options.instrument_whitelist == expected
    assert capsys.readouterr().out.strip() == "{}"
