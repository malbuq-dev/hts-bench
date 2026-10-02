import os

import pandas as pd
import pytest

from hts_bench.report.recording import load_records, save_record


@pytest.fixture
def comparison():
    return pd.DataFrame(
        {
            "method": ["Naive", "Naive"],
            "series_id": ["s1", "s2"],
            "level": ["bottom", "bottom"],
            "mae": [1.0, 2.0],
        }
    ).set_index(["method", "series_id"])


def test_save_record_writes_a_csv_with_index_as_columns(tmp_path, comparison):
    path = save_record(comparison, str(tmp_path), file_prefix="labour")

    assert os.path.exists(path)
    assert os.path.dirname(path) == str(tmp_path)
    reloaded = pd.read_csv(path)
    assert list(reloaded["method"]) == ["Naive", "Naive"]
    assert list(reloaded["mae"]) == [1.0, 2.0]


def test_save_record_creates_the_directory_if_missing(tmp_path, comparison):
    save_dir = tmp_path / "nested" / "result"
    save_record(comparison, str(save_dir), file_prefix="labour")

    assert save_dir.exists()
    assert len(list(save_dir.iterdir())) == 1


def test_load_records_from_a_single_file(tmp_path, comparison):
    path = save_record(comparison, str(tmp_path), file_prefix="labour")

    loaded = load_records([path])

    assert len(loaded) == 2
    assert set(loaded["method"]) == {"Naive"}


def test_load_records_from_a_directory_concatenates_all_csvs(tmp_path, comparison):
    save_record(comparison, str(tmp_path), file_prefix="labour")
    save_record(comparison, str(tmp_path), file_prefix="tourism")

    loaded = load_records([str(tmp_path)])

    assert len(loaded) == 4


def test_load_records_raises_when_nothing_found(tmp_path):
    with pytest.raises(ValueError, match="no record files"):
        load_records([str(tmp_path)])
