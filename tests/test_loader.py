import pandas as pd
import pytest

from hts_bench.data.loader import aggregate_from_bottom, load_dataset


def test_single_file_load(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta)
    ds = load_dataset("toy", root=root)

    pd.testing.assert_frame_equal(ds.data, toy_data, check_freq=False)
    assert ds.bottom_series == ["s1", "s2"]


def test_sharded_load_matches_single_file_load(write_dataset, toy_series_meta, toy_data):
    single_root = write_dataset("toy_single", toy_data, toy_series_meta)
    sharded_root = write_dataset(
        "toy_sharded",
        toy_data,
        toy_series_meta,
        data_files={"part1.csv": ["s0"], "part2.csv": ["s1", "s2"]},
    )

    single = load_dataset("toy_single", root=single_root)
    sharded = load_dataset("toy_sharded", root=sharded_root)

    pd.testing.assert_frame_equal(
        sharded.data.sort_index(axis=1), single.data.sort_index(axis=1)
    )


def test_load_dataset_raises_on_n_series_mismatch(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta, n_series=99)
    with pytest.raises(ValueError, match="series_meta.csv has"):
        load_dataset("toy", root=root)


def test_load_dataset_raises_on_n_bottom_mismatch(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta, n_bottom=1)
    with pytest.raises(ValueError, match="bottom rows"):
        load_dataset("toy", root=root)


def test_load_dataset_raises_on_series_missing_from_data(write_dataset, toy_series_meta, toy_data):
    incomplete_data = toy_data.drop(columns=["s2"])
    root = write_dataset("toy", incomplete_data, toy_series_meta)
    with pytest.raises(ValueError, match="missing from data"):
        load_dataset("toy", root=root)


def test_aggregate_from_bottom_reproduces_stored_total(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta)
    ds = load_dataset("toy", root=root)

    reconstructed = aggregate_from_bottom(ds.summing_matrix, ds.get_bottom_data())
    pd.testing.assert_series_equal(reconstructed["s0"], ds.data["s0"], check_names=False)
