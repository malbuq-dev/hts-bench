import os

import numpy as np
import pandas as pd
import pytest

from hts_bench.data.loader import aggregate_from_bottom, load_dataset
from hts_bench.reconciliation.reconcile import min_trace_shrink, shrinkage_covariance
from hts_bench.method.naive import Naive
from hts_bench.method.statsmodels_adapter import ETS


def test_shrinkage_covariance_is_symmetric_and_positive_definite():
    rng = np.random.default_rng(0)
    mixing = np.array([[1, 0.3, 0], [0.3, 1, 0.2], [0, 0.2, 1]])
    x = rng.standard_normal((30, 3)) @ mixing
    residuals = pd.DataFrame(x)

    W = shrinkage_covariance(residuals)

    assert np.allclose(W, W.T)
    assert (np.linalg.eigvalsh(W) > 0).all()


def test_shrinkage_covariance_preserves_variances():
    rng = np.random.default_rng(1)
    residuals = pd.DataFrame(rng.standard_normal((25, 4)))

    W = shrinkage_covariance(residuals)
    sample_variance = (residuals.to_numpy() ** 2).mean(axis=0)

    np.testing.assert_allclose(np.diag(W), sample_variance)


def test_shrinkage_covariance_shrinks_off_diagonal_toward_zero_with_little_data():
    # T close to n -> a noisy, near-singular sample covariance - the case
    # shrinkage exists for. Off-diagonal entries should move toward 0.
    rng = np.random.default_rng(2)
    x = rng.standard_normal((6, 5))
    residuals = pd.DataFrame(x)

    W = shrinkage_covariance(residuals)
    sample_cov = (x.T @ x) / 6

    off_diag = ~np.eye(5, dtype=bool)
    assert np.abs(W[off_diag]).sum() < np.abs(sample_cov[off_diag]).sum()


@pytest.fixture
def toy_ds_long(write_dataset, toy_series_meta):
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    rng = np.random.default_rng(42)
    north = 50 + np.arange(20) * 0.5 + rng.standard_normal(20)
    south = 30 - np.arange(20) * 0.2 + rng.standard_normal(20)
    data = pd.DataFrame({"s0": north + south, "s1": north, "s2": south}, index=dates)
    data.index.name = "date"
    root = write_dataset("toy_long", data, toy_series_meta)
    return load_dataset("toy_long", root=root)


def test_min_trace_shrink_raises_with_too_little_history(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta)
    ds = load_dataset("toy", root=root)

    with pytest.raises(ValueError, match="not enough in-sample history"):
        min_trace_shrink(ds, Naive, horizon=1)


def test_min_trace_shrink_reconciles_disagreeing_levels_coherently(toy_ds_long):
    reconcile_fn = min_trace_shrink(toy_ds_long, Naive, horizon=2)

    incoherent = pd.DataFrame(
        {"s0": [100.0], "s1": [60.0], "s2": [20.0]}, index=toy_ds_long.data.index[-1:]
    )
    reconciled = reconcile_fn(toy_ds_long, incoherent)

    assert reconciled["s0"].iloc[0] == pytest.approx(
        reconciled["s1"].iloc[0] + reconciled["s2"].iloc[0]
    )
    # A real blend happened - not just the raw bottom values passed through.
    assert reconciled["s1"].iloc[0] != pytest.approx(60.0)


@pytest.mark.parametrize("name", ["labour", "tourism"])  # same runtime reasoning as min_trace's m5 skip
def test_min_trace_shrink_is_coherent_on_real_datasets(name):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")
    ds = load_dataset(name)
    horizon = 4
    method_factory = lambda: ETS(seasonal_period=12)  # noqa: E731

    from hts_bench.method.runner import run_forecast

    forecasts = run_forecast(ds, method_factory, horizon, series_ids=ds.summing_matrix.row_ids)
    reconcile_fn = min_trace_shrink(ds, method_factory, horizon)

    reconciled = reconcile_fn(ds, forecasts)

    S = ds.summing_matrix
    re_aggregated = aggregate_from_bottom(S, reconciled[S.col_ids])
    pd.testing.assert_frame_equal(reconciled[S.row_ids], re_aggregated, check_exact=False)
