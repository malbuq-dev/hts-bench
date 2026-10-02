import os

import pandas as pd
import pytest

from hts_bench.data.loader import aggregate_from_bottom, load_dataset
from hts_bench.evaluation.reconcile import bottom_up, min_trace, top_down
from hts_bench.method.runner import run_forecast
from hts_bench.method.statsmodels_adapter import ETS


@pytest.fixture
def toy_ds(write_dataset, toy_series_meta, toy_data):
    root = write_dataset("toy", toy_data, toy_series_meta)
    return load_dataset("toy", root=root)


@pytest.fixture
def toy_forecasts(toy_ds):
    # One forecast day (2020-01-03), independent of each bottom series' own history.
    return pd.DataFrame({"s1": [60.0], "s2": [20.0]}, index=toy_ds.data.index[-1:])


def test_bottom_up_matches_aggregate_from_bottom(toy_ds, toy_forecasts):
    reconciled = bottom_up(toy_ds, toy_forecasts)
    expected = aggregate_from_bottom(toy_ds.summing_matrix, toy_forecasts)
    pd.testing.assert_frame_equal(reconciled, expected)


def test_top_down_splits_by_average_historical_proportions(toy_ds, toy_forecasts):
    # History (days 1-2): north=[50,51], south=[30,29], total=[80,80].
    # proportions: north = mean(50/80, 51/80) = 0.63125, south = mean(30/80, 29/80) = 0.36875
    reconciled = top_down(toy_ds, toy_forecasts)

    assert reconciled.loc[toy_forecasts.index[0], "s1"] == pytest.approx(80.0 * 0.63125)
    assert reconciled.loc[toy_forecasts.index[0], "s2"] == pytest.approx(80.0 * 0.36875)


def test_top_down_preserves_the_bottom_up_implied_total(toy_ds, toy_forecasts):
    # top_down only changes the split across bottom series, not the root total -
    # there's no independent top-level forecast to change it (see reconcile.py).
    bu = bottom_up(toy_ds, toy_forecasts)
    td = top_down(toy_ds, toy_forecasts)

    assert td["s0"].iloc[0] == pytest.approx(bu["s0"].iloc[0])


def test_top_down_is_coherent(toy_ds, toy_forecasts):
    reconciled = top_down(toy_ds, toy_forecasts)
    assert reconciled["s0"].iloc[0] == pytest.approx(
        reconciled["s1"].iloc[0] + reconciled["s2"].iloc[0]
    )


def test_top_down_uses_an_independent_root_forecast_when_given_one(toy_ds, toy_forecasts):
    # Textbook top-down: when the root is forecast independently (not just
    # derived from the bottom series), that's the total redistributed - not
    # the bottom-up-implied one.
    with_root = toy_forecasts.copy()
    with_root["s0"] = 999.0  # deliberately not s1 + s2 (=80)

    reconciled = top_down(toy_ds, with_root)

    assert reconciled["s0"].iloc[0] == pytest.approx(999.0)
    assert reconciled["s1"].iloc[0] == pytest.approx(999.0 * 0.63125)
    assert reconciled["s2"].iloc[0] == pytest.approx(999.0 * 0.36875)
    assert reconciled["s0"].iloc[0] == pytest.approx(
        reconciled["s1"].iloc[0] + reconciled["s2"].iloc[0]
    )


def test_min_trace_rejects_bottom_only_forecasts(toy_ds, toy_forecasts):
    with pytest.raises(ValueError, match="independent forecast for every series"):
        min_trace(toy_ds, toy_forecasts)


def test_min_trace_matches_bottom_up_when_input_is_already_coherent(toy_ds, toy_forecasts):
    # If the aggregate columns are derived from the bottom ones (S @ b_hat, same
    # as bottom_up/top_down get), MinT's projection is provably a no-op - see
    # min_trace's docstring. Sanity-checks that theoretical property directly.
    all_forecasts = aggregate_from_bottom(toy_ds.summing_matrix, toy_forecasts)

    reconciled = min_trace(toy_ds, all_forecasts)

    pd.testing.assert_frame_equal(reconciled[["s1", "s2"]], toy_forecasts, check_exact=False)


def test_min_trace_blends_disagreeing_levels(toy_ds):
    # Deliberately incoherent: top says 100, bottom says 60+20=80. Structural
    # scaling here (weights [2,1,1] for root/north/south - the row sums of S)
    # works out to G = [[0.25,0.75,-0.25],[0.25,-0.25,0.75]] (rows s1,s2, cols
    # s0,s1,s2), hand-derived from (S'W^-1S)^-1 S'W^-1 - see reconcile.py.
    all_forecasts = pd.DataFrame(
        {"s0": [100.0], "s1": [60.0], "s2": [20.0]}, index=toy_ds.data.index[-1:]
    )

    reconciled = min_trace(toy_ds, all_forecasts)

    assert reconciled["s1"].iloc[0] == pytest.approx(65.0)
    assert reconciled["s2"].iloc[0] == pytest.approx(25.0)
    assert reconciled["s0"].iloc[0] == pytest.approx(90.0)  # blend of 100 and 80, not either raw value


@pytest.mark.parametrize("name", ["labour", "tourism"])  # m5's ~3k-bottom S'W^-1S inversion is too slow for tests
def test_min_trace_is_coherent_on_real_datasets_with_a_nonlinear_method(name):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")
    ds = load_dataset(name)
    horizon = 4
    seasonal_period = 12

    # ETS is nonlinear, so independently-fit forecasts at each level genuinely
    # disagree - unlike a linear method (e.g. SeasonalNaive), which stays
    # coherent on its own and would make this indistinguishable from bottom_up.
    forecasts = run_forecast(
        ds, lambda: ETS(seasonal_period=seasonal_period), horizon, series_ids=ds.summing_matrix.row_ids
    )

    reconciled = min_trace(ds, forecasts)

    S = ds.summing_matrix
    re_aggregated = aggregate_from_bottom(S, reconciled[S.col_ids])
    pd.testing.assert_frame_equal(reconciled[S.row_ids], re_aggregated, check_exact=False)


@pytest.mark.parametrize("name", ["labour", "tourism", "m5"])
@pytest.mark.parametrize("reconcile_fn", [bottom_up, top_down], ids=["bottom_up", "top_down"])
def test_reconciliation_is_coherent_on_real_datasets(name, reconcile_fn):
    if not os.path.exists(os.path.join("dataset", name, "meta.json")):
        pytest.skip(f"dataset/{name} not converted yet - run scripts/convert_{name}.py")
    ds = load_dataset(name)
    horizon = 4
    bottom = ds.get_bottom_data()
    train, test_dates = bottom.iloc[:-horizon], bottom.index[-horizon:]
    forecasts = pd.DataFrame(
        {sid: train[sid].iloc[-1] for sid in ds.bottom_series},
        index=test_dates,
    )

    reconciled = reconcile_fn(ds, forecasts)

    # Coherence: every series in `reconciled` must equal what re-aggregating its
    # own reconciled bottom-level values through S would give - i.e. y_t = S @ b_t
    # holds for the reconciled output itself, not just before/after separately.
    S = ds.summing_matrix
    re_aggregated = aggregate_from_bottom(S, reconciled[S.col_ids])
    pd.testing.assert_frame_equal(reconciled[S.row_ids], re_aggregated, check_exact=False)
