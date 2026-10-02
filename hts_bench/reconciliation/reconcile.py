import functools
import time
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from hts_bench.data.dataset import HierarchicalDataset
from hts_bench.data.loader import aggregate_from_bottom
from hts_bench.method.base import MethodBase
from hts_bench.method.runner import compute_residuals


def bottom_up(ds: HierarchicalDataset, forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Reconcile by summing raw bottom-level forecasts up through S - the cheapest
    coherent method, and the baseline every HTS paper compares others against.

    This has no separate implementation: aggregating actual bottom-level data up
    (loader.aggregate_from_bottom, used by data/coherence.py to check y_t = S @ b_t
    holds) and aggregating bottom-level *forecasts* up are the same operation on
    a different b. MinT/OLS reconciliation will need their own functions here -
    they reweight by forecast-error covariance, which bottom-up ignores entirely.
    """
    return aggregate_from_bottom(ds.summing_matrix, forecasts)


def _root_series_id(ds: HierarchicalDataset) -> str:
    """
    The one series with no hierarchy dimensions set at all - the same "root
    level" condition hierarchy.build_summing_matrix uses (every dimension
    column NaN for the whole level).

    Not "the row of S that sums to every bottom series": that's unreliable on
    a dataset subset, where a lower node (e.g. one store's worth of M5) can
    cover 100% of *that subset's* bottom series without being the semantic
    root - series_meta's dimension columns are the source of truth, S is
    derived from them.
    """
    dim_cols = [c for c in ds.series_meta.columns if c not in ("level", "is_bottom")]
    is_root = ~ds.series_meta[dim_cols].notna().any(axis=1)
    root_ids = ds.series_meta.index[is_root]
    if len(root_ids) != 1:
        raise ValueError(f"expected exactly one root series, found {len(root_ids)}: {list(root_ids)}")
    return root_ids[0]


def top_down(ds: HierarchicalDataset, forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Reconcile using average historical proportions (Gross & Sohl 1990, method A):
    each bottom series gets a fixed share of the top-level total, equal to its
    mean historical (bottom / total) ratio, computed on data strictly before
    `forecasts.index` so there's no test-period leakage.

    Textbook top-down when `forecasts` includes the root series' own forecast
    (pass series_ids=ds.summing_matrix.row_ids to run_forecast, same as
    min_trace needs): the total redistributed is then the root's genuinely
    independent forecast, not derived from the bottom ones - forecast the top,
    then disaggregate it, per the original method.

    Falls back to the bottom-up-implied total (S @ b_hat via
    aggregate_from_bottom) when the root isn't in `forecasts` - the bottom-only
    default run_forecast produces. In that case only the *split* across bottom
    series is genuinely top-down; the total itself is bottom-up's. This keeps
    every existing bottom-only caller working unchanged, while letting callers
    that already fit every level (for min_trace, say) get the textbook version
    for free from the same forecasts.
    """
    S = ds.summing_matrix
    root_id = _root_series_id(ds)

    hist = ds.data[ds.data.index < forecasts.index.min()]
    proportions = hist[S.col_ids].div(hist[root_id], axis=0).mean()

    if root_id in forecasts.columns:
        total_forecast = forecasts[root_id]
    else:
        total_forecast = aggregate_from_bottom(S, forecasts)[root_id]
    disaggregated = pd.DataFrame(
        np.outer(total_forecast.to_numpy(), proportions.to_numpy()),
        index=forecasts.index,
        columns=S.col_ids,
    )
    return aggregate_from_bottom(S, disaggregated)


def min_trace(
    ds: HierarchicalDataset, forecasts: pd.DataFrame, W: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    MinT reconciliation (Wickramasuriya, Athanasopoulos & Hyndman 2019):
    b_tilde = (S'W^-1 S)^-1 S'W^-1 y_hat, where y_hat holds an independent
    forecast for every series in the hierarchy.

    W defaults to structural scaling - each node's weight is the number of
    bottom series it aggregates (row sums of S, "WLSS" in Hyndman &
    Athanasopoulos' terminology), no residuals needed. Pass a real (n, n)
    forecast-error covariance matrix, ordered like ds.summing_matrix.row_ids in
    both dimensions, for the literature's actual MinT(shrink) - see
    shrinkage_covariance/min_trace_shrink below, which build one from in-sample
    residuals. Dense W is only practical up to a few hundred series (needs an
    (n, n) inverse); the structural default stays sparse throughout except the
    final (m, m) solve, so it scales further.

    Unlike bottom_up/top_down, `forecasts` here must hold every series in
    ds.summing_matrix.row_ids, not just the bottom - produced by
    run_forecast(ds, method_factory, horizon, series_ids=ds.summing_matrix.row_ids).
    Reconciliation only has something to do when levels were forecast
    independently and can disagree: every member of the MinT/GLS family
    satisfies G@S = I by construction, so if the aggregate columns were instead
    derived from the bottom ones (S @ b_hat, as bottom_up/top_down do), this
    collapses to exactly bottom_up - reconciling an already-coherent forecast
    is a no-op regardless of W.
    """
    S = ds.summing_matrix
    missing = set(S.row_ids) - set(forecasts.columns)
    if missing:
        raise ValueError(
            "min_trace needs an independent forecast for every series in the hierarchy "
            f"({len(missing)} missing, e.g. {sorted(missing)[:3]}) - use "
            "run_forecast(..., series_ids=ds.summing_matrix.row_ids), not the bottom-only default."
        )

    if W is None:
        row_sums = np.asarray(S.matrix.sum(axis=1)).ravel()  # structural weights, per row_id
        inv_w = 1.0 / row_sums
        weighted = S.matrix.multiply(inv_w[:, None]).tocsr()  # W^-1 S, (n, m)
        StWinvS = (S.matrix.T @ weighted).toarray()  # (m, m) dense - small enough to invert
        StWinv = S.matrix.T.multiply(inv_w[None, :]).toarray()  # S'W^-1, (m, n)
    else:
        Sm = S.matrix.toarray()  # dense throughout - W itself already is
        Winv = np.linalg.inv(W)
        StWinv = Sm.T @ Winv  # S'W^-1, (m, n)
        StWinvS = StWinv @ Sm  # (m, m)

    G = np.linalg.solve(StWinvS, StWinv)  # (m, n) = (S'W^-1S)^-1 S'W^-1

    y_hat = forecasts[S.row_ids].to_numpy()  # (T, n), independently forecast at every level
    b_tilde = y_hat @ G.T  # (T, m)
    reconciled_bottom = pd.DataFrame(b_tilde, index=forecasts.index, columns=S.col_ids)
    return aggregate_from_bottom(S, reconciled_bottom)


def shrinkage_covariance(residuals: pd.DataFrame) -> np.ndarray:
    """
    Schafer & Strimmer (2005) shrinkage-to-diagonal-target covariance estimator
    - the one Wickramasuriya, Athanasopoulos & Hyndman (2019) use for
    MinT(shrink), same algorithm as R's hts/FoReco `shrink.estim` (itself based
    on corpcor::cov.shrink).

    `residuals` must already be a common window with no NaNs: one column per
    series, one row per time point where every column has a fitted value (see
    method.runner.compute_residuals - callers typically pass
    compute_residuals(...).dropna()). Shrinks the *correlation* structure
    toward zero (independence) while leaving each series' own variance
    unchanged; the shrinkage intensity is estimated from the data itself
    (Ledoit-Wolf-style optimal intensity), not a free parameter - a hierarchy
    with genuinely correlated errors keeps most of that structure, while a
    noisy/small-sample estimate gets pulled toward diagonal.
    """
    x = residuals.to_numpy()
    T = x.shape[0]

    covm = (x.T @ x) / T  # uncentered: forecast errors are assumed zero-mean
    variances = np.diag(covm)
    target = np.diag(variances)

    std = np.sqrt(variances)
    corm = covm / np.outer(std, std)

    xs = x / std  # each column standardized by its own sample sd
    xtx = xs.T @ xs
    cross_xs2 = (xs**2).T @ (xs**2)
    v = (cross_xs2 - (xtx**2) / T) / (T * (T - 1))  # per-pair variance of the correlation estimate
    np.fill_diagonal(v, 0.0)

    d = (corm - np.eye(len(variances))) ** 2  # squared distance from the identity-correlation target

    lam = float(np.clip(v.sum() / d.sum(), 0.0, 1.0))
    return lam * target + (1 - lam) * covm


def min_trace_shrink(
    ds: HierarchicalDataset, method_factory: Callable[[], MethodBase], horizon: int
) -> Tuple[Callable[[HierarchicalDataset, pd.DataFrame], pd.DataFrame], float]:
    """
    Builds a ready-to-use reconcile_fn for MinT(shrink) - the actual MinT most
    of the literature means by that name, using shrinkage_covariance's
    data-driven W instead of min_trace's structural-scaling default.

    Needs `method_factory` and `horizon` up front (unlike every other
    reconcile_fn here, which only need ds/forecasts) because the covariance
    estimate comes from a second, independent fitting pass over every series
    in the hierarchy - method.runner.compute_residuals - not from the
    forecasts being reconciled. Call this once to get a plain reconcile_fn,
    then use it like bottom_up/top_down/min_trace everywhere else:

        reconcile_fn, setup_seconds = min_trace_shrink(ds, lambda: ETS(seasonal_period=12), horizon)
        forecasts, times = run_forecast(ds, method_factory, horizon, series_ids=ds.summing_matrix.row_ids)
        evaluate(ds, forecasts, horizon, reconcile_fn=reconcile_fn, times=times)

    method_factory must build a method whose fitted_values() is implemented
    (Naive, SeasonalNaive, ETS, ARIMA, LightGBM - not Theta, see
    statsmodels_adapter.py) - it's the model used to estimate residuals at
    every level, not necessarily the same method being scored.

    Returns (reconcile_fn, setup_seconds): setup_seconds times compute_residuals
    + shrinkage_covariance - the cost of building W, done once here rather than
    inside reconcile_fn (which just does min_trace's usual (S'W^-1S)^-1 S'W^-1
    solve - see min_trace's own per-call cost, captured separately by
    evaluate()'s reconcile_seconds). This setup cost is typically the larger of
    the two, and - unlike reconcile_seconds - isn't a per-method quantity: a
    caller sweeping several methods under the same min_trace_shrink builds it
    once and reuses the same reconcile_fn for all of them (see
    scripts/run_experiments.py), so attributing it to any one method would be
    misleading. Report it separately instead of folding it into a per-method
    results table.
    """
    start = time.perf_counter()
    residuals = compute_residuals(ds, method_factory, horizon, series_ids=ds.summing_matrix.row_ids)
    common = residuals.dropna()
    if len(common) < 2:
        raise ValueError(
            f"only {len(common)} time points have fitted values for every series after "
            "warm-up - not enough in-sample history to estimate a residual covariance"
        )

    W = shrinkage_covariance(common[ds.summing_matrix.row_ids])
    setup_seconds = time.perf_counter() - start
    return functools.partial(min_trace, W=W), setup_seconds
