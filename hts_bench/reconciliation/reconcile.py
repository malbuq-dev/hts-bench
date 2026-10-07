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
    return aggregate_from_bottom(ds.summing_matrix, forecasts)


def _root_series_id(ds: HierarchicalDataset) -> str:
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

    S = ds.summing_matrix
    missing = set(S.row_ids) - set(forecasts.columns)
    if missing:
        raise ValueError(
            "min_trace needs an independent forecast for every series in the hierarchy "
            f"({len(missing)} missing, e.g. {sorted(missing)[:3]}) - use "
            "run_forecast(..., series_ids=ds.summing_matrix.row_ids), not the bottom-only default."
        )

    if W is None:
        row_sums = np.asarray(S.matrix.sum(axis=1)).ravel()
        inv_w = 1.0 / row_sums
        weighted = S.matrix.multiply(inv_w[:, None]).tocsr()  
        StWinvS = (S.matrix.T @ weighted).toarray()  
        StWinv = S.matrix.T.multiply(inv_w[None, :]).toarray()  
    else:
        Sm = S.matrix.toarray()  
        Winv = np.linalg.inv(W)
        StWinv = Sm.T @ Winv 
        StWinvS = StWinv @ Sm 

    G = np.linalg.solve(StWinvS, StWinv) 

    y_hat = forecasts[S.row_ids].to_numpy()  
    b_tilde = y_hat @ G.T 
    reconciled_bottom = pd.DataFrame(b_tilde, index=forecasts.index, columns=S.col_ids)
    return aggregate_from_bottom(S, reconciled_bottom)


def shrinkage_covariance(residuals: pd.DataFrame) -> np.ndarray:
    x = residuals.to_numpy()
    T = x.shape[0]

    covm = (x.T @ x) / T  
    variances = np.diag(covm)
    target = np.diag(variances)

    std = np.sqrt(variances)
    corm = covm / np.outer(std, std)

    xs = x / std  
    xtx = xs.T @ xs
    cross_xs2 = (xs**2).T @ (xs**2)
    v = (cross_xs2 - (xtx**2) / T) / (T * (T - 1)) 
    np.fill_diagonal(v, 0.0)

    d = (corm - np.eye(len(variances))) ** 2  

    lam = float(np.clip(v.sum() / d.sum(), 0.0, 1.0))
    return lam * target + (1 - lam) * covm


def min_trace_shrink(
    ds: HierarchicalDataset, method_factory: Callable[[], MethodBase], horizon: int
) -> Tuple[Callable[[HierarchicalDataset, pd.DataFrame], pd.DataFrame], float]:
 
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
