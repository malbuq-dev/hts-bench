import numpy as np

__all__ = ["mae", "rmse", "mase"]


def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def mase(
    actual: np.ndarray, predicted: np.ndarray, hist_data: np.ndarray, seasonality: int = 1, **kwargs
) -> float:
    """
    Mean Absolute Scaled Error - MAE scaled by the seasonal-naive MAE on training
    history. Same definition TFB uses (ts_benchmark/evaluation/metrics/
    regression_metrics.py:mase), rewritten as the standard closed form
    (mean(|actual-predicted|) / mean(|hist[t]-hist[t-seasonality]|)) instead of
    TFB's sum-and-rescale version - same result, without its length/seasonality
    off-by-one edge case.
    """
    naive_errors = np.abs(hist_data[seasonality:] - hist_data[:-seasonality])
    scale = naive_errors.mean()
    if scale == 0:
        # Flat training history (e.g. an all-zero M5 series) -> seasonal-naive
        # baseline has zero error itself, so the scaled ratio is undefined rather
        # than infinite/misleadingly-large. NaN, not a crash: one degenerate
        # series shouldn't take down a whole-hierarchy evaluate() run.
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / scale)


METRICS = {"mae": mae, "rmse": rmse, "mase": mase}
