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
    naive_errors = np.abs(hist_data[seasonality:] - hist_data[:-seasonality])
    scale = naive_errors.mean()
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / scale)


METRICS = {"mae": mae, "rmse": rmse, "mase": mase}
