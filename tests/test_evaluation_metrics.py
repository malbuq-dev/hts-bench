import numpy as np

from hts_bench.evaluation.metrics import mae, mase, rmse


def test_mae():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])
    assert mae(actual, predicted) == np.mean([2.0, 2.0, 3.0])


def test_rmse():
    actual = np.array([10.0, 20.0])
    predicted = np.array([12.0, 16.0])
    assert rmse(actual, predicted) == np.sqrt(np.mean([4.0, 16.0]))


def test_mase():
    actual = np.array([10.0, 12.0])
    predicted = np.array([11.0, 11.0])
    hist_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # seasonal-naive(1) error = 1.0 every step

    result = mase(actual, predicted, hist_data=hist_data, seasonality=1)

    assert result == np.mean([1.0, 1.0]) / 1.0


def test_mase_returns_nan_for_flat_history():
    actual = np.array([10.0, 12.0])
    predicted = np.array([11.0, 11.0])
    hist_data = np.array([5.0, 5.0, 5.0, 5.0])  # constant -> seasonal-naive error is 0 everywhere

    result = mase(actual, predicted, hist_data=hist_data, seasonality=1)

    assert np.isnan(result)
