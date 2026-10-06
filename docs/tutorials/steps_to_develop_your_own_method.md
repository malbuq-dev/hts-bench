# Steps to develop your own method

Every HTSBench method implements `MethodBase` (`hts_bench/method/base.py`): a univariate, hierarchy-agnostic interface - it receives a series, returns a forecast. The method never sees the hierarchy or the summing matrix S; that's the Reconciliation module's responsibility, applied afterward, over the forecasts already generated.

This tutorial uses the **Drift** method as an example: a simple linear extrapolation between the first and last point of the training data (the same "Drift" method from Hyndman & Athanasopoulos, in the same family as `naive`/`seasonal_naive`). The code below was tested and runs end to end.

## 1. Create a file for your method

It doesn't need to live inside `hts_bench/`; any importable Python module works, since `method_factories` (the dictionary passed to `compare_methods`/`run_benchmark`) accepts any `Callable[[], MethodBase]`. For this tutorial, create `drift.py` anywhere in your project.

## 2. Inherit from `MethodBase`

```python
import numpy as np
import pandas as pd

from hts_bench.method.base import MethodBase


class Drift(MethodBase):
    def __init__(self):
        self._last_value = None
        self._slope = None
```

## 3. Implement `forecast_fit`

Receives the training series (a `pd.Series`, already sliced to exclude the test horizon) and must return `self`.

```python
    def forecast_fit(self, train_data: pd.Series) -> "Drift":
        n = len(train_data)
        self._last_value = train_data.iloc[-1]
        self._slope = (train_data.iloc[-1] - train_data.iloc[0]) / (n - 1)
        return self
```

## 4. Implement `forecast`

Receives the desired horizon and the same series used in `forecast_fit` (passed again explicitly, for methods that need it directly, such as lag-window ones) and must return a 1-D array of size `horizon`.

```python
    def forecast(self, horizon: int, series: pd.Series) -> np.ndarray:
        steps = np.arange(1, horizon + 1)
        return self._last_value + self._slope * steps
```

## 5. Implement the `name` property

```python
    @property
    def name(self) -> str:
        return "Drift"
```

## 6. (Optional) Implement `fitted_values`

Only needed if the method is used to estimate residuals for `min_trace_shrink` (see `shrinkage_covariance` in `reconciliation/reconcile.py`). If omitted, `MethodBase`'s default implementation raises `NotImplementedError` - correct for a method that, like `statsmodels`' Theta, has no natural concept of an in-sample fitted value. Drift does have one: the fitted line at each training point.

```python
    def fitted_values(self) -> pd.Series:
        n = len(self._train_data)
        steps = np.arange(n)
        return pd.Series(self._train_data.iloc[0] + self._slope * steps, index=self._train_data.index)
```

(This requires storing `train_data` in `forecast_fit` - `self._train_data = train_data` - omitted above for simplicity; see `hts_bench/method/naive.py` for the full pattern with `fitted_values`.)

## Putting it all together

```python
import numpy as np
import pandas as pd

from hts_bench.method.base import MethodBase


class Drift(MethodBase):
    def __init__(self):
        self._last_value = None
        self._slope = None

    def forecast_fit(self, train_data: pd.Series) -> "Drift":
        n = len(train_data)
        self._last_value = train_data.iloc[-1]
        self._slope = (train_data.iloc[-1] - train_data.iloc[0]) / (n - 1)
        return self

    def forecast(self, horizon: int, series: pd.Series) -> np.ndarray:
        steps = np.arange(1, horizon + 1)
        return self._last_value + self._slope * steps

    @property
    def name(self) -> str:
        return "Drift"
```

## 7. Use the method

There's no generic, configuration-driven CLI for this - `method_factories` is just a Python dictionary `{name: factory}`, used directly through the API:

```python
from hts_bench.pipeline import run_benchmark
from drift import Drift
from hts_bench.method.naive import Naive

result = run_benchmark(
    "labour",
    {"drift": Drift, "naive": Naive},
    horizon=8,
)
print(result)
```

Real output, obtained by running this exact code:

```
             mae       rmse      mase
method
drift    9.207740  10.871704  1.561329
naive   11.475307  13.525983  1.764651
```

To use your method from the CLI (`scripts/run_benchmark.py --methods ...`), add it to the `registry` dictionary inside `build_method_factories` in that script - the CLI only recognizes names already registered there (`naive`, `seasonal_naive`, `ets`, `arima`, `theta`, `lightgbm`).
