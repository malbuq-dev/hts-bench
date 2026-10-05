# Como desenvolver seu próprio método

Todo método do HTSBench implementa `MethodBase` (`hts_bench/method/base.py`): uma interface univariada e agnóstica à hierarquia - recebe uma série, devolve uma previsão. O método nunca enxerga a hierarquia nem a matriz de somação S; isso é responsabilidade do módulo de Reconciliação, aplicado depois, sobre as previsões já geradas.

Este tutorial usa como exemplo o método **Drift**: uma extrapolação linear simples entre o primeiro e o último ponto do treino (o mesmo método "Drift" de Hyndman & Athanasopoulos, da mesma família de `naive`/`seasonal_naive`). O código abaixo foi testado e roda de ponta a ponta.

## 1. Crie um arquivo para o seu método

Não precisa viver dentro de `hts_bench/`; qualquer módulo Python importável serve, já que `method_factories` (o dicionário passado para `compare_methods`/`run_benchmark`) aceita qualquer `Callable[[], MethodBase]`. Para este tutorial, crie `drift.py` em qualquer lugar do seu projeto.

## 2. Herde de `MethodBase`

```python
import numpy as np
import pandas as pd

from hts_bench.method.base import MethodBase


class Drift(MethodBase):
    def __init__(self):
        self._last_value = None
        self._slope = None
```

## 3. Implemente `forecast_fit`

Recebe a série de treino (um `pd.Series`, já recortada para excluir o horizonte de teste) e deve retornar `self`.

```python
    def forecast_fit(self, train_data: pd.Series) -> "Drift":
        n = len(train_data)
        self._last_value = train_data.iloc[-1]
        self._slope = (train_data.iloc[-1] - train_data.iloc[0]) / (n - 1)
        return self
```

## 4. Implemente `forecast`

Recebe o horizonte desejado e a mesma série usada em `forecast_fit` (passada de novo explicitamente, para métodos que precisem dela diretamente, como os de janela de lag) e deve retornar um array 1-D de tamanho `horizon`.

```python
    def forecast(self, horizon: int, series: pd.Series) -> np.ndarray:
        steps = np.arange(1, horizon + 1)
        return self._last_value + self._slope * steps
```

## 5. Implemente a propriedade `name`

```python
    @property
    def name(self) -> str:
        return "Drift"
```

## 6. (Opcional) Implemente `fitted_values`

Só é necessário se o método for usado para estimar resíduos em `min_trace_shrink` (veja `reconciliation/reconcile.py`'s `shrinkage_covariance`). Se omitido, `MethodBase`'s implementação padrão levanta `NotImplementedError` - correto para um método que, como o Theta do `statsmodels`, não tem um conceito natural de valor ajustado in-sample. O Drift tem: a reta ajustada em cada ponto do treino.

```python
    def fitted_values(self) -> pd.Series:
        n = len(self._train_data)
        steps = np.arange(n)
        return pd.Series(self._train_data.iloc[0] + self._slope * steps, index=self._train_data.index)
```

(Isso exige guardar `train_data` em `forecast_fit` - `self._train_data = train_data` - omitido acima por simplicidade; veja `hts_bench/method/naive.py` para o padrão completo com `fitted_values`.)

## Juntando tudo

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

## 7. Use o método

Não existe um CLI genérico controlado por configuração - `method_factories` é só um dicionário Python `{nome: fábrica}`, usado diretamente na API:

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

Saída real, obtida rodando este exato código:

```
             mae       rmse      mase
method
drift    9.207740  10.871704  1.561329
naive   11.475307  13.525983  1.764651
```

Para usar seu método a partir do CLI (`scripts/run_benchmark.py --methods ...`), adicione-o ao dicionário `registry` dentro de `build_method_factories` nesse script - o CLI só reconhece os nomes já cadastrados ali (`naive`, `seasonal_naive`, `ets`, `arima`, `theta`, `lightgbm`).
