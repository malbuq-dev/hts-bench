# Como avaliar sobre suas próprias séries

> Diferente do TFB, o HTSBench não trabalha com séries soltas: todo dataset precisa declarar uma **hierarquia** (quais séries são agregados de quais outras). Se você só tem uma série isolada, sem estrutura hierárquica nenhuma, veja a seção [Caso sem hierarquia](#caso-sem-hierarquia) ao final — é o caso trivial do formato abaixo.

## O formato de dataset

Um dataset é um diretório `dataset/<nome>/` com três arquivos:

| Arquivo | Conteúdo |
|---|---|
| `data.csv` | Índice `date`, uma coluna por série — **todos** os níveis, de nível-base aos agregados, lado a lado |
| `series_meta.csv` | Índice `series_id`, colunas `level`, `is_bottom` e uma coluna por dimensão da hierarquia |
| `meta.json` | `name`, `freq`, `horizon_suggested`, `n_series`, `n_bottom`, `data_files` |

Em `series_meta.csv`, uma série de nível agregado deixa em branco (`NaN`) as dimensões que ela não especifica. A matriz de agregação S é derivada automaticamente desse arquivo por `hts_bench/data/hierarchy.py` — você nunca escreve S à mão.

## Caso geral: você tem dados de nível-base e conhece a hierarquia

Esse é o caso mais comum na prática (é exatamente a situação do M5 — veja `scripts/convert_m5.py`): você tem os valores reais apenas para as séries de nível-base, e quer que o HTSBench calcule os agregados automaticamente a partir delas.

Usamos aqui a mesma hierarquia didática Região × Produto já usada nas demais figuras do projeto (CA/NY × Trousers/T-shirts). O código abaixo foi testado e roda de ponta a ponta.

### 1. Descreva a hierarquia em `series_meta`

```python
import pandas as pd

series_meta = pd.DataFrame(
    [
        {"series_id": "s0", "level": "Total", "is_bottom": False, "region": None, "product": None},
        {"series_id": "s1", "level": "Region", "is_bottom": False, "region": "CA", "product": None},
        {"series_id": "s2", "level": "Region", "is_bottom": False, "region": "NY", "product": None},
        {"series_id": "s3", "level": "Product", "is_bottom": False, "region": None, "product": "Trousers"},
        {"series_id": "s4", "level": "Product", "is_bottom": False, "region": None, "product": "T-shirts"},
        {"series_id": "s5", "level": "Region/Product", "is_bottom": True, "region": "CA", "product": "Trousers"},
        {"series_id": "s6", "level": "Region/Product", "is_bottom": True, "region": "CA", "product": "T-shirts"},
        {"series_id": "s7", "level": "Region/Product", "is_bottom": True, "region": "NY", "product": "Trousers"},
        {"series_id": "s8", "level": "Region/Product", "is_bottom": True, "region": "NY", "product": "T-shirts"},
    ]
).set_index("series_id")
```

### 2. Derive S e calcule os agregados a partir dos seus dados de nível-base

Troque `bottom_data` pelos seus valores reais (mesmas colunas que as séries `is_bottom=True` acima, uma coluna por `series_id`).

```python
from hts_bench.data.hierarchy import build_summing_matrix
from hts_bench.data.loader import aggregate_from_bottom

S = build_summing_matrix(series_meta)

# bottom_data: suas séries reais, índice de datas, uma coluna por série de nível-base.
full_data = aggregate_from_bottom(S, bottom_data)
```

`full_data` já sai com uma coluna por série — nível-base e agregados juntos, exatamente o que `data.csv` espera.

### 3. Escreva os três arquivos

```python
import json
import os

out_dir = "dataset/meu_dataset"
os.makedirs(out_dir, exist_ok=True)

full_data.to_csv(os.path.join(out_dir, "data.csv"))
series_meta.to_csv(os.path.join(out_dir, "series_meta.csv"))

meta = {
    "name": "meu_dataset",
    "freq": "D",                                   # ou "MS", conforme sua série
    "horizon_suggested": 5,
    "n_series": len(series_meta),
    "n_bottom": int(series_meta["is_bottom"].sum()),
    "data_files": ["data.csv"],
}
with open(os.path.join(out_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
```

### 4. Carregue e verifique a coerência

`check_coherence` confirma que y = S·b realmente vale em todo o `data.csv` escrito — útil tanto aqui quanto para pegar erros de digitação na hierarquia.

```python
from hts_bench.data.loader import load_dataset
from hts_bench.data.coherence import check_coherence

ds = load_dataset("meu_dataset")
violations = check_coherence(ds)
assert violations.empty, violations
```

### 5. Rode um benchmark

```python
from hts_bench.pipeline import run_benchmark
from hts_bench.method.naive import Naive, SeasonalNaive

result = run_benchmark(
    "meu_dataset",
    {"naive": Naive, "seasonal_naive": lambda: SeasonalNaive(seasonal_period=7)},
    horizon=5,
)
print(result)
```

Ou pelo CLI, sem escrever nenhum código:

```bash
python scripts/run_benchmark.py --dataset meu_dataset --methods naive seasonal_naive --horizon 5
```

## Caso sem hierarquia

Se você só tem uma série solta, sem agregados, ela é o caso degenerado do mesmo formato: uma única linha em `series_meta.csv`, `is_bottom=True`, sem nenhuma coluna de dimensão, e `n_series == n_bottom == 1`. `data.csv` tem só essa coluna. Nenhum passo de agregação é necessário.
