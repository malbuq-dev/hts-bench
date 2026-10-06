# Steps to evaluate on your own time series

> HTSBench doesn't work with standalone series: every dataset needs to declare a **hierarchy** (which series are aggregates of which others). If you only have a single, isolated series with no hierarchical structure at all, see the [No-hierarchy case](#no-hierarchy-case) section at the end - it's the trivial case of the format below.

## The dataset format

A dataset is a `dataset/<name>/` directory with three files:

| File | Content |
|---|---|
| `data.csv` | `date` index, one column per series - **every** level, from bottom to aggregates, side by side |
| `series_meta.csv` | `series_id` index, columns `level`, `is_bottom`, and one column per hierarchy dimension |
| `meta.json` | `name`, `freq`, `horizon_suggested`, `n_series`, `n_bottom`, `data_files` |

In `series_meta.csv`, an aggregate-level series leaves blank (`NaN`) the dimensions it doesn't specify. The aggregation matrix S is derived automatically from this file by `hts_bench/data/hierarchy.py` - you never write S by hand.

## General case: you have bottom-level data and know the hierarchy

This is the most common case in practice (it's exactly the situation for M5_lite - see `scripts/convert_m5.py`): you only have the real values for the bottom-level series, and want HTSBench to compute the aggregates automatically from them.

Here we use the same Region × Product toy hierarchy already used in the project's other figures (CA/NY × Trousers/T-shirts). The code below was tested and runs end to end.

### 1. Describe the hierarchy in `series_meta`

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

### 2. Derive S and compute the aggregates from your bottom-level data

Swap `bottom_data` for your real values (same columns as the `is_bottom=True` series above, one column per `series_id`).

```python
from hts_bench.data.hierarchy import build_summing_matrix
from hts_bench.data.loader import aggregate_from_bottom

S = build_summing_matrix(series_meta)

# bottom_data: your real series, date-indexed, one column per bottom-level series.
full_data = aggregate_from_bottom(S, bottom_data)
```

`full_data` comes out with one column per series - bottom-level and aggregates together, exactly what `data.csv` expects.

### 3. Write the three files

```python
import json
import os

out_dir = "dataset/my_dataset"
os.makedirs(out_dir, exist_ok=True)

full_data.to_csv(os.path.join(out_dir, "data.csv"))
series_meta.to_csv(os.path.join(out_dir, "series_meta.csv"))

meta = {
    "name": "my_dataset",
    "freq": "D",                                   # or "MS", depending on your series
    "horizon_suggested": 5,
    "n_series": len(series_meta),
    "n_bottom": int(series_meta["is_bottom"].sum()),
    "data_files": ["data.csv"],
}
with open(os.path.join(out_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
```

### 4. Load and check coherence

`check_coherence` confirms that y = S·b actually holds across the whole `data.csv` you wrote - useful here and for catching typos in the hierarchy.

```python
from hts_bench.data.loader import load_dataset
from hts_bench.data.coherence import check_coherence

ds = load_dataset("my_dataset")
violations = check_coherence(ds)
assert violations.empty, violations
```

### 5. Run a benchmark

```python
from hts_bench.pipeline import run_benchmark
from hts_bench.method.naive import Naive, SeasonalNaive

result = run_benchmark(
    "my_dataset",
    {"naive": Naive, "seasonal_naive": lambda: SeasonalNaive(seasonal_period=7)},
    horizon=5,
)
print(result)
```

Or through the CLI, without writing any code:

```bash
python scripts/run_benchmark.py --dataset my_dataset --methods naive seasonal_naive --horizon 5
```

## No-hierarchy case

If you only have a single, standalone series with no aggregates, it's the degenerate case of the same format: a single row in `series_meta.csv`, `is_bottom=True`, no dimension columns at all, and `n_series == n_bottom == 1`. `data.csv` has just that one column. No aggregation step is needed.
