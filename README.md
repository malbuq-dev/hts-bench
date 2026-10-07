<div align="center">
  <img src="docs/assets/HTSBench-logo-light.svg" alt="HTSBench" width="220" />
  <p><b>A Benchmarking Platform for Hierarchical Time Series</b></p>
  <p>
    <a href="https://malbuq-dev.github.io/hts-bench/leaderboard.html">HTSBench</a>
  </p>
</div>

<p align="center">English | <a href="README.pt-br.md">🇧🇷 Português</a></p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue)

![Docker](https://img.shields.io/badge/Docker-ready-blue)

![Tests](https://img.shields.io/badge/pytest-automated%20suite-green)

</div>


## Table of contents

1. [Introduction](#introduction)

2. [Datasets](#datasets)

3. [Forecasting methods](#forecasting-methods)

4. [Hierarchical reconciliation](#hierarchical-reconciliation)

5. [Metrics](#metrics)

6. [Installation](#installation)

7. [Quickstart](#quickstart)

8. [Reproducing a full experiment](#reproducing-a-full-experiment)

9. [Updating the leaderboard](#updating-the-leaderboard)

10. [Preparing the data](#preparing-the-data)

11. [Extending the platform](#extending-the-platform)

12. [Tests](#tests)

13. [Project structure](#project-structure)

14. [Acknowledgments](#acknowledgments)

15. [Contact](#contact)

## Introduction

HTSBench evaluates, end to end, forecasting methods over time series organized in a **hierarchy**: lower-level series (e.g., sales per store) that sum up into aggregated series (e.g., sales per state, total sales). The platform runs each method independently per series, applies a **reconciliation** strategy to make the full set of forecasts coherent with the hierarchy (the sum of the parts matches the whole), and computes error metrics that are comparable across methods and datasets.

The figure below summarizes the flow: the five components on the left (Data, Methods, Reconciliation, Evaluation, Reports) and the execution sequence they implement on the right.

<div align="center">

<img alt="HTSBench pipeline" src="docs/figures/pipeline.png" width="85%"/>

</div>

In broad strokes:

- **Data** loads a hierarchical dataset (`dataset/<name>/`) and derives the summing matrix S that describes the hierarchy from `series_meta.csv` - no hierarchy is hand-coded anywhere else in the codebase.

- **Methods** implement a single, univariate interface (`MethodBase`); every method is fit and forecast series by series, never seeing the hierarchy.

- **Reconciliation** takes the raw forecasts for every series and produces a set coherent with the matrix S (the sum of the parts matches the whole), through one of four strategies (`bottom_up`, `top_down`, `min_trace`, `min_trace_shrink`).

- **Evaluation** orchestrates forecasting series by series, applies the chosen reconciliation strategy, and computes the metrics, with support for single-origin evaluation or multi-origin evaluation (**rolling-origin**).

- **Reports** aggregates the per-series result into a single table per method (a **leaderboard**) and can persist both the leaderboard and the raw per-series table for later reprocessing without re-running any model.

## Datasets

| Dataset   | Frequency | Series (bottom level) | Suggested horizon | Hierarchy | Source |
|-----------|:----------:|:--------------------:|:-------------------:|:----------:|-------|
| `labour`  | Monthly     | 57 (32)               | 8                    | Crossed (region × gender × employment status) | Australian Labour Force, via [Nixtla `datasetsforecast`](https://github.com/Nixtla/datasetsforecast) |
| `tourism` | Monthly     | 555 (304)              | 24                   | Crossed (geography × travel purpose) | Australian Tourism (Athanasopoulos et al.), via `datasetsforecast` |
| `traffic` | Daily     | 207 (200)               | 1                    | Tree-structured | San Francisco Traffic, Rangapuram et al., **End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series**, ICML 2021 (PMLR 139:8832–8843) |
| `wiki2`   | Daily     | 199 (150)               | 1                    | Tree-structured | Wikipedia page views, same source as `traffic` above |
| `m5` (M5_lite) | Daily     | 9,180 (3,049)¹          | 28                   | Crossed (state × store × category × department × item) | M5 Forecasting Competition, via `datasetsforecast.m5` (mirror of the original Kaggle files, no account needed) |

¹ By default `m5` is converted for store `CA_1` only - referred to throughout this README as **M5_lite**, to keep it clearly distinct from the full M5 competition dataset - to keep the build fast; `python scripts/convert_m5.py --store all` rebuilds the full M5 dataset (~30,490 bottom-level series), using the exact same code.

Each dataset is a `dataset/<name>/` directory with three files: `data.csv` (one column per series, all sharing a single date index), `series_meta.csv` (one row per series, with the hierarchy's dimension columns) and `meta.json` (frequency, suggested horizon, series counts). See [Extending the platform](#extending-the-platform) to add a new dataset.

## Forecasting methods

Every method implements the `MethodBase` interface (`hts_bench/method/base.py`): univariate, hierarchy-agnostic - it receives a series, returns a forecast. Reconciliation is the Reconciliation module's responsibility, not the method's.

| Method | Description | Library |
|---|---|---|
| `naive` | Repeats the last observed value | custom implementation |
| `seasonal_naive` | Repeats the value from the same point in the previous seasonal cycle | custom implementation |
| `ets` | Exponential smoothing, with a seasonal component | `statsmodels` |
| `arima` | ARIMA | `statsmodels` |
| `theta` | Theta method | `statsmodels` |
| `lightgbm` | Gradient boosting over a window of **lags**, per series, with recursive forecasting | `lightgbm` |

`lightgbm` is a **local** variant (one model per series) - not the "global" LightGBM (a single model trained across every series in a dataset) that won the M5 competition. Its seed (`random_state=42`) is fixed by default so that re-running the same experiment doesn't change the reported numbers.

`theta` doesn't implement `fitted_values()` (statsmodels' Theta result object doesn't expose in-sample fitted values), so it can't be used as a residual estimator for `min_trace_shrink` - the other methods can.

## Hierarchical reconciliation

Given a set of forecasts (possibly incoherent with one another), reconciliation produces a set coherent with the hierarchy's summing matrix S (y = S·b).

| Strategy | Idea |
|---|---|
| `bottom_up` | Sums the bottom-level forecasts through S. The standard baseline in the literature. |
| `top_down` | Disaggregates the top total by average historical proportions (Gross & Sohl, 1990). When an independent forecast for the root series is available, it's used as the total to disaggregate (textbook version); otherwise, the total falls back to the value implied by bottom-up. |
| `min_trace` | MinT (Wickramasuriya, Athanasopoulos & Hyndman, 2019), with structural weighting (WLSS, based on the number of bottom series each node aggregates) by default. Requires an independent forecast for **every** level of the hierarchy, not just the bottom level. |
| `min_trace_shrink` | MinT(shrink): same formulation as `min_trace`, but with the covariance matrix estimated via **shrinkage** (Schäfer & Strimmer, 2005) from the in-sample residuals of an auxiliary method. Not applied to `m5` (M5_lite) in the default **sweep** - inverting a dense (n×n) covariance stops being practical at M5_lite's scale, a limitation also present in the original RHiOTS paper, which subsampled the full M5 dataset for the same reason. |

## Metrics

`mae`, `rmse`, and `mase` (Mean Absolute Scaled Error - mean absolute error scaled by the error of the seasonal **naive** method over the training history, which makes it comparable across series of different scales and is notably safer than MAPE on series with zero values, as in part of M5_lite).

## Installation

Dependencies pinned to fixed versions for reproducibility (see `requirements.txt`).

```bash
pip install -r requirements-dev.txt
```

### Docker

```bash
docker build -t hts-bench:latest .
docker run --rm -v "$(pwd)/result:/app/result" hts-bench:latest \
  --dataset labour --methods naive seasonal_naive --horizon 8 --records-dir result
```

The image runs the full test suite as part of the build - if any test fails, the image isn't built. The `ENTRYPOINT` is `scripts/run_benchmark.py` itself, so any argument passed to `docker run` after the image name goes straight to the CLI.

## Quickstart

```bash
python scripts/run_benchmark.py \
  --dataset labour \
  --methods naive seasonal_naive ets \
  --horizon 8 \
  --reconcile bottom_up
```

For multi-origin evaluation (more robust than a single train/test split):

```bash
python scripts/run_benchmark.py \
  --dataset labour --methods naive seasonal_naive ets \
  --horizon 8 --n-origins 5 --reconcile min_trace --by-level
```

`--reconcile min_trace` automatically implies independent forecasting at every level of the hierarchy (the CLI handles this); `--by-level` breaks the leaderboard down by hierarchy level instead of collapsing everything into a single row per method.

## Reproducing a full experiment

`scripts/run_experiments.py` sweeps every relevant dataset, method, and reconciliation strategy combination (skipping `min_trace_shrink` for `m5`, for the reason already described), saving the raw per-series table for each run to `result/`:

```bash
python scripts/run_experiments.py
```

`scripts/analyze_experiments.py` then reads everything saved under `result/` - without refitting any model - and produces the comparison views used in the results analysis: method × dataset, comparison between reconciliation strategies, reconciliation benefit by hierarchy depth, and grouping by hierarchy type (crossed vs. tree-structured):

```bash
python scripts/analyze_experiments.py
```

## Updating the leaderboard

The [live leaderboard page](https://malbuq-dev.github.io/hts-bench/leaderboard.html) (`docs/leaderboard.html`) doesn't read `result/` directly - it fetches a pre-aggregated `docs/data/leaderboard.json`, built by a separate export step:

```bash
python scripts/export_leaderboard_json.py
```

This reads everything under `result/` (same as `analyze_experiments.py`, without refitting any model) and overwrites `docs/data/leaderboard.json` with one aggregated row per `(dataset, reconcile, method)`. Commit and push the updated JSON and the live page picks it up automatically on the next load - `docs/leaderboard.html` fetches it client-side, and GitHub Pages serves straight from the `/docs` folder on `main`, so there's no separate build or deploy step.

One thing to know if you've added a new dataset or reconciliation strategy: `scripts/export_leaderboard_json.py` has the datasets and reconciliation strategies it looks for hardcoded at the top of the file (`DATASETS`, `RECONCILE_CHOICES`). A new one won't show up in the export until it's added to those two lists too, even if `result/` already has the data for it.

## Preparing the data

The already-converted datasets live in `dataset/`; the scripts below regenerate each one from its original source, if needed:

```bash
python scripts/convert_labour.py
python scripts/convert_tourism.py
python scripts/convert_traffic.py
python scripts/convert_wiki2.py
python scripts/convert_m5.py            # CA_1 store only (M5_lite, default)
python scripts/convert_m5.py --store all  # full M5 dataset
```

## Extending the platform

**New method**: implement `MethodBase` (`hts_bench/method/base.py`) - `forecast_fit`, `forecast`, the `name` property, and, optionally, `fitted_values()` (only needed if the method is used to estimate residuals for `min_trace_shrink`). Full tutorial, building a real method step by step: [docs/tutorials/steps_to_develop_your_own_method.md](docs/tutorials/steps_to_develop_your_own_method.md).

**New dataset**: create `dataset/<name>/` with:

- `data.csv` - `date` index, one column per series (every level, including aggregates);

- `series_meta.csv` - `series_id` index, columns `level`, `is_bottom`, and one column per hierarchy dimension (an aggregate-level series leaves `NaN` in the dimensions it doesn't specify);

- `meta.json` - `name`, `freq`, `horizon_suggested`, `n_series`, `n_bottom`, `data_files`.

Full tutorial, including how to derive aggregates automatically from bottom-level data: [docs/tutorials/steps_to_evaluate_your_own_time_series.md](docs/tutorials/steps_to_evaluate_your_own_time_series.md).

## Tests

```bash
pytest tests/ -q
```

Covers data (loading, summing matrix, coherence checking), methods (including an end-to-end integration test per method), reconciliation (including numerical coherence on real datasets), and the reports module (persistence and **leaderboard**). Tests that depend on the full M5 dataset skip automatically when only the default subset is available.

## Project structure

```
hts_bench/
  data/           # dataset loading, summing matrix S, coherence checking
  method/         # MethodBase interface + adapters (naive, statsmodels, lightgbm)
  reconciliation/ # hierarchical reconciliation strategies (bottom_up, top_down, min_trace, min_trace_shrink)
  evaluation/     # forecast execution, reconciliation application, metrics, method comparison
  report/         # raw result persistence and leaderboard aggregation
  pipeline.py     # ties the five modules above into a single call
scripts/
  convert_*.py            # conversion of each dataset into the project's format
  run_benchmark.py         # CLI for a single run
  run_experiments.py       # full sweep over datasets × methods × reconciliations
  analyze_experiments.py   # reading and analyzing saved results, without re-running models
dataset/    # already-converted datasets (Labour, Tourism, Traffic, Wiki2, M5_lite)
tests/      # automated test suite
```

## Acknowledgments

The split into modules (Data, Methods, Reconciliation, Evaluation, Reports), the standardized method interface, and the pattern of persisting raw results for later reprocessing were inspired by [TFB](https://github.com/decisionintelligence/TFB):

```
@article{qiu2024tfb,
  title   = {TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods},
  author  = {Xiangfei Qiu and Jilin Hu and Lekui Zhou and Xingjian Wu and Junyang Du and Buang Zhang and Chenjuan Guo and Aoying Zhou and Christian S. Jensen and Zhenli Sheng and Bin Yang},
  journal = {Proc. {VLDB} Endow.},
  volume  = {17},
  number  = {9},
  pages   = {2363--2377},
  year    = {2024}
}
```

The `traffic` and `wiki2` datasets, and the horizon=1 convention used for both, come from:

```
@inproceedings{rangapuram2021end,
  title     = {End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series},
  author    = {Rangapuram, Syama Sundar and Werner, Lucien D. and Benidis, Konstantinos and Mercado, Pedro and Gasthaus, Jan and Januschowski, Tim},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {139},
  pages     = {8832--8843},
  year      = {2021}
}
```

`min_trace`/`min_trace_shrink` reconciliation follows Wickramasuriya, Athanasopoulos & Hyndman (2019), **Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization**, JASA.

## Contact

Questions, suggestions, or issues: open an [issue](https://github.com/malbuq-dev/hts-bench/issues) in this repository.
