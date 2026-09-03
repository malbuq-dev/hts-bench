"""
Converts M5 (via datasetsforecast.m5, a Nixtla GitHub mirror of the Kaggle
competition files - no Kaggle account needed) into hts_bench's format.

Unlike Labour/Tourism, M5's source gives ONLY bottom-level (item x store) sales
plus a flat category table - no pre-built aggregates. This script builds the
standard 12-level M5 grouped hierarchy itself from the category columns, then
derives every aggregate level by dogfooding hts_bench's own
loader.aggregate_from_bottom (S @ b) rather than summing independently - which
also doubles as an integration check on that code path before it's ever loaded
back through load_dataset.

--store filters to a single store (default CA_1, ~3,049 bottom series) to keep
today's build fast; the exact same code path run with --store all produces the
full ~30,490-series dataset - nothing to rewrite, just longer runtime and larger
files.
"""
import argparse
import json
import os
import sys

import pandas as pd
from datasetsforecast.m5 import M5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_CACHE_DIR = os.path.join(PROJECT_ROOT, "_raw_cache")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "m5")

sys.path.insert(0, PROJECT_ROOT)
from hts_bench.data.hierarchy import build_summing_matrix  # noqa: E402
from hts_bench.data.loader import aggregate_from_bottom  # noqa: E402

# Root-first; last entry is the bottom (item x store) level.
LEVELS = [
    [],
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id"],
    ["state_id", "item_id"],
    ["store_id", "item_id"],
]


def build_series_meta(cat: pd.DataFrame):
    bottom_cols = LEVELS[-1]
    # Every dimension a bottom item genuinely has, not just the ones bottom_cols
    # names (store_id, item_id): state_id/dept_id/cat_id are functionally
    # determined by store_id/item_id, but bottom rows need them set explicitly
    # too, or hierarchy.build_summing_matrix's per-level merge (e.g. on dept_id
    # alone) matches zero bottom rows and that aggregate ends up all-zero.
    all_dim_cols = sorted({c for cols in LEVELS for c in cols})
    series_rows = []
    unique_id_to_sid = {}
    next_idx = 0

    for cols in LEVELS:
        level_name = "/".join(cols) if cols else "Total"
        is_bottom = cols == bottom_cols
        row_cols = all_dim_cols if is_bottom else cols
        if not cols:
            # cat[[]].drop_duplicates() is a no-op on zero columns in this pandas
            # version (keeps every row instead of collapsing to one) - special-case
            # the root level explicitly instead.
            groups = pd.DataFrame([{}])
        else:
            extra = ["unique_id"] if is_bottom else []
            groups = cat[row_cols + extra].drop_duplicates().reset_index(drop=True)

        for _, vals in groups.iterrows():
            sid = f"s{next_idx:05d}"
            next_idx += 1
            row = {"series_id": sid, "level": level_name, "is_bottom": is_bottom}
            row.update({c: vals[c] for c in row_cols})
            series_rows.append(row)
            if is_bottom:
                unique_id_to_sid[vals["unique_id"]] = sid

    series_meta = pd.DataFrame(series_rows).set_index("series_id")
    return series_meta, unique_id_to_sid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="CA_1", help="store_id to subset to, or 'all'")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Y_df, _X_df, S_df = M5.load(RAW_CACHE_DIR)

    cat = S_df if args.store == "all" else S_df[S_df["store_id"] == args.store]
    cat = cat.copy()
    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        cat[col] = cat[col].astype(str)
    keep_ids = set(cat["unique_id"].astype(str))
    Y_df = Y_df[Y_df["unique_id"].astype(str).isin(keep_ids)]

    series_meta, unique_id_to_sid = build_series_meta(cat)

    bottom_wide = Y_df.pivot(index="ds", columns="unique_id", values="y")
    bottom_wide.index.name = "date"
    bottom_wide = bottom_wide.rename(columns=lambda c: unique_id_to_sid[str(c)])
    bottom_wide = bottom_wide.fillna(0.0).sort_index()  # ragged item-launch dates -> 0 sales

    summing_matrix = build_summing_matrix(series_meta)
    full_wide = aggregate_from_bottom(summing_matrix, bottom_wide)
    full_wide = full_wide[series_meta.index.tolist()]  # order s00000, s00001, ...

    # Shard by dept_id (a natural partition: bottom + dept-level aggregates all carry
    # it); series with no dept_id (Total, state, store, cat, item, state x item levels)
    # go in one "other" shard.
    dept_of = series_meta["dept_id"] if "dept_id" in series_meta.columns else pd.Series(dtype=object)
    shard_files = []
    for dept, sids in series_meta.groupby(dept_of.reindex(series_meta.index).fillna("other")):
        fname = f"data_{dept}.csv"
        full_wide[sids.index.tolist()].to_csv(os.path.join(OUTPUT_DIR, fname))
        shard_files.append(fname)

    series_meta.to_csv(os.path.join(OUTPUT_DIR, "series_meta.csv"))

    freq = pd.infer_freq(full_wide.index) or "D"
    meta = {
        "name": "m5",
        "freq": freq,
        "horizon_suggested": 28,  # M5 competition's own forecast horizon
        "n_series": len(series_meta),
        "n_bottom": int(series_meta["is_bottom"].sum()),
        "data_files": sorted(shard_files),
        "source_store_filter": args.store,
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[m5] store filter: {args.store}")
    print(f"[m5] wrote {meta['n_series']} series ({meta['n_bottom']} bottom) to {OUTPUT_DIR}")
    print(f"[m5] data shape: {full_wide.shape}, freq: {freq}, shards: {len(shard_files)}")


if __name__ == "__main__":
    main()
