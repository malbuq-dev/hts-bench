"""
Shared conversion logic for datasets sourced via datasetsforecast.hierarchical
(Labour, Tourism, ...): tags-walk -> synthetic ids -> series_meta -> data.csv -> meta.json.

Requires `datasetsforecast` at conversion time only (not a runtime dependency
of hts_bench itself - see requirements.txt).
"""
import ast
import json
import os
import sys

import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_CACHE_DIR = os.path.join(PROJECT_ROOT, "_raw_cache")

sys.path.insert(0, PROJECT_ROOT)
from hts_bench.data.hierarchy import build_summing_matrix  # noqa: E402


def _decode_labour_style(raw_id: str, dim_names: list) -> dict:
    """Default decoder: raw_id is a Python-list repr, e.g. "['Females', 'NSW']"."""
    if not dim_names:
        return {}
    values = ast.literal_eval(raw_id)
    return dict(zip(dim_names, values))


def convert_group(
    group_name: str,
    output_name: str,
    horizon_suggested: int,
    decode_raw_id=_decode_labour_style,
) -> None:
    output_dir = os.path.join(PROJECT_ROOT, "dataset", output_name)
    os.makedirs(output_dir, exist_ok=True)

    Y_df, S_df, tags = HierarchicalData.load(RAW_CACHE_DIR, group_name)

    # Root-first ordering (fewer '/' = higher in the hierarchy).
    level_names = sorted(tags.keys(), key=lambda lv: lv.count("/"))
    bottom_level = level_names[-1]

    raw_to_id = {}
    series_rows = []
    next_idx = 0
    for level in level_names:
        dim_names = [d.lower() for d in level.split("/")[1:]]
        for raw_id in tags[level]:
            sid = f"s{next_idx:05d}"
            next_idx += 1
            raw_to_id[raw_id] = sid

            row = {"series_id": sid, "level": level, "is_bottom": level == bottom_level}
            row.update(decode_raw_id(raw_id, dim_names))
            series_rows.append(row)

    series_meta = pd.DataFrame(series_rows).set_index("series_id")

    wide = Y_df.pivot(index="ds", columns="unique_id", values="y")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.rename(columns=raw_to_id).sort_index()
    wide = wide[list(raw_to_id.values())]  # order columns s00000, s00001, ... for readability
    wide.index.name = "date"
    wide.to_csv(os.path.join(output_dir, "data.csv"))

    series_meta.to_csv(os.path.join(output_dir, "series_meta.csv"))

    freq = pd.infer_freq(wide.index) or "UNKNOWN"
    meta = {
        "name": output_name,
        "freq": freq,
        "horizon_suggested": horizon_suggested,
        "n_series": len(series_meta),
        "n_bottom": int(series_meta["is_bottom"].sum()),
        "data_files": ["data.csv"],
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Correctness check: derive S from the freshly-written series_meta and diff
    # against Nixtla's original dense S_df - confirms the derive-from-metadata
    # approach reproduces ground truth before trusting it for datasets that never
    # ship a pre-built S at all (M5).
    derived = build_summing_matrix(series_meta)
    original = S_df.rename(index=raw_to_id, columns=raw_to_id).loc[
        derived.row_ids, derived.col_ids
    ]
    derived_dense = pd.DataFrame(
        derived.matrix.toarray(), index=derived.row_ids, columns=derived.col_ids
    )
    mismatches = int((derived_dense.values != original.values).sum())

    print(f"[{output_name}] wrote {meta['n_series']} series ({meta['n_bottom']} bottom) to {output_dir}")
    print(f"[{output_name}] data.csv shape: {wide.shape}, freq: {freq}")
    print(f"[{output_name}] S derivation check: {mismatches} mismatches vs Nixtla's original S")


def convert_group_from_matrix(
    group_name: str,
    output_name: str,
    horizon_suggested: int,
    level_order: list,
) -> None:
    """
    Alternative to convert_group, for datasets whose raw ids don't self-encode
    ancestry the way Labour's Python-list-repr ids or Wiki2's underscore-joined
    ids do - e.g. Traffic's bottom leaves are opaque "Bottom1".."Bottom200",
    with no shared prefix linking them to their "y11"/"y1"-style parents.
    Derives every row's dimension columns directly from Nixtla's own S_df
    (which bottom series each aggregate sums) instead of parsing raw_id
    strings - correct by construction for any strict tree regardless of how
    ids happen to be spelled, at the cost of being O(n_bottom) per ancestor
    lookup (fine for datasets this size; not meant to scale to M5).

    level_order: `tags`'s keys in root-first order - HierarchicalData doesn't
    expose this itself, it has to be known per dataset (see e.g. the
    "level_names" lists in the ICML-2021 paper's own dataset_config.py).
    Column names are just each non-root level's own name, lowercased.
    """
    output_dir = os.path.join(PROJECT_ROOT, "dataset", output_name)
    os.makedirs(output_dir, exist_ok=True)

    Y_df, S_df, tags = HierarchicalData.load(RAW_CACHE_DIR, group_name)
    bottom_level = level_order[-1]
    dim_cols = [lv.lower() for lv in level_order[1:]]

    def first_bottom_descendant(agg_id):
        row = S_df.loc[agg_id]
        return row[row == 1].index[0]

    def ancestor_at(level_name, bottom_id):
        col = S_df.loc[tags[level_name], bottom_id]
        return col[col == 1].index[0]

    raw_to_id = {}
    series_rows = []
    next_idx = 0
    for depth, level_name in enumerate(level_order):
        for raw_id in tags[level_name]:
            sid = f"s{next_idx:05d}"
            next_idx += 1
            raw_to_id[raw_id] = sid

            row = {"series_id": sid, "level": level_name, "is_bottom": level_name == bottom_level}
            if depth > 0:
                representative = raw_id if level_name == bottom_level else first_bottom_descendant(raw_id)
                for anc_depth in range(1, depth + 1):
                    anc_level, col = level_order[anc_depth], dim_cols[anc_depth - 1]
                    row[col] = raw_id if anc_level == level_name else ancestor_at(anc_level, representative)
            series_rows.append(row)

    series_meta = pd.DataFrame(series_rows).set_index("series_id")

    wide = Y_df.pivot(index="ds", columns="unique_id", values="y")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.rename(columns=raw_to_id).sort_index()
    wide = wide[list(raw_to_id.values())]
    wide.index.name = "date"
    wide.to_csv(os.path.join(output_dir, "data.csv"))

    series_meta.to_csv(os.path.join(output_dir, "series_meta.csv"))

    freq = pd.infer_freq(wide.index) or "UNKNOWN"
    meta = {
        "name": output_name,
        "freq": freq,
        "horizon_suggested": horizon_suggested,
        "n_series": len(series_meta),
        "n_bottom": int(series_meta["is_bottom"].sum()),
        "data_files": ["data.csv"],
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    derived = build_summing_matrix(series_meta)
    original = S_df.rename(index=raw_to_id, columns=raw_to_id).loc[derived.row_ids, derived.col_ids]
    derived_dense = pd.DataFrame(
        derived.matrix.toarray(), index=derived.row_ids, columns=derived.col_ids
    )
    mismatches = int((derived_dense.values != original.values).sum())

    print(f"[{output_name}] wrote {meta['n_series']} series ({meta['n_bottom']} bottom) to {output_dir}")
    print(f"[{output_name}] data.csv shape: {wide.shape}, freq: {freq}")
    print(f"[{output_name}] S derivation check: {mismatches} mismatches vs Nixtla's original S")
