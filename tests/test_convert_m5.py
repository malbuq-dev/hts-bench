import pandas as pd

from convert_m5 import build_series_meta


def _toy_category_table():
    # 2 items x 2 stores (2 states), full cross - synthetic, no network/download needed.
    rows = [
        {"unique_id": "I1_ST1", "item_id": "I1", "dept_id": "D1", "cat_id": "C1", "store_id": "ST1", "state_id": "S1"},
        {"unique_id": "I1_ST2", "item_id": "I1", "dept_id": "D1", "cat_id": "C1", "store_id": "ST2", "state_id": "S2"},
        {"unique_id": "I2_ST1", "item_id": "I2", "dept_id": "D2", "cat_id": "C1", "store_id": "ST1", "state_id": "S1"},
        {"unique_id": "I2_ST2", "item_id": "I2", "dept_id": "D2", "cat_id": "C1", "store_id": "ST2", "state_id": "S2"},
    ]
    return pd.DataFrame(rows)


def test_root_level_has_exactly_one_row():
    # Regression: cat[[]].drop_duplicates() is a no-op on zero columns in this
    # pandas version, which previously produced one "Total" row per input row
    # (3049 duplicates for the real M5-CA_1 run) instead of collapsing to one.
    series_meta, _ = build_series_meta(_toy_category_table())
    assert (series_meta["level"] == "Total").sum() == 1


def test_level_counts_match_expected_cross_structure():
    series_meta, unique_id_to_sid = build_series_meta(_toy_category_table())

    counts = series_meta["level"].value_counts().to_dict()
    assert counts == {
        "Total": 1,
        "state_id": 2,
        "store_id": 2,
        "cat_id": 1,
        "dept_id": 2,
        "state_id/cat_id": 2,
        "state_id/dept_id": 4,
        "store_id/cat_id": 2,
        "store_id/dept_id": 4,
        "item_id": 2,
        "state_id/item_id": 4,
        "store_id/item_id": 4,
    }
    assert len(series_meta) == 30
    assert series_meta["is_bottom"].sum() == 4
    assert len(unique_id_to_sid) == 4


def test_series_ids_unique_and_bottom_ids_map_correctly():
    series_meta, unique_id_to_sid = build_series_meta(_toy_category_table())

    assert series_meta.index.is_unique
    for unique_id, sid in unique_id_to_sid.items():
        row = series_meta.loc[sid]
        assert row["is_bottom"]
        assert unique_id == f"{row['item_id']}_{row['store_id']}"
