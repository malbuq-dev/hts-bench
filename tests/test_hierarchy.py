import numpy as np

from hts_bench.data.hierarchy import build_summing_matrix


def test_toy_hierarchy_matches_expected_matrix(toy_series_meta):
    S = build_summing_matrix(toy_series_meta)

    assert S.row_ids == ["s0", "s1", "s2"]
    assert S.col_ids == ["s1", "s2"]
    expected = np.array(
        [
            [1, 1],  # Total = North + South
            [1, 0],  # North = North
            [0, 1],  # South = South
        ]
    )
    np.testing.assert_array_equal(S.matrix.toarray(), expected)


def test_bottom_rows_map_only_to_themselves(toy_series_meta):
    S = build_summing_matrix(toy_series_meta)
    dense = S.matrix.toarray()
    bottom_rows = [S.row_ids.index(sid) for sid in S.col_ids]
    identity = np.eye(len(S.col_ids), dtype=int)
    np.testing.assert_array_equal(dense[bottom_rows, :], identity)


def test_root_level_collapses_to_single_row(toy_series_meta):
    # Regression: cat[[]].drop_duplicates() is a no-op on zero columns in this
    # pandas version and silently multiplies the root row - build_summing_matrix
    # must never produce more than one row for a level with no dimension columns.
    S = build_summing_matrix(toy_series_meta)
    assert S.row_ids.count("s0") == 1


def test_crossed_grouped_hierarchy(crossed_series_meta):
    S = build_summing_matrix(crossed_series_meta)

    assert S.col_ids == ["s3", "s4", "s5", "s6"]  # North-A, North-B, South-A, South-B
    dense = S.matrix.toarray()
    row = dict(zip(S.row_ids, dense))

    # Total sums all four bottom series.
    np.testing.assert_array_equal(row["s0"], [1, 1, 1, 1])
    # Region "North" (s1) sums only North-A, North-B.
    np.testing.assert_array_equal(row["s1"], [1, 1, 0, 0])
    # Region "South" (s2) sums only South-A, South-B.
    np.testing.assert_array_equal(row["s2"], [0, 0, 1, 1])
