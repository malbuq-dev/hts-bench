from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from hts_bench.data.hierarchy import SummingMatrix, build_summing_matrix


@dataclass
class HierarchicalDataset:
    """
    In-memory representation of a hierarchical/grouped time series dataset.

    """

    name: str
    freq: str
    data: pd.DataFrame
    series_meta: pd.DataFrame

    @cached_property
    def summing_matrix(self) -> SummingMatrix:
        return build_summing_matrix(self.series_meta)

    @property
    def bottom_series(self) -> list:
        return self.series_meta.index[self.series_meta["is_bottom"]].tolist()

    @property
    def levels(self) -> dict:
        return {
            level: rows.index.tolist()
            for level, rows in self.series_meta.groupby("level")
        }

    def get_series(self, series_id: str) -> pd.Series:
        return self.data[series_id]

    def get_bottom_data(self) -> pd.DataFrame:
        return self.data[self.bottom_series]
