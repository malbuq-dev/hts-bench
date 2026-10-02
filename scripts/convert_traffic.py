from _convert_hierarchical_common import convert_group_from_matrix

if __name__ == "__main__":
    # Traffic's own bottom-level ids ("Bottom1".."Bottom200") don't encode
    # ancestry at all - see convert_group_from_matrix's docstring - so this
    # uses the S_df-derived path, not convert_group's raw_id decoding.
    # prediction_length=1 is the value the ICML-2021 paper (Rangapuram et al.,
    # "End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical
    # Time Series") itself uses for this dataset (experiments/config/
    # dataset_config.py in https://github.com/rshyamsundar/gluonts-hierarchical-ICML-2021).
    convert_group_from_matrix(
        "Traffic", "traffic", horizon_suggested=1, level_order=["Level1", "Level2", "Level3", "Level4"]
    )
