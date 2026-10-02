from _convert_hierarchical_common import convert_group


def _decode_wiki2(raw_id: str, dim_names: list) -> dict:
    """
    Wiki2's raw ids are underscore-joined, e.g. "de_AAC_AAG_001" = country 'de'
    + access 'AAC' + agent 'AAG' + topic '001' - not Python-list reprs like
    Labour's, so the default decoder (ast.literal_eval) can't parse them.
    maxsplit keeps this correct even if a topic code ever contained "_" itself.
    """
    if not dim_names:
        return {}
    values = raw_id.split("_", maxsplit=len(dim_names) - 1)
    return dict(zip(dim_names, values))


if __name__ == "__main__":
    # prediction_length=1 is the value the ICML-2021 paper (Rangapuram et al.,
    # "End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical
    # Time Series") itself uses for this dataset (experiments/config/
    # dataset_config.py in https://github.com/rshyamsundar/gluonts-hierarchical-ICML-2021).
    convert_group("Wiki2", "wiki2", horizon_suggested=1, decode_raw_id=_decode_wiki2)
