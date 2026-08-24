from _convert_hierarchical_common import convert_group


def _decode_tourism(raw_id: str, dim_names: list) -> dict:
    """
    Tourism's raw ids are compact positional codes, not Python-list reprs, e.g.
    "AAAHol" = state 'A' + zone 'A' + region 'A' + purpose 'Hol'. The country-level
    root uses the literal sentinel "Total" in place of geo letters (0 of them), and
    "All" as the purpose suffix wherever purpose isn't specified at that level - both
    positionally derivable from which dims this level's path actually specifies, no
    hardcoded vocabulary needed. Codes are opaque (single letters, not real state
    names - datasetsforecast doesn't expose a name lookup) but still fully structured.
    """
    geo_dims = [d for d in dim_names if d != "purpose"]
    k = len(geo_dims)
    has_purpose = "purpose" in dim_names
    prefix_len = 5 if k == 0 else k  # "Total" (5 chars) at the root, else 1 char/level

    result = {}
    if k > 0:
        geo_code = raw_id[:prefix_len]
        for i, dim in enumerate(geo_dims):
            result[dim] = geo_code[i]
    if has_purpose:
        result["purpose"] = raw_id[prefix_len:]
    return result


if __name__ == "__main__":
    # TourismLarge: the 555-series (304 bottom) Australian domestic tourism
    # hierarchy from Athanasopoulos et al. - the canonical benchmark in the
    # hierarchical forecasting literature. Monthly; 24-month horizon is standard
    # in that literature (original Tourism forecasting competition).
    convert_group("TourismLarge", "tourism", horizon_suggested=24, decode_raw_id=_decode_tourism)
