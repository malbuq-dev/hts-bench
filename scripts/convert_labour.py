from _convert_hierarchical_common import convert_group

if __name__ == "__main__":
    # Nixtla's own spec for Labour: freq='MS', horizon=8 (papers commonly use 12).
    convert_group("Labour", "labour", horizon_suggested=8)
