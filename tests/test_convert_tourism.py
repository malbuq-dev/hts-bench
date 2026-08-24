from convert_tourism import _decode_tourism


def test_decode_root_level():
    assert _decode_tourism("TotalAll", []) == {}


def test_decode_state_only():
    assert _decode_tourism("AAll", ["state"]) == {"state": "A"}


def test_decode_state_zone_region():
    assert _decode_tourism("AABAll", ["state", "zone", "region"]) == {
        "state": "A",
        "zone": "A",
        "region": "B",
    }


def test_decode_country_purpose():
    assert _decode_tourism("TotalHol", ["purpose"]) == {"purpose": "Hol"}


def test_decode_bottom_level_state_zone_region_purpose():
    assert _decode_tourism("AAAHol", ["state", "zone", "region", "purpose"]) == {
        "state": "A",
        "zone": "A",
        "region": "A",
        "purpose": "Hol",
    }
