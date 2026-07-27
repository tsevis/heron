"""Material table tests (CLAUDE.md §7)."""

from heron.materials import load_material_table


def test_table_loads_with_seed_classes():
    t = load_material_table()
    for cls in ("skin", "water", "metal_polished", "foliage", "unknown"):
        assert cls in t.materials


def test_skin_is_hot_high_emissivity():
    t = load_material_table()
    skin = t.get("skin")
    assert skin.emissivity_lwir > 0.95
    assert skin.temp_mean_c > t.get("sky").temp_mean_c  # skin warmer than sky


def test_reflective_materials_flagged():
    t = load_material_table()
    refl = t.reflective_map()
    assert refl["water"] > 0.0 and refl["metal_polished"] > 0.0
    assert refl["skin"] == 0.0


def test_alias_resolution():
    t = load_material_table()
    assert t.resolve_label("Person") == "skin"
    assert t.resolve_label("cotton t-shirt") == "fabric"
    assert t.resolve_label("a shiny chrome bumper") == "metal_polished"
    assert t.resolve_label("wibblefish") == "unknown"


def test_skin_has_facial_zones():
    t = load_material_table()
    zones = t.get("skin").zones
    assert zones["nose"] < 0 < zones["tearduct"]  # nose cold, tearduct hot
