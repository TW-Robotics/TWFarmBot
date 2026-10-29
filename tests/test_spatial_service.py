from spatial_service import format_world_context, load_world, resolve_named
from spatial_service.observations import ObservationStore, class_score, try_record_ndre, try_record_segment
from spatial_service.positions import list_positions, save_positions
from spatial_service.scripts import list_scripts, resolve_script, save_scripts


def test_load_world_has_usable_spatial_frame() -> None:
    world = load_world()
    assert world.bounds.width > 0
    assert world.bounds.height > 0
    assert world.camera.position.z > 0
    assert isinstance(world.zones, tuple)


def test_latest_cells_keep_both_samples() -> None:
    store = ObservationStore(":memory:")
    store.record(x=200, y=100, source="capture_ndre", ndre_mean=0.3, action_hint="ok")
    store.record(
        x=1200,
        y=400,
        source="capture_ndre",
        ndre_mean=0.04,
        stress_fraction=0.6,
        action_hint="consider_water",
    )
    cells = store.latest_cells()
    assert len(cells) == 2
    by_x = {round(cell["x"]): cell for cell in cells}
    assert by_x[200]["action_hint"] == "ok"
    assert by_x[1200]["stress_fraction"] == 0.6


def test_find_locations_ranks_stressed() -> None:
    store = ObservationStore(":memory:")
    store.record(x=200, y=100, ndre_mean=0.32, stress_fraction=0.05, action_hint="ok")
    store.record(
        x=1200, y=400, ndre_mean=0.04, stress_fraction=0.55, action_hint="consider_water"
    )
    found = store.find_locations("stressed")
    assert found["count"] >= 1
    assert found["locations"][0]["x"] == 1200
    healthy = store.find_locations("healthy")
    assert healthy["locations"][0]["x"] == 200


def test_find_locations_ripe_is_not_a_layer() -> None:
    store = ObservationStore(":memory:")
    out = store.find_locations("ripe plants")
    assert out["locations"] == []
    assert "ripeness" in out["error"]
    assert "plant" in out["try"]


def test_cell_merges_ndre_then_segment() -> None:
    store = ObservationStore(":memory:")
    store.record(x=1000, y=250, source="capture_ndre", ndre_mean=0.2, action_hint="ok")
    store.record(x=1010, y=260, source="segment_image", plant=0.7, weed=0.1, soil=0.2)
    cells = store.latest_cells()
    assert len(cells) == 1
    cell = cells[0]
    assert cell["ndre_mean"] == 0.2
    assert cell["plant"] == 0.7
    assert cell["action_hint"] == "ok"


def test_try_record_ndre_and_segment(monkeypatch) -> None:
    store = ObservationStore(":memory:")
    monkeypatch.setattr(
        "spatial_service.observations.get_observation_store",
        lambda db_path=None: store,
    )
    try_record_ndre(
        {
            "gantry_mm": {"x": 500, "y": 100, "z": 0},
            "ndre": {"mean": 0.18, "stress_fraction": 0.2, "vegetation_fraction": 0.4},
            "interpretation": {"action_hint": "monitor"},
        }
    )
    try_record_segment(
        {"xyz": {"x": 500, "y": 100, "z": 0}},
        {"plants": 0.6, "weeds": 0.1, "soil": 0.3},
        "plants",
    )
    cells = store.latest_cells()
    assert len(cells) == 1
    assert cells[0]["ndre_mean"] == 0.18
    assert cells[0]["plant"] == 0.6


def test_try_record_ndre_accepts_paired_gantry(monkeypatch) -> None:
    store = ObservationStore(":memory:")
    monkeypatch.setattr(
        "spatial_service.observations.get_observation_store",
        lambda db_path=None: store,
    )
    try_record_ndre(
        {
            "gantry_mm": {
                "nir": {"x": 1298.0, "y": 150.0, "z": 0.0},
                "rededge": {"x": 1298.0, "y": 190.0, "z": 0.0},
            },
            "ndre": {"mean": 0.12, "stress_fraction": 0.3, "vegetation_fraction": 0.5},
            "interpretation": {"action_hint": "consider_water"},
        }
    )
    cells = store.latest_cells()
    assert len(cells) == 1
    assert cells[0]["x"] == 1298.0
    assert cells[0]["y"] == 150.0
    assert cells[0]["ndre_mean"] == 0.12
    assert cells[0]["action_hint"] == "consider_water"


def test_class_score_aliases() -> None:
    assert class_score({"plants": 0.4, "weeds": 0.2}, "plant") == 0.4
    assert class_score({"Weed": 0.9}, "weed") == 0.9


def test_format_world_context_mentions_bed_map(monkeypatch) -> None:
    monkeypatch.setattr("spatial_service.latest_cells", lambda cell_mm=250: [])
    text = format_world_context()
    assert "find_locations" in text
    assert "ask the user for mm" not in text
    assert "Ripeness is not a map layer" in text
    assert "goto_named" in text
    assert "Home" in text
    assert "run_script" in text
    assert "Get tool servo" in text


def test_save_and_resolve_named_positions(tmp_path, monkeypatch) -> None:
    path = tmp_path / "dev.yaml"
    path.write_text("positions: []\n", encoding="utf-8")
    monkeypatch.setenv("TWFB_CONFIG", str(path))
    saved = save_positions(
        [
            {"label": "Home", "group": "Travel", "x": 0, "y": 0, "z": 0},
            {"label": "Cutting", "group": "Tools"},
        ]
    )
    assert saved[0]["set"] is True
    assert saved[1]["set"] is False
    listed = list_positions()
    assert listed[1]["label"] == "Cutting"
    assert resolve_named("Home")["x"] == 0
    try:
        resolve_named("Cutting")
        raise AssertionError("unset preset should not resolve")
    except ValueError as err:
        assert "no saved gantry pose" in str(err)


def test_save_and_resolve_scripts(tmp_path, monkeypatch) -> None:
    path = tmp_path / "dev.yaml"
    path.write_text("scripts: []\n", encoding="utf-8")
    monkeypatch.setenv("TWFB_CONFIG", str(path))
    save_scripts(
        [
            {
                "label": "Get tool servo",
                "waypoints": [
                    {"x": 1927, "y": 186, "z": 0},
                    {"x": 1927, "y": 186, "z": -297},
                    {"x": 1850, "y": 186, "z": -297},
                ],
            }
        ]
    )
    found = resolve_script("tool servo")
    assert found["waypoints"][0]["x"] == 1927
    assert list_scripts()[0]["label"] == "Get tool servo"
