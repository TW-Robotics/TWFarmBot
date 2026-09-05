from twfarmbot_core.domain import Action
from twfarmbot_api_server.handlers.jobs import axis_stops, handle_scan_ndre


def test_axis_stops_inclusive() -> None:
    assert axis_stops(0, 300, 100) == [0.0, 100.0, 200.0, 300.0]


def test_axis_stops_appends_end() -> None:
    assert axis_stops(0, 250, 100) == [0.0, 100.0, 200.0, 250.0]


def test_handle_scan_ndre_visits_each_stop(monkeypatch) -> None:
    moves: list[dict] = []
    captures = {"n": 0}

    monkeypatch.setattr(
        "twfarmbot_api_server.handlers.jobs.farmbot.backend.get_xyz",
        lambda: {"x": 10.0, "y": 0.0, "z": 0.0},
    )

    def fake_move(action: Action) -> Action:
        moves.append(action.params)
        return action

    def fake_ndre(action: Action) -> Action:
        captures["n"] += 1
        return Action(
            kind="capture_ndre",
            params={"ndre": {"mean": captures["n"]}, "interpretation": {"label": "ok"}},
        )

    monkeypatch.setattr("twfarmbot_api_server.handlers.jobs.handle_move", fake_move)
    monkeypatch.setattr(
        "twfarmbot_api_server.handlers.jobs.handle_capture_ndre", fake_ndre
    )

    out = handle_scan_ndre(
        Action(
            kind="scan_ndre",
            params={"axis": "y", "end_mm": 300, "step_mm": 100, "start_mm": 0},
        )
    )
    assert out.params["count"] == 4
    assert [sample["y"] for sample in out.params["samples"]] == [0.0, 100.0, 200.0, 300.0]
    assert captures["n"] == 4
    assert moves[-1] == {"x": 10.0, "y": 0.0, "z": 0.0}
