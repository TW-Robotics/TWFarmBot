import pytest

from farmduino.motion import resolve_axis_speed, resolve_axis_speeds


def test_speed_percent_maps_to_steps_per_second() -> None:
    assert resolve_axis_speed(1, min_steps_s=50, max_steps_s=400) == 50
    assert resolve_axis_speed(100, min_steps_s=50, max_steps_s=400) == 400
    mid = resolve_axis_speed(50, min_steps_s=50, max_steps_s=400)
    assert 220 < mid < 230
    assert isinstance(mid, int)


def test_axis_speeds_are_integers() -> None:
    speeds = resolve_axis_speeds(50, min_steps_s=(50, 50, 50), max_steps_s=(400, 400, 400))
    assert speeds == (223, 223, 223)


def test_speed_above_100_is_raw_steps_per_second() -> None:
    assert resolve_axis_speed(250, min_steps_s=50, max_steps_s=400) == 250
    assert resolve_axis_speed(500, min_steps_s=50, max_steps_s=400) == 400


def test_speed_must_be_positive() -> None:
    with pytest.raises(ValueError):
        resolve_axis_speed(0)
