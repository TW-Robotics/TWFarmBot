from twfarmbot_core.domain import Action, Bed, Plant, SensorReading


def test_domain_types_are_importable() -> None:
    plant = Plant(id="p1", species="tomato", bed_id="b1")
    bed = Bed(id="b1", name="Greenhouse A", width_mm=1000.0, height_mm=2000.0)
    reading = SensorReading(
        sensor_id="s1", metric="soil_moisture", value=42.0, unit="%", taken_at=__import__("datetime").datetime.now()
    )
    action = Action(kind="water", params={"bed_id": "b1"})

    assert plant.bed_id == bed.id
    assert reading.value == 42.0
    assert action.kind == "water"
