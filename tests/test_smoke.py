from twfarmbot_core.domain import Action, CameraPose, GardenWorld, GardenEntity, Point3D, Rectangle


def test_domain_types_are_importable() -> None:
    world = GardenWorld(
        bounds=Rectangle(x=0, y=0, width=1000, height=500),
        camera=CameraPose(position=Point3D(0, 0, 0)),
        entities=(GardenEntity(id="e1", kind="plant", name="Tomato", position=Point3D(100, 200, 0)),),
    )
    action = Action(kind="water", params={"seconds": 5})

    assert world.bounds.width == 1000
    assert world.entities[0].name == "Tomato"
    assert action.kind == "water"
