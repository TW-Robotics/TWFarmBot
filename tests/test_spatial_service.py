from spatial_service import load_world


def test_load_world_contains_required_spatial_layers() -> None:
    world = load_world()
    assert world.bounds.width > 0
    assert world.bounds.height > 0
    assert world.camera.position.z > 0
    assert {entity.kind for entity in world.entities} >= {"plant", "tool"}
    assert {zone.kind for zone in world.zones} >= {"watering", "obstacle"}
