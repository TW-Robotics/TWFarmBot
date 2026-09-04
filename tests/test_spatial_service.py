from spatial_service import load_world


def test_load_world_has_usable_spatial_frame() -> None:
    world = load_world()
    assert world.bounds.width > 0
    assert world.bounds.height > 0
    assert world.camera.position.z > 0
    assert isinstance(world.zones, tuple)
