"""Action handlers — one tiny module per kind, each calls a backend method.

Adding a new action kind = one new file here + one line in __init__.py.
"""

from __future__ import annotations

from twfarmbot_core.actions import ActionRegistry


def register_default_handlers(registry: ActionRegistry) -> None:
    from .watering import handle_water
    from .move import handle_move
    from .path import handle_move_path
    from .mount_tool import handle_mount_tool, handle_dismount_tool
    from .pin import handle_read_pin, handle_write_pin
    from .feedback import handle_e_stop
    from .find_home import handle_find_home
    from .camera import handle_take_photo

    registry.register("water", handle_water)
    registry.register("move", handle_move)
    registry.register("move_path", handle_move_path)
    registry.register("mount_tool", handle_mount_tool)
    registry.register("dismount_tool", handle_dismount_tool)
    registry.register("read_pin", handle_read_pin)
    registry.register("write_pin", handle_write_pin)
    registry.register("e_stop", handle_e_stop)
    registry.register("find_home", handle_find_home)
    registry.register("take_photo", handle_take_photo)
