"""twfarmbot_core — shared building blocks.

Re-exports the subpackages so they can be imported as
``twfarmbot_core.domain``, ``twfarmbot_core.config``,
``twfarmbot_core.logging``, and ``twfarmbot_core.actions``.
"""

from . import actions, config, domain, logging

__all__ = ["actions", "config", "domain", "logging"]
__version__ = "0.1.0"
