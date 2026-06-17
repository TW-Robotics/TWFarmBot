"""twfarmbot_core — shared building blocks.

Re-exports the subpackages so they can be imported as
``twfarmbot_core.domain``, ``twfarmbot_core.config``,
``twfarmbot_core.logging`` and ``twfarmbot_core.events``.
"""

from . import config, domain, events, logging

__all__ = ["config", "domain", "events", "logging"]
__version__ = "0.1.0"
