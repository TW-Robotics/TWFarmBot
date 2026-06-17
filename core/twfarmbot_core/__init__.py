"""twfarmbot_core — shared building blocks.

Re-exports the subpackages so they can be imported as
``twfarmbot_core.domain``, ``twfarmbot_core.config``,
``twfarmbot_core.logging`` and ``twfarmbot_core.events``.

``twfarmbot_core.actions`` is intentionally not eagerly imported here to
avoid a circular dependency with ``services/safety_service`` (which itself
imports ``twfarmbot_core.domain``). Import it directly when needed::

    from twfarmbot_core.actions import ActionRegistry
"""

from . import config, domain, events, logging

__all__ = ["config", "domain", "events", "logging"]
__version__ = "0.1.0"
