"""Parse, validate, and safety-check the LLM's plan.

Three responsibilities, in order:

1. **Extract JSON** from the model's free-form output. LLMs wrap JSON
   in markdown fences, prose, or both; we use a tolerant extractor and
   fall back to a clear error.
2. **Validate structurally** against :class:`PlannerResponse` (Pydantic).
   Unknown action kinds and wrong param shapes fail here.
3. **Validate against safety** by running each action through
   :func:`safety_service.validate`. This is the critical line: the
   planner cannot bypass the safety gate because every action still
   goes through it. The result is a list of ``Action`` objects that is
   safe to feed to ``ActionRegistry.dispatch``.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from safety_service import validate as safety_validate
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action

from .prompt import PlannerAction, PlannerResponse


class PlanError(ValueError):
    """The LLM's plan could not be turned into a safe Action[]."""


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of ``text``."""
    for match in _JSON_FENCE.finditer(text):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Fall back to the first balanced {...} block.
    start = text.find("{")
    if start == -1:
        raise PlanError(f"planner output contains no JSON object: {text!r}")
    depth = 0
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as err:
                    raise PlanError(f"planner output is not valid JSON: {err}") from err
    raise PlanError(f"planner output has unbalanced braces: {text!r}")


def parse_plan(text: str, registry: ActionRegistry) -> list[Action]:
    """Turn raw LLM text into a safety-validated list of ``Action``.

    Raises :class:`PlanError` for any structural problem, and
    :class:`UnsafeActionError` (re-raised from ``safety_service``) for
    any action the safety gate rejects.
    """
    raw = _extract_json(text)
    try:
        response = PlannerResponse.model_validate(raw)
    except Exception as err:  # pydantic.ValidationError
        raise PlanError(f"planner output does not match schema: {err}") from err

    known_kinds = set(registry.kinds())
    return [_to_action(a, known_kinds) for a in response.actions]


def _to_action(item: PlannerAction, known_kinds: Iterable[str]) -> Action:
    if item.kind not in known_kinds:
        raise PlanError(
            f"planner emitted unknown action kind {item.kind!r}; "
            f"known kinds: {sorted(known_kinds)}"
        )
    action = Action(kind=item.kind, params=dict(item.params))
    # The safety gate is the single source of truth for "is this safe".
    safety_validate(action)
    return action
