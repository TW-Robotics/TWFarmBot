from planning_service.reasoning import (
    effort_from_budget,
    model_uses_reasoning,
    openai_effort,
)


def test_effort_from_budget_snaps_to_named_levels() -> None:
    assert effort_from_budget(0) == "none"
    assert effort_from_budget(2048) == "low"
    assert effort_from_budget(8192) == "medium"
    assert effort_from_budget(16384) == "high"
    assert effort_from_budget(32768) == "xhigh"


def test_openai_extra_high_is_model_specific() -> None:
    assert openai_effort("gpt-5", "xhigh") == "high"
    assert openai_effort("gpt-5.6", "xhigh") == "xhigh"
    assert openai_effort("openai/gpt-5.6", "medium") == "medium"


def test_model_uses_reasoning_is_id_specific() -> None:
    assert model_uses_reasoning("openai/gpt-4o") is False
    assert model_uses_reasoning("llama3") is False
    assert model_uses_reasoning("gpt-5.6") is True
    assert model_uses_reasoning("z-ai/glm-5.3-flash") is True
    assert model_uses_reasoning("google/gemini-2.5-flash") is True
    assert model_uses_reasoning("anthropic/claude-3.5-sonnet") is False
    assert model_uses_reasoning("o3-mini") is True
