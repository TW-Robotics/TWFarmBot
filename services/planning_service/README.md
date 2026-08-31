# planning_service

LLM task planner for the FarmBot. Takes a natural-language request and
returns a validated list of [`Action`][twfarmbot_core.domain.Action]
objects ready to dispatch through the existing
[`ActionRegistry`][twfarmbot_core.actions.ActionRegistry].

## Backends

The planner defaults to the OpenAI Responses API with native Programmatic
Tool Calling (PTC). PTC runs bounded read-only orchestration in OpenAI's
hosted JavaScript runtime. Physical FarmBot actions remain direct,
approval-gated calls.

The previous LangChain `/chat/completions` providers and local script
prototype are retained but disabled by default. Set
`PLANNING_LLM_ENABLE_LEGACY=1` to re-enable them.

OpenAI configuration:
```
PLANNING_LLM_PROVIDER=openai
PLANNING_LLM_BASE_URL=https://api.openai.com/v1
PLANNING_LLM_MODEL=gpt-5.6
PLANNING_LLM_API_KEY=sk-...
```

- **Legacy OpenRouter** — dozens of hosted models, one API key.
  ```
  PLANNING_LLM_BASE_URL=https://openrouter.ai/api/v1
  PLANNING_LLM_MODEL=anthropic/claude-3.5-sonnet
  PLANNING_LLM_API_KEY=sk-or-...
  ```
- **Legacy self-hosted checkpoint** — anything that speaks OpenAI's
  `/chat/completions` (`llama.cpp`, vLLM, Ollama with
  `OLLAMA_OPENAI_COMPAT=true`, TGI, etc.).
  ```
  PLANNING_LLM_BASE_URL=http://localhost:8000/v1
  PLANNING_LLM_MODEL=my-checkpoint-name
  PLANNING_LLM_API_KEY=                   # optional for local
  ```

The legacy backends are opt-in with `PLANNING_LLM_ENABLE_LEGACY=1`.

## Safety

Every physical action runs through [`safety_service.validate`][safety_service]
before execution. The planner cannot bypass the safety gate; unsafe actions
are rejected and surfaced to the caller.

Read-only tools declare `output_schema` and `allowed_callers: ["programmatic"]`
so hosted PTC can loop, filter, and reduce results. Physical jobs
(`inspect_zone`, `water_zone`, `goto_named`, `move`, `water`) stay direct,
approval-gated function calls.

## Usage

```python
from planning_service import plan

actions = plan("water the garden for 90 seconds, then move to home")
# -> [Action(kind='water', params={...}), Action(kind='move', params={...})]
```

The returned list is safe to feed to `ActionRegistry.dispatch(...)` or
to the `POST /actions` endpoint.
