# planning_service

LLM task planner for the FarmBot. Takes a natural-language request and
returns a validated list of [`Action`][twfarmbot_core.domain.Action]
objects ready to dispatch through the existing
[`ActionRegistry`][twfarmbot_core.actions.ActionRegistry].

## Backends

The planner uses JSON function/tool calling. Native OpenAI GPT-5 uses the
Responses API so it can reason while using tools; OpenRouter and local models
use OpenAI-compatible `/chat/completions`. The model receives every tool
result in the next turn, then decides what information to gather or action to
take next. Physical FarmBot actions remain approval-gated and never talk to
motors or serial from the model.

`reasoning_budget` defaults to 8192 and is adjustable in **Settings → Chat
assistant** (or per browser model profile in **Advanced**). Vertex receives it
as Gemini's `thinking_budget`; OpenRouter receives `reasoning.max_tokens`;
GPT-5 maps it to low (≤2048), medium (≤8192), or high effort. Set it to `0`
to omit the explicit reasoning request.

Any backend that speaks `/chat/completions` works:

OpenAI:
```
PLANNING_LLM_PROVIDER=openai
PLANNING_LLM_BASE_URL=https://api.openai.com/v1
PLANNING_LLM_MODEL=gpt-5.6
PLANNING_LLM_API_KEY=sk-...
```

- **OpenRouter** — dozens of hosted models, one API key.
  ```
  PLANNING_LLM_PROVIDER=openrouter
  PLANNING_LLM_BASE_URL=https://openrouter.ai/api/v1
  PLANNING_LLM_MODEL=anthropic/claude-3.5-sonnet
  PLANNING_LLM_API_KEY=sk-or-...
  ```
- **Self-hosted checkpoint** — anything that speaks OpenAI's
  `/chat/completions` (`llama.cpp`, vLLM, Ollama with
  `OLLAMA_OPENAI_COMPAT=true`, TGI, etc.).
  ```
  PLANNING_LLM_PROVIDER=local
  PLANNING_LLM_BASE_URL=http://localhost:8000/v1
  PLANNING_LLM_MODEL=my-checkpoint-name
  PLANNING_LLM_API_KEY=                   # optional for local
  ```

## Safety

Every physical action runs through [`safety_service.validate`][safety_service]
before execution. The planner cannot bypass the safety gate; unsafe actions
are rejected and surfaced to the caller.

Read-only tools (`get_position`, `list_zones`, `get_images`, `get_garden`,
`take_photo`, `capture`, and the rest of introspection) are ordinary JSON tools. Physical jobs
(`inspect_zone`, `water_zone`, `goto_named`, `move`, `water`) stay
approval-gated function calls. The model never talks to motors or serial.

## Usage

```python
from planning_service import plan

actions = plan("water the garden for 90 seconds, then move to home")
# -> [Action(kind='water', params={...}), Action(kind='move', params={...})]
```

The returned list is safe to feed to `ActionRegistry.dispatch(...)` or
to the `POST /actions` endpoint.
