# planning_service

LLM task planner for the FarmBot. Takes a natural-language request and
returns a validated list of [`Action`][twfarmbot_core.domain.Action]
objects ready to dispatch through the existing
[`ActionRegistry`][twfarmbot_core.actions.ActionRegistry].

## Backends

The planner uses [LangChain](https://python.langchain.com/) and the
OpenAI-compatible `/chat/completions` protocol, so it works with:

- **OpenRouter** — dozens of hosted models, one API key.
  ```
  PLANNING_LLM_BASE_URL=https://openrouter.ai/api/v1
  PLANNING_LLM_MODEL=anthropic/claude-3.5-sonnet
  PLANNING_LLM_API_KEY=sk-or-...
  ```
- **Self-hosted checkpoint** — anything that speaks OpenAI's
  `/chat/completions` (`llama.cpp`, vLLM, Ollama with
  `OLLAMA_OPENAI_COMPAT=true`, TGI, etc.).
  ```
  PLANNING_LLM_BASE_URL=http://localhost:8000/v1
  PLANNING_LLM_MODEL=my-checkpoint-name
  PLANNING_LLM_API_KEY=                   # optional for local
  ```

No code change is needed to swap backends — set the env vars.

## Safety

Every plan runs through [`safety_service.validate`][safety_service]
before it is returned. The planner cannot bypass the safety gate;
unsafe actions are rejected and surfaced to the caller.

## Usage

```python
from planning_service import plan

actions = plan("water the garden for 90 seconds, then move to home")
# -> [Action(kind='water', params={...}), Action(kind='move', params={...})]
```

The returned list is safe to feed to `ActionRegistry.dispatch(...)` or
to the `POST /actions` endpoint.
