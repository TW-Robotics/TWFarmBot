"""API client used by the Streamlit app.

Kept separate from ``app.py`` so tests can import it without dragging in
Streamlit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import httpx


@dataclass
class ApiResult:
    ok: bool
    code: int
    body: Any

    def error_message(self) -> str:
        """Return a human-readable backend error from the response body.

        FastAPI errors are shaped like ``{"detail": "..."}``; legacy or
        connection-failure bodies may use ``{"error": "..."}``. Fall back to
        the raw body text when neither key is present.
        """
        if isinstance(self.body, dict):
            if "detail" in self.body:
                return str(self.body["detail"])
            if "error" in self.body:
                return str(self.body["error"])
        if isinstance(self.body, str):
            return self.body
        return str(self.body)


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 1.0)),
        )

    def request(self, method: str, path: str, **kwargs: Any) -> ApiResult:
        try:
            if method.upper() == "POST" and path == "/actions":
                params = dict(kwargs.pop("params", {}) or {})
                params.setdefault("wait", "false")
                kwargs["params"] = params
            per_request_timeout = kwargs.pop("timeout", None)
            if per_request_timeout is not None:
                kwargs["timeout"] = httpx.Timeout(
                    per_request_timeout, connect=min(per_request_timeout, 1.0)
                )
            r = self._client.request(method, path, **kwargs)
            try:
                body: Any = r.json()
            except ValueError:
                body = r.text
            return ApiResult(ok=r.is_success, code=r.status_code, body=body)
        except httpx.HTTPError as err:
            return ApiResult(
                ok=False, code=0, body={"error": f"{type(err).__name__}: {err}"}
            )

    def stream(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream server-sent events from the API.

        Yields parsed ``data:`` payloads as dicts. Raises on HTTP errors so
        the caller can fall back to a non-streaming request.
        """
        per_request_timeout = kwargs.pop("timeout", None)
        if per_request_timeout is not None:
            kwargs["timeout"] = httpx.Timeout(
                per_request_timeout, connect=min(per_request_timeout, 1.0)
            )
        with self._client.stream(method, path, **kwargs) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                yield json.loads(line[6:])

    # ---- Refresh helpers ------------------------------------------------

    def get_position(self) -> dict[str, Any] | None:
        r = self.request("GET", "/position")
        if r.ok and isinstance(r.body, dict):
            return r.body.get("xyz") or {}
        return None

    def get_health(self) -> dict[str, Any] | None:
        r = self.request("GET", "/health")
        if r.ok and isinstance(r.body, dict):
            return r.body
        return None

    def get_messages(self) -> list[str]:
        r = self.request("GET", "/messages")
        if not r.ok or not isinstance(r.body, dict):
            return []
        raw = r.body.get("last_messages")
        if isinstance(raw, list):
            return [str(m) for m in raw[-20:]]
        return []
