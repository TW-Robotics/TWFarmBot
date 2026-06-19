"""API client used by the Streamlit app.

Kept separate from ``app.py`` so tests can import it without dragging in
Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ApiResult:
    ok: bool
    code: int
    body: Any


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
            return ApiResult(ok=False, code=0, body={"error": f"{type(err).__name__}: {err}"})
