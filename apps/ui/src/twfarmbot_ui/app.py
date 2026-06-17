"""Minimal UI for TWFarmBot.

A single-page Streamlit app that proxies HTTP calls to the api_server.
No business logic — every read hits a ``GET /...`` route on the API,
every write hits ``POST /actions``. The API is the only thing that
talks to the FarmBot.

Run:
    # terminal 1
    uv run twfarmbot-api
    # terminal 2
    uv run twfarmbot-ui
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("TWFB_API_URL", "http://127.0.0.1:8000")


# ---------- API client -----------------------------------------------------

@dataclass
class ApiResult:
    ok: bool
    code: int
    body: Any


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, **kwargs: Any) -> ApiResult:
        try:
            r = httpx.request(
                method, f"{self.base_url}{path}", timeout=10.0, **kwargs
            )
            try:
                body: Any = r.json()
            except ValueError:
                body = r.text
            return ApiResult(ok=r.is_success, code=r.status_code, body=body)
        except httpx.HTTPError as err:
            return ApiResult(ok=False, code=0, body={"error": f"{type(err).__name__}: {err}"})


@st.cache_resource
def _client(base_url: str) -> ApiClient:
    return ApiClient(base_url)


# ---------- Page setup -----------------------------------------------------

st.set_page_config(
    page_title="TWFarmBot",
    page_icon="\U0001F33E",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS — keep it inline + minimal so it works without a separate theme file.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1100px; }
      h1 { font-weight: 600; letter-spacing: -0.02em; }
      .pill {
        display: inline-block; padding: 0.15rem 0.6rem;
        border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        border: 1px solid rgba(0,0,0,0.08);
      }
      .pill.ok      { background: #d1fae5; color: #065f46; }
      .pill.warn    { background: #fef3c7; color: #92400e; }
      .pill.bad     { background: #fee2e2; color: #991b1b; }
      .pill.idle    { background: #e5e7eb; color: #374151; }
      .stat-label   { color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
      .stat-value   { font-size: 1.4rem; font-weight: 600; }
      section[data-testid="stSidebar"] { display: none; }
      div[data-testid="stMetric"] { background: #f9fafb; padding: 0.75rem 1rem; border-radius: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def status_pill(text: str, kind: str = "idle") -> None:
    st.markdown(f'<span class="pill {kind}">{text}</span>', unsafe_allow_html=True)


# ---------- Header ---------------------------------------------------------

col_title, col_status = st.columns([3, 2], vertical_alignment="center")
with col_title:
    st.markdown("# \U0001F33E TWFarmBot")
with col_status:
    api_url = st.text_input("API URL", value=API_URL, label_visibility="collapsed", placeholder=API_URL)
    client = _client(api_url or API_URL)

    health = client.request("GET", "/health")
    if health.ok and isinstance(health.body, dict):
        fb = health.body.get("farmbot", "?")
        if fb == "connected":
            status_pill(f"● FarmBot connected  ·  {api_url or API_URL}", "ok")
        elif fb == "skipped":
            status_pill(f"○ FarmBot skipped (FARMBOT_REQUIRED=0)  ·  {api_url or API_URL}", "warn")
        elif str(fb).startswith("failed"):
            status_pill(f"✕ {fb}", "bad")
        else:
            status_pill(f"○ FarmBot: {fb}  ·  {api_url or API_URL}", "warn")
    else:
        status_pill(f"API unreachable  ·  {api_url or API_URL}", "bad")

st.caption("Live reads · inspection · actions — every request is routed through the api_server.")
st.divider()


# ---------- Tabs -----------------------------------------------------------

tab_live, tab_inspect, tab_act = st.tabs(["\U0001F4CA  Live", "\U0001F50D  Inspect", "\U0001F4E4  Act"])


# ---------- Live -----------------------------------------------------------

with tab_live:
    st.markdown("##### Position")
    pos_row = st.columns([1, 1, 1, 1])
    if pos_row[3].button("Refresh", key="refresh_position", use_container_width=True):
        st.session_state["position"] = client.request("GET", "/position")
    pos = st.session_state.get("position")
    if pos and pos.ok and isinstance(pos.body, dict):
        xyz = pos.body.get("xyz") or {}
        pos_row[0].markdown(
            f'<div class="stat-label">X (mm)</div><div class="stat-value">{xyz.get("x", "—")}</div>',
            unsafe_allow_html=True,
        )
        pos_row[1].markdown(
            f'<div class="stat-label">Y (mm)</div><div class="stat-value">{xyz.get("y", "—")}</div>',
            unsafe_allow_html=True,
        )
        pos_row[2].markdown(
            f'<div class="stat-label">Z (mm)</div><div class="stat-value">{xyz.get("z", "—")}</div>',
            unsafe_allow_html=True,
        )
    elif pos and not pos.ok:
        st.error(f"position: http {pos.code} — {pos.body}")

    st.markdown("##### Messages")
    msg_row = st.columns([1, 4])
    if msg_row[0].button("Refresh", key="refresh_messages", use_container_width=True):
        st.session_state["messages"] = client.request("GET", "/messages")
    msgs = st.session_state.get("messages")
    if msgs and msgs.ok and isinstance(msgs.body, dict):
        last = msgs.body.get("last_messages")
        if isinstance(last, list) and last:
            st.code("\n".join(str(m) for m in last[-20:]), language="text")
        elif isinstance(last, dict) and last:
            st.json(last)
        else:
            st.info("No messages from the FarmBot yet.")
    elif msgs and not msgs.ok:
        st.error(f"messages: http {msgs.code} — {msgs.body}")

    st.markdown("##### Pin")
    with st.form("read_pin"):
        pin_cols = st.columns([1, 1, 1])
        pin = pin_cols[0].number_input("Pin", min_value=0, max_value=64, value=13, step=1)
        mode = pin_cols[1].selectbox("Mode", ["digital", "analog"])
        read = pin_cols[2].form_submit_button("Read", use_container_width=True)
        if read:
            st.session_state["pin"] = client.request(
                "GET", f"/pin/{int(pin)}", params={"mode": mode}
            )
    pin_result = st.session_state.get("pin")
    if pin_result and pin_result.ok and isinstance(pin_result.body, dict):
        v = pin_result.body.get("value")
        st.markdown(
            f'<span class="pill ok">pin {pin_result.body.get("pin")} '
            f'({pin_result.body.get("mode")}) = {v}</span>',
            unsafe_allow_html=True,
        )
    elif pin_result and not pin_result.ok:
        st.error(f"pin: http {pin_result.code} — {pin_result.body}")


# ---------- Inspect --------------------------------------------------------

with tab_inspect:
    st.markdown("##### Status")
    with st.form("status_form"):
        cols = st.columns([3, 1])
        path = cols[0].text_input(
            "Path (optional, e.g. 'position')", value="", label_visibility="visible"
        )
        fetch = cols[1].form_submit_button("Fetch", use_container_width=True)
        if fetch:
            params = {"path": path} if path.strip() else None
            st.session_state["status"] = client.request("GET", "/status", params=params)
    status = st.session_state.get("status")
    if status and status.ok:
        st.json(status.body)
    elif status and not status.ok:
        st.error(f"status: http {status.code} — {status.body}")

    st.markdown("##### Health")
    if st.button("Refresh health", key="refresh_health"):
        st.session_state["health"] = client.request("GET", "/health")
    health = st.session_state.get("health")
    if health and health.ok and isinstance(health.body, dict):
        st.json(health.body)
    elif health and not health.ok:
        st.error(f"health: http {health.code} — {health.body}")


# ---------- Act ------------------------------------------------------------

with tab_act:
    st.markdown("##### Water a bed")
    with st.form("water_form", clear_on_submit=False):
        cols = st.columns([1, 1, 1])
        bed_id = cols[0].text_input("Bed ID", value="b1")
        seconds = cols[1].number_input("Seconds", min_value=0.1, max_value=300.0, value=2.0, step=0.5)
        go = cols[2].form_submit_button("💧 Water now", use_container_width=True, type="primary")
        if go:
            r = client.request(
                "POST", "/actions",
                json={"kind": "water", "params": {"bed_id": bed_id, "seconds": seconds}},
            )
            if r.ok:
                st.success(f"Watered {bed_id} for {seconds}s")
            else:
                st.error(f"HTTP {r.code}: {r.body}")

    st.markdown("##### Move the gantry")
    with st.form("move_form", clear_on_submit=False):
        cols = st.columns([1, 1, 1, 1])
        x = cols[0].number_input("X (mm)", value=0.0, step=10.0)
        y = cols[1].number_input("Y (mm)", value=0.0, step=10.0)
        z = cols[2].number_input("Z (mm)", value=0.0, step=10.0)
        go = cols[3].form_submit_button("➡️ Move", use_container_width=True, type="primary")
        if go:
            r = client.request(
                "POST", "/actions",
                json={"kind": "move", "params": {"x": x, "y": y, "z": z}},
            )
            if r.ok:
                st.success(f"Moved to ({x}, {y}, {z})")
            else:
                st.error(f"HTTP {r.code}: {r.body}")

    with st.expander("Advanced: raw action", expanded=False):
        st.caption("Bypasses the typed forms. The action still goes through safety_service on the server.")
        with st.form("raw_form"):
            kind = st.text_input("Kind", value="water")
            params_json = st.text_area(
                "Params (JSON)",
                value='{"bed_id": "b1", "seconds": 1.0}',
                height=120,
            )
            go = st.form_submit_button("Dispatch")
            if go:
                try:
                    params = json.loads(params_json) if params_json.strip() else {}
                except json.JSONDecodeError as err:
                    st.error(f"Bad JSON: {err}")
                    params = None
                if params is not None:
                    r = client.request(
                        "POST", "/actions", json={"kind": kind, "params": params}
                    )
                    if r.ok:
                        st.success("OK")
                        st.json(r.body)
                    else:
                        st.error(f"HTTP {r.code}: {r.body}")
