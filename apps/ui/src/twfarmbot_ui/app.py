"""Minimal UI for TWFarmBot.

A single-page Streamlit app that proxies HTTP calls to the api_server.
No business logic — every action goes through ``POST /actions`` and
every read goes through the matching ``GET /...`` route on the API,
which is the only place that talks to the FarmBot.

Run:
    # in one terminal
    uv run twfarmbot-api
    # in another
    uv run twfarmbot-ui
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("TWFB_API_URL", "http://127.0.0.1:8000")


def _request(method: str, path: str, **kwargs: Any) -> tuple[int, Any]:
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=10.0, **kwargs)
        try:
            body: Any = r.json()
        except ValueError:
            body = r.text
        return r.status_code, body
    except httpx.HTTPError as err:
        return 0, {"error": f"{type(err).__name__}: {err}"}


st.set_page_config(page_title="TWFarmBot", page_icon="\U0001F33E", layout="centered")
st.title("TWFarmBot")

with st.sidebar:
    st.header("API")
    api_url = st.text_input("API URL", value=API_URL)
    if api_url != API_URL:
        API_URL = api_url  # noqa: F841 — picked up on next rerun
    if st.button("Check status"):
        st.session_state["last_status"] = _request("GET", "/health")

status = st.session_state.get("last_status")
if status:
    code, body = status
    if code == 200 and isinstance(body, dict):
        st.success(
            f"API up — actions: {', '.join(body.get('actions', []))} · "
            f"farmbot: {body.get('farmbot', '?')}"
        )
    else:
        st.error(f"status {code}: {body}")

st.divider()

# ---------- Reads ---------------------------------------------------------

st.subheader("Live reads")
col1, col2 = st.columns(2)

with col1:
    if st.button("Position"):
        st.session_state["last_position"] = _request("GET", "/position")
    pos = st.session_state.get("last_position")
    if pos:
        code, body = pos
        if code == 200:
            xyz = body.get("xyz")
            st.metric("X (mm)", value=xyz.get("x") if isinstance(xyz, dict) else "—")
            st.metric("Y (mm)", value=xyz.get("y") if isinstance(xyz, dict) else "—")
            st.metric("Z (mm)", value=xyz.get("z") if isinstance(xyz, dict) else "—")
        else:
            st.error(f"http {code}: {body}")

with col2:
    if st.button("Messages"):
        st.session_state["last_messages"] = _request("GET", "/messages")
    msgs = st.session_state.get("last_messages")
    if msgs:
        code, body = msgs
        if code == 200:
            last = body.get("last_messages") or []
            if isinstance(last, list) and last:
                st.write("\n".join(f"• {m}" for m in last[-10:]))
            else:
                st.info("no messages yet")
        else:
            st.error(f"http {code}: {body}")

st.divider()

# Read pin by number
st.subheader("Read pin")
with st.form("read_pin"):
    pin = st.number_input("Pin number", min_value=0, max_value=64, value=13, step=1)
    mode = st.selectbox("Mode", ["digital", "analog"])
    submitted = st.form_submit_button("Read")
    if submitted:
        code, body = _request("GET", f"/pin/{int(pin)}", params={"mode": mode})
        if code == 200:
            st.success(f"pin {body['pin']} ({body['mode']}) = {body['value']}")
        else:
            st.error(f"http {code}: {body}")

st.divider()

# Read status (optionally filtered to a sub-path)
st.subheader("Status")
with st.form("status"):
    path = st.text_input("Path (optional, e.g. 'position')", value="")
    submitted = st.form_submit_button("Fetch")
    if submitted:
        params = {"path": path} if path.strip() else None
        code, body = _request("GET", "/status", params=params)
        if code == 200:
            st.json(body)
        else:
            st.error(f"http {code}: {body}")

st.divider()

# ---------- Writes --------------------------------------------------------

st.subheader("Water")
with st.form("water"):
    bed_id = st.text_input("Bed ID", value="b1")
    seconds = st.number_input("Seconds", min_value=0.1, max_value=300.0, value=2.0, step=0.5)
    submitted = st.form_submit_button("Water now")
    if submitted:
        code, body = _request(
            "POST", "/actions",
            json={"kind": "water", "params": {"bed_id": bed_id, "seconds": seconds}},
        )
        if code == 200:
            st.success(f"OK — {body}")
        else:
            st.error(f"HTTP {code}: {body}")

st.divider()

st.subheader("Move")
with st.form("move"):
    x = st.number_input("X (mm)", value=0.0, step=10.0)
    y = st.number_input("Y (mm)", value=0.0, step=10.0)
    z = st.number_input("Z (mm)", value=0.0, step=10.0)
    submitted = st.form_submit_button("Move")
    if submitted:
        code, body = _request(
            "POST", "/actions",
            json={"kind": "move", "params": {"x": x, "y": y, "z": z}},
        )
        if code == 200:
            st.success(f"OK — {body}")
        else:
            st.error(f"HTTP {code}: {body}")

st.divider()

st.subheader("Raw action")
with st.form("raw_action"):
    kind = st.text_input("Kind", value="water")
    params_json = st.text_area(
        "Params (JSON)", value='{"bed_id": "b1", "seconds": 1.0}'
    )
    submitted = st.form_submit_button("Dispatch")
    if submitted:
        try:
            params = json.loads(params_json) if params_json.strip() else {}
        except json.JSONDecodeError as err:
            st.error(f"bad params JSON: {err}")
            params = None
        if params is not None:
            code, body = _request(
                "POST", "/actions", json={"kind": kind, "params": params}
            )
            if code == 200:
                st.success(f"OK — {body}")
            else:
                st.error(f"HTTP {code}: {body}")
