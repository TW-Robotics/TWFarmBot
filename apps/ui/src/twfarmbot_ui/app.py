"""Minimal UI for TWFarmBot.

A single-page Streamlit app that proxies HTTP calls to the api_server.
No business logic — every read hits a ``GET /...`` route, every write
hits ``POST /actions``. The API is the only thing that talks to the
FarmBot.

Run:
    # terminal 1
    uv run twfarmbot-api
    # terminal 2
    uv run twfarmbot-ui
"""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st

from twfarmbot_ui.client import ApiClient, ApiResult

API_URL = os.getenv("TWFB_API_URL", "http://127.0.0.1:8000")


# ---------- API client -----------------------------------------------------

@st.cache_resource
def _client(base_url: str) -> ApiClient:
    return ApiClient(base_url)


# ---------- Action helpers -------------------------------------------------

def _do_pin_write(client: ApiClient, selected: dict, value: int) -> None:
    r = client.request(
        "POST", "/actions",
        json={
            "kind": "write_pin",
            "params": {
                "pin": int(selected["pin"]),
                "value": value,
                "mode": selected.get("mode", "digital"),
            },
        },
    )
    if r.ok:
        st.toast(f"pin {selected['pin']} = {value}", icon="✏️")
        st.session_state["pin"] = client.request(
            "GET", f"/pin/{int(selected['pin'])}",
            params={"mode": selected.get("mode", "digital")},
        )
    else:
        st.error(f"HTTP {r.code}: {r.body}")


def _do_move(client: ApiClient, x: float, y: float, z: float, label: str = "") -> None:
    r = client.request(
        "POST", "/actions",
        json={"kind": "move", "params": {"x": x, "y": y, "z": z}},
    )
    if r.ok:
        msg = f"→ {label}" if label else f"→ ({x:.0f}, {y:.0f}, {z:.0f})"
        st.toast(msg, icon="➡️")
        # Keep the UI responsive. The explicit telemetry refresh reconciles
        # these optimistic values with the robot when the user asks for it.
        st.session_state["pos_x"] = _as_num(x)
        st.session_state["pos_y"] = _as_num(y)
        st.session_state["pos_z"] = _as_num(z)
    else:
        st.error(f"HTTP {r.code}: {r.body}")


# ---------- Telemetry helpers ---------------------------------------------

def _as_num(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "—"


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert cached telemetry to a coordinate without crashing on placeholders."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def refresh_position(client: ApiClient) -> None:
    """Refresh the small position snapshot without probing full robot state."""
    pos = client.request("GET", "/position")
    if pos.ok and isinstance(pos.body, dict):
        xyz = pos.body.get("xyz") or {}
        st.session_state["pos_x"] = _as_num(xyz.get("x"))
        st.session_state["pos_y"] = _as_num(xyz.get("y"))
        st.session_state["pos_z"] = _as_num(xyz.get("z"))
    else:
        st.session_state["pos_x"] = "—"
        st.session_state["pos_y"] = "—"
        st.session_state["pos_z"] = "—"


@st.fragment(run_every="1s")
def live_position(client: ApiClient) -> None:
    """Render and continuously reconcile the cached MQTT position."""
    pos = client.request("GET", "/position")
    if pos.ok and isinstance(pos.body, dict):
        xyz = pos.body.get("xyz") or {}
        if xyz:
            st.session_state["pos_x"] = _as_num(xyz.get("x"))
            st.session_state["pos_y"] = _as_num(xyz.get("y"))
            st.session_state["pos_z"] = _as_num(xyz.get("z"))

    cur_cols = st.columns(3)
    cur_cols[0].metric("X (mm)", st.session_state.get("pos_x", "—"))
    cur_cols[1].metric("Y (mm)", st.session_state.get("pos_y", "—"))
    cur_cols[2].metric("Z (mm)", st.session_state.get("pos_z", "—"))
    if not pos.ok:
        st.caption("Position temporarily unavailable; retaining the last reading.")


def refresh_messages(client: ApiClient) -> None:
    msgs = client.request("GET", "/messages")
    if msgs.ok and isinstance(msgs.body, dict):
        raw = msgs.body.get("last_messages")
        if isinstance(raw, list):
            st.session_state["messages"] = [str(m) for m in raw[-20:]]
        else:
            # FarmBot sometimes returns its entire status tree here. That is
            # diagnostic state, not a human-readable message feed.
            st.session_state["messages"] = []
    else:
        st.session_state["messages"] = []


def refresh_health(client: ApiClient) -> None:
    health = client.request("GET", "/health")
    if health.ok and isinstance(health.body, dict):
        st.session_state["actions"] = health.body.get("actions", [])
        st.session_state["farmbot_status"] = health.body.get("farmbot", "?")
    else:
        st.session_state["actions"] = []
        st.session_state["farmbot_status"] = "unreachable"


def refresh_telemetry(client: ApiClient) -> None:
    """Explicit full refresh initiated by the user."""
    refresh_health(client)
    refresh_position(client)
    refresh_messages(client)


# ---------- Page setup -----------------------------------------------------

st.set_page_config(
    page_title="TWFarmBot",
    page_icon="\U0001F33E",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 4rem; }
      h1 { font-weight: 600; letter-spacing: -0.02em; font-size: 1.8rem; }
      .stat-card {
        background: #f9fafb;
        border-radius: 0.5rem;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
      }
      .stat-label { color: #6b7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
      .stat-value { font-size: 1.3rem; font-weight: 600; }
      .pill {
        display: inline-block; padding: 0.15rem 0.6rem;
        border-radius: 999px; font-size: 0.75rem; font-weight: 600;
      }
      .pill.ok      { background: #d1fae5; color: #065f46; }
      .pill.warn    { background: #fef3c7; color: #92400e; }
      .pill.bad     { background: #fee2e2; color: #991b1b; }
      .pill.idle    { background: #e5e7eb; color: #374151; }
      .log-line { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; color: #374151; padding: 0.1rem 0; border-bottom: 1px solid #f3f4f6; }
      .log-empty { color: #9ca3af; font-style: italic; }
    </style>
    """,
    unsafe_allow_html=True,
)


def status_pill(text: str, kind: str = "idle") -> None:
    st.markdown(f'<span class="pill {kind}">{text}</span>', unsafe_allow_html=True)


# ---------- Sidebar --------------------------------------------------------

api_url = st.sidebar.text_input("API URL", value=API_URL)
client = _client(api_url or API_URL)

# On first load, probe health and load the API's cached position snapshot.
# This does not reconnect to FarmBot or issue a robot command.
if "farmbot_status" not in st.session_state:
    refresh_health(client)
    refresh_position(client)
    st.session_state.setdefault("messages", [])

fb_status = st.session_state.get("farmbot_status", "unknown")
if fb_status == "connected":
    status_pill("● FarmBot connected", "ok")
elif fb_status == "skipped":
    status_pill("○ FarmBot skipped", "warn")
elif str(fb_status).startswith("failed"):
    status_pill(f"✕ {fb_status}", "bad")
else:
    status_pill(f"○ {fb_status}", "warn")

st.sidebar.divider()
st.sidebar.markdown("**Telemetry**")
if st.sidebar.button("↻ Refresh", use_container_width=True):
    refresh_telemetry(client)
    st.rerun()

st.sidebar.markdown(
    f"""
    <div class="stat-card">
      <div class="stat-label">Position (mm)</div>
      <div class="stat-value">X {st.session_state.get('pos_x', '—')}  ·  Y {st.session_state.get('pos_y', '—')}  ·  Z {st.session_state.get('pos_z', '—')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("**Recent messages**")
messages = st.session_state.get("messages", [])
if messages:
    for line in messages[-8:]:
        st.sidebar.markdown(f'<div class="log-line">{line}</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="log-empty">No messages yet</div>', unsafe_allow_html=True)

actions = st.session_state.get("actions", [])
if actions:
    st.sidebar.divider()
    st.sidebar.caption(f"Actions: {', '.join(actions)}")


# ---------- Main header ---------------------------------------------------

st.markdown("# \U0001F33E TWFarmBot")
st.caption("Control panel for the FarmBot at UAS Technikum Wien.")
st.divider()


# ---------- Tabs ----------------------------------------------------------

tab_dashboard, tab_move, tab_camera, tab_pins, tab_actions = st.tabs(
    ["Dashboard", "Move", "Camera", "Pins", "Actions"]
)


# ---------- Dashboard -----------------------------------------------------

with tab_dashboard:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
              <div class="stat-label">X position</div>
              <div class="stat-value">{st.session_state.get('pos_x', '—')} mm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
              <div class="stat-label">Y position</div>
              <div class="stat-value">{st.session_state.get('pos_y', '—')} mm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
              <div class="stat-label">Z position</div>
              <div class="stat-value">{st.session_state.get('pos_z', '—')} mm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Messages**")
        if messages:
            st.code("\n".join(messages[-20:]), language="text")
        else:
            st.info("No messages from the FarmBot yet.")

    with col_right:
        st.markdown("**Status**")
        if st.button("Load diagnostic status", use_container_width=True):
            st.session_state["diagnostic_status"] = client.request("GET", "/status")
        status_resp = st.session_state.get("diagnostic_status")
        if status_resp and status_resp.ok:
            st.json(status_resp.body)
        elif status_resp:
            st.error(f"status: http {status_resp.code} — {status_resp.body}")
        else:
            st.caption("Not loaded. Full robot status is fetched only on demand.")


# ---------- Move ----------------------------------------------------------

with tab_move:
    st.markdown("**Current position**")
    live_position(client)

    st.divider()
    st.markdown("**Jog**")

    step = float(st.radio("Step", options=["1", "10", "50", "100"], index=1, horizontal=True))

    cur_x = _as_float(st.session_state.get("pos_x"))
    cur_y = _as_float(st.session_state.get("pos_y"))
    cur_z = _as_float(st.session_state.get("pos_z"))

    pad = st.columns([1, 1, 1, 1, 1])
    if pad[1].button("▲ Y+", use_container_width=True):
        _do_move(client, cur_x, cur_y + step, cur_z, f"Y+{step:.0f}")
    if pad[2].button("🏠 Home", use_container_width=True):
        _do_move(client, 0.0, 0.0, 0.0, "Home")
    if pad[3].button("▲ Y-", use_container_width=True):
        _do_move(client, cur_x, cur_y - step, cur_z, f"Y-{step:.0f}")
    if pad[0].button("◀ X-", use_container_width=True):
        _do_move(client, cur_x - step, cur_y, cur_z, f"X-{step:.0f}")
    if pad[4].button("X+ ▶", use_container_width=True):
        _do_move(client, cur_x + step, cur_y, cur_z, f"X+{step:.0f}")

    zrow = st.columns([1, 1, 1])
    if zrow[0].button("⬆ Z+", use_container_width=True):
        _do_move(client, cur_x, cur_y, cur_z + step, f"Z+{step:.0f}")
    if zrow[1].button("🔍 Find home", use_container_width=True):
        r = client.request("POST", "/actions", json={"kind": "find_home", "params": {}})
        if r.ok:
            st.toast("Find home sequence sent", icon="🏠")
            st.session_state["pos_x"] = "0.0"
            st.session_state["pos_y"] = "0.0"
            st.session_state["pos_z"] = "0.0"
        else:
            st.error(f"find_home failed: HTTP {r.code} {r.body}")
    if zrow[2].button("⬇ Z-", use_container_width=True):
        _do_move(client, cur_x, cur_y, cur_z - step, f"Z-{step:.0f}")

    st.divider()
    st.markdown("**Presets**")
    if "presets" not in st.session_state:
        preset_resp = client.request("GET", "/positions")
        st.session_state["presets"] = preset_resp.body.get("positions", []) if preset_resp.ok else []
    presets = st.session_state.get("presets", [])
    if presets:
        pcols = st.columns(min(len(presets), 6))
        for i, p in enumerate(presets):
            if pcols[i].button(
                f"📍 {p.get('label', '?')}",
                key=f"preset_{i}",
                use_container_width=True,
                help=f"x={p.get('x')}, y={p.get('y')}, z={p.get('z')}",
            ):
                _do_move(
                    client,
                    float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)),
                    p.get("label", "preset"),
                )

    with st.expander("Go to absolute coordinates"):
        gx, gy, gz = st.columns(3)
        gtx = gx.number_input("X", value=cur_x, step=10.0)
        gty = gy.number_input("Y", value=cur_y, step=10.0)
        gtz = gz.number_input("Z", value=cur_z, step=10.0)
        if st.button("🎯 Go", type="primary"):
            _do_move(client, float(gtx), float(gty), float(gtz), f"({gtx:.0f}, {gty:.0f}, {gtz:.0f})")


# ---------- Pins ----------------------------------------------------------

with tab_camera:
    camera_actions = st.columns([1, 1, 4])
    if camera_actions[0].button("📷 Take photo", type="primary", use_container_width=True):
        result = client.request(
            "POST", "/actions", json={"kind": "take_photo", "params": {}}
        )
        if result.ok:
            st.toast("Photo capture queued", icon="📷")
            st.info("FarmBot is capturing and uploading the image. Refresh photos in a few seconds.")
        else:
            st.error(f"take_photo failed: HTTP {result.code} {result.body}")

    if camera_actions[1].button("↻ Refresh photos", use_container_width=True):
        st.session_state["images"] = client.request("GET", "/images", timeout=10.0)
        st.session_state["images_loaded"] = True

    image_result = st.session_state.get("images")
    if image_result and image_result.ok and isinstance(image_result.body, dict):
        images = image_result.body.get("images", [])
        if images:
            latest = images[0]
            meta = latest.get("meta") or {}
            st.image(
                latest.get("attachment_url"),
                caption=(
                    f"{latest.get('created_at', 'Unknown time')} · "
                    f"X {meta.get('x', '—')} · Y {meta.get('y', '—')} · Z {meta.get('z', '—')} mm"
                ),
                use_container_width=True,
            )
            with st.expander(f"Earlier photos ({max(0, len(images) - 1)})"):
                for image in images[1:]:
                    image_meta = image.get("meta") or {}
                    st.image(
                        image.get("attachment_url"),
                        caption=(
                            f"{image.get('created_at', 'Unknown time')} · "
                            f"X {image_meta.get('x', '—')} · Y {image_meta.get('y', '—')}"
                        ),
                        width=420,
                    )
        else:
            st.info("No uploaded FarmBot photos found.")
    elif image_result:
        st.error(f"images: HTTP {image_result.code} — {image_result.body}")
    else:
        st.caption("Photos load on request and are cached for one minute to protect the FarmBot API.")


# ---------- Pins ----------------------------------------------------------

with tab_pins:
    if "named_pins" not in st.session_state:
        pin_resp = client.request("GET", "/pins")
        st.session_state["named_pins"] = pin_resp.body.get("pins", []) if pin_resp.ok else []
    named_pins = st.session_state.get("named_pins", [])

    if named_pins:
        seen_pins: set[int] = set()
        dups: set[int] = set()
        for p in named_pins:
            pin_num = p.get("pin")
            if isinstance(pin_num, int):
                if pin_num in seen_pins:
                    dups.add(pin_num)
                seen_pins.add(pin_num)
        if dups:
            st.warning(
                f"Duplicate pin numbers in configs/dev.yaml: {sorted(dups)}. "
                "A GPIO can only have one function — fix the config."
            )

        groups: dict[str, list[dict]] = {}
        for p in named_pins:
            groups.setdefault(p.get("group", "Other"), []).append(p)

        for group, items in groups.items():
            st.markdown(f"**{group}**")
            grid = st.columns(min(4, len(items)))
            for i, p in enumerate(items):
                col = grid[i % len(grid)]
                kind = p.get("kind", "io")
                icon = {"valve": "💧", "servo": "⚙️", "sensor": "📈", "io": "🔌"}.get(kind, "•")
                label = f"{icon} {p.get('label', '?')}"
                if col.button(label, key=f"pinbtn_{group}_{i}", use_container_width=True):
                    st.session_state["pin_preselect"] = p

    selected = st.session_state.get("pin_preselect")
    if selected is None and named_pins:
        selected = named_pins[0]
        st.session_state["pin_preselect"] = selected

    if selected:
        st.divider()
        head = st.columns([4, 1])
        head[0].markdown(
            f"**{selected.get('label', '?')}** — pin {selected.get('pin')} · "
            f"{selected.get('mode', 'digital')} · `{selected.get('kind', 'io')}`"
        )
        if head[1].button("↻ Read", key="pin_read", use_container_width=True):
            r = client.request(
                "GET", f"/pin/{int(selected['pin'])}",
                params={"mode": selected.get("mode", "digital")},
            )
            st.session_state["pin"] = r

        pin_result = st.session_state.get("pin")
        if pin_result and pin_result.ok and isinstance(pin_result.body, dict):
            v = pin_result.body.get("value")
            kind = selected.get("kind", "io")
            if kind == "sensor":
                st.metric("Reading", value=v)
            else:
                st.metric("State", value="HIGH" if v else "LOW")
        elif pin_result and not pin_result.ok:
            st.error(f"http {pin_result.code}: {pin_result.body}")

        if selected.get("kind") != "sensor":
            st.markdown("**Write**")
            wcols = st.columns([1, 1, 1])
            label_key = selected.get("label", f"pin{selected.get('pin')}")
            if wcols[0].button("Set 0", key=f"w0_{label_key}", use_container_width=True):
                _do_pin_write(client, selected, 0)
            if wcols[1].button("Set 1", key=f"w1_{label_key}", use_container_width=True):
                _do_pin_write(client, selected, 1)
            if wcols[2].button("Toggle", key=f"wt_{label_key}", use_container_width=True):
                current_val = (
                    pin_result.body.get("value")
                    if pin_result and pin_result.ok and isinstance(pin_result.body, dict)
                    else None
                )
                if current_val is None:
                    st.warning("Read the pin first to know its current value.")
                else:
                    _do_pin_write(client, selected, 0 if current_val else 1)


# ---------- Actions -------------------------------------------------------

with tab_actions:
    st.markdown("**Water a bed**")
    with st.form("water_form", clear_on_submit=False):
        cols = st.columns([1, 1, 1])
        bed_id = cols[0].text_input("Bed ID", value="b1")
        seconds = cols[1].number_input("Seconds", min_value=0.1, max_value=300.0, value=2.0, step=0.5)
        go = cols[2].form_submit_button("💧 Water", use_container_width=True, type="primary")
        if go:
            r = client.request(
                "POST", "/actions",
                json={"kind": "water", "params": {"bed_id": bed_id, "seconds": seconds}},
            )
            if r.ok:
                st.success(f"Watered {bed_id} for {seconds}s")
            else:
                st.error(f"HTTP {r.code}: {r.body}")

    st.divider()
    st.markdown("**Raw action**")
    with st.form("raw_form"):
        kind = st.text_input("Kind", value="water")
        params_json = st.text_area(
            "Params (JSON)", value='{"bed_id": "b1", "seconds": 1.0}', height=120
        )
        go = st.form_submit_button("Dispatch")
        if go:
            try:
                params = json.loads(params_json) if params_json.strip() else {}
            except json.JSONDecodeError as err:
                st.error(f"Bad JSON: {err}")
                params = None
            if params is not None:
                r = client.request("POST", "/actions", json={"kind": kind, "params": params})
                if r.ok:
                    st.success("OK")
                    st.json(r.body)
                else:
                    st.error(f"HTTP {r.code}: {r.body}")

    st.divider()
    if st.button("🛑 Emergency stop", type="primary", use_container_width=True):
        r = client.request("POST", "/actions", json={"kind": "e_stop", "params": {}})
        if r.ok:
            st.toast("Emergency stop sent", icon="🛑")
        else:
            st.error(f"e_stop failed: HTTP {r.code} {r.body}")
