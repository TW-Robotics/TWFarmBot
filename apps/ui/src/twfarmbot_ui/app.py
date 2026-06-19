"""TWFarmBot Research UI.

Sidebar navigation with a clean main canvas.  Each tab renders a focused,
compact card-based view.  Zero business logic — every read and write is
proxied through the api_server.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import altair as alt
import httpx
import streamlit as st
from twfarmbot_ml_utils import HuggingFaceImageProcessor

from twfarmbot_ui.client import ApiClient

# ── config ────────────────────────────────────────────────────────────────────

API_URL = os.getenv("TWFB_API_URL", "http://127.0.0.1:8000")
AI_SPACE_ID = os.getenv("TWFB_AI_SPACE_ID", "DavidSeyserHF/Eupe-Lang")
PLAN_TIMEOUT = 60.0  # LLM planning can take longer than the default 2s API timeout


@st.cache_resource
def _client(base_url: str) -> ApiClient:
    return ApiClient(base_url)


@st.cache_resource
def _image_processor(space_id: str) -> HuggingFaceImageProcessor:
    return HuggingFaceImageProcessor(space_id)


# ── helpers ───────────────────────────────────────────────────────────────────

def _num(value: Any) -> str:
    try:    return f"{float(value):.1f}"
    except (TypeError, ValueError):  return "—"


def _float(value: Any, default: float = 0.0) -> float:
    try:    return float(value)
    except (TypeError, ValueError):  return default


_NUMBER_RE = re.compile(r"^\s*-?\d+(?:[.,]\d+)?\s*$")


def _parse_number(value: Any, default: float = 0.0) -> float | None:
    """Parse a user-typed number, accepting both '.' and ',' as decimals.

    Returns ``None`` on invalid input rather than silently defaulting — a
    silent fallback here would let a mistyped "1.234,5" (German thousands
    style) drive the FarmBot to (0, 0, 0). Callers should treat ``None``
    as a user-visible error.
    """
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not _NUMBER_RE.match(text):
        return None
    try:
        return float(text.replace(",", "."))
    except (TypeError, ValueError):
        return None


def _action_summary(action: dict[str, Any]) -> str:
    """Return a compact, human-readable summary of an action."""
    kind = action.get("kind", "action")
    params = action.get("params") or {}
    if kind == "move":
        return (
            f"🛠️ **move** → "
            f"({_num(params.get('x'))}, {_num(params.get('y'))}, {_num(params.get('z'))})"
        )
    if kind == "water":
        return f"🌊 **water** bed **{params.get('bed_id', '—')}** for {_num(params.get('seconds'))} s"
    if kind == "find_home":
        return f"🏠 **find_home** (axis={params.get('axis', 'all')}, speed={params.get('speed', '—')})"
    if kind == "take_photo":
        return "📷 **take_photo**"
    if kind == "read_pin":
        return f"📖 **read_pin** {params.get('pin', '—')} ({params.get('mode', 'digital')})"
    if kind == "write_pin":
        return f"✏️ **write_pin** {params.get('pin', '—')} = {params.get('value', '—')}"
    if kind == "send_message":
        msg = str(params.get("message", ""))[:40]
        return f"💬 **send_message**: {msg}"
    if kind == "mount_tool":
        return f"🔧 **mount_tool** {params.get('tool_name', '—')}"
    if kind == "dismount_tool":
        return "🔧 **dismount_tool**"
    if kind == "e_stop":
        return "🛑 **e_stop**"
    return f"🛠️ **{kind}**"


def _render_action_cards(actions: list[dict[str, Any]]) -> None:
    """Render proposed actions as compact cards with collapsible details."""
    for action in actions:
        with st.container(border=True):
            st.markdown(_action_summary(action))
            with st.expander("Details"):
                st.json(action.get("params", {}))


_APPROVAL_WORDS = {
    "yes", "y", "approve", "approved", "ok", "okay", "sure",
    "go ahead", "do it", "confirm", "confirmed", "execute", "run it",
}
_REJECTION_WORDS = {
    "no", "n", "reject", "rejected", "cancel", "cancelled",
    "don't", "dont", "stop", "abort",
}


def _is_approval(text: str) -> bool:
    return text.strip("!.? ").lower() in _APPROVAL_WORDS


def _is_rejection(text: str) -> bool:
    return text.strip("!.? ").lower() in _REJECTION_WORDS


def _refresh_position(client: ApiClient) -> None:
    r = client.request("GET", "/position")
    if r.ok and isinstance(r.body, dict):
        xyz = (r.body.get("xyz") or {}) if r.ok else {}
        st.session_state["pos_x"] = _num(xyz.get("x"))
        st.session_state["pos_y"] = _num(xyz.get("y"))
        st.session_state["pos_z"] = _num(xyz.get("z"))


def _refresh_health(client: ApiClient) -> None:
    r = client.request("GET", "/health")
    if r.ok and isinstance(r.body, dict):
        st.session_state["farmbot_status"] = r.body.get("farmbot", "?")
        st.session_state["actions"] = r.body.get("actions", [])


def _refresh_messages(client: ApiClient) -> None:
    r = client.request("GET", "/messages")
    if r.ok and isinstance(r.body, dict):
        raw = r.body.get("last_messages")
        if isinstance(raw, list):
            st.session_state["messages"] = [str(m) for m in raw[-20:]]
        else:
            st.session_state["messages"] = []


def _refresh_telemetry(client: ApiClient) -> None:
    _refresh_position(client)
    _refresh_health(client)
    _refresh_messages(client)


TABS = [
    "Overview", "Garden", "Motion", "Camera", "Sensors", "Operations",
    "Assistant", "Diagnostics", "Settings",
]


def _qp_tab() -> str:
    """Return the tab key currently set in the URL query string."""
    raw = st.query_params.get("tab")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return (raw or "").lower()


def _tab_from_key(key: str) -> str:
    """Map a URL-safe tab key back to the display name."""
    low = key.lower()
    for tab in TABS:
        if tab.lower() == low:
            return tab
    return TABS[0]


def _sync_tab_url() -> None:
    """Callback that updates the URL when the user switches tabs."""
    selected = st.session_state.get("nav_tab")
    if selected:
        st.query_params["tab"] = selected.lower()


def _do_move(client: ApiClient, x: float, y: float, z: float, label: str = "") -> None:
    r = client.request("POST", "/actions", json={"kind": "move", "params": {"x": x, "y": y, "z": z}})
    if r.ok:
        msg = f"→ {label}" if label else f"→ ({x:.0f}, {y:.0f}, {z:.0f})"
        st.toast(msg, icon="➡️")
        _refresh_position(client)
    else:
        st.error(f"HTTP {r.code}: {r.body}")


def _do_pin_write(client: ApiClient, pin: int, value: int, mode: str = "digital") -> None:
    r = client.request("POST", "/actions", json={
        "kind": "write_pin", "params": {"pin": pin, "value": value, "mode": mode},
    })
    if r.ok:
        st.toast(f"pin {pin} = {value}", icon="✏️")
    else:
        st.error(f"HTTP {r.code}: {r.body}")


# ── page shell ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="TWFarmBot Research", page_icon="🌾", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  header[data-testid="stHeader"], [data-testid="stToolbar"],
  [data-testid="stDecoration"] { display:none !important; }
  section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    transform: none !important;
    position: fixed !important;
    inset: 0 auto 0 0 !important;
    width: 18rem !important;
    min-width: 18rem !important;
    z-index: 999 !important;
    border-right: 1px solid rgba(128,128,128,0.15);
    background: var(--secondary-background-color);
  }
  section[data-testid="stSidebar"] > div {
    display: block !important;
    visibility: visible !important;
  }
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="stSidebarCollapseButton"] { display:none !important; }
  .stMain { margin-left: 18rem !important; }
  .block-container { max-width: 1300px; padding-top: 1.25rem; }
  h1 { font-size: 1.6rem; letter-spacing: -0.03em; }
  .eyebrow { color: #3f8f64; font-size: .66rem; font-weight: 750;
             letter-spacing: .14em; text-transform: uppercase; }
  .card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.12);
    border-radius: 10px; padding: .8rem 1rem; min-height: 5rem;
  }
  .card-label { font-size: .66rem; font-weight: 700; letter-spacing: .09em;
                text-transform: uppercase; opacity: .5; }
  .card-value { font-size: 1.3rem; font-weight: 650; margin-top: .4rem; }
  .empty { opacity: .4; font-size: .8rem; }
  div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.10);
    border-radius: 9px; padding: .7rem .9rem;
  }
  div[data-testid="stRadio"] label {
    border-radius: 7px; padding: .52rem .6rem; margin: .04rem 0;
  }
  div[data-testid="stRadio"] label:hover,
  div[data-testid="stRadio"] label:has(input:checked) {
    background: var(--secondary-background-color);
  }
  div[data-testid="stRadio"] label:has(input:checked) { color: #3f8f64; font-weight: 700; }
  div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display:none; }
  .sidebar-brand { font-size: 1.05rem; font-weight: 750; letter-spacing: -.02em; }
  .sidebar-kicker { font-size: .63rem; color: #3f8f64; font-weight: 750;
                    letter-spacing: .1em; text-transform: uppercase; }
  .pill { display: inline-block; padding: .1rem .55rem; border-radius: 999px;
          font-size: .72rem; font-weight: 650; }
  .pill.ok { background: #d1fae5; color: #065f46; }
  .pill.warn { background: #fef3c7; color: #92400e; }
  .pill.bad { background: #fee2e2; color: #991b1b; }
  .st-key-analysis_source img,
  .st-key-analysis_processed img {
    width: 100% !important;
    height: 320px !important;
    object-fit: contain !important;
    background: var(--secondary-background-color);
    border-radius: 9px;
  }
  @media (max-width: 760px) {
    section[data-testid="stSidebar"] { width: 14rem !important; min-width: 14rem !important; }
    .stMain { margin-left: 14rem !important; }
  }
</style>
""", unsafe_allow_html=True)

api_url = st.session_state.setdefault("api_url", API_URL)
client = _client(api_url)
if "farmbot_status" not in st.session_state:
    _refresh_health(client)
    _refresh_position(client)
    st.session_state.setdefault("messages", [])

# ── sidebar  ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-kicker">Field robotics</div>'
                '<div class="sidebar-brand">TWFarmBot</div>', unsafe_allow_html=True)
    # Sync the navigation radio with the URL ?tab=... query parameter so
    # refreshing the browser returns to the same tab.
    url_tab = _tab_from_key(_qp_tab())
    if st.session_state.get("nav_tab") != url_tab:
        st.session_state["nav_tab"] = url_tab
    tab = st.radio("Navigation", TABS,
                   key="nav_tab", on_change=_sync_tab_url, label_visibility="collapsed")

    st.divider()
    fb = st.session_state.get("farmbot_status", "?")
    pill_css = "ok" if fb == "connected" else ("warn" if fb == "skipped" else "bad")
    st.markdown(f'<span class="pill {pill_css}">● {fb}</span>', unsafe_allow_html=True)
    st.caption(f"X {st.session_state.get('pos_x', '—')} · "
               f"Y {st.session_state.get('pos_y', '—')} · "
               f"Z {st.session_state.get('pos_z', '—')} mm")
    if st.button("↻ Refresh", use_container_width=True):
        _refresh_telemetry(client)
        st.rerun()

    st.divider()
    if st.button("🛑 ESTOP", type="primary", use_container_width=True):
        r = client.request("POST", "/actions", json={"kind": "e_stop", "params": {}})
        if r.ok:
            st.toast("ESTOP sent", icon="🛑")
        else:
            st.error(str(r.body))

# ── tab content ───────────────────────────────────────────────────────────────

def _render_overview() -> None:
    st.markdown('<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>', unsafe_allow_html=True)
    st.markdown("# Research overview")

    row = st.columns(3)
    row[0].metric("X · mm", st.session_state.get("pos_x", "—"))
    row[1].metric("Y · mm", st.session_state.get("pos_y", "—"))
    row[2].metric("Z · mm", st.session_state.get("pos_z", "—"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Experiment**")
        st.text_input("Run", key="run", placeholder="e.g. soil-map-07", label_visibility="collapsed")
        st.text_input("Operator", key="op", placeholder="initials", label_visibility="collapsed")
        st.text_area("Notes", key="notes", placeholder="Conditions, observations…", height=90, label_visibility="collapsed")

    with c2:
        st.markdown("**Recent events**")
        msgs = st.session_state.get("messages", [])
        if msgs:
            st.code("\n".join(msgs[-10:]), language="text")
        else:
            st.caption("No events recorded.")


def _render_garden() -> None:
    st.markdown('<div class="eyebrow">Spatial model · configured world state</div>', unsafe_allow_html=True)
    st.markdown("# Garden map")

    result = client.request("GET", "/garden")
    if not result.ok or not isinstance(result.body, dict):
        st.error(f"Garden model unavailable: {result.body}")
        return

    world = result.body
    bounds = world.get("bounds", {})
    camera = world.get("camera", {})
    robot = world.get("robot", {})
    entities = world.get("entities", [])
    zones = world.get("zones", [])

    metrics = st.columns(4)
    metrics[0].metric("Garden X", f"{_num(bounds.get('width'))} mm")
    metrics[1].metric("Garden Y", f"{_num(bounds.get('height'))} mm")
    metrics[2].metric("Known objects", len(entities))
    metrics[3].metric("Mapped zones", len(zones))

    zone_rows = [{
        **zone["bounds"],
        "x2": zone["bounds"]["x"] + zone["bounds"]["width"],
        "y2": zone["bounds"]["y"] + zone["bounds"]["height"],
        "kind": zone["kind"],
        "name": zone["name"],
    } for zone in zones]
    point_rows = [{
        "x": entity["position"]["x"],
        "y": entity["position"]["y"],
        "kind": entity["kind"],
        "name": entity["name"],
        "radius_mm": entity["radius_mm"],
    } for entity in entities]
    point_rows.extend([
        {"x": robot.get("x", 0), "y": robot.get("y", 0), "kind": "robot",
         "name": "FarmBot", "radius_mm": 35},
        {"x": camera.get("position", {}).get("x", 0),
         "y": camera.get("position", {}).get("y", 0), "kind": "camera",
         "name": "Camera", "radius_mm": 25},
    ])

    x_min = bounds.get("x", 0)
    x_max = x_min + bounds.get("width", 1)
    y_min = bounds.get("y", 0)
    y_max = y_min + bounds.get("height", 1)

    def _map_scale(lo: float, hi: float) -> alt.Scale:
        # Clamp pan/zoom so the user cannot scroll far outside the garden.
        return alt.Scale(
            domain=[lo, hi],
            domainMin=lo,
            domainMax=hi,
            clamp=True,
            nice=False,
        )

    x_scale = _map_scale(x_min, x_max)
    y_scale = _map_scale(y_min, y_max)

    bounds_chart = alt.Chart(alt.Data(values=[{
        "x": x_min, "y": y_min, "x2": x_max, "y2": y_max,
    }])).mark_rect(
        filled=False, stroke="#888888", strokeWidth=2
    ).encode(
        x=alt.X("x:Q", scale=x_scale, title="X · mm"),
        x2="x2:Q",
        y=alt.Y("y:Q", scale=y_scale, title="Y · mm"),
        y2="y2:Q",
    )
    zones_chart = alt.Chart(alt.Data(values=zone_rows)).mark_rect(
        opacity=0.18, strokeWidth=2
    ).encode(
        x=alt.X("x:Q", scale=x_scale, title="X · mm"),
        x2="x2:Q",
        y=alt.Y("y:Q", scale=y_scale, title="Y · mm"),
        y2="y2:Q",
        color=alt.Color("kind:N", title="Layer"),
        stroke=alt.Stroke("kind:N", legend=None),
        tooltip=["name:N", "kind:N", "x:Q", "y:Q", "width:Q", "height:Q"],
    )
    points_chart = alt.Chart(alt.Data(values=point_rows)).mark_point(
        filled=True, stroke="white", strokeWidth=1
    ).encode(
        x=alt.X("x:Q", scale=x_scale),
        y=alt.Y("y:Q", scale=y_scale),
        color=alt.Color("kind:N", title="Object"),
        shape=alt.Shape("kind:N", title="Object"),
        size=alt.Size("radius_mm:Q", scale=alt.Scale(range=[90, 500]), legend=None),
        tooltip=["name:N", "kind:N", "x:Q", "y:Q"],
    )

    map_col, details = st.columns([2.3, 1])
    with map_col:
        st.altair_chart(
            (bounds_chart + zones_chart + points_chart).properties(height=520).interactive(),
            width="stretch",
        )
    with details:
        st.markdown("**Live pose**")
        pose = st.columns(3)
        pose[0].metric("X", _num(robot.get("x")))
        pose[1].metric("Y", _num(robot.get("y")))
        pose[2].metric("Z", _num(robot.get("z")))
        st.markdown("**Camera pose**")
        st.caption(
            f"X {_num(camera.get('position', {}).get('x'))} · "
            f"Y {_num(camera.get('position', {}).get('y'))} · "
            f"Z {_num(camera.get('position', {}).get('z'))} mm"
        )
        st.caption(
            f"Yaw {_num(camera.get('yaw_deg'))}° · "
            f"Pitch {_num(camera.get('pitch_deg'))}° · "
            f"Roll {_num(camera.get('roll_deg'))}°"
        )
        camera_offset = world.get("camera_offset") or {}
        if camera_offset:
            st.caption(
                f"Offset from FarmBot: "
                f"X {_num(camera_offset.get('x'))} · "
                f"Y {_num(camera_offset.get('y'))} · "
                f"Z {_num(camera_offset.get('z'))} mm"
            )
        st.markdown("**Mapped objects**")
        st.dataframe(
            [{"name": item["name"], "kind": item["kind"]} for item in entities],
            hide_index=True,
            width="stretch",
        )


def _render_motion() -> None:
    cur_x = _float(st.session_state.get("pos_x"))
    cur_y = _float(st.session_state.get("pos_y"))
    cur_z = _float(st.session_state.get("pos_z"))

    st.markdown('<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>', unsafe_allow_html=True)
    st.markdown("# Motion workspace")

    row = st.columns(3)
    row[0].metric("X · mm", st.session_state.get("pos_x", "—"))
    row[1].metric("Y · mm", st.session_state.get("pos_y", "—"))
    row[2].metric("Z · mm", st.session_state.get("pos_z", "—"))

    step = float(st.segmented_control("Jog step · mm", [1, 10, 50, 100], default=10))

    # D-pad
    _, u, _ = st.columns(3)
    if u.button("▲ Y+", use_container_width=True): _do_move(client, cur_x, cur_y + step, cur_z, f"Y+{step:.0f}")
    l, m, r = st.columns(3)
    if l.button("◀ X−", use_container_width=True): _do_move(client, cur_x - step, cur_y, cur_z, f"X-{step:.0f}")
    if m.button("🏠 Home", use_container_width=True): _do_move(client, 0, 0, 0, "Home")
    if r.button("X+ ▶", use_container_width=True): _do_move(client, cur_x + step, cur_y, cur_z, f"X+{step:.0f}")
    _, d, _ = st.columns(3)
    if d.button("▼ Y−", use_container_width=True): _do_move(client, cur_x, cur_y - step, cur_z, f"Y-{step:.0f}")

    zl, zr = st.columns(2)
    if zl.button("⬆ Z+", use_container_width=True): _do_move(client, cur_x, cur_y, cur_z + step, f"Z+{step:.0f}")
    if zr.button("⬇ Z−", use_container_width=True): _do_move(client, cur_x, cur_y, cur_z - step, f"Z-{step:.0f}")

    st.divider()
    with st.form("absolute"):
        tx, ty, tz = st.columns(3)
        gx = tx.text_input("X", value=f"{cur_x:.2f}")
        gy = ty.text_input("Y", value=f"{cur_y:.2f}")
        gz = tz.text_input("Z", value=f"{cur_z:.2f}")
        if st.form_submit_button("Go to", use_container_width=True):
            x = _parse_number(gx)
            y = _parse_number(gy)
            z = _parse_number(gz)
            if None in (x, y, z):
                st.error(
                    f"Invalid coordinates: X={gx!r}, Y={gy!r}, Z={gz!r}. "
                    "Use a plain number like '123' or '123.4' (comma also accepted)."
                )
            else:
                _do_move(client, x, y, z)

    if st.button("Find home"):
        r = client.request("POST", "/actions", json={"kind": "find_home", "params": {}})
        if r.ok:
            st.toast("Homing queued")
        else:
            st.error(str(r.body))

    # Presets
    if "presets" not in st.session_state:
        r = client.request("GET", "/positions")
        st.session_state["presets"] = r.body.get("positions", []) if r.ok else []
    presets = st.session_state["presets"]
    if presets:
        st.markdown("**Locations**")
        cols = st.columns(min(5, len(presets)))
        for i, p in enumerate(presets):
            if cols[i].button(p.get("label", "?"), key=f"preset_{i}", use_container_width=True):
                _do_move(client, float(p["x"]), float(p["y"]), float(p["z"]), p["label"])


def _render_sensors() -> None:
    st.markdown('<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>', unsafe_allow_html=True)
    st.markdown("# Sensor workspace")

    if "named_pins" not in st.session_state:
        r = client.request("GET", "/pins")
        st.session_state["named_pins"] = r.body.get("pins", []) if r.ok else []
    named = st.session_state["named_pins"]

    if not named:
        st.info("No pins configured.")
        return

    sensors = [p for p in named if p.get("kind") == "sensor"]
    if sensors:
        st.markdown("**Instruments**")
        cols = st.columns(min(3, len(sensors)))
        for i, s in enumerate(sensors):
            with cols[i]:
                st.caption(f"{s['label']} · pin {s['pin']} · {s.get('mode', 'analog')}")
                if st.button("Read", key=f"sensor_{i}", use_container_width=True):
                    r = client.request("GET", f"/pin/{s['pin']}", params={"mode": s.get("mode", "analog")})
                    st.session_state[f"sv_{s['pin']}"] = r.body.get("value") if r.ok else "—"
                st.metric("Value", st.session_state.get(f"sv_{s['pin']}", "—"))


def _render_camera() -> None:
    st.markdown('<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>', unsafe_allow_html=True)
    st.markdown("# Camera")

    capture, refresh, _ = st.columns([1, 1, 4])
    if capture.button("📷 Take photo", type="primary", use_container_width=True):
        r = client.request("POST", "/actions", json={"kind": "take_photo", "params": {}})
        if r.ok:
            st.toast("Capture queued", icon="📷")
        else:
            st.error(str(r.body))
    if refresh.button("↻ Refresh gallery", use_container_width=True):
        st.session_state["images"] = client.request(
            "GET", "/images", params={"refresh": "true"}, timeout=10.0
        )

    result = st.session_state.get("images")
    images = (
        result.body.get("images", [])
        if result and result.ok and isinstance(result.body, dict)
        else []
    )
    if not images:
        st.info("Refresh the gallery to load FarmBot photos.")
        return

    selected = st.selectbox(
        "Research image",
        images,
        format_func=lambda image: (
            f"{image.get('created_at', 'Unknown time')} · "
            f"image {image.get('id', '—')}"
        ),
    )
    meta = selected.get("meta") or {}
    photo, details = st.columns([1.55, 1])
    with photo:
        st.image(selected.get("attachment_url"), use_container_width=True)
    with details:
        st.markdown("**Selected capture**")
        st.caption(selected.get("created_at", "Unknown time"))
        xyz = st.columns(3)
        xyz[0].metric("X", meta.get("x", "—"))
        xyz[1].metric("Y", meta.get("y", "—"))
        xyz[2].metric("Z", meta.get("z", "—"))
        st.caption(f"Image ID {selected.get('id', '—')}")

        st.markdown("**AI analysis**")
        prompt = st.text_input(
            "Target",
            placeholder="e.g. green leaves, dry soil, red marker",
            key=f"ai_prompt_{selected.get('id', 'unknown')}",
        )
        if st.button(
            "Analyze selected image",
            type="primary",
            use_container_width=True,
            disabled=not prompt.strip(),
        ):
            try:
                with st.spinner("Processing image…"):
                    result_path = _image_processor(AI_SPACE_ID).process(
                        selected["attachment_url"],
                        prompt.strip(),
                    )
                st.session_state["ai_result"] = {
                    "image_id": selected.get("id"),
                    "source_url": selected.get("attachment_url"),
                    "path": str(result_path),
                    "prompt": prompt.strip(),
                }
            except Exception as exc:
                st.error(f"AI processing failed: {exc}")

    result = st.session_state.get("ai_result")
    if result:
        st.markdown("### Analysis result")
        source, processed = st.columns(2)
        with source:
            with st.container(key="analysis_source"):
                st.image(
                    result["source_url"],
                    caption="Source image",
                    width="stretch",
                )
        with processed:
            with st.container(key="analysis_processed"):
                st.image(
                    result["path"],
                    caption=f"Similarity map · {result['prompt']}",
                    width="stretch",
                )

    if len(images) > 1:
        st.markdown("**Recent captures**")
        gallery = st.columns(3)
        for index, image in enumerate(images[1:7]):
            image_meta = image.get("meta") or {}
            gallery[index % 3].image(
                image.get("attachment_url"),
                caption=f"X {image_meta.get('x', '—')} · Y {image_meta.get('y', '—')}",
                use_container_width=True,
            )


def _render_assistant() -> None:
    st.markdown('<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>', unsafe_allow_html=True)
    st.markdown("# Assistant")

    mode = st.segmented_control(
        "Mode", ["Chat", "Plan"],
        default=st.session_state.get("assistant_mode", "Chat"),
        key="assistant_mode",
    )
    if mode == "Plan":
        _render_plan()
    else:
        _render_chat()


def _render_chat() -> None:
    header, clear_col = st.columns([3, 1])
    with header:
        st.caption(
            "Chat with the FarmBot. Ask about status and zones, or tell it to "
            "water, take photos, move, and more."
        )
    with clear_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state["assistant_messages"] = []
            st.rerun()

    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = []

    for idx, msg in enumerate(st.session_state["assistant_messages"]):
        if msg.get("role") == "tool":
            with st.chat_message("assistant"):
                st.markdown(f"🔧 **{msg.get('name', 'tool')}**")
                with st.expander("Tool result"):
                    st.json({"args": msg.get("args"), "result": msg.get("result")})
            continue

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander("Thinking"):
                    st.json(msg["tool_calls"])

            proposed_actions = msg.get("proposed_actions", [])
            if proposed_actions and not msg.get("approved") and not msg.get("rejected"):
                st.markdown("**Proposed actions**")
                _render_action_cards(proposed_actions)
                approve_col, reject_col = st.columns([1, 1])
                if approve_col.button("✅ Approve", key=f"approve_{idx}", use_container_width=True):
                    _execute_proposed_actions(proposed_actions)
                    msg["approved"] = True
                    msg["content"] += f"\n\n✅ Approved and queued {len(proposed_actions)} action(s)."
                    st.rerun()
                if reject_col.button("❌ Reject", key=f"reject_{idx}", use_container_width=True):
                    msg["rejected"] = True
                    msg["content"] += "\n\n❌ Cancelled."
                    st.rerun()
            elif msg.get("approved"):
                st.caption("Approved")
            elif msg.get("rejected"):
                st.caption("Rejected")

    if prompt := st.chat_input("Ask the FarmBot assistant…"):
        messages = st.session_state["assistant_messages"]

        # Natural-language approval/rejection: if the user replies "yes",
        # "approve", "no", "cancel", etc. to a proposal, handle it immediately
        # instead of sending it back to the model and getting a confused answer.
        if messages and messages[-1].get("role") == "assistant":
            last_assistant = messages[-1]
            proposed = last_assistant.get("proposed_actions", [])
            pending = proposed and not last_assistant.get("approved") and not last_assistant.get("rejected")
            approval = _is_approval(prompt)
            rejection = _is_rejection(prompt)
            if approval or rejection:
                if pending:
                    st.session_state["assistant_messages"].append({"role": "user", "content": prompt})
                    if approval:
                        _execute_proposed_actions(proposed)
                        last_assistant["approved"] = True
                        last_assistant["content"] += (
                            f"\n\n✅ Approved and queued {len(proposed)} action(s)."
                        )
                    else:
                        last_assistant["rejected"] = True
                        last_assistant["content"] += "\n\n❌ Cancelled."
                    st.rerun()
                else:
                    st.toast("No pending proposal to approve or reject.", icon="⚠️")
                    st.rerun()

        st.session_state["assistant_messages"].append({"role": "user", "content": prompt})
        thinking = st.empty()
        thinking.info("🤖 Assistant is thinking…")
        with st.chat_message("assistant"):
            placeholder = st.empty()
            stream_meta = {"tool_calls": [], "proposed_actions": []}
            stream_error = None
            accumulated = ""
            try:
                with httpx.Client() as http:
                    with http.stream(
                        "POST", f"{api_url}/chat/stream",
                        json={"messages": st.session_state["assistant_messages"]},
                        timeout=PLAN_TIMEOUT,
                    ) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])
                            etype = event.get("type")
                            if etype == "delta":
                                accumulated += event.get("content", "")
                                placeholder.markdown(accumulated)
                            elif etype == "tool_call":
                                st.session_state["assistant_messages"].append({
                                    "role": "tool",
                                    "name": event.get("name"),
                                    "args": event.get("args"),
                                    "result": event.get("result"),
                                })
                            elif etype == "meta":
                                stream_meta["tool_calls"] = event.get("tool_calls", [])
                                stream_meta["proposed_actions"] = event.get("proposed_actions", [])
                            elif etype == "error":
                                stream_error = event.get("error", "stream error")
            except Exception as exc:  # noqa: BLE001
                stream_error = f"{type(exc).__name__}: {exc}"

            # If the stream produced nothing useful, fall back to the
            # non-streaming endpoint so the chat still works even when the
            # SSE path is blocked or misbehaving.
            if not accumulated and not stream_meta["tool_calls"] and not stream_meta["proposed_actions"]:
                try:
                    r = client.request(
                        "POST", "/chat",
                        json={"messages": st.session_state["assistant_messages"]},
                        timeout=PLAN_TIMEOUT,
                    )
                    if r.ok and isinstance(r.body, dict):
                        accumulated = str(r.body.get("response", ""))
                        stream_meta["tool_calls"] = r.body.get("tool_calls", []) or []
                        stream_meta["proposed_actions"] = [
                            {"kind": tc["result"].get("kind", tc["name"]),
                             "params": tc["result"].get("params", tc.get("args", {}))}
                            for tc in stream_meta["tool_calls"]
                            if isinstance(tc.get("result"), dict) and tc["result"].get("status") == "proposed"
                        ]
                        stream_error = None
                    else:
                        stream_error = f"Fallback failed: HTTP {r.code}: {r.body}"
                except Exception as exc:  # noqa: BLE001
                    stream_error = f"Fallback failed: {type(exc).__name__}: {exc}"

            thinking.empty()
            if accumulated:
                placeholder.markdown(accumulated)
            if stream_error:
                st.error(f"Assistant error: {stream_error}")

            if accumulated or stream_meta["tool_calls"] or stream_meta["proposed_actions"]:
                st.session_state["assistant_messages"].append({
                    "role": "assistant",
                    "content": accumulated,
                    "tool_calls": stream_meta["tool_calls"],
                    "proposed_actions": stream_meta["proposed_actions"],
                })
        st.rerun()


def _execute_proposed_actions(actions: list[dict[str, Any]]) -> None:
    for action in actions:
        client.request(
            "POST", "/actions",
            json={"kind": action["kind"], "params": action.get("params", {})},
        )


def _render_plan() -> None:
    st.caption("Describe a task. The LLM builds a step-by-step plan; review it before running.")

    if "assistant_plan_response" not in st.session_state:
        st.session_state["assistant_plan_response"] = None
    if "assistant_plan_status" not in st.session_state:
        st.session_state["assistant_plan_status"] = None
    if "assistant_plan_request" not in st.session_state:
        st.session_state["assistant_plan_request"] = ""

    examples = [
        "Water the tomato zone for 90 seconds, then go home",
        "Take a photo and send me the result",
        "Move to x=500 y=200 z=0",
    ]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True, key=f"plan_ex_{example[:20]}"):
            st.session_state["assistant_plan_request"] = example
            st.session_state["assistant_plan_response"] = None
            st.session_state["assistant_plan_status"] = None
            st.rerun()

    request = st.text_area(
        "Task",
        value=st.session_state["assistant_plan_request"],
        placeholder="e.g. water bed 1 for 60 seconds, then home",
        height=80,
        label_visibility="collapsed",
    )
    st.session_state["assistant_plan_request"] = request

    plan_col, _ = st.columns([1, 3])
    preview_clicked = plan_col.button(
        "Preview plan", type="primary", use_container_width=True,
        disabled=not request.strip(),
    )

    if preview_clicked and request.strip():
        with st.spinner("Asking the planner…"):
            r = client.request(
                "POST", "/plan",
                json={"request": request, "debug": True},
                timeout=PLAN_TIMEOUT,
            )
        st.session_state["assistant_plan_response"] = r.body if r.ok else {"error": r.body}
        st.session_state["assistant_plan_status"] = r.code

    response = st.session_state.get("assistant_plan_response")
    status = st.session_state.get("assistant_plan_status")

    if not response:
        st.info("No plan yet. Type a task above and click **Preview plan**.")
        return

    with st.expander("Debug · raw response", expanded=False):
        st.json(response)

    if status and status >= 400:
        st.error(f"Planner error (HTTP {status}): {response.get('error', response)}")
        return

    actions = response.get("actions", []) or []
    rationale = response.get("rationale") if isinstance(response, dict) else None
    st.success(f"Plan ready · {len(actions)} action(s)")
    if rationale:
        st.caption(f"Model rationale: {rationale}")

    if not actions:
        st.warning("The planner returned an empty plan.")
        return

    st.markdown("**Proposed actions**")
    for idx, action in enumerate(actions, start=1):
        with st.container(border=True):
            st.markdown(f"{idx}. {_action_summary(action)}")
            with st.expander("Details"):
                st.json(action.get("params", {}))

    run_col, clear_col = st.columns([1, 1])
    if clear_col.button("Clear", use_container_width=True):
        st.session_state["assistant_plan_response"] = None
        st.session_state["assistant_plan_status"] = None
        st.rerun()

    if run_col.button("Run plan", type="primary", use_container_width=True):
        queued = 0
        failed = 0
        for action in actions:
            r = client.request(
                "POST", "/actions",
                json={"kind": action["kind"], "params": action.get("params", {})},
            )
            if r.ok:
                queued += 1
                st.toast(f"Queued {action['kind']}", icon="➡️")
            else:
                failed += 1
                st.error(f"Failed to queue {action['kind']}: {r.body}")
        if failed == 0:
            st.success(f"Plan queued · {queued} action(s)")
        else:
            st.warning(f"Plan partially queued · {queued} ok, {failed} failed")
        st.session_state["assistant_plan_response"] = None
        st.session_state["assistant_plan_status"] = None
        st.rerun()


def _render_operations() -> None:
    st.markdown('<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>', unsafe_allow_html=True)
    st.markdown("# Operations")

    a, b = st.columns(2)
    with a:
        st.markdown("**Irrigation**")
        with st.form("water"):
            bed = st.selectbox("Bed", ["b1", "b2", "b3"])
            secs = st.number_input("Seconds", 0.1, 300.0, 2.0, 0.5)
            if st.form_submit_button("Water", use_container_width=True):
                r = client.request("POST", "/actions",
                                   json={"kind": "water", "params": {"bed_id": bed, "seconds": secs}})
                if r.ok:
                    st.success("Queued")
                else:
                    st.error(str(r.body))

    with b:
        st.markdown("**Peripheral control**")
        if "named_pins" not in st.session_state:
            r = client.request("GET", "/pins")
            st.session_state["named_pins"] = r.body.get("pins", []) if r.ok else []
        outputs = [p for p in st.session_state["named_pins"] if p.get("kind") != "sensor"]
        sel = st.selectbox("Output", outputs,
                           format_func=lambda p: f"{p['label']} · pin {p['pin']}")
        if sel:
            off, on = st.columns(2)
            if off.button("Set LOW", use_container_width=True):
                _do_pin_write(client, sel["pin"], 0, sel.get("mode", "digital"))
            if on.button("Set HIGH", use_container_width=True):
                _do_pin_write(client, sel["pin"], 1, sel.get("mode", "digital"))


def _render_diagnostics() -> None:
    st.markdown('<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>', unsafe_allow_html=True)
    st.markdown("# Diagnostics")

    if st.button("Load /status"):
        d = client.request("GET", "/status")
        if d.ok and isinstance(d.body, dict):
            st.session_state["diag"] = d.body.get("state", {})
        else:
            st.error(f"Read failed: {d.body}")

    payload = st.session_state.get("diag", {}) or {}
    info = payload.get("informational_settings", {}) or {}
    loc = payload.get("location_data", {}) or {}
    axes = loc.get("axis_states", {}) or {}
    pins = payload.get("pins", {}) or {}
    jobs = payload.get("jobs", {}) or {}

    if not info and not axes and not pins and not jobs:
        st.info("Click 'Load /status' to fetch diagnostic state.")
        return

    top = st.columns(4)
    top[0].metric("Controller", info.get("controller_version", "—"))
    top[1].metric("Firmware", info.get("firmware_version", "—"))
    top[2].metric("Wi-Fi", f"{info.get('wifi_level_percent', '—')}%")
    top[3].metric("Uptime", f"{info.get('uptime', '—')} s")

    res = st.columns(3)
    with res[0]:
        st.markdown(
            f'<div class="card"><div class="card-label">Resources</div>'
            f'<div class="card-value">CPU {info.get("cpu_usage", "—")}%</div>'
            f'<div>Memory {info.get("memory_usage", "—")}% · Disk {info.get("disk_usage", "—")}%</div>'
            f'<div>SoC {info.get("soc_temp", "—")} °C</div></div>',
            unsafe_allow_html=True,
        )
    with res[1]:
        st.markdown(
            f'<div class="card"><div class="card-label">Axis state</div>'
            f'<div class="card-value">X {axes.get("x", "—")}</div>'
            f'<div>Y {axes.get("y", "—")} · Z {axes.get("z", "—")}</div>'
            f'<div>Busy: {info.get("busy", "—")}</div></div>',
            unsafe_allow_html=True,
        )
    with res[2]:
        st.markdown(
            f'<div class="card"><div class="card-label">Network</div>'
            f'<div class="card-value">{info.get("wifi_level", "—")} dBm</div>'
            f'<div>{info.get("private_ip", "—")}</div>'
            f'<div>Sync: {info.get("sync_status", "—")}</div></div>',
            unsafe_allow_html=True,
        )

    if pins:
        st.markdown("**Pin snapshot**")
        st.dataframe(
            [{"pin": pn, "value": pd.get("value"), "mode": pd.get("mode")} for pn, pd in pins.items()],
            use_container_width=True, hide_index=True,
        )


def _render_settings() -> None:
    st.markdown('<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>', unsafe_allow_html=True)
    st.markdown("# Settings")

    st.markdown("**Connection**")
    url = st.text_input("API URL", value=api_url)
    if url != st.session_state["api_url"]:
        st.session_state["api_url"] = url
        st.cache_resource.clear()
        st.rerun()

    if st.button("Health check"):
        _refresh_health(client)
        st.rerun()

    st.json({"farmbot": st.session_state.get("farmbot_status", "?"),
             "api": st.session_state["api_url"],
             "actions": st.session_state.get("actions", [])})

    with st.expander("Raw action"):
        with st.form("raw"):
            kind = st.text_input("Kind", "send_message")
            raw = st.text_area("Params (JSON)", '{"message":"hello"}', height=100)
            if st.form_submit_button("Fire"):
                try:
                    p = json.loads(raw)
                except json.JSONDecodeError as e:
                    st.error(f"Bad JSON: {e}")
                else:
                    r = client.request("POST", "/actions", json={"kind": kind, "params": p})
                    st.json(r.body)


# ── dispatch ──────────────────────────────────────────────────────────────────

renderers = {
    "Overview":     _render_overview,
    "Garden":       _render_garden,
    "Motion":       _render_motion,
    "Camera":       _render_camera,
    "Sensors":      _render_sensors,
    "Operations":   _render_operations,
    "Assistant":    _render_assistant,
    "Diagnostics":  _render_diagnostics,
    "Settings":     _render_settings,
}
renderers[tab]()
