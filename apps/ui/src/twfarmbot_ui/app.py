"""TWFarmBot Research UI.

Sidebar navigation with a clean main canvas.  Each tab renders a focused,
compact card-based view.  Zero business logic — every read and write is
proxied through the api_server.
"""

from __future__ import annotations

import json
import os
from typing import Any

import altair as alt
import streamlit as st
from twfarmbot_ml_utils import HuggingFaceImageProcessor

from twfarmbot_ui.client import ApiClient

# ── config ────────────────────────────────────────────────────────────────────

API_URL = os.getenv("TWFB_API_URL", "http://127.0.0.1:8000")
AI_SPACE_ID = os.getenv("TWFB_AI_SPACE_ID", "DavidSeyserHF/Eupe-Lang")


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
    tab = st.radio("Navigation",
                    ["Overview", "Garden", "Motion", "Camera", "Sensors", "Operations",
                     "Diagnostics", "Settings"],
                    label_visibility="collapsed")

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

    x_domain = [bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 1)]
    y_domain = [bounds.get("y", 0) + bounds.get("height", 1), bounds.get("y", 0)]
    zones_chart = alt.Chart(alt.Data(values=zone_rows)).mark_rect(
        opacity=0.18, strokeWidth=2
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=x_domain), title="X · mm"),
        x2="x2:Q",
        y=alt.Y("y:Q", scale=alt.Scale(domain=y_domain), title="Y · mm"),
        y2="y2:Q",
        color=alt.Color("kind:N", title="Layer"),
        stroke=alt.Stroke("kind:N", legend=None),
        tooltip=["name:N", "kind:N", "x:Q", "y:Q", "width:Q", "height:Q"],
    )
    points_chart = alt.Chart(alt.Data(values=point_rows)).mark_point(
        filled=True, stroke="white", strokeWidth=1
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=x_domain)),
        y=alt.Y("y:Q", scale=alt.Scale(domain=y_domain)),
        color=alt.Color("kind:N", title="Object"),
        shape=alt.Shape("kind:N", title="Object"),
        size=alt.Size("radius_mm:Q", scale=alt.Scale(range=[90, 500]), legend=None),
        tooltip=["name:N", "kind:N", "x:Q", "y:Q"],
    )

    map_col, details = st.columns([2.3, 1])
    with map_col:
        st.altair_chart(
            (zones_chart + points_chart).properties(height=520).interactive(),
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
        gx = tx.number_input("X", value=cur_x, step=10.0)
        gy = ty.number_input("Y", value=cur_y, step=10.0)
        gz = tz.number_input("Z", value=cur_z, step=10.0)
        if st.form_submit_button("Go to", use_container_width=True):
            _do_move(client, float(gx), float(gy), float(gz))

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
    "Diagnostics":  _render_diagnostics,
    "Settings":     _render_settings,
}
renderers[tab]()
