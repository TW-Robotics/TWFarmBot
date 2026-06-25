"""TWFarmBot Research UI.

Sidebar navigation with a clean main canvas.  Each tab renders a focused,
compact card-based view.  Zero business logic — every read and write is
proxied through the api_server.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import altair as alt
import streamlit as st
from ruamel.yaml import YAML
from twfarmbot_ml_utils import (
    HuggingFaceImageProcessor,
    parse_segmentation_labels,
)

from twfarmbot_core.actions import summarize_action

from twfarmbot_ui.client import ApiClient, ApiResult
from twfarmbot_ui import history

# ── config ────────────────────────────────────────────────────────────────────

API_URL = os.getenv("TWFB_API_URL", "http://127.0.0.1:8000")
AI_SPACE_ID = os.getenv("TWFB_AI_SPACE_ID", "SimonSchwaiger/resireg-playground")
PLAN_TIMEOUT = 60.0  # LLM planning can take longer than the default 2s API timeout


@st.cache_resource
def _client(base_url: str) -> ApiClient:
    return ApiClient(base_url)


@st.cache_resource
def _image_processor(space_id: str) -> HuggingFaceImageProcessor:
    return HuggingFaceImageProcessor(space_id)


# ── helpers ───────────────────────────────────────────────────────────────────


def _num(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "—"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    return summarize_action(action)


def _render_tool_call(
    name: str, args: Any, result: Any, *, show_image: bool = True
) -> None:
    """Render a compact tool call; shows AI-analysis images inline if present."""
    label = f"🔧 {name}"
    if name == "analyze_image" and isinstance(args, dict) and args.get("prompt"):
        label += f" · '{args['prompt']}'"
    elif name == "segment_image" and isinstance(args, dict) and args.get("classes"):
        label += f" · '{args['classes']}'"
    elif name == "visualize_image_features" and isinstance(args, dict):
        label += f" · clusters={args.get('n_clusters', 6)}"
    elif (
        name == "estimate_traversability"
        and isinstance(args, dict)
        and args.get("prompt")
    ):
        label += f" · '{args['prompt']}'"
    elif name == "get_images" and isinstance(args, dict) and args.get("limit"):
        label += f" · limit={args['limit']}"

    with st.expander(label, expanded=False):
        st.json({"args": args, "result": result})

    if not show_image or not isinstance(result, dict):
        return

    if name == "analyze_image" and result.get("image_url"):
        st.image(result["image_url"], use_container_width=True)
    elif name == "estimate_traversability" and result.get("image_url"):
        st.image(result["image_url"], use_container_width=True)
    elif name in {"segment_image", "visualize_image_features"} and result.get(
        "image_urls"
    ):
        cols = st.columns(min(len(result["image_urls"]), 3))
        for idx, url in enumerate(result["image_urls"]):
            cols[idx % len(cols)].image(url, use_container_width=True)
        for label_text in result.get("labels", []):
            st.caption(label_text)
    elif result.get("image_url"):
        # Fallback for any other tool that returns a single image (e.g. take_photo).
        st.image(result["image_url"], use_container_width=True)


def _render_proposed_actions_inline(
    message: dict[str, Any], actions: list[dict[str, Any]], idx: int
) -> None:
    """Render proposed actions as compact inline chat-style approval."""
    with st.container(key=f"proposal_{idx}"):
        st.markdown("*I can do this:*")
        for action in actions:
            st.markdown(f"• {_action_summary(action)}")
        approve_col, reject_col = st.columns([1, 1])
        if approve_col.button(
            "✓ Approve", key=f"approve_{idx}", use_container_width=True
        ):
            with st.spinner("Executing actions…"):
                results = _execute_proposed_actions(actions, message, wait=True)
            message["approved"] = True
            message["content"] += "\n\n" + _format_execution_results(results)
            _persist_session()
            st.rerun()
        if reject_col.button("✕ Reject", key=f"reject_{idx}", use_container_width=True):
            message["rejected"] = True
            message["content"] += "\n\n❌ Cancelled."
            _persist_session()
            st.rerun()


_APPROVAL_WORDS = {
    "yes",
    "y",
    "approve",
    "approved",
    "ok",
    "okay",
    "sure",
    "go ahead",
    "do it",
    "confirm",
    "confirmed",
    "execute",
    "run it",
}
_REJECTION_WORDS = {
    "no",
    "n",
    "reject",
    "rejected",
    "cancel",
    "cancelled",
    "don't",
    "dont",
    "stop",
    "abort",
}


def _is_approval(text: str) -> bool:
    return text.strip("!.? ").lower() in _APPROVAL_WORDS


def _is_rejection(text: str) -> bool:
    return text.strip("!.? ").lower() in _REJECTION_WORDS


_CONFIG_PATH = Path(os.getenv("TWFB_CONFIG", "configs/dev.yaml"))


def _load_config_yaml() -> tuple[YAML, Any, Path]:
    """Load the project YAML while preserving comments and formatting."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    path = Path(_CONFIG_PATH)
    with path.open() as fh:
        data = yaml.load(fh)
    return yaml, data, path


def _entity_id(name: str) -> str:
    base = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    return base or "entity"


def _add_garden_entity(x: float, y: float, kind: str, name: str) -> None:
    """Append a new entity to ``configs/dev.yaml`` and reload the world model."""
    yaml, data, path = _load_config_yaml()
    spatial = data.setdefault("spatial", {})
    entities = spatial.setdefault("entities", [])
    entity = {
        "id": _entity_id(name),
        "kind": kind,
        "name": name,
        "x": float(x),
        "y": float(y),
        "z": 0.0,
        "radius_mm": 50,
        "metadata": {},
    }
    entities.append(entity)
    with path.open("w") as fh:
        yaml.dump(data, fh)


def _selected_garden_points(event: Any) -> list[tuple[float, float]]:
    """Extract all selected grid coordinates from an Altair on_select event."""
    if not event:
        return []
    selection = event.get("selection") if isinstance(event, dict) else None
    if not isinstance(selection, dict):
        selection = event if isinstance(event, dict) else {}
    points: list[tuple[float, float]] = []
    for value in selection.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    points.append((float(item.get("x", 0)), float(item.get("y", 0))))
    return points


def _garden_grid(
    bounds: dict[str, float], step: float = 100.0
) -> list[dict[str, float]]:
    x0 = bounds.get("x", 0)
    y0 = bounds.get("y", 0)
    width = bounds.get("width", 0)
    height = bounds.get("height", 0)
    rows: list[dict[str, float]] = []
    xi = x0
    while xi <= x0 + width:
        yi = y0
        while yi <= y0 + height:
            rows.append({"x": round(xi, 1), "y": round(yi, 1)})
            yi += step
        xi += step
    return rows


def _refresh_position(client: ApiClient) -> None:
    xyz = client.get_position() or {}
    st.session_state["pos_x"] = _num(xyz.get("x"))
    st.session_state["pos_y"] = _num(xyz.get("y"))
    st.session_state["pos_z"] = _num(xyz.get("z"))


def _refresh_health(client: ApiClient) -> None:
    health = client.get_health()
    if health is not None:
        st.session_state["farmbot_status"] = health.get("farmbot", "?")
        st.session_state["actions"] = health.get("actions", [])


def _refresh_messages(client: ApiClient) -> None:
    st.session_state["messages"] = client.get_messages()


def _refresh_telemetry(client: ApiClient) -> None:
    _refresh_position(client)
    _refresh_health(client)
    _refresh_messages(client)


TABS = [
    "Overview",
    "Garden",
    "Motion",
    "Camera",
    "I/O",
    "Assistant",
    "History",
    "Diagnostics",
    "Settings",
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
    # Legacy routes from the old Sensors / Operations tabs now live under I/O.
    if low in {"sensors", "operations"}:
        return "I/O"
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
    r = client.request(
        "POST", "/actions", json={"kind": "move", "params": {"x": x, "y": y, "z": z}}
    )
    if r.ok:
        msg = f"→ {label}" if label else f"→ ({x:.0f}, {y:.0f}, {z:.0f})"
        st.toast(msg, icon="➡️")
        _refresh_position(client)
    else:
        st.error(r.error_message())


def _do_pin_write(
    client: ApiClient, pin: int, value: int, mode: str = "digital"
) -> None:
    r = client.request(
        "POST",
        "/actions",
        json={
            "kind": "write_pin",
            "params": {"pin": pin, "value": value, "mode": mode},
        },
    )
    if r.ok:
        st.toast(f"pin {pin} = {value}", icon="✏️")
    else:
        st.error(r.error_message())


def _do_pin_pulse(
    client: ApiClient, pin: int, seconds: float, mode: str = "digital"
) -> None:
    r = client.request(
        "POST",
        "/actions",
        json={
            "kind": "write_pin",
            "params": {"pin": pin, "value": 1, "mode": mode, "seconds": seconds},
        },
    )
    if r.ok:
        st.toast(f"pin {pin} HIGH for {seconds}s", icon="✏️")
    else:
        st.error(r.error_message())


# ── page shell ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TWFarmBot Research",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
  .stMain {
    margin-left: 18rem !important;
    width: calc(100% - 18rem) !important;
    max-width: calc(100% - 18rem) !important;
  }
  .block-container {
    max-width: 100% !important;
    padding: 0.5rem 1rem 5rem 1rem;
  }
  .eyebrow { margin-bottom: 0 !important; }
  h1 { margin-top: 0.1rem !important; margin-bottom: 0.1rem !important; }
  [data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 18rem !important;
    width: calc(100% - 18rem) !important;
    z-index: 100 !important;
    background: var(--background-color) !important;
    padding: 0.5rem 1rem 1rem 1rem !important;
    border-top: 1px solid rgba(128,128,128,0.1) !important;
  }
  h1 { font-size: 1.15rem; letter-spacing: -0.01em; margin-bottom: 0.35rem; }
  .eyebrow { color: #3f8f64; font-size: .55rem; font-weight: 750;
             letter-spacing: .12em; text-transform: uppercase; }
  .card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.12);
    border-radius: 10px; padding: .8rem 1rem; min-height: 5rem;
  }
  .card-label { font-size: .66rem; font-weight: 700; letter-spacing: .09em;
                text-transform: uppercase; opacity: .5; }
  .card-value { font-size: 1.3rem; font-weight: 650; margin-top: .4rem; }
  .sensor-value {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.10);
    border-radius: 9px;
    padding: .55rem .75rem;
    font-size: 1.25rem;
    font-weight: 650;
    min-height: 2.2rem;
    display: flex;
    align-items: center;
    margin-top: .4rem;
  }
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
    max-height: 70vh !important;
    object-fit: contain !important;
    background: var(--secondary-background-color);
    border-radius: 9px;
  }
  [data-testid="stChatMessage"] img,
  [data-testid="stImage"] img {
    max-width: 100% !important;
    max-height: 70vh !important;
    border-radius: 9px;
  }
  [data-testid="stChatMessage"] {
    padding: 0.35rem 0 !important;
    margin-bottom: 0.25rem !important;
    background: transparent !important;
  }
  [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] li {
    font-size: 0.92rem !important;
    line-height: 1.35 !important;
    margin-bottom: 0.15rem !important;
  }
  [data-testid="stChatMessage"] .stCaption {
    font-size: 0.72rem !important;
  }
  [class*="st-key-proposal_"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.12);
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
    margin: 0.35rem 0 0.5rem 0;
  }
  [class*="st-key-user_msg_"] [data-testid="stChatMessage"] {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
  }
  [class*="st-key-user_msg_"] [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
    margin-left: 0.5rem !important;
    margin-right: 0 !important;
  }
  [class*="st-key-user_msg_"] [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    align-items: flex-end !important;
    margin-right: 0.75rem !important;
  }
  [class*="st-key-user_msg_"] [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p {
    text-align: right !important;
  }
  @media (max-width: 760px) {
    section[data-testid="stSidebar"] { width: 14rem !important; min-width: 14rem !important; }
    .stMain {
      margin-left: 14rem !important;
      width: calc(100% - 14rem) !important;
      max-width: calc(100% - 14rem) !important;
    }
    .block-container { max-width: 100% !important; }
    [data-testid="stChatInput"] {
      left: 14rem !important;
      width: calc(100% - 14rem) !important;
    }
  }
</style>
""",
    unsafe_allow_html=True,
)

api_url = st.session_state.setdefault("api_url", API_URL)
client = _client(api_url)
if "farmbot_status" not in st.session_state:
    _refresh_health(client)
    _refresh_position(client)
    st.session_state.setdefault("messages", [])

# ── sidebar  ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-kicker">Field robotics</div>'
        '<div class="sidebar-brand">TWFarmBot</div>',
        unsafe_allow_html=True,
    )
    # Sync the navigation radio with the URL ?tab=... query parameter so
    # refreshing the browser returns to the same tab.
    url_tab = _tab_from_key(_qp_tab())
    if st.session_state.get("nav_tab") != url_tab:
        st.session_state["nav_tab"] = url_tab
    tab = st.radio(
        "Navigation",
        TABS,
        key="nav_tab",
        on_change=_sync_tab_url,
        label_visibility="collapsed",
    )

    st.divider()
    fb = st.session_state.get("farmbot_status", "?")
    pill_css = "ok" if fb == "connected" else ("warn" if fb == "skipped" else "bad")
    st.markdown(f'<span class="pill {pill_css}">● {fb}</span>', unsafe_allow_html=True)
    st.caption(
        f"X {st.session_state.get('pos_x', '—')} · "
        f"Y {st.session_state.get('pos_y', '—')} · "
        f"Z {st.session_state.get('pos_z', '—')} mm"
    )
    if st.button("↻ Refresh", use_container_width=True):
        _refresh_telemetry(client)
        st.rerun()

    st.divider()
    if st.button("🛑 ESTOP", type="primary", use_container_width=True):
        r = client.request("POST", "/actions", json={"kind": "e_stop", "params": {}})
        if r.ok:
            st.toast("ESTOP sent", icon="🛑")
        else:
            st.error(r.error_message())

# ── tab content ───────────────────────────────────────────────────────────────


def _render_overview() -> None:
    st.markdown(
        '<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>',
        unsafe_allow_html=True,
    )
    st.markdown("# Research overview")

    # ── Live position ──────────────────────────────────────────────────────────
    row = st.columns(3)
    row[0].metric("X · mm", st.session_state.get("pos_x", "—"))
    row[1].metric("Y · mm", st.session_state.get("pos_y", "—"))
    row[2].metric("Z · mm", st.session_state.get("pos_z", "—"))

    # ── System status ──────────────────────────────────────────────────────────
    st.markdown("### System status")
    st.session_state.setdefault("history", [])

    refresh_col, clear_col = st.columns([3, 1])
    with refresh_col:
        refresh_clicked = st.button("🔄 Refresh status", use_container_width=True)
    with clear_col:
        if st.button("Clear history", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    if refresh_clicked:
        _refresh_health(client)
        _refresh_position(client)
        d = client.request("GET", "/status")
        st.session_state["diag"] = (
            d.body.get("state", {}) if d.ok and isinstance(d.body, dict) else {}
        )
        info_for_history = (st.session_state.get("diag") or {}).get(
            "informational_settings", {}
        ) or {}
        st.session_state["history"].append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cpu": _float(info_for_history.get("cpu_usage")),
                "memory": _float(info_for_history.get("memory_usage")),
                "disk": _float(info_for_history.get("disk_usage")),
                "wifi": _float(info_for_history.get("wifi_level_percent")),
                "soc": _float(info_for_history.get("soc_temp")),
                "uptime": _float(info_for_history.get("uptime")),
            }
        )
        # Keep the last 60 samples so the chart stays readable.
        st.session_state["history"] = st.session_state["history"][-60:]
        st.rerun()

    payload = st.session_state.get("diag", {}) or {}
    info = payload.get("informational_settings", {}) or {}
    loc = payload.get("location_data", {}) or {}

    status_cols = st.columns(5)
    status_cols[0].metric("FarmBot", st.session_state.get("farmbot_status", "—"))
    status_cols[1].metric("Uptime", f"{_num(info.get('uptime'))} s")
    status_cols[2].metric("Wi-Fi", f"{_num(info.get('wifi_level_percent'))}%")
    status_cols[3].metric("Sync", info.get("sync_status", "—"))
    status_cols[4].metric("Busy", "Yes" if info.get("busy") else "No")

    # ── Resources over time ────────────────────────────────────────────────────
    st.markdown("### Resources over time")
    cpu = info.get("cpu_usage")
    mem = info.get("memory_usage")
    disk = info.get("disk_usage")
    soc = info.get("soc_temp")

    res_cols = st.columns(4)
    res_cols[0].metric("CPU", f"{cpu}%" if cpu is not None else "—")
    res_cols[1].metric("Memory", f"{mem}%" if mem is not None else "—")
    res_cols[2].metric("Disk", f"{disk}%" if disk is not None else "—")
    res_cols[3].metric("SoC temp", f"{soc}°C" if soc is not None else "—")

    hist = st.session_state["history"]
    if hist:
        base = alt.Chart(alt.Data(values=hist))
        usage_lines = (
            base.transform_fold(
                fold=["cpu", "memory", "disk"],
                as_=["metric", "value"],
            )
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("time:N", title=None),
                y=alt.Y("value:Q", title="Usage %", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "metric:N",
                    scale=alt.Scale(
                        domain=["cpu", "memory", "disk"],
                        range=["#3f8f64", "#5b8fc7", "#c7a15b"],
                    ),
                    legend=alt.Legend(title="Metric"),
                ),
            )
            .properties(height=240)
        )
        st.altair_chart(usage_lines, use_container_width=True)

        extra_cols = st.columns(2)
        with extra_cols[0]:
            wifi_chart = (
                base.mark_line(point=True, color="#5b8fc7", strokeWidth=2)
                .encode(
                    x=alt.X("time:N", title=None),
                    y=alt.Y(
                        "wifi:Q", title="Wi-Fi %", scale=alt.Scale(domain=[0, 100])
                    ),
                )
                .properties(height=180)
            )
            st.altair_chart(wifi_chart, use_container_width=True)
        with extra_cols[1]:
            soc_chart = (
                base.mark_line(point=True, color="#c75b5b", strokeWidth=2)
                .encode(
                    x=alt.X("time:N", title=None),
                    y=alt.Y("soc:Q", title="SoC temp °C"),
                )
                .properties(height=180)
            )
            st.altair_chart(soc_chart, use_container_width=True)
    else:
        st.info("Click **Refresh status** to start collecting data for the charts.")

    # ── Network & details ──────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Network & hardware**")
        st.caption(f"Private IP: `{info.get('private_ip', '—')}`")
        st.caption(f"Wi-Fi signal: {_num(info.get('wifi_level'))} dBm")
        st.caption(f"Controller: {info.get('controller_version', '—')}")
        st.caption(f"Firmware: {info.get('firmware_version', '—')}")
        axes = loc.get("axis_states", {}) or {}
        st.caption(
            f"Axis states · X {axes.get('x', '—')} · Y {axes.get('y', '—')} · Z {axes.get('z', '—')}"
        )

    with c2:
        st.markdown("**Recent events**")
        msgs = st.session_state.get("messages", [])
        if msgs:
            st.code("\n".join(msgs[-10:]), language="text")
        else:
            st.caption("No events recorded.")

    # ── Experiment notes ───────────────────────────────────────────────────────
    st.markdown("### Experiment")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input(
            "Run",
            key="run",
            placeholder="e.g. soil-map-07",
            label_visibility="collapsed",
        )
    with c2:
        st.text_input(
            "Operator", key="op", placeholder="initials", label_visibility="collapsed"
        )
    with c3:
        st.text_area(
            "Notes",
            key="notes",
            placeholder="Conditions, observations…",
            height=90,
            label_visibility="collapsed",
        )


def _render_garden() -> None:
    st.markdown(
        '<div class="eyebrow">Spatial model · configured world state</div>',
        unsafe_allow_html=True,
    )
    st.markdown("# Garden map")

    result = client.request("GET", "/garden")
    if not result.ok or not isinstance(result.body, dict):
        st.error(f"Garden model unavailable: {result.error_message()}")
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

    zone_rows = [
        {
            **zone["bounds"],
            "x2": zone["bounds"]["x"] + zone["bounds"]["width"],
            "y2": zone["bounds"]["y"] + zone["bounds"]["height"],
            "kind": zone["kind"],
            "name": zone["name"],
        }
        for zone in zones
    ]
    point_rows = [
        {
            "x": entity["position"]["x"],
            "y": entity["position"]["y"],
            "kind": entity["kind"],
            "name": entity["name"],
            "radius_mm": entity["radius_mm"],
        }
        for entity in entities
    ]
    point_rows.extend(
        [
            {
                "x": robot.get("x", 0),
                "y": robot.get("y", 0),
                "kind": "robot",
                "name": "FarmBot",
                "radius_mm": 35,
            },
            {
                "x": camera.get("position", {}).get("x", 0),
                "y": camera.get("position", {}).get("y", 0),
                "kind": "camera",
                "name": "Camera",
                "radius_mm": 25,
            },
        ]
    )

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

    bounds_chart = (
        alt.Chart(
            alt.Data(
                values=[
                    {
                        "x": x_min,
                        "y": y_min,
                        "x2": x_max,
                        "y2": y_max,
                    }
                ]
            )
        )
        .mark_rect(filled=False, stroke="#888888", strokeWidth=2)
        .encode(
            x=alt.X("x:Q", scale=x_scale, title="X · mm"),
            x2="x2:Q",
            y=alt.Y("y:Q", scale=y_scale, title="Y · mm"),
            y2="y2:Q",
        )
    )
    zones_chart = (
        alt.Chart(alt.Data(values=zone_rows))
        .mark_rect(opacity=0.18, strokeWidth=2)
        .encode(
            x=alt.X("x:Q", scale=x_scale, title="X · mm"),
            x2="x2:Q",
            y=alt.Y("y:Q", scale=y_scale, title="Y · mm"),
            y2="y2:Q",
            color=alt.Color("kind:N", title="Layer"),
            stroke=alt.Stroke("kind:N", legend=None),
            tooltip=["name:N", "kind:N", "x:Q", "y:Q", "width:Q", "height:Q"],
        )
    )
    points_chart = (
        alt.Chart(alt.Data(values=point_rows))
        .mark_point(filled=True, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", scale=x_scale),
            y=alt.Y("y:Q", scale=y_scale),
            color=alt.Color("kind:N", title="Object"),
            shape=alt.value("circle"),
            size=alt.Size("radius_mm:Q", scale=alt.Scale(range=[90, 500]), legend=None),
            tooltip=["name:N", "kind:N", "x:Q", "y:Q"],
        )
    )

    grid_rows = _garden_grid(bounds, step=25)
    click_selection = alt.selection_point(
        name="garden_click",
        fields=["x", "y"],
        on="click",
        toggle=True,
        nearest=True,
        empty=False,
    )
    click_layer = (
        alt.Chart(alt.Data(values=grid_rows))
        .mark_point(size=120)
        .encode(
            x=alt.X("x:Q", scale=x_scale),
            y=alt.Y("y:Q", scale=y_scale),
            opacity=alt.condition(click_selection, alt.value(0.7), alt.value(0)),
            color=alt.value("#ff4b4b"),
        )
        .add_params(click_selection)
    )

    map_col, details = st.columns([2.3, 1])
    with map_col:
        event = st.altair_chart(
            (bounds_chart + zones_chart + points_chart + click_layer)
            .properties(height=520)
            .interactive(),
            width="stretch",
            on_select="rerun",
            key="garden_map",
        )
    selected_points = _selected_garden_points(st.session_state.get("garden_map"))
    with details:
        if selected_points:
            with st.container(border=True):
                st.markdown(f"**🌱 {len(selected_points)} selected**")
                kind = st.pills(
                    "Kind",
                    [
                        "plant",
                        "obstacle",
                        "tool",
                        "marker",
                        "sensor",
                        "valve",
                        "custom",
                    ],
                    default="plant",
                    selection_mode="single",
                    label_visibility="collapsed",
                    key="garden_assign_kind",
                )
                custom_kind = ""
                if kind == "custom":
                    custom_kind = st.text_input(
                        "Custom kind",
                        placeholder="e.g. watering",
                        key="garden_assign_custom_kind",
                    )
                name = st.text_input(
                    "Name prefix",
                    placeholder="e.g. Tomato",
                    key="garden_assign_name",
                )
                c1, c2 = st.columns(2)
                if c1.button(
                    f"Assign {len(selected_points)}",
                    key="garden_assign_save",
                    use_container_width=True,
                ):
                    final_kind = (
                        custom_kind
                        if kind == "custom" and custom_kind
                        else (kind or "plant")
                    )
                    if name:
                        for i, (px, py) in enumerate(selected_points, start=1):
                            _add_garden_entity(px, py, final_kind, f"{name}-{i}")
                        st.success(f"Added {len(selected_points)} {final_kind}(s)")
                        st.session_state.pop("garden_map", None)
                        st.rerun()
                    else:
                        st.warning("Please enter a name prefix.")
                if c2.button(
                    "Clear", key="garden_assign_cancel", use_container_width=True
                ):
                    st.session_state.pop("garden_map", None)
                    st.rerun()
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

    st.markdown(
        '<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>',
        unsafe_allow_html=True,
    )
    st.markdown("# Motion workspace")

    row = st.columns(3)
    row[0].metric("X · mm", st.session_state.get("pos_x", "—"))
    row[1].metric("Y · mm", st.session_state.get("pos_y", "—"))
    row[2].metric("Z · mm", st.session_state.get("pos_z", "—"))

    step = float(st.segmented_control("Jog step · mm", [1, 10, 50, 100], default=10))

    # D-pad
    _, u, _ = st.columns(3)
    if u.button("▲ Y+", use_container_width=True):
        _do_move(client, cur_x, cur_y + step, cur_z, f"Y+{step:.0f}")
    l, m, r = st.columns(3)
    if l.button("◀ X−", use_container_width=True):
        _do_move(client, cur_x - step, cur_y, cur_z, f"X-{step:.0f}")
    if m.button("🏠 Home", use_container_width=True):
        _do_move(client, 0, 0, 0, "Home")
    if r.button("X+ ▶", use_container_width=True):
        _do_move(client, cur_x + step, cur_y, cur_z, f"X+{step:.0f}")
    _, d, _ = st.columns(3)
    if d.button("▼ Y−", use_container_width=True):
        _do_move(client, cur_x, cur_y - step, cur_z, f"Y-{step:.0f}")

    zl, zr = st.columns(2)
    if zl.button("⬆ Z+", use_container_width=True):
        _do_move(client, cur_x, cur_y, cur_z + step, f"Z+{step:.0f}")
    if zr.button("⬇ Z−", use_container_width=True):
        _do_move(client, cur_x, cur_y, cur_z - step, f"Z-{step:.0f}")

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
            st.error(r.error_message())

    # Presets
    if "presets" not in st.session_state:
        r = client.request("GET", "/positions")
        st.session_state["presets"] = r.body.get("positions", []) if r.ok else []
    presets = st.session_state["presets"]
    if presets:
        st.markdown("**Locations**")
        cols = st.columns(min(5, len(presets)))
        for i, p in enumerate(presets):
            if cols[i].button(
                p.get("label", "?"), key=f"preset_{i}", use_container_width=True
            ):
                _do_move(
                    client, float(p["x"]), float(p["y"]), float(p["z"]), p["label"]
                )


def _render_io() -> None:
    st.markdown(
        '<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>',
        unsafe_allow_html=True,
    )
    st.markdown("# I/O workspace")

    if "named_pins" not in st.session_state:
        r = client.request("GET", "/pins")
        st.session_state["named_pins"] = r.body.get("pins", []) if r.ok else []
    named = st.session_state["named_pins"]

    # ── Sensors ───────────────────────────────────────────────────────────────
    sensors = [p for p in named if p.get("kind") == "sensor"]
    if sensors:
        st.markdown("### 🔍 Sensors")
        cols = st.columns(min(3, len(sensors)))
        for i, s in enumerate(sensors):
            with cols[i]:
                with st.container(border=True, height=170):
                    mode = s.get("mode", "analog")
                    st.markdown(
                        f"**{s['label']}**  <span class='pill'>{mode}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"pin {s['pin']}")
                    if st.button("Read", key=f"sensor_{i}", use_container_width=True):
                        r = client.request(
                            "GET",
                            f"/pin/{s['pin']}",
                            params={"mode": mode},
                        )
                        st.session_state[f"sv_{s['pin']}"] = (
                            r.body.get("value") if r.ok else "—"
                        )
                    sensor_value = st.session_state.get(f"sv_{s['pin']}", "—")
                    st.markdown(
                        f"<div class='sensor-value'>{sensor_value}</div>",
                        unsafe_allow_html=True,
                    )
    elif named:
        st.info("No sensor pins configured.")

    st.divider()

    # ── Actuators ─────────────────────────────────────────────────────────────
    st.markdown("### ⚡ Actuators")
    a, b = st.columns(2)
    with a:
        with st.container(border=True, height=320):
            st.markdown("**💧 Irrigation**")
            secs = st.number_input("Seconds", 0.1, 300.0, 2.0, 0.5, key="water_secs")
            if st.button("Water", use_container_width=True, type="primary"):
                r = client.request(
                    "POST",
                    "/actions",
                    json={"kind": "water", "params": {"seconds": secs}},
                )
                if r.ok:
                    st.success("Queued")
                else:
                    st.error(r.error_message())
            st.caption("Runs the pump for the selected duration.")

    with b:
        with st.container(border=True, height=320):
            st.markdown("**🔌 Peripheral control**")
            outputs = [p for p in named if p.get("kind") != "sensor"]
            if not outputs:
                st.info("No output pins configured.")
            else:
                sel = st.selectbox(
                    "Output",
                    outputs,
                    format_func=lambda p: f"{p['label']} · pin {p['pin']}",
                    label_visibility="collapsed",
                )
                if sel:
                    mode = sel.get("mode", "digital")
                    st.markdown(
                        f"<span class='pill'>{mode}</span>",
                        unsafe_allow_html=True,
                    )

                    if mode == "analog":
                        val_col, btn_col = st.columns([4, 1])
                        with val_col:
                            analog_value = st.slider(
                                "PWM value",
                                min_value=0,
                                max_value=255,
                                value=0,
                                key=f"analog_value_{sel['pin']}",
                            )
                        with btn_col:
                            st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
                            if st.button("Apply", use_container_width=True):
                                _do_pin_write(client, sel["pin"], analog_value, mode)
                    else:
                        pulse = st.toggle("Timed pulse", value=True)
                        if pulse:
                            pulse_secs = st.number_input(
                                "Seconds",
                                0.1,
                                300.0,
                                2.0,
                                0.5,
                                key=f"pulse_secs_{sel['pin']}",
                            )

                        off, on = st.columns(2)
                        if off.button("⏻ OFF", use_container_width=True):
                            _do_pin_write(client, sel["pin"], 0, mode)
                        if on.button("⏻ ON", use_container_width=True, type="primary"):
                            if pulse and pulse_secs is not None:
                                _do_pin_pulse(client, sel["pin"], pulse_secs, mode)
                            else:
                                _do_pin_write(client, sel["pin"], 1, mode)


def _render_camera() -> None:
    st.markdown(
        '<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>',
        unsafe_allow_html=True,
    )
    st.markdown("# Camera")

    capture, refresh, _ = st.columns([1, 1, 4])
    if capture.button("📷 Take photo", type="primary", use_container_width=True):
        r = client.request(
            "POST", "/actions", json={"kind": "take_photo", "params": {}}
        )
        if r.ok:
            st.toast("Capture queued", icon="📷")
        else:
            st.error(r.error_message())
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

    # Show the source image and analysis controls side-by-side so both fit
    # above the fold without scrolling. The analysis results still appear below.
    img_col, ctrl_col = st.columns([1.6, 1])

    with img_col:
        st.image(selected.get("attachment_url"), use_container_width=True)

    with ctrl_col:
        st.markdown("**AI analysis**")
        mode = st.selectbox(
            "Mode",
            [
                "Open Language Similarity",
                "Zero-Shot Segmentation",
                "PCA Feature Visualization",
                "Traversability Estimation",
            ],
            key=f"ai_mode_{selected.get('id', 'unknown')}",
            label_visibility="collapsed",
        )

        processor = _image_processor(AI_SPACE_ID)
        inputs: dict[str, Any] = {}
        button_disabled = False

        if mode == "Open Language Similarity":
            inputs["prompt"] = st.text_input(
                "Target prompt",
                placeholder="e.g. green leaves, dry soil, red marker",
                key=f"ai_prompt_{selected.get('id', 'unknown')}",
                label_visibility="collapsed",
            )
            button_disabled = not inputs["prompt"].strip()
        elif mode == "Zero-Shot Segmentation":
            inputs["classes"] = st.text_input(
                "Classes (comma-separated)",
                value="plant, weed, soil, path",
                key=f"ai_classes_{selected.get('id', 'unknown')}",
            )
            inputs["negative"] = st.text_input(
                "Background prompt (optional)",
                placeholder="e.g. thing, object, stuff",
                key=f"ai_negative_{selected.get('id', 'unknown')}",
            )
            button_disabled = not inputs["classes"].strip()
        elif mode == "PCA Feature Visualization":
            inputs["n_clusters"] = st.slider(
                "K-means clusters",
                min_value=2,
                max_value=20,
                value=6,
                key=f"ai_clusters_{selected.get('id', 'unknown')}",
            )
        elif mode == "Traversability Estimation":
            inputs["prompt"] = st.text_input(
                "Traversable prompt",
                placeholder="e.g. path, road, flat ground",
                key=f"ai_trav_prompt_{selected.get('id', 'unknown')}",
                label_visibility="collapsed",
            )
            inputs["negatives"] = st.text_input(
                "Background prompts (optional, comma-separated)",
                placeholder="e.g. thing, object, stuff, scenery",
                key=f"ai_trav_negatives_{selected.get('id', 'unknown')}",
            )
            button_disabled = not inputs["prompt"].strip()

        if st.button(
            "Analyze selected image",
            type="primary",
            use_container_width=True,
            disabled=button_disabled,
        ):
            try:
                with st.spinner("Processing image…"):
                    if mode == "Open Language Similarity":
                        result_path = processor.process(
                            selected["attachment_url"],
                            inputs["prompt"].strip(),
                            negatives="",
                        )
                        st.session_state["ai_result"] = {
                            "image_id": selected.get("id"),
                            "source_url": selected.get("attachment_url"),
                            "paths": [str(result_path)],
                            "captions": [f"Similarity map · {inputs['prompt']}"],
                            "labels": [],
                            "mode": mode,
                        }
                    elif mode == "Zero-Shot Segmentation":
                        raw = processor.predict(
                            selected["attachment_url"],
                            api_name="/run_seg",
                            classes=inputs["classes"].strip(),
                            negative=inputs["negative"].strip(),
                        )
                        labels = [str(raw[2]), str(raw[3])]
                        class_scores = parse_segmentation_labels(labels)
                        st.session_state["ai_result"] = {
                            "image_id": selected.get("id"),
                            "source_url": selected.get("attachment_url"),
                            "paths": [str(raw[0]), str(raw[1])],
                            "captions": ["Segmentation overlay", "Segmentation map"],
                            "labels": labels,
                            "class_scores": class_scores,
                            "dominant_class": (
                                max(class_scores, key=class_scores.get)
                                if class_scores
                                else None
                            ),
                            "classes": inputs["classes"].strip(),
                            "mode": mode,
                        }
                    elif mode == "PCA Feature Visualization":
                        raw = processor.predict(
                            selected["attachment_url"],
                            api_name="/run_pca",
                            n_clusters=int(inputs["n_clusters"]),
                        )
                        st.session_state["ai_result"] = {
                            "image_id": selected.get("id"),
                            "source_url": selected.get("attachment_url"),
                            "paths": [str(raw[0]), str(raw[1]), str(raw[2])],
                            "captions": [
                                "PCA visualization 1",
                                "PCA visualization 2",
                                "PCA visualization 3",
                            ],
                            "labels": [],
                            "n_clusters": int(inputs["n_clusters"]),
                            "mode": mode,
                        }
                    elif mode == "Traversability Estimation":
                        result_path = processor.predict(
                            selected["attachment_url"],
                            api_name="/run_trav",
                            prompt=inputs["prompt"].strip(),
                            negatives=inputs["negatives"].strip(),
                        )
                        st.session_state["ai_result"] = {
                            "image_id": selected.get("id"),
                            "source_url": selected.get("attachment_url"),
                            "paths": [str(result_path)],
                            "captions": [f"Traversability map · {inputs['prompt']}"],
                            "labels": [],
                            "mode": mode,
                        }
            except Exception as exc:
                st.error(f"AI processing failed: {exc}")

    result = st.session_state.get("ai_result")
    if result:
        st.markdown("### Analysis result")
        result_cols = st.columns(len(result["paths"]))
        for idx, (path, caption) in enumerate(zip(result["paths"], result["captions"])):
            with result_cols[idx]:
                st.image(path, caption=caption, use_container_width=True)

        st.markdown("**Raw output**")
        if result.get("class_scores"):
            score_cols = st.columns(len(result["class_scores"]))
            for idx, (cls, score) in enumerate(result["class_scores"].items()):
                score_cols[idx].metric(cls, f"{score * 100:.1f}%")
            if result.get("dominant_class"):
                st.caption(f"Dominant class: **{result['dominant_class']}**")
        elif result.get("n_clusters"):
            st.caption(f"PCA with **{result['n_clusters']}** K-means clusters")

        for label in result.get("labels", []):
            st.caption(label)

    if len(images) > 1:
        st.markdown("**Recent captures**")
        gallery = st.columns(3)
        for index, image in enumerate(images[1:7]):
            image_meta = image.get("meta") or {}
            gallery[index % 3].image(
                image.get("attachment_url"),
                caption=f"X {image_meta.get('x', '—')} · Y {image_meta.get('y', '—')}",
                width=240,
            )


def _render_model_picker() -> str | None:
    """Render provider + model selectors and return the selected model id."""
    if "assistant_providers" not in st.session_state:
        r = client.request("GET", "/providers")
        if r.ok and isinstance(r.body, dict):
            st.session_state["assistant_providers"] = r.body.get("providers", [])
            st.session_state["assistant_provider"] = r.body.get("current", "openrouter")
        else:
            st.session_state["assistant_providers"] = ["openrouter", "local"]
            st.session_state["assistant_provider"] = "openrouter"

    providers = st.session_state["assistant_providers"]
    provider_col, model_col = st.columns([1, 2])
    with provider_col:
        provider = st.selectbox(
            "Provider",
            providers,
            key="assistant_provider",
        )

    cache = st.session_state.setdefault("assistant_models_cache", {})
    if provider not in cache:
        r = client.request("GET", "/models", params={"provider": provider})
        if r.ok and isinstance(r.body, dict):
            cache[provider] = r.body.get("models", [])
            if not st.session_state.get("assistant_model"):
                st.session_state["assistant_model"] = r.body.get("current")
        else:
            cache[provider] = []

    models = cache.get(provider, [])
    selected_model: str | None = None
    with model_col:
        if models:
            current = st.session_state.get("assistant_model")
            # Avoid defaulting to the first option from a raw provider list,
            # which may be a meta/safeguard model that cannot chat.
            if current not in models:
                preferred = [
                    "openai/gpt-4o-mini",
                    "openai/gpt-4o",
                    "anthropic/claude-3.5-sonnet",
                    "anthropic/claude-3.5-haiku",
                    "deepseek/deepseek-v4-flash",
                ]
                current = next((m for m in preferred if m in models), models[0])
                st.session_state["assistant_model"] = current
            index = models.index(current)
            selected_model = st.selectbox(
                "Model",
                models,
                index=index,
                key="assistant_model",
            )
        else:
            selected_model = st.text_input(
                "Model",
                value=st.session_state.get("assistant_model", ""),
                key="assistant_model",
            ) or None
    return selected_model


def _render_assistant() -> None:
    st.markdown(
        '<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>',
        unsafe_allow_html=True,
    )
    title_col, clear_col = st.columns([5, 1])
    with title_col:
        st.markdown("# Assistant")
    with clear_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state["assistant_messages"] = []
            _persist_session()
            st.rerun()
    _render_session_controls()
    selected_model = _render_model_picker()
    st.session_state["assistant_selected_model"] = selected_model
    _render_chat()
    _persist_session()


def _render_chat() -> None:
    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = []

    for idx, msg in enumerate(st.session_state["assistant_messages"]):
        if msg.get("role") == "tool":
            with st.chat_message("assistant"):
                _render_tool_call(
                    msg.get("name", "tool"),
                    msg.get("args"),
                    msg.get("result"),
                    show_image=True,
                )
            continue

        if msg.get("role") == "user":
            with st.container(key=f"user_msg_{idx}"):
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            continue

        # Render the model's reasoning as its own collapsible assistant pill,
        # similar to how tool calls are shown, so the conversation flow is clear.
        if msg.get("thinking"):
            with st.chat_message("assistant"):
                with st.expander("🧠 Thinking", expanded=False):
                    st.markdown(msg["thinking"])

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            images = msg.get("images", [])
            if images:
                cols = st.columns(min(len(images), 3))
                for i, image in enumerate(images):
                    cols[i % len(cols)].image(
                        image.get("attachment_url"), width=220
                    )

            proposed_actions = msg.get("proposed_actions", [])
            if proposed_actions and not msg.get("approved") and not msg.get("rejected"):
                _render_proposed_actions_inline(msg, proposed_actions, idx)
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
            pending = (
                proposed
                and not last_assistant.get("approved")
                and not last_assistant.get("rejected")
            )
            approval = _is_approval(prompt)
            rejection = _is_rejection(prompt)
            if approval or rejection:
                if pending:
                    st.session_state["assistant_messages"].append(
                        {"role": "user", "content": prompt}
                    )
                    with st.container(key="user_msg_current"):
                        with st.chat_message("user"):
                            st.markdown(prompt)
                    if approval:
                        with st.spinner("Executing actions…"):
                            results = _execute_proposed_actions(proposed, last_assistant, wait=True)
                        last_assistant["approved"] = True
                        last_assistant["content"] += "\n\n" + _format_execution_results(results)
                    else:
                        last_assistant["rejected"] = True
                        last_assistant["content"] += "\n\n❌ Cancelled."
                    _persist_session()
                    st.rerun()
                else:
                    st.toast("No pending proposal to approve or reject.", icon="⚠️")
                    _persist_session()
                    st.rerun()

        st.session_state["assistant_messages"].append(
            {"role": "user", "content": prompt}
        )
        with st.container(key="user_msg_current"):
            with st.chat_message("user"):
                st.markdown(prompt)

        thinking = st.empty()
        thinking.caption("🤖 Assistant is thinking…")
        with st.chat_message("assistant"):
            stream_meta = {"tool_calls": [], "proposed_actions": []}
            stream_thinking: list[str] = []
            stream_error = None
            accumulated = ""
            # Preserve the order of streamed text relative to tools/thinking.
            # Each open segment is a placeholder that gets updated in place.
            # When a tool or thinking block starts, the current text segment is
            # closed so subsequent text appears *after* that block.
            text_segments: list[list[Any, str] | None] = []

            def _current_text_segment() -> list[Any, str]:
                if not text_segments or text_segments[-1] is None:
                    ph = st.empty()
                    text_segments.append([ph, ""])
                return text_segments[-1]

            def _close_text_segment() -> None:
                if text_segments and text_segments[-1] is not None:
                    text_segments.append(None)

            try:
                for event in client.stream(
                    "POST",
                    "/chat/stream",
                    json={
                        "messages": st.session_state["assistant_messages"],
                        "model": st.session_state.get("assistant_selected_model"),
                    },
                    timeout=PLAN_TIMEOUT,
                ):
                    etype = event.get("type")
                    if etype == "delta":
                        accumulated += event.get("content", "")
                        seg = _current_text_segment()
                        seg[1] = accumulated
                        seg[0].markdown(accumulated)
                    elif etype == "thinking":
                        _close_text_segment()
                        think_text = str(event.get("content", ""))
                        with st.expander("🧠 Thinking", expanded=False):
                            st.markdown(think_text)
                        stream_thinking.append(think_text)
                    elif etype == "tool_call":
                        _close_text_segment()
                        thinking.caption("🤖 Assistant is using tools…")
                        name = event.get("name")
                        args = event.get("args")
                        result = event.get("result")
                        st.session_state["assistant_messages"].append(
                            {
                                "role": "tool",
                                "name": name,
                                "args": args,
                                "result": result,
                            }
                        )
                        if (
                            name == "take_photo"
                            and isinstance(result, dict)
                            and result.get("status") == "ok"
                        ):
                            image = _capture_photo_image()
                            if image:
                                result["image_url"] = image.get("attachment_url")
                        _render_tool_call(name, args, result)
                    elif etype == "meta":
                        stream_meta["tool_calls"] = event.get("tool_calls", [])
                        stream_meta["proposed_actions"] = event.get(
                            "proposed_actions", []
                        )
                    elif etype == "error":
                        stream_error = event.get("error", "stream error")
            except Exception as exc:  # noqa: BLE001
                stream_error = f"{type(exc).__name__}: {exc}"

            # If the stream produced nothing useful, fall back to the
            # non-streaming endpoint so the chat still works even when the
            # SSE path is blocked or misbehaving.
            if (
                not accumulated
                and not stream_meta["tool_calls"]
                and not stream_meta["proposed_actions"]
            ):
                try:
                    r = client.request(
                        "POST",
                        "/chat",
                        json={
                            "messages": st.session_state["assistant_messages"],
                            "model": st.session_state.get("assistant_selected_model"),
                        },
                        timeout=PLAN_TIMEOUT,
                    )
                    if r.ok and isinstance(r.body, dict):
                        accumulated = str(r.body.get("response", ""))
                        stream_meta["tool_calls"] = r.body.get("tool_calls", []) or []
                        for tc in stream_meta["tool_calls"]:
                            st.session_state["assistant_messages"].append(
                                {
                                    "role": "tool",
                                    "name": tc.get("name"),
                                    "args": tc.get("args"),
                                    "result": tc.get("result"),
                                }
                            )
                        stream_meta["proposed_actions"] = [
                            {
                                "kind": tc["result"].get("kind", tc["name"]),
                                "params": tc["result"].get(
                                    "params", tc.get("args", {})
                                ),
                            }
                            for tc in stream_meta["tool_calls"]
                            if isinstance(tc.get("result"), dict)
                            and tc["result"].get("status") == "proposed"
                        ]
                        stream_thinking = [str(r.body.get("thinking", ""))]
                        stream_error = None
                        if accumulated:
                            seg = _current_text_segment()
                            seg[1] = accumulated
                            seg[0].markdown(accumulated)
                    else:
                        stream_error = f"Fallback failed: {r.error_message()}"
                except Exception as exc:  # noqa: BLE001
                    stream_error = f"Fallback failed: {type(exc).__name__}: {exc}"

            thinking.empty()
            if stream_error:
                st.error(f"Assistant error: {stream_error}")

            if (
                accumulated
                or stream_meta["tool_calls"]
                or stream_meta["proposed_actions"]
            ):
                # Analysis images are shown inline with their tool calls above,
                # so we only keep plain photo attachments on the assistant message.
                photo_images = [
                    {"attachment_url": img.get("attachment_url")}
                    for img in stream_meta.get("images", [])
                    if isinstance(img, dict) and img.get("attachment_url")
                ]
                st.session_state["assistant_messages"].append(
                    {
                        "role": "assistant",
                        "content": accumulated,
                        "thinking": "".join(stream_thinking),
                        "tool_calls": stream_meta["tool_calls"],
                        "proposed_actions": stream_meta["proposed_actions"],
                        "images": photo_images,
                    }
                )
        _persist_session()
        st.rerun()


def _fetch_latest_image() -> dict[str, Any] | None:
    """Return the most recent FarmBot image via the existing /images endpoint."""
    result = client.request(
        "GET", "/images", params={"limit": "1", "refresh": "true"}, timeout=10.0
    )
    if result.ok and isinstance(result.body, dict):
        images = result.body.get("images", [])
        if images:
            return images[0]
    return None


def _image_is_newer(image: dict[str, Any], previous: dict[str, Any]) -> bool:
    """Return True if image is strictly newer/different than previous."""
    if image.get("id") is not None and previous.get("id") is not None:
        return image["id"] != previous["id"]
    new_ts = image.get("created_at", "")
    old_ts = previous.get("created_at", "")
    if new_ts and old_ts:
        return new_ts > old_ts
    return True


def _wait_for_new_image(
    previous: dict[str, Any] | None,
    max_attempts: int = 15,
    delay: float = 2.0,
) -> dict[str, Any] | None:
    """Poll /images until an image newer than ``previous`` appears."""
    for _ in range(max_attempts):
        image = _fetch_latest_image()
        if image and (previous is None or _image_is_newer(image, previous)):
            return image
        time.sleep(delay)
    return None


def _capture_photo_image() -> dict[str, Any] | None:
    """Fetch the latest image after take_photo, polling for a fresh upload."""
    baseline = _fetch_latest_image()
    for _ in range(8):
        image = _fetch_latest_image()
        if image:
            if baseline is None or _image_is_newer(image, baseline):
                return image
            # If baseline is already the newest, wait briefly in case the
            # just-triggered photo is still uploading.
        time.sleep(1.5)
    return baseline


def _execute_proposed_actions(
    actions: list[dict[str, Any]],
    message: dict[str, Any] | None = None,
    *,
    wait: bool = True,
) -> list[dict[str, Any]]:
    """Dispatch proposed actions and return per-action results.

    By default this waits for each action to finish so the UI can give
    immediate feedback. For fire-and-forget dispatch, pass ``wait=False``.
    """
    will_capture = message is not None and any(
        action.get("kind") == "take_photo" for action in actions
    )
    previous_image = _fetch_latest_image() if will_capture else None

    results: list[dict[str, Any]] = []
    for action in actions:
        r = client.request(
            "POST",
            "/actions",
            json={"kind": action["kind"], "params": action.get("params", {})},
            params={"wait": "true" if wait else "false"},
        )
        results.append(
            {
                "kind": action["kind"],
                "ok": r.ok,
                "status": "ok" if r.ok else "error",
                "detail": r.body if isinstance(r.body, str) else r.body.get("detail") if isinstance(r.body, dict) else str(r.body),
            }
        )

    if will_capture:
        new_image = _wait_for_new_image(previous_image)
        if new_image:
            message.setdefault("images", []).append(new_image)

    return results


def _format_execution_results(results: list[dict[str, Any]]) -> str:
    """Turn per-action results into a short, human-readable summary."""
    if not results:
        return "✅ Approved (no actions)."
    lines: list[str] = []
    for res in results:
        summary = _action_summary({"kind": res["kind"], "params": {}})
        if res.get("ok"):
            lines.append(f"✅ {summary}")
        else:
            lines.append(f"❌ {summary} — {res.get('detail', 'unknown error')}")
    return "\n".join(lines)


# ── session persistence ───────────────────────────────────────────────────────


def _has_session_state() -> bool:
    """Return True if any chat/plan state has been initialised."""
    return (
        "assistant_messages" in st.session_state
        or "assistant_plan_response" in st.session_state
        or "executed_plans" in st.session_state
    )


def _is_session_empty() -> bool:
    """Return True if the current session has nothing worth saving."""
    messages = st.session_state.get("assistant_messages") or []
    plan_response = st.session_state.get("assistant_plan_response")
    executed = st.session_state.get("executed_plans") or []
    return not messages and not plan_response and not executed


def _restore_session() -> None:
    """Load the latest or URL-specified session on first app load."""
    if _has_session_state():
        return

    session_id = st.query_params.get("session")
    if isinstance(session_id, list):
        session_id = session_id[0] if session_id else None

    snapshot: dict[str, Any] | None = None
    if session_id:
        snapshot = history.load_session(session_id)
    if snapshot is None:
        sessions = history.list_sessions(limit=1)
        if sessions:
            snapshot = history.load_session(sessions[0]["session_id"])

    if snapshot is None:
        snapshot = history.empty_snapshot()

    st.session_state["assistant_session_id"] = snapshot["session_id"]
    st.session_state["assistant_session_label"] = snapshot.get("label")
    st.session_state["assistant_messages"] = snapshot.get("assistant_messages", [])
    st.session_state["assistant_plan_request"] = snapshot.get(
        "assistant_plan_request", ""
    )
    st.session_state["assistant_plan_response"] = snapshot.get(
        "assistant_plan_response"
    )
    st.session_state["assistant_plan_status"] = snapshot.get(
        "assistant_plan_status"
    )
    st.session_state["assistant_selected_model"] = snapshot.get(
        "assistant_selected_model"
    )
    st.session_state["executed_plans"] = snapshot.get("executed_plans", [])


def _persist_session() -> None:
    """Save the current chat/plan state to disk."""
    if _is_session_empty():
        return
    snapshot = history.empty_snapshot(
        session_id=st.session_state.get("assistant_session_id")
    )
    snapshot["label"] = st.session_state.get("assistant_session_label")
    snapshot["created_at"] = st.session_state.get(
        "assistant_session_created_at", snapshot["created_at"]
    )
    snapshot["assistant_messages"] = st.session_state.get("assistant_messages", [])
    snapshot["assistant_plan_request"] = st.session_state.get(
        "assistant_plan_request", ""
    )
    snapshot["assistant_plan_response"] = st.session_state.get(
        "assistant_plan_response"
    )
    snapshot["assistant_plan_status"] = st.session_state.get("assistant_plan_status")
    snapshot["assistant_selected_model"] = st.session_state.get(
        "assistant_selected_model"
    )
    snapshot["executed_plans"] = st.session_state.get("executed_plans", [])
    history.save_session(snapshot)


def _render_session_controls() -> None:
    """Render session management widgets inside the Assistant tab."""
    with st.expander("🗂️ Session", expanded=False):
        current_label = st.text_input(
            "Session label",
            value=st.session_state.get("assistant_session_label") or "",
            key="assistant_session_label_input",
            placeholder="e.g. watering experiment",
        )
        st.session_state["assistant_session_label"] = (
            current_label.strip() or None
        )

        new_col, save_col = st.columns([1, 1])
        if new_col.button("New session", use_container_width=True):
            _persist_session()
            new_id = history.new_session_id()
            st.session_state["assistant_session_id"] = new_id
            st.session_state["assistant_session_label"] = None
            st.session_state["assistant_messages"] = []
            st.session_state["assistant_plan_request"] = ""
            st.session_state["assistant_plan_response"] = None
            st.session_state["assistant_plan_status"] = None
            st.session_state["executed_plans"] = []
            st.query_params.pop("session", None)
            st.rerun()
        if save_col.button("Save now", use_container_width=True):
            _persist_session()
            st.toast("Session saved", icon="💾")

        sessions = history.list_sessions(limit=20)
        if sessions:
            st.divider()
            st.markdown("**Previous sessions**")
        for sess in sessions:
            if sess["session_id"] == st.session_state.get("assistant_session_id"):
                continue
            label = sess["label"] or sess["session_id"]
            preview = sess["preview"]
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.caption(f"{label}" + (f" · {preview}" if preview else ""))
            if c2.button("Load", key=f"load_sess_{sess['session_id']}", use_container_width=True):
                snapshot = history.load_session(sess["session_id"])
                if snapshot is None:
                    st.error("Session not found")
                    continue
                st.session_state["assistant_session_id"] = snapshot["session_id"]
                st.session_state["assistant_session_label"] = snapshot.get("label")
                st.session_state["assistant_messages"] = snapshot.get(
                    "assistant_messages", []
                )
                st.session_state["assistant_plan_request"] = snapshot.get(
                    "assistant_plan_request", ""
                )
                st.session_state["assistant_plan_response"] = snapshot.get(
                    "assistant_plan_response"
                )
                st.session_state["assistant_plan_status"] = snapshot.get(
                    "assistant_plan_status"
                )
                st.session_state["assistant_selected_model"] = snapshot.get(
                    "assistant_selected_model"
                )
                st.session_state["executed_plans"] = snapshot.get(
                    "executed_plans", []
                )
                st.query_params["session"] = snapshot["session_id"]
                st.rerun()
            if c3.button("🗑", key=f"del_sess_{sess['session_id']}", use_container_width=True):
                history.delete_session(sess["session_id"])
                st.rerun()


def _render_plan() -> None:
    st.caption(
        "Describe a task. The LLM builds a step-by-step plan; review it before running."
    )

    selected_model = _render_model_picker()
    st.session_state["assistant_selected_model"] = selected_model

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
        placeholder="e.g. water bed for 60 seconds, then home",
        height=80,
        label_visibility="collapsed",
    )
    st.session_state["assistant_plan_request"] = request

    plan_col, _ = st.columns([1, 3])
    preview_clicked = plan_col.button(
        "Preview plan",
        type="primary",
        use_container_width=True,
        disabled=not request.strip(),
    )

    if preview_clicked and request.strip():
        with st.spinner("Asking the planner…"):
            r = client.request(
                "POST",
                "/plan",
                json={
                    "request": request,
                    "debug": True,
                    "model": st.session_state.get("assistant_selected_model"),
                },
                timeout=PLAN_TIMEOUT,
            )
        st.session_state["assistant_plan_response"] = (
            r.body if r.ok else {"error": r.body}
        )
        st.session_state["assistant_plan_status"] = r.code
        _persist_session()

    response = st.session_state.get("assistant_plan_response")
    status = st.session_state.get("assistant_plan_status")

    if not response:
        st.info("No plan yet. Type a task above and click **Preview plan**.")
        return

    with st.expander("Debug · raw response", expanded=False):
        st.json(response)

    if status and status >= 400:
        err_body = response.get("error", response)
        st.error(
            f"Planner error (HTTP {status}): "
            f"{ApiResult(ok=False, code=status, body=err_body).error_message()}"
        )
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
        _persist_session()
        st.rerun()

    if run_col.button("Run plan", type="primary", use_container_width=True):
        queued = 0
        failed = 0
        action_results: list[dict[str, Any]] = []
        for action in actions:
            r = client.request(
                "POST",
                "/actions",
                json={"kind": action["kind"], "params": action.get("params", {})},
            )
            action_results.append(
                {
                    "kind": action["kind"],
                    "ok": r.ok,
                    "detail": r.error_message() if not r.ok else None,
                }
            )
            if r.ok:
                queued += 1
                st.toast(f"Queued {action['kind']}", icon="➡️")
            else:
                failed += 1
                st.error(f"Failed to queue {action['kind']}: {r.error_message()}")
        if failed == 0:
            st.success(f"Plan queued · {queued} action(s)")
        else:
            st.warning(f"Plan partially queued · {queued} ok, {failed} failed")
        executed = st.session_state.get("executed_plans") or []
        executed.append(
            {
                "request": response.get("request", ""),
                "actions": actions,
                "results": action_results,
                "queued_at": datetime.now().isoformat(),
                "status": "ok" if failed == 0 else ("partial" if queued > 0 else "failed"),
            }
        )
        st.session_state["executed_plans"] = executed
        st.session_state["assistant_plan_response"] = None
        st.session_state["assistant_plan_status"] = None
        _persist_session()
        st.rerun()


def _render_history() -> None:
    st.markdown(
        '<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>',
        unsafe_allow_html=True,
    )
    st.markdown("# History")

    sessions = history.list_sessions(limit=50)
    if not sessions:
        st.info("No saved sessions yet. Chat and plans are saved automatically.")
        return

    st.markdown("## Chat sessions")
    for sess in sessions:
        label = sess["label"] or sess["session_id"]
        updated = sess["updated_at"][:19].replace("T", " ") if sess["updated_at"] else ""
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**{label}**")
            st.caption(
                f"Updated {updated}" + (f" · {sess['preview']}" if sess["preview"] else "")
            )
        if c2.button("Load", key=f"hist_load_{sess['session_id']}", use_container_width=True):
            snapshot = history.load_session(sess["session_id"])
            if snapshot is None:
                st.error("Session not found")
            else:
                st.session_state["assistant_session_id"] = snapshot["session_id"]
                st.session_state["assistant_session_label"] = snapshot.get("label")
                st.session_state["assistant_messages"] = snapshot.get(
                    "assistant_messages", []
                )
                st.session_state["assistant_plan_request"] = snapshot.get(
                    "assistant_plan_request", ""
                )
                st.session_state["assistant_plan_response"] = snapshot.get(
                    "assistant_plan_response"
                )
                st.session_state["assistant_plan_status"] = snapshot.get(
                    "assistant_plan_status"
                )
                st.session_state["assistant_selected_model"] = snapshot.get(
                    "assistant_selected_model"
                )
                st.session_state["executed_plans"] = snapshot.get("executed_plans", [])
                st.query_params["session"] = snapshot["session_id"]
                st.rerun()

    executed = st.session_state.get("executed_plans") or []
    if executed:
        st.markdown("## Executed plans")
        for idx, plan in enumerate(reversed(executed), start=1):
            with st.container(border=True):
                queued_at = plan.get("queued_at", "")
                ts = queued_at[:19].replace("T", " ") if queued_at else ""
                status = plan.get("status", "unknown")
                status_emoji = {"ok": "✅", "partial": "⚠️", "failed": "❌"}.get(
                    status, "❓"
                )
                st.markdown(
                    f"{status_emoji} **Plan {idx}** · {plan.get('request', '')}"
                )
                st.caption(f"{ts} · {len(plan.get('actions', []))} action(s) · {status}")
                with st.expander("Actions"):
                    for action in plan.get("actions", []):
                        st.markdown(f"• {_action_summary(action)}")
                results = plan.get("results", [])
                if results:
                    with st.expander("Results"):
                        for res in results:
                            icon = "✅" if res.get("ok") else "❌"
                            detail = res.get("detail")
                            line = f"{icon} {_action_summary({'kind': res['kind'], 'params': {}})}"
                            if detail:
                                line += f" — {detail}"
                            st.markdown(line)


def _render_diagnostics() -> None:
    st.markdown(
        '<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>',
        unsafe_allow_html=True,
    )
    st.markdown("# Diagnostics")

    if st.button("Load /status"):
        d = client.request("GET", "/status")
        if d.ok and isinstance(d.body, dict):
            st.session_state["diag"] = d.body.get("state", {})
        else:
            st.error(f"Read failed: {d.error_message()}")

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
            [
                {"pin": pn, "value": pd.get("value"), "mode": pd.get("mode")}
                for pn, pd in pins.items()
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_settings() -> None:
    st.markdown(
        '<div class="eyebrow">TWFarmBot · UAS Technikum Wien</div>',
        unsafe_allow_html=True,
    )
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

    st.json(
        {
            "farmbot": st.session_state.get("farmbot_status", "?"),
            "api": st.session_state["api_url"],
            "actions": st.session_state.get("actions", []),
        }
    )

    with st.expander("Raw action"):
        with st.form("raw"):
            kind = st.text_input("Kind", "move")
            raw = st.text_area("Params (JSON)", '{"message":"hello"}', height=100)
            if st.form_submit_button("Fire"):
                try:
                    p = json.loads(raw)
                except json.JSONDecodeError as e:
                    st.error(f"Bad JSON: {e}")
                else:
                    r = client.request(
                        "POST", "/actions", json={"kind": kind, "params": p}
                    )
                    st.json(r.body)


# ── dispatch ──────────────────────────────────────────────────────────────────

renderers = {
    "Overview": _render_overview,
    "Garden": _render_garden,
    "Motion": _render_motion,
    "Camera": _render_camera,
    "I/O": _render_io,
    "Assistant": _render_assistant,
    "History": _render_history,
    "Diagnostics": _render_diagnostics,
    "Settings": _render_settings,
}

# Restore the latest or URL-specified chat/plan session on first load.
_restore_session()
renderers[tab]()
