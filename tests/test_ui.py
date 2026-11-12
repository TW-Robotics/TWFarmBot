"""Tests for the UI's API client.

The dashboard itself is hard to unit-test (renders DOM), but the
``ApiClient`` it uses to talk to the API is testable in isolation.
"""

from __future__ import annotations

from typing import Any

import pytest


def test_api_client_returns_ok_on_2xx(monkeypatch: "pytest.MonkeyPatch") -> None:
    import httpx

    def fake_request(self, method, url, **kwargs):
        return httpx.Response(
            200,
            json={"status": "ok", "actions": ["water"], "farmbot": "connected"},
        )

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    from twfarmbot_ui import ApiClient

    c = ApiClient("http://api")
    r = c.request("GET", "/health")
    assert r.ok is True
    assert r.code == 200
    assert r.body["status"] == "ok"
    assert "water" in r.body["actions"]


def test_api_client_returns_not_ok_on_connection_failure(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    import httpx

    def fake_request(self, method, url, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    from twfarmbot_ui import ApiClient

    c = ApiClient("http://api")
    r = c.request("GET", "/health")
    assert r.ok is False
    assert r.code == 0
    assert "ConnectError" in str(r.body.get("error", ""))


def test_api_client_passes_query_params(monkeypatch: "pytest.MonkeyPatch") -> None:
    import httpx

    seen: list[dict[str, Any]] = []

    def fake_request(self, method, url, **kwargs):
        seen.append({"method": method, "url": str(url), "kwargs": kwargs})
        return httpx.Response(200, json={"pin": 13, "mode": "analog", "value": 1})

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    from twfarmbot_ui import ApiClient

    c = ApiClient("http://api")
    r = c.request("GET", "/pin/13", params={"mode": "analog"})
    assert r.ok is True
    assert seen[0]["url"].endswith("/pin/13")
    assert seen[0]["kwargs"]["params"] == {"mode": "analog"}


def test_api_client_strips_trailing_slash() -> None:
    from twfarmbot_ui import ApiClient

    c = ApiClient("http://api/")
    assert c.base_url == "http://api"


def test_api_result_error_message_prefers_detail() -> None:
    from twfarmbot_ui.client import ApiResult

    r = ApiResult(ok=False, code=500, body={"detail": "FarmBot not connected"})
    assert r.error_message() == "FarmBot not connected"


def test_api_result_error_message_falls_back_to_error_key() -> None:
    from twfarmbot_ui.client import ApiResult

    r = ApiResult(ok=False, code=0, body={"error": "ConnectError: refused"})
    assert r.error_message() == "ConnectError: refused"


def test_api_result_error_message_falls_back_to_string_body() -> None:
    from twfarmbot_ui.client import ApiResult

    r = ApiResult(ok=False, code=500, body="raw error text")
    assert r.error_message() == "raw error text"


def test_api_result_error_message_stringifies_unknown_body() -> None:
    from twfarmbot_ui.client import ApiResult

    r = ApiResult(ok=False, code=500, body={"nested": ["info"]})
    assert "nested" in r.error_message()


def test_vision_processor_calls_local_server(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from twfarmbot_ml_utils import huggingface as module

    seen: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            return {
                "choices": [
                    {"message": {"content": '{"result_image_base64": "%s"}' % b64}}
                ]
            }

    def fake_post(url: str, *, json: dict[str, Any], timeout: float, **kwargs: Any):
        seen["url"] = url
        seen["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)

    processor = module.VisionProcessor("http://127.0.0.1:8080")
    result = processor.process("https://example.test/photo.jpg", "green leaves")

    assert result.exists()
    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert seen["payload"]["messages"][0]["content"][0]["text"] == "green leaves"
    assert (
        seen["payload"]["messages"][0]["content"][1]["image_url"]["url"]
        == "https://example.test/photo.jpg"
    )


def test_tablet_motion_tab_has_unlock_and_find_home() -> None:
    """The tablet Motion layout must offer Unlock + Find home like desktop.

    There is no JS test harness in this repo, so this guards the TSX source:
    both buttons, wired to the ``unlock`` / ``find_home`` actions, must render
    inside the ``isTablet`` branch of ``MotionPage`` (not only on desktop).
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent
        / "apps" / "ui" / "web" / "pages" / "MotionPage.tsx"
    ).read_text(encoding="utf-8")
    tablet = source.split("if (isTablet)", 1)[1].split("return (", 1)[1]
    tablet = tablet.split("\n  return (", 1)[0]
    assert 'label="Unlock"' in tablet
    assert 'label="Find home"' in tablet
    assert 'run("unlock"' in tablet
    assert 'run("find_home", { axis: "all" }' in tablet


def test_tablet_assistant_is_transcribe_only_no_live_voice() -> None:
    """Tablet Assistant keeps record-and-send but must not offer Live voice.

    ``VoiceDock`` is only rendered in the tablet branch, so guard its source
    plus that branch: no Live voice control, and the record button is labeled
    "Speak" (not the jargon "Transcribe") for a broad audience.
    Desktop keeps its separate Voice mode button (out of scope here).
    """
    from pathlib import Path

    web = Path(__file__).resolve().parent.parent / "apps" / "ui" / "web"
    dock = (web / "components" / "VoiceDock.tsx").read_text(encoding="utf-8")
    assert "Live voice" not in dock
    assert "onLive" not in dock
    assert 'label={recording ? "Done" : "Speak"}' in dock
    assert '"Transcribe"' not in dock
    assert '"Transcribing…"' not in dock
    # Transcribed text waits in a confirmation bubble: "You said:" + Send /
    # Discard, instead of being sent to the assistant automatically.
    assert '"You said:"' in dock
    assert 'label="Send"' in dock
    assert 'label="Discard"' in dock
    assert "pendingText" in dock
    # An Edit button opens an inline text field so the text can be fixed
    # by typing before sending.
    assert '"Edit"' in dock
    assert "TextArea" in dock
    assert "onEditPending" in dock

    page = (web / "pages" / "AssistantPage.tsx").read_text(encoding="utf-8")
    tablet = page.split("{isTablet ? (", 1)[1].split(") : (", 1)[0]
    assert "VoiceDock" in tablet
    assert "onLive" not in tablet
    assert "onSpeak" in tablet
    assert "setPendingSpeech(text)" in tablet
    assert "(text) => void send(text)" not in tablet
    assert "onSendPending" in tablet
    assert "onDiscardPending" in tablet
    assert "onEditPending" in tablet
    # Empty-state hint must not tell tablet users to tap Live voice or
    # Transcribe, and the desktop Voice mode control must survive this change.
    assert "Tap Live voice" not in page
    assert "Tap Transcribe" not in page
    assert "Tap Speak and say what you need. You can check it before it sends." in page
    assert '"Voice mode"' in page


def test_tablet_hides_overview_and_lands_on_assistant() -> None:
    """Tablets skip the Overview tab and land straight on the Assistant."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent
        / "apps" / "ui" / "web" / "App.tsx"
    ).read_text(encoding="utf-8")
    assert '{ id: "overview", label: "Overview", icon: HomeIcon, tabletHidden: true }' in source
    assert 'if (isTablet && isHiddenOnTablet(next)) return "assistant";' in source
    assert 'if (isTablet && isHiddenOnTablet(tab)) selectTab("assistant");' in source
    assert "item.desktopOnly || item.tabletHidden" in source


def test_parse_segmentation_labels() -> None:
    from twfarmbot_ml_utils import parse_segmentation_labels

    labels = ["weed (23.4%), soil (45.1%)", "plant (31.5%)"]
    scores = parse_segmentation_labels(labels)

    assert scores == {"weed": 0.234, "soil": 0.451, "plant": 0.315}
