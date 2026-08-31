from planning_service.responses_client import _append_result_images
from twfarmbot_api_server.app import _CHAT_MEDIA, _materialize_chat_images


def test_append_result_images_prefers_segmentation_artifacts() -> None:
    text = _append_result_images(
        "Done.",
        [
            {
                "name": "get_images",
                "result": {
                    "images": [
                        {"attachment_url": "https://example.test/source-1.jpg"},
                        {"attachment_url": "https://example.test/source-2.jpg"},
                    ]
                },
            },
            {
                "name": "segment_image",
                "result": {
                    "image_urls": ["data:image/png;base64,overlay", "data:image/png;base64,mask"]
                },
            },
        ],
    )

    assert "source-1.jpg" not in text
    assert "source-2.jpg" not in text
    assert "![Segmentation overlay](data:image/png;base64,overlay)" in text
    assert "![Segmentation mask](data:image/png;base64,mask)" in text


def test_append_result_images_shows_only_latest_plain_photo() -> None:
    text = _append_result_images(
        "Photo taken.",
        [
            {
                "name": "get_images",
                "result": {
                    "images": [
                        {"attachment_url": "https://example.test/latest.jpg"},
                        {"attachment_url": "https://example.test/older.jpg"},
                    ]
                },
            }
        ],
    )

    assert "![FarmBot photo](https://example.test/latest.jpg)" in text
    assert "older.jpg" not in text


def test_materialize_chat_images_replaces_embedded_markdown_data_uri() -> None:
    _CHAT_MEDIA.clear()

    result = _materialize_chat_images(
        {"type": "delta", "content": "![Mask](data:image/png;base64,aW1hZ2U=)"},
        "http://api.test",
    )

    assert result["content"].startswith("![Mask](http://api.test/chat/media/")
    assert "base64" not in result["content"]
    assert list(_CHAT_MEDIA.values()) == [(b"image", "image/png")]
