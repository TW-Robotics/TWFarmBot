from planning_service.tool_results import append_result_images


def test_append_result_images_prefers_segmentation_artifacts() -> None:
    text = append_result_images(
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
                    "image_urls": [
                        "data:image/png;base64,overlay",
                        "data:image/png;base64,mask",
                    ]
                },
            },
        ],
    )

    assert "source-1.jpg" not in text
    assert "source-2.jpg" not in text
    assert "![Segmentation overlay](data:image/png;base64,overlay)" in text
    assert "![Segmentation mask](data:image/png;base64,mask)" in text


def test_append_result_images_shows_only_latest_plain_photo() -> None:
    text = append_result_images(
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
