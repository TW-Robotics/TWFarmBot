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


def test_append_result_images_includes_capture_artifact() -> None:
    text = append_result_images(
        "NIR image captured successfully.",
        [
            {
                "name": "capture",
                "result": {
                    "status": "ok",
                    "kind": "capture",
                    "params": {
                        "band": "nir",
                        "artifact_id": "f4bdddfcb97d41e2b160d76ca28159a4",
                    },
                },
            }
        ],
    )

    assert (
        "![NIR capture](/captures/f4bdddfcb97d41e2b160d76ca28159a4/nir)" in text
    )


def test_append_result_images_includes_capture_ndre() -> None:
    text = append_result_images(
        "NDRE done.",
        [
            {
                "name": "capture",
                "result": {
                    "status": "ok",
                    "params": {
                        "artifact_id": "aaa",
                        "band": "nir",
                    },
                },
            },
            {
                "name": "capture",
                "result": {
                    "status": "ok",
                    "params": {
                        "artifact_id": "bbb",
                        "band": "rededge",
                    },
                },
            },
            {
                "name": "capture_ndre",
                "result": {
                    "status": "ok",
                    "kind": "capture_ndre",
                    "params": {
                        "summary": "NDRE mean 0.22",
                        "ndre": {"mean": 0.22},
                        "ndre_preview": "/captures/aaa/ndre",
                        "nir": {"attachment_url": "/captures/aaa/nir"},
                        "rededge": {"attachment_url": "/captures/bbb/rededge"},
                    },
                },
            },
        ],
    )

    assert "![NDRE map](/captures/aaa/ndre)" in text
    assert "NIR capture" not in text
    assert "Red-edge capture" not in text
    assert "/captures/aaa/nir" not in text
