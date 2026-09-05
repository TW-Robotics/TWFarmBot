import base64

from planning_service.tool_results import (
    _MAX_PROVIDER_IMAGE_BYTES,
    _ndre_preview_from_sample,
    _path_to_data_uri,
    _sse_images,
    append_result_images,
    compact_tool_result,
)


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

    assert "![NIR capture](/captures/f4bdddfcb97d41e2b160d76ca28159a4/nir)" in text


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


def test_append_result_images_includes_scan_ndre_previews() -> None:
    text = append_result_images(
        "Sweep done.",
        [
            {
                "name": "scan_ndre",
                "result": {
                    "status": "ok",
                    "kind": "scan_ndre",
                    "params": {
                        "samples": [
                            {
                                "y": 0,
                                "ndre_preview": "/captures/a/ndre",
                            },
                            {
                                "y": 50,
                                "ndre_preview": "/captures/b/ndre",
                            },
                        ]
                    },
                },
            }
        ],
    )
    assert "![NDRE 1 (0 mm)](/captures/a/ndre)" in text
    assert "![NDRE 2 (50 mm)](/captures/b/ndre)" in text
    compacted = compact_tool_result(
        {
            "params": {
                "samples": [{"ndre_preview": "/captures/a/ndre"}],
            }
        }
    )
    assert compacted["params"]["samples"][0]["ndre_preview"] == "/captures/a/ndre"


def test_ndre_preview_from_sample_falls_back_to_nir_artifact() -> None:
    assert (
        _ndre_preview_from_sample({"nir": {"artifact_id": "abc123", "band": "nir"}})
        == "/captures/abc123/ndre"
    )


def test_sse_images_for_scan_ndre() -> None:
    images = _sse_images(
        "scan_ndre",
        {
            "status": "ok",
            "params": {
                "samples": [
                    {"y": 0, "nir": {"artifact_id": "aaa", "band": "nir"}},
                    {"y": 50, "ndre_preview": "/captures/bbb/ndre"},
                ]
            },
        },
    )
    assert images == [
        {"label": "NDRE 1 (0 mm)", "url": "/captures/aaa/ndre"},
        {"label": "NDRE 2 (50 mm)", "url": "/captures/bbb/ndre"},
    ]


def test_path_to_data_uri_downscales_large_jpeg(tmp_path) -> None:
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (2400, 2400, 3), dtype=np.uint8)
    path = tmp_path / "big.jpg"
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert path.stat().st_size > _MAX_PROVIDER_IMAGE_BYTES
    uri = _path_to_data_uri(path)
    assert uri and uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert len(raw) <= _MAX_PROVIDER_IMAGE_BYTES
