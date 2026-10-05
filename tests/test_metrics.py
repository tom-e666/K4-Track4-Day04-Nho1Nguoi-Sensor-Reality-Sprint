import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_demo import add_relative_health, raw_metrics, synthetic_road, variants


def _rows():
    img = synthetic_road()
    rows = []
    for level, (name, frame) in enumerate(variants(img)):
        rows.append({"level": level, "condition": name, **raw_metrics(frame)})
    add_relative_health(rows)
    return rows


def test_blur_reduces_laplacian_variance():
    rows = _rows()
    assert rows[3]["laplacian_variance"] < rows[0]["laplacian_variance"]


def test_glare_increases_highlight_saturation():
    rows = _rows()
    assert rows[4]["saturation_ratio"] > rows[0]["saturation_ratio"]


def test_strong_blur_reduces_demo_health():
    rows = _rows()
    assert rows[3]["health_score"] < rows[0]["health_score"]


def test_synthetic_frame_is_valid_bgr():
    img = synthetic_road()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img.ndim == 3 and img.shape[2] == 3
    assert gray.shape == img.shape[:2]
