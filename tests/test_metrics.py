import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_demo import (
    CONDITIONS,
    HEALTH_THRESHOLD,
    PERSISTENCE_FRAMES,
    _ratio_similarity,
    add_relative_health,
    add_salt_pepper,
    condition_params,
    match_to_reference,
    persistence_policy,
    raw_metrics,
    simulate_policy,
    synthetic_road,
    variants,
)


def _rows():
    img = synthetic_road()
    rows = []
    for level, (name, frame) in enumerate(variants(img)):
        rows.append({"level": level, "condition": name, **raw_metrics(frame)})
    add_relative_health(rows)
    return {r["condition"]: r for r in rows}, rows


def test_condition_names_encode_parameters():
    for cond in CONDITIONS[1:]:
        for value in cond.params.values():
            assert str(value).rstrip("0") in cond.name


def test_condition_params_are_recorded():
    img = synthetic_road()
    params = {c.name: condition_params(img, c) for c in CONDITIONS}
    assert params["blur_gauss_k21"]["sigma_px"] == pytest.approx(3.5)
    assert {"center_px", "radius_px", "gain"} <= params["glare_gain1.05"].keys()
    assert params["saltpepper_p0.025"]["seed"] == 19


def test_blur_reduces_laplacian_monotonically():
    by_name, _ = _rows()
    laps = [by_name[n]["laplacian_variance"]
            for n in ["clean", "blur_gauss_k5", "blur_gauss_k11", "blur_gauss_k21"]]
    assert laps == sorted(laps, reverse=True)


def test_glare_increases_highlight_saturation_monotonically():
    by_name, _ = _rows()
    sats = [by_name[n]["saturation_ratio"]
            for n in ["clean", "glare_gain0.35", "glare_gain0.70", "glare_gain1.05"]]
    assert sats == sorted(sats)
    assert sats[-1] > sats[0]


def test_strong_blur_reduces_demo_health():
    by_name, _ = _rows()
    assert by_name["blur_gauss_k21"]["health_score"] < HEALTH_THRESHOLD
    assert by_name["clean"]["health_score"] == pytest.approx(1.0)


def test_impulse_noise_raises_sharpness_but_lowers_health():
    by_name, _ = _rows()
    noisy, clean = by_name["saltpepper_p0.025"], by_name["clean"]
    assert noisy["laplacian_variance"] > clean["laplacian_variance"]
    assert noisy["health_score"] < clean["health_score"]


def test_ratio_similarity_is_symmetric_and_bounded():
    assert _ratio_similarity(50, 100) == pytest.approx(0.5)
    assert _ratio_similarity(200, 100) == pytest.approx(0.5)
    assert _ratio_similarity(100, 100) == 1.0
    assert 0.0 <= _ratio_similarity(0.0, 100) <= 1.0


def test_salt_pepper_is_deterministic():
    img = synthetic_road()
    assert np.array_equal(add_salt_pepper(img, 0.01), add_salt_pepper(img, 0.01))


def test_persistence_policy_ignores_short_transients():
    low, high = HEALTH_THRESHOLD - 0.1, HEALTH_THRESHOLD + 0.1
    scores = [high, low, high, low, low, high] + [low] * PERSISTENCE_FRAMES
    states = persistence_policy(scores)
    assert states[:6] == ["OK"] * 6
    assert states[-1] == "DEGRADED"
    assert states.index("DEGRADED") == 6 + PERSISTENCE_FRAMES - 1


def test_policy_simulation_suppresses_transient_alarms():
    _, rows = _rows()
    _, stats = simulate_policy(rows)
    assert stats["policy_alarms_on_transients"] == 0
    assert stats["policy_alarm_latency_frames"] == PERSISTENCE_FRAMES - 1


def test_match_to_reference_detects_lost_and_new_boxes():
    ref = [{"cls": "bus", "conf": 0.9, "xyxy": [0, 0, 100, 100]},
           {"cls": "person", "conf": 0.8, "xyxy": [200, 0, 250, 100]}]
    dets = [{"cls": "airplane", "conf": 0.5, "xyxy": [0, 0, 100, 100]},
            {"cls": "person", "conf": 0.7, "xyxy": [202, 0, 252, 100]}]
    out = match_to_reference(ref, dets)
    assert out["pseudo_label_agreement"] == pytest.approx(0.5)
    assert out["lost_vs_clean"] == "bus:0.90"
    assert out["new_vs_clean"] == "airplane:0.50"


def test_synthetic_frame_is_valid_bgr():
    img = synthetic_road()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img.ndim == 3 and img.shape[2] == 3
    assert gray.shape == img.shape[:2]
