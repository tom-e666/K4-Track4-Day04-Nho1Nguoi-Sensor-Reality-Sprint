from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results"

# Frozen before the multi-level run (commit 40af73b). Deliberately NOT re-tuned
# after adding the new glare/noise levels, so those levels test it out of sample.
HEALTH_THRESHOLD = 0.65
PERSISTENCE_FRAMES = 3
HEALTH_WEIGHTS = {
    "sharpness_similarity": 0.40,
    "edge_similarity": 0.15,
    "brightness_similarity": 0.10,
    "contrast_similarity": 0.10,
    "clipping_quality": 0.25,
}

SALT_PEPPER_SEED = 19
GLARE_CENTER_FRAC = (0.73, 0.30)
GLARE_RADIUS_FRAC = 0.30
PSEUDO_LABEL_IOU = 0.5


@dataclass
class Condition:
    name: str
    family: str
    params: dict = field(default_factory=dict)


CONDITIONS = [
    Condition("clean", "baseline"),
    Condition("blur_gauss_k5", "blur", {"kernel": 5}),
    Condition("blur_gauss_k11", "blur", {"kernel": 11}),
    Condition("blur_gauss_k21", "blur", {"kernel": 21}),
    Condition("glare_gain0.35", "glare", {"gain": 0.35}),
    Condition("glare_gain0.70", "glare", {"gain": 0.70}),
    Condition("glare_gain1.05", "glare", {"gain": 1.05}),
    Condition("saltpepper_p0.005", "impulse_noise", {"p": 0.005}),
    Condition("saltpepper_p0.010", "impulse_noise", {"p": 0.010}),
    Condition("saltpepper_p0.025", "impulse_noise", {"p": 0.025}),
]


def synthetic_road(w: int = 960, h: int = 540) -> np.ndarray:
    """Create a deterministic road-like frame for offline smoke testing."""
    img = np.full((h, w, 3), 185, np.uint8)
    cv2.rectangle(img, (0, 330), (w, h), (55, 55, 55), -1)
    cv2.line(img, (w // 2, 340), (w // 2, h), (235, 235, 235), 8)
    cv2.rectangle(img, (120, 250), (350, 390), (30, 90, 190), -1)
    cv2.rectangle(img, (165, 285), (305, 350), (185, 220, 245), -1)
    cv2.circle(img, (165, 395), 30, (20, 20, 20), -1)
    cv2.circle(img, (305, 395), 30, (20, 20, 20), -1)
    cv2.rectangle(img, (650, 235), (700, 365), (45, 45, 45), -1)
    cv2.circle(img, (675, 205), 25, (70, 70, 70), -1)
    cv2.putText(img, "ADAS CAMERA HEALTH", (35, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    return img


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input(path: str | None) -> tuple[np.ndarray, str]:
    if not path:
        return synthetic_road(), "synthetic_road"
    image_path = Path(path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot decode input image: {image_path}")
    return img, image_path.name


def add_glare(img: np.ndarray, gain: float = 1.05) -> np.ndarray:
    """Add a local over-exposure/glare region with soft falloff.

    `gain` scales the white overlay before saturation; larger = stronger glare.
    """
    h, w = img.shape[:2]
    center = (int(w * GLARE_CENTER_FRAC[0]), int(h * GLARE_CENTER_FRAC[1]))
    radius = max(40, int(min(h, w) * GLARE_RADIUS_FRAC))
    overlay = np.zeros_like(img)
    cv2.circle(overlay, center, radius, (255, 255, 255), -1)
    mask = cv2.GaussianBlur(overlay, (0, 0), max(12, radius / 3))
    return cv2.addWeighted(img, 1.0, mask, gain, 0)


def add_salt_pepper(img: np.ndarray, p: float = 0.025, seed: int = SALT_PEPPER_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = img.copy()
    r = rng.random(img.shape[:2])
    x[r < p / 2] = 0
    x[(r >= p / 2) & (r < p)] = 255
    return x


def gaussian_sigma(kernel: int) -> float:
    """Sigma OpenCV derives when GaussianBlur is called with sigma=0."""
    return 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8


def apply_condition(img: np.ndarray, cond: Condition) -> np.ndarray:
    if cond.family == "baseline":
        return img
    if cond.family == "blur":
        k = int(cond.params["kernel"])
        return cv2.GaussianBlur(img, (k, k), 0)
    if cond.family == "glare":
        return add_glare(img, float(cond.params["gain"]))
    if cond.family == "impulse_noise":
        return add_salt_pepper(img, float(cond.params["p"]))
    raise ValueError(f"Unknown condition family: {cond.family}")


def condition_params(img: np.ndarray, cond: Condition) -> dict:
    """Full, explicit corruption parameters for provenance."""
    h, w = img.shape[:2]
    params = dict(cond.params)
    if cond.family == "blur":
        params["sigma_px"] = round(gaussian_sigma(int(params["kernel"])), 3)
    elif cond.family == "glare":
        radius = max(40, int(min(h, w) * GLARE_RADIUS_FRAC))
        params.update(
            center_px=[int(w * GLARE_CENTER_FRAC[0]), int(h * GLARE_CENTER_FRAC[1])],
            radius_px=radius,
            falloff_sigma_px=round(max(12, radius / 3), 2),
        )
    elif cond.family == "impulse_noise":
        params["seed"] = SALT_PEPPER_SEED
    return params


def variants(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [(c.name, apply_condition(img, c)) for c in CONDITIONS]


def entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(float(hist.sum()), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def raw_metrics(img: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    # "saturation" means highlight/sensor saturation (near-white clipping),
    # not HSV chroma saturation.
    saturation_ratio = float((gray >= 245).mean())
    shadow_clip_ratio = float((gray <= 10).mean())
    clipped_pixel_ratio = float(((gray >= 245) | (gray <= 10)).mean())
    return {
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "saturation_ratio": saturation_ratio,
        "shadow_clip_ratio": shadow_clip_ratio,
        "clipped_pixel_ratio": clipped_pixel_ratio,
        "entropy": entropy(gray),
        "edge_density": float((edges > 0).mean()),
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
    }


def _ratio_similarity(value: float, baseline: float, eps: float = 1e-9) -> float:
    """Symmetric ratio similarity in [0,1]; penalizes too-low and too-high values."""
    value = max(value, eps)
    baseline = max(baseline, eps)
    return float(min(value / baseline, baseline / value, 1.0))


def add_relative_health(rows: list[dict[str, float | int | str]]) -> None:
    """Reference-based score: every row is compared with rows[0] (clean)."""
    base = rows[0]
    base_lap = float(base["laplacian_variance"])
    base_edge = float(base["edge_density"])
    base_brightness = float(base["brightness_mean"])
    base_contrast = float(base["contrast_std"])
    base_clip = float(base["clipped_pixel_ratio"])

    for row in rows:
        lap = float(row["laplacian_variance"])
        edge = float(row["edge_density"])
        brightness = float(row["brightness_mean"])
        contrast = float(row["contrast_std"])
        clip = float(row["clipped_pixel_ratio"])

        components = {
            "sharpness_similarity": _ratio_similarity(lap, base_lap),
            "edge_similarity": _ratio_similarity(edge, base_edge),
            "brightness_similarity": 1.0 - min(abs(brightness - base_brightness) / 128.0, 1.0),
            "contrast_similarity": _ratio_similarity(contrast, base_contrast),
            "clipping_quality": 1.0 - min(max(0.0, clip - base_clip) / 0.20, 1.0),
        }
        # Demo heuristic only. Weights are interpretable choices, not fitted
        # on a safety-labelled production dataset.
        health = sum(HEALTH_WEIGHTS[k] * v for k, v in components.items())
        row["edge_retention"] = edge / max(base_edge, 1e-9)
        row["health_score"] = float(np.clip(health, 0.0, 1.0))
        row["health_state"] = "DEGRADED" if row["health_score"] < HEALTH_THRESHOLD else "OK"


def persistence_policy(
    scores: list[float],
    threshold: float = HEALTH_THRESHOLD,
    n_frames: int = PERSISTENCE_FRAMES,
) -> list[str]:
    """CAMERA_DEGRADED only after `n_frames` consecutive frames below threshold."""
    states, run = [], 0
    for s in scores:
        run = run + 1 if s < threshold else 0
        states.append("DEGRADED" if run >= n_frames else "OK")
    return states


# Synthetic frame sequence built from the per-condition health scores of the
# *same* input image: a 1-frame glare flash and a 2-frame noise burst
# (transients), then a persistent blur onset. Not real video.
POLICY_SEQUENCE = (
    ["clean"] * 5
    + ["glare_gain1.05"]
    + ["clean"] * 4
    + ["saltpepper_p0.025"] * 2
    + ["clean"] * 3
    + ["blur_gauss_k21"] * 8
)


def simulate_policy(rows: list[dict], sequence: list[str] = POLICY_SEQUENCE) -> tuple[list[dict], dict]:
    score = {str(r["condition"]): float(r["health_score"]) for r in rows}
    scores = [score[c] for c in sequence]
    raw = ["DEGRADED" if s < HEALTH_THRESHOLD else "OK" for s in scores]
    policy = persistence_policy(scores)
    frames = [
        {"frame": i, "condition": c, "health_score": s, "single_frame_state": r, "policy_state": p}
        for i, (c, s, r, p) in enumerate(zip(sequence, scores, raw, policy))
    ]
    onset = next(i for i, c in enumerate(sequence) if c == "blur_gauss_k21")
    transient = [i for i in range(onset) if sequence[i] != "clean"]
    first_alarm = next((i for i in range(onset, len(sequence)) if policy[i] == "DEGRADED"), None)
    stats = {
        "sequence_length": len(sequence),
        "transient_frames": len(transient),
        "single_frame_alarms_on_transients": sum(raw[i] == "DEGRADED" for i in transient),
        "policy_alarms_on_transients": sum(policy[i] == "DEGRADED" for i in transient),
        "persistent_onset_frame": onset,
        "policy_first_alarm_frame": first_alarm,
        "policy_alarm_latency_frames": None if first_alarm is None else first_alarm - onset,
    }
    return frames, stats


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def match_to_reference(ref: list[dict], dets: list[dict], thr: float = PSEUDO_LABEL_IOU) -> dict:
    """Greedy same-class IoU matching of detections to clean-image pseudo-labels.

    This is agreement with the detector's own clean output, NOT recall against
    ground truth: clean-image mistakes are inherited as "truth".
    """
    used: set[int] = set()
    matched = 0
    lost = []
    for r in sorted(ref, key=lambda d: -d["conf"]):
        best, best_iou = None, thr
        for j, d in enumerate(dets):
            if j in used or d["cls"] != r["cls"]:
                continue
            o = iou(np.asarray(r["xyxy"]), np.asarray(d["xyxy"]))
            if o >= best_iou:
                best, best_iou = j, o
        if best is None:
            lost.append(f'{r["cls"]}:{r["conf"]:.2f}')
        else:
            used.add(best)
            matched += 1
    new = [f'{d["cls"]}:{d["conf"]:.2f}' for j, d in enumerate(dets) if j not in used]
    return {
        "pseudo_label_agreement": matched / len(ref) if ref else 1.0,
        "lost_vs_clean": ";".join(lost),
        "new_vs_clean": ";".join(new),
    }


def yolo_detect(model, img: np.ndarray) -> tuple[list[dict], np.ndarray]:
    result = model.predict(img, verbose=False)[0]
    dets = []
    if result.boxes is not None:
        for xyxy, conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy()):
            dets.append({"cls": result.names[int(cls)], "conf": float(conf),
                         "xyxy": [float(v) for v in xyxy]})
    return dets, result.plot()


def run_bremola(script: Path, img: np.ndarray) -> float:
    """Run the unmodified upstream bremola.py on a lossless copy of `img`."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frame.png"
        cv2.imwrite(str(path), img)
        out = subprocess.run([sys.executable, str(script), "-i", str(path)],
                             capture_output=True, text=True, check=True).stdout
    m = re.search(r"bremola:\s*([-\d.eE+]+)", out)
    if not m:
        raise RuntimeError(f"Unexpected BREMOLA output: {out!r}")
    return float(m.group(1))


def bremola_meta(script: Path) -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(script.parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "name": "BREMOLA (Nam et al., Vehicles 2025)",
        "repository": "https://github.com/woongchan789/BREMOLA",
        "commit": commit,
        "script_sha256": sha256_file(script),
        "note": "Upstream script run unmodified; it resizes every frame to 1920x1080.",
    }


def save_metric_plot(rows: list[dict], out_dir: Path) -> None:
    labels = [str(r["condition"]) for r in rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(11, 5.4))
    plt.plot(x, [float(r["health_score"]) for r in rows], marker="o", label="health score (reference-based)")
    plt.plot(x, [min(float(r["edge_retention"]), 1.5) for r in rows],
             marker="o", label="edge retention vs clean (capped at 1.5)")
    if "bremola" in rows[0]:
        base = max(float(rows[0]["bremola"]), 1e-9)
        plt.plot(x, [min(float(r["bremola"]) / base, 1.5) for r in rows], marker="^",
                 label="BREMOLA / clean BREMOLA (no-reference, capped at 1.5)")
    plt.axhline(HEALTH_THRESHOLD, linestyle="--", linewidth=1.4,
                label=f"demo threshold {HEALTH_THRESHOLD}")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("normalized proxy (1.0 = clean)")
    plt.title("Camera degradation health proxies")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "degradation_metrics.png", dpi=170)
    plt.close()


def save_detector_plot(rows: list[dict], out_dir: Path) -> None:
    if "mean_confidence" not in rows[0]:
        return
    labels = [str(r["condition"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(11, 5.4))
    ax1.plot(x, [float(r["mean_confidence"]) for r in rows], marker="o", label="mean confidence")
    ax1.plot(x, [float(r["pseudo_label_agreement"]) for r in rows], marker="D",
             label="agreement with clean pseudo-labels")
    ax1.set_ylabel("fraction")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x, labels, rotation=30, ha="right")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, [int(r["detections"]) for r in rows], marker="s", linestyle="--",
             color="gray", label="detection count")
    ax2.set_ylabel("detection count")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    plt.title("YOLOv8n downstream proxy under camera degradation (no ground truth)")
    fig.tight_layout()
    fig.savefig(out_dir / "detector_proxy.png", dpi=170)
    plt.close(fig)


def save_policy_plot(frames: list[dict], out_dir: Path) -> None:
    x = [f["frame"] for f in frames]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.step(x, [f["health_score"] for f in frames], where="mid", label="per-frame health")
    ax.axhline(HEALTH_THRESHOLD, linestyle="--", linewidth=1.2, label=f"threshold {HEALTH_THRESHOLD}")
    for f in frames:
        if f["single_frame_state"] == "DEGRADED":
            ax.scatter(f["frame"], 0.05, marker="v", color="tab:orange")
        if f["policy_state"] == "DEGRADED":
            ax.scatter(f["frame"], 0.12, marker="s", color="tab:red")
    ax.scatter([], [], marker="v", color="tab:orange", label="single-frame alarm")
    ax.scatter([], [], marker="s", color="tab:red", label=f"policy alarm ({PERSISTENCE_FRAMES} consecutive)")
    ax.set_xticks(x, [f["condition"].split("_")[0] for f in frames], rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("health score")
    ax.set_title("Persistence policy on a synthetic sequence (frames reuse single-image variants)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(out_dir / "policy_timeline.png", dpi=170)
    plt.close(fig)


def titled(frame: np.ndarray, title: str) -> np.ndarray:
    framed = cv2.copyMakeBorder(frame, 58, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(framed, title, (18, 39), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    return framed


def save_pair(frames: dict[str, np.ndarray], a: str, b: str, prefix: str, path: Path) -> None:
    if a in frames and b in frames:
        pair = np.hstack([titled(frames[a], f"{prefix}{a}"), titled(frames[b], f"{prefix}{b}")])
        cv2.imwrite(str(path), pair)


def runtime_versions() -> dict:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass
    return versions


def run(
    input_path: str | None,
    use_yolo: bool,
    out_dir: Path,
    source_url: str | None = None,
    bremola_script: str | None = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base, source_name = load_input(input_path)
    input_sha256 = sha256_file(Path(input_path)) if input_path else None
    model = None
    detector_meta = None
    if use_yolo:
        import ultralytics
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        weight_path = Path("yolov8n.pt")
        detector_meta = {
            "name": "YOLOv8n",
            "ultralytics_version": ultralytics.__version__,
            "weight_file": weight_path.name,
            "weight_sha256": sha256_file(weight_path) if weight_path.exists() else None,
            "pseudo_label_iou": PSEUDO_LABEL_IOU,
        }
    bremola_path = Path(bremola_script) if bremola_script else None

    rows: list[dict] = []
    images: dict[str, np.ndarray] = {}
    detector_frames: dict[str, np.ndarray] = {}
    clean_dets: list[dict] = []
    for level, cond in enumerate(CONDITIONS):
        img = apply_condition(base, cond)
        images[cond.name] = img
        row: dict = {"level": level, "condition": cond.name, "family": cond.family,
                     **raw_metrics(img)}
        if bremola_path is not None:
            row["bremola"] = run_bremola(bremola_path, img)
        if model is not None:
            dets, annotated = yolo_detect(model, img)
            if cond.family == "baseline":
                clean_dets = dets
            row.update(
                detections=len(dets),
                mean_confidence=float(np.mean([d["conf"] for d in dets])) if dets else 0.0,
                detected_classes=";".join(f'{d["cls"]}:{d["conf"]:.2f}' for d in dets),
                **match_to_reference(clean_dets, dets),
            )
            detector_frames[cond.name] = annotated
            cv2.imwrite(str(out_dir / f"detector_{level:02d}_{cond.name}.jpg"), annotated)
        rows.append(row)
        cv2.imwrite(str(out_dir / f"{level:02d}_{cond.name}.jpg"), img)

    add_relative_health(rows)
    policy_frames, policy_stats = simulate_policy(rows)

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "policy_sequence.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(policy_frames[0].keys()))
        writer.writeheader()
        writer.writerows(policy_frames)

    save_metric_plot(rows, out_dir)
    save_detector_plot(rows, out_dir)
    save_policy_plot(policy_frames, out_dir)
    save_pair(images, "clean", "glare_gain1.05", "", out_dir / "before_after.png")
    save_pair(detector_frames, "clean", "glare_gain1.05", "YOLO: ", out_dir / "detector_before_after.png")
    save_pair(detector_frames, "clean", "blur_gauss_k21", "YOLO: ", out_dir / "detector_blur_compare.png")

    summary = {
        "input": source_name,
        "input_source_url": source_url,
        "input_sha256": input_sha256,
        "input_shape_hw": list(base.shape[:2]),
        "detector": detector_meta,
        "no_reference_iqa": bremola_meta(bremola_path) if bremola_path else None,
        "runtime_provenance": {
            "context": "github_actions" if os.environ.get("GITHUB_ACTIONS") else "local",
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_repository": os.environ.get("GITHUB_REPOSITORY"),
            "versions": runtime_versions(),
        },
        "conditions": [
            {"level": i, "name": c.name, "family": c.family, "params": condition_params(base, c)}
            for i, c in enumerate(CONDITIONS)
        ],
        "health_score_formula": HEALTH_WEIGHTS,
        "health_score_reference": "clean image of the same scene (reference-based, not deployable as-is)",
        "health_threshold": HEALTH_THRESHOLD,
        "policy_demo_only": f"health_score < {HEALTH_THRESHOLD} for {PERSISTENCE_FRAMES} consecutive frames -> DEGRADED",
        "policy_simulation": {"sequence": POLICY_SEQUENCE, **policy_stats},
        "claim_boundary": (
            "Health score is a heuristic proxy; detector confidence and agreement with "
            "clean-image pseudo-labels are not mAP/recall."
        ),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input: {source_name}")
    print(f"Wrote results to: {out_dir}")
    for row in rows:
        print(row)
    print("policy:", policy_stats)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Optional path to an RGB road image")
    parser.add_argument("--yolo", action="store_true", help="Run optional YOLOv8n detector proxy")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--source-url", default=None, help="Optional provenance URL for the input")
    parser.add_argument("--bremola-script", default=None,
                        help="Optional path to upstream BREMOLA bremola.py (no-reference IQA)")
    args = parser.parse_args()
    run(args.input, args.yolo, Path(args.output_dir), args.source_url, args.bremola_script)


if __name__ == "__main__":
    main()
