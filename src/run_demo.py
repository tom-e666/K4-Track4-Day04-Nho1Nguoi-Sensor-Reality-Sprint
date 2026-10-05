from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results"


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


def load_input(path: str | None) -> tuple[np.ndarray, str]:
    if not path:
        return synthetic_road(), "synthetic_road"
    image_path = Path(path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot decode input image: {image_path}")
    return img, image_path.name


def add_glare(img: np.ndarray) -> np.ndarray:
    """Add a local over-exposure/glare region with soft falloff."""
    h, w = img.shape[:2]
    center = (int(w * 0.73), int(h * 0.30))
    radius = max(40, int(min(h, w) * 0.30))
    overlay = np.zeros_like(img)
    cv2.circle(overlay, center, radius, (255, 255, 255), -1)
    mask = cv2.GaussianBlur(overlay, (0, 0), max(12, radius / 3))
    return cv2.addWeighted(img, 1.0, mask, 1.05, 0)


def add_salt_pepper(img: np.ndarray, p: float = 0.025, seed: int = 19) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = img.copy()
    r = rng.random(img.shape[:2])
    x[r < p / 2] = 0
    x[(r >= p / 2) & (r < p)] = 255
    return x


def variants(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("clean", img),
        ("blur_k5", cv2.GaussianBlur(img, (5, 5), 0)),
        ("blur_k11", cv2.GaussianBlur(img, (11, 11), 0)),
        ("blur_k21", cv2.GaussianBlur(img, (21, 21), 0)),
        ("strong_glare", add_glare(img)),
        ("salt_pepper", add_salt_pepper(img)),
    ]


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

        sharpness_similarity = _ratio_similarity(lap, base_lap)
        edge_similarity = _ratio_similarity(edge, base_edge)
        brightness_similarity = 1.0 - min(abs(brightness - base_brightness) / 128.0, 1.0)
        contrast_similarity = _ratio_similarity(contrast, base_contrast)
        excess_clip = max(0.0, clip - base_clip)
        clipping_quality = 1.0 - min(excess_clip / 0.20, 1.0)

        # Demo heuristic only. Weights are interpretable choices, not fitted
        # on a safety-labelled production dataset.
        health = (
            0.40 * sharpness_similarity
            + 0.15 * edge_similarity
            + 0.10 * brightness_similarity
            + 0.10 * contrast_similarity
            + 0.25 * clipping_quality
        )
        row["edge_retention"] = edge / max(base_edge, 1e-9)
        row["health_score"] = float(np.clip(health, 0.0, 1.0))
        row["health_state"] = "DEGRADED" if row["health_score"] < 0.65 else "OK"


def yolo_metrics(model, img: np.ndarray) -> tuple[int, float, np.ndarray]:
    result = model.predict(img, verbose=False)[0]
    conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
    return int(len(conf)), float(conf.mean()) if len(conf) else 0.0, result.plot()


def save_metric_plot(rows: list[dict], out_dir: Path) -> None:
    labels = [str(r["condition"]) for r in rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(9.5, 5.2))
    plt.plot(x, [float(r["health_score"]) for r in rows], marker="o", label="health score")
    plt.plot(x, [min(float(r["edge_retention"]), 1.5) for r in rows],
             marker="o", label="edge retention")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("normalized proxy")
    plt.title("Camera degradation health proxies")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "degradation_metrics.png", dpi=170)
    plt.close()


def save_detector_plot(rows: list[dict], out_dir: Path) -> None:
    if "mean_confidence" not in rows[0]:
        return
    labels = [str(r["condition"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    ax1.plot(x, [float(r["mean_confidence"]) for r in rows], marker="o")
    ax1.set_ylabel("YOLO mean confidence")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x, labels, rotation=25, ha="right")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, [int(r["detections"]) for r in rows], marker="s", linestyle="--")
    ax2.set_ylabel("detection count")
    plt.title("Detector proxy under camera degradation")
    fig.tight_layout()
    fig.savefig(out_dir / "detector_proxy.png", dpi=170)
    plt.close(fig)


def run(input_path: str | None, use_yolo: bool, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base, source_name = load_input(input_path)
    model = None
    if use_yolo:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")

    rows: list[dict] = []
    for level, (name, img) in enumerate(variants(base)):
        row: dict = {"level": level, "condition": name, **raw_metrics(img)}
        if model is not None:
            n, c, annotated = yolo_metrics(model, img)
            row.update(detections=n, mean_confidence=c)
            cv2.imwrite(str(out_dir / f"detector_{level}_{name}.jpg"), annotated)
        rows.append(row)
        cv2.imwrite(str(out_dir / f"{level}_{name}.jpg"), img)

    add_relative_health(rows)

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_metric_plot(rows, out_dir)
    save_detector_plot(rows, out_dir)

    glare_img = variants(base)[4][1]
    canvas = np.hstack([base, glare_img])
    cv2.putText(canvas, "CLEAN", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    cv2.putText(canvas, "STRONG GLARE", (base.shape[1] + 20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    cv2.imwrite(str(out_dir / "before_after.png"), canvas)

    summary = {
        "input": source_name,
        "detector": "YOLOv8n" if use_yolo else None,
        "health_score_formula": {
            "sharpness_similarity": 0.40,
            "edge_similarity": 0.15,
            "brightness_similarity": 0.10,
            "contrast_similarity": 0.10,
            "clipping_quality": 0.25,
        },
        "policy_demo_only": "health_score < 0.65 for 3 consecutive frames -> DEGRADED",
        "claim_boundary": "Health score is a heuristic proxy; detector confidence is not mAP/recall.",
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input: {source_name}")
    print(f"Wrote results to: {out_dir}")
    for row in rows:
        print(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Optional path to an RGB road image")
    parser.add_argument("--yolo", action="store_true", help="Run optional YOLOv8n detector proxy")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run(args.input, args.yolo, Path(args.output_dir))


if __name__ == "__main__":
    main()
