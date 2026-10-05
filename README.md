# Day 19 — AI Sensor Health Demo

**Topic:** T1 — Camera Degradation Health Score for ADAS

A lightweight, reproducible demo that injects camera degradation (blur, glare, noise), measures sensor-health indicators, optionally runs YOLOv8n as a detector proxy, exports CSV evidence, plots results, and documents a concrete failure case + fallback engineering decision.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/run_demo.py
```

The default demo is **offline and synthetic**: it generates a road-like test image, so no dataset download is required. YOLO is optional:

```bash
pip install ultralytics
python src/run_demo.py --yolo
```

## What is measured

- **Laplacian variance** — sharpness/blur proxy. Lower usually means blurrier.
- **Saturation ratio** — fraction of near-clipped pixels, useful for glare/over-exposure.
- **Entropy** — image-information proxy.
- **Edge retention** — ratio of Canny edge density versus the clean baseline.
- **YOLO mean confidence / detection count** — only when `--yolo` is enabled.

These are engineering health indicators, **not a substitute for labeled mAP/recall**.

## Experiment

Five conditions are generated from the same baseline:

| Level | Condition |
|---|---|
| 0 | clean |
| 1 | Gaussian blur k=5 |
| 2 | Gaussian blur k=11 |
| 3 | Gaussian blur k=21 |
| 4 | strong glare / local over-exposure |
| 5 | salt-and-pepper noise |

Artifacts are written to `results/`:
- `metrics.csv`
- `degradation_metrics.png`
- `before_after.png`
- degraded images

## Failure case

**Strong glare** can saturate a large image region. A blur-only monitor may still report non-trivial edge energy outside the glare region while an important object inside it becomes unusable. Therefore the health monitor combines sharpness with saturation and should not treat Laplacian variance as a universal camera-health score.

## Engineering decision

Prototype fallback policy:

> If normalized health score < 0.40 for 3 consecutive frames, mark the camera DEGRADED. Log exposure/gain/shutter metadata where available; down-weight camera evidence in fusion and request a safer operating mode. Radar/LiDAR fallback is a system-level recommendation, not implemented by this demo.

The threshold is a **demo policy**, not a validated production safety threshold.

## Team split

1. Tech Lead — degradation pipeline + metrics + integration.
2. Model Runner — YOLOv8n optional inference + logs.
3. Benchmark/Visualization — CSV + plots + before/after.
4. Research/Method — assumptions, failure case, source review.
5. Presenter — one-page report/site and 3-minute pitch.

## Repository layout

```text
src/                 experiment code
tests/               smoke tests
results/             generated evidence
docs/                static GitHub Pages report
.github/workflows/   CI + Pages deployment
REPORT.md             one-page rubric-aligned report
PITCH.md              3-minute speaking script
```

## Reproducibility / claim policy

Numbers in generated CSV/plots come from the local run. The report deliberately separates **our measurements** from **external literature**. No statistical significance, detector recall, or mAP claim is made without labels and repeated experiments.

## Static report

The repository includes a static site in `docs/index.html` and a GitHub Pages deployment workflow. In repository **Settings → Pages**, choose **GitHub Actions** as the source if GitHub has not enabled it automatically.
