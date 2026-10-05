# 3–5 Minute Pitch — Camera Degradation Health Score

## 0:00–0:40 — Problem

Our question is not simply **“Does the detector still output boxes?”**

For an ADAS camera, blur, glare, contamination or noise can damage the input itself. If we only watch detector confidence, the system may not know that the camera has become unreliable.

Corruption benchmarks support this concern. Dong et al. tested 27 autonomous-driving corruptions on KITTI-C, nuScenes-C and Waymo-C, and RoboBEV later studied eight camera corruption/failure types across 33 BEV models.

So our small engineering question is:

**Can we monitor camera health independently of the detector?**

---

## 0:40–1:20 — Method

We use one public road image and create five controlled corruptions:

- Gaussian blur with kernels 5, 11 and 21
- strong glare / local over-exposure
- salt-and-pepper noise

For each image we compute lightweight health signals:

- Laplacian variance
- edge retention
- brightness and contrast
- highlight/shadow clipping
- entropy

Then we combine sharpness, edges, brightness, contrast and clipping into an interpretable health score.

Separately, we run YOLOv8n and record only two proxy values:

- number of detections
- mean confidence

We do **not** call these mAP or recall because we have no ground-truth annotations.

---

## 1:20–2:30 — Result

The most interesting result is the strong blur case.

On the clean image:

- Laplacian variance is **3619**
- edge retention is **100%**
- YOLO mean confidence is **0.656**

With blur kernel 21:

- Laplacian falls to **3.4**, about **99.91% lower**
- edge retention falls to only **4.09%**
- our health score becomes **0.444 — DEGRADED**

But YOLO mean confidence actually rises to **0.675**, around **3% higher**, and the detector still outputs 6 boxes.

So detector confidence alone would not tell us that the camera image has lost most of its high-frequency structure.

Glare shows another failure mode:

- highlight saturation jumps from **1.27% to 20.00%**
- mean confidence drops by about **16.3%**
- but detection count changes from 6 to 7

Again, raw detection count is not a quality metric.

Noise gives the opposite problem: it creates fake high-frequency structure.

- Laplacian becomes about **3.86× larger than clean**
- edge retention becomes **139%**

So a rule such as “high Laplacian means healthy” would also fail.

---

## 2:30–3:20 — Failure case and research connection

This is why we use a multi-signal health monitor.

The BREMOLA paper in *Vehicles 2025* is directly relevant. It proposes a no-reference image-quality metric for autonomous-driving blur using Fourier information and a Laplacian complexity term. Its Table 2 reports Laplacian as the best correlated high-pass filter in their comparison, with SROCC **0.9236**.

But BREMOLA explicitly focuses on blur; glare and noise are outside its main scope.

RoboBEV also shows the broader lesson: simple image-distribution changes do not necessarily track downstream perception damage.

So our conclusion is not “Laplacian solves camera health.”

Our conclusion is:

**Laplacian is useful, but camera health needs multiple independent signals.**

---

## 3:20–4:00 — Engineering decision

For the demo we use:

**health score below 0.65 for 3 consecutive frames → CAMERA DEGRADED**

The threshold is only a lab heuristic.

On a real vehicle we would also log:

- exposure time
- gain / ISO
- shutter
- auto-exposure state
- frame timestamp
- brightness/clipping signals
- camera temperature and diagnostics where available

When degradation persists, the vehicle should raise a health event, down-weight camera evidence in fusion, and request a safer operating mode.

If Radar or LiDAR is available, those independent sensors can support fallback.

We are **not claiming that this repository implements sensor fusion or a production safety controller**.

---

## 4:00–4:20 — Closing

The key result is simple:

> A camera can be severely degraded while a detector still looks confident.

That is why sensor health should be monitored **before and independently from perception confidence**.

All numbers, plots, degraded images, detector overlays, paper references and reproducibility metadata are committed in the repository.
