# 3–5 Minute Pitch — Camera Degradation Health Score

Order: problem → method → benchmark → failure case → decision. Every number below is in `results/metrics.csv` or `results/summary.json`; open them when asked.

## 0:00–0:35 — Problem

ADAS forward camera, object detection feeding AEB/ACC-type functions.

Blur, glare and sensor noise damage the camera input itself. If we only watch detector confidence, the system may not know the camera has become unreliable.

Sources: Dong et al. (CVPR 2023) tested 27 corruptions on KITTI-C/nuScenes-C/Waymo-C. RoboBEV (TPAMI 2025) tested 8 camera corruptions × 3 severities on 33 BEV models. Both report clear robustness loss. These are their numbers, not ours.

**Our question: can we monitor camera health independently of the detector, and where does that monitor fail?**

## 0:35–1:20 — Method

- One public street image. Nine corruptions from the same baseline, one factor at a time:
  - Gaussian blur, kernels 5 / 11 / 21;
  - glare gain 0.35 / 0.70 / 1.05;
  - salt-and-pepper 0.5% / 1% / 2.5%, seed 19.
- Health signals: Laplacian variance, edge retention, highlight clipping, brightness, contrast and entropy, combined into a health score with threshold 0.65.
- We also **ran the BREMOLA paper's own code** (Vehicles 2025, pinned commit), a no-reference blur metric.
- YOLOv8n: count, mean confidence, and agreement with its own clean-image boxes. **Not recall.** We have no ground truth.

## 1:20–2:30 — Benchmark results

**Blur k=21.** Laplacian falls 3619 → 3.4 (−99.9%), and edge retention falls to 4%. Health is 0.44, DEGRADED. Yet mean confidence *rises* 0.656 → 0.675. Why? The detector lost a "stop sign 0.26" and invented a "dog 0.36". Same count, wrong box, higher average.

**Light noise, 0.5%.** The only change is that a weak box disappears, and mean confidence jumps +13%. Mean confidence moves for reasons unrelated to image health.

**BREMOLA** catches blur onset (−25% at k=5) but stays flat from k=5 to k=21. It *rises* under glare and noise, which matches the paper's own blur-only scope.

## 2:30–3:30 — Failure case: glare

At glare gain 1.05:
- saturated pixels rise from 1.3% to 20%;
- the detector **loses the bus (0.87)** and outputs "airplane 0.45" and "truck 0.29";
- our health score is 0.636, just under the threshold.

At gain 0.70, 11% of pixels are saturated and our score says **OK (0.80)**.

Why: clipping has weight 0.25, so glare alone can never pull the score below 0.75. A global average hides a regional fault.

To be honest about the process: we set the weights after the first run, then froze them. The new mild levels are out-of-sample, and the rule missed two of them. *(Hypothesis)* For AEB, a lead vehicle flipping class at that moment could change the braking decision. We did not test a tracker or planner.

## 3:30–4:20 — Engineering decision

1. Do not use detector confidence as the camera-health signal.
2. Add a **glare-specific saturation rule** beside the composite score.
3. Require **3 consecutive bad frames**. On our synthetic sequence, transient alarms go from 3 to 0 at a cost of 2 frames of latency (~67 ms at 30 fps).
4. Replace the clean reference, which a car never has, with a rolling healthy-frame reference.

On a persistent fault: log the event, down-weight camera in fusion, and request a safer mode. Radar or LiDAR can support fallback. We do **not** implement fusion or control.

Next round we measure:
- flag rate per glare level on 20+ images;
- the policy on real video with labelled fault intervals;
- real recall on a labelled nuScenes-C slice.

## 4:20–4:40 — Closing

> A camera can be badly degraded while the detector looks just as confident, and our own global health score can miss regional glare.

Monitor sensor health before, and independently of, perception confidence, and test the monitor on levels it was not tuned on.
