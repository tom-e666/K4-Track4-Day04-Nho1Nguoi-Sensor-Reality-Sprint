# 3-minute pitch

**0:00–0:40 — Problem.** Our question is not only “does the detector fail?”, but “can the vehicle know that its camera input is becoming unhealthy?” Blur and glare can reduce useful visual evidence before a perception model gives an obvious error.

**0:40–1:20 — Method.** We built a tiny reproducible pipeline. One clean road-like frame is transformed into several controlled corruptions: three Gaussian blur levels, strong glare and salt-and-pepper noise. We measure Laplacian variance, saturation ratio, entropy and edge retention. YOLOv8n can be enabled as an optional downstream detector proxy.

**1:20–2:15 — Benchmark and failure case.** The experiment writes every measurement to CSV and plots degradation against the normalized health proxy. The key failure case is glare: a blur-only score can miss it because edges remain in unaffected regions, even though a bright region has lost information. That means camera health cannot be represented by one sharpness metric.

**2:15–3:00 — Engineering decision.** Our demo policy marks the camera degraded if health stays below 0.40 for three frames. A real vehicle should also log exposure, gain and shutter metadata, down-weight unhealthy camera evidence in fusion, and move toward a safer operating mode using independent sensors where available. The threshold is a prototype choice, not a production safety claim.
