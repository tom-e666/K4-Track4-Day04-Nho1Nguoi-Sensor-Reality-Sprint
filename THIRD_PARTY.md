# Third-party provenance

Third-party inputs/models are referenced rather than treated as project-owned artifacts.

## Road-scene sample

- Runtime URL: https://ultralytics.com/images/bus.jpg
- Purpose: tiny public road/street-scene input for the real-image benchmark.
- Storage policy: downloaded by GitHub Actions at benchmark time; the source file itself is not committed.
- Reproducibility: `results/summary.json` records the source URL and SHA-256 of the downloaded input.

## YOLO detector proxy

- Model: YOLOv8n pretrained weight `yolov8n.pt`
- Python package: `ultralytics==8.4.172` for the committed benchmark workflow
- Ecosystem/source: https://github.com/ultralytics/ultralytics
- Package: https://pypi.org/project/ultralytics/
- License note: Ultralytics publishes open-source software under AGPL-3.0 alongside commercial licensing options. Check upstream terms for your own deployment scenario.
- Storage policy: the weight is downloaded at runtime and not committed.
- Reproducibility: `results/summary.json` records package version and the local weight SHA-256 when available.

## BREMOLA no-reference IQA (upstream code)

- Repository: https://github.com/woongchan789/BREMOLA
- Pinned commit: `7ba26999c265692bb8e44c5a3f2d91c06746830f`
- Used file: `bremola.py`, run **unmodified** as a subprocess on a lossless PNG of each benchmark variant.
- Storage policy: cloned into git-ignored `third_party/BREMOLA` at benchmark time; no upstream code is committed here.
- Reproducibility: `results/summary.json → no_reference_iqa` records the commit and the script's SHA-256.
- License note: no license file was found in the repository at the pinned commit; we therefore only execute it and do not redistribute it.

## Research papers

Paper PDFs are **not copied into this repository**. Stable DOI/official-paper/code URLs and full citations are stored in:

- `docs/research.md`
- `REFERENCES.bib`

This keeps the repository lightweight and avoids implying ownership of external publications.

## Claim boundary

YOLO detection count and mean confidence are used only as **downstream proxy signals**. They are not mAP, recall, missed-detection rate, or safety validation.
