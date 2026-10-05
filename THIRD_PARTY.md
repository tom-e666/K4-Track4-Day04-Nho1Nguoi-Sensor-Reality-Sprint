# Third-party assets and models

This project keeps third-party inputs separate from the experiment code.

## Ultralytics sample image and YOLOv8n

The real-image benchmark downloads:

- Sample image: `https://ultralytics.com/images/bus.jpg`
- Detector: Ultralytics YOLOv8n (`yolov8n.pt`), downloaded by the `ultralytics` package.

Source ecosystem: https://github.com/ultralytics

Ultralytics publishes its open-source software/assets under AGPL-3.0 alongside commercial licensing options. The sample image and detector are used here only as a lightweight demonstration/benchmark input. Generated experiment artifacts record this provenance.

The repository does **not** claim ownership of the source sample image or pretrained model.

## Claim boundary

YOLO detection count and mean confidence are used as **downstream proxy signals** only. They are not mAP, recall, missed-detection rate, or a safety validation.
