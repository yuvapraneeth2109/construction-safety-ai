from src.config import CONF_THRESHOLD
import numpy as np


def run_inference(models, frame):
    """
    Runs inference using all loaded YOLO models
    Returns unified detection list
    """

    all_detections = []

    for model_name, model in models.items():
        results = model(frame, verbose=False)

        for r in results:
            boxes = r.boxes

            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                class_name = model.names[class_id]

                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "source_model": model_name
                }

                all_detections.append(detection)

    return all_detections
