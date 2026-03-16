import numpy as np
from src.config import IOU_THRESHOLD


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def global_nms(detections):
    """
    Apply Class-Aware Non-Maximum Suppression
    """

    if len(detections) == 0:
        return []

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    final_detections = []

    while detections:
        best = detections.pop(0)
        final_detections.append(best)

        remaining = []

        for det in detections:
            # Only suppress if same class
            if det["class_id"] != best["class_id"]:
                remaining.append(det)
                continue

            iou = compute_iou(best["bbox"], det["bbox"])

            if iou < IOU_THRESHOLD:
                remaining.append(det)

        detections = remaining

    return final_detections

