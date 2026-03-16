import cv2
from src.model_loader import load_models
from src.inference_engine import run_inference

models = load_models()

# Load a test image (use any construction image)
image = cv2.imread("/Users/yuvapraneeth/Desktop/construction_safety_system/test.jpeg")

detections = run_inference(models, image)

from src.fusion import global_nms

print("Before NMS:", len(detections))

filtered = global_nms(detections)

print("After NMS:", len(filtered))

for d in filtered:
    print(d)

from src.violation_logic import evaluate_violations

violations, summary = evaluate_violations(filtered)

print("\nSummary:", summary)
print("Violations:", violations)

from src.logger import SafetyLogger

logger = SafetyLogger()
logger.log(filtered, violations)

report_path = logger.export_excel()

print("\nReport saved at:", report_path)
