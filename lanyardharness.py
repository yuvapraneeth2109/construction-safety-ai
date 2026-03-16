import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Harness + Lanyard + Hook Detection", layout="wide")
st.title("🦺 Safety Harness, Lanyard & Hook Detection System")

# ---------------- LOAD MODELS ----------------
# Model 1 → Harness model
harness_model = YOLO("models/ppeharness.pt")

# Model 2 → Lanyard + Hook model
lanyard_hook_model = YOLO("models/lanyard.pt") 

# ---------------- UPLOAD VIDEO ----------------
uploaded_video = st.file_uploader("Upload construction site video", type=["mp4", "avi", "mov"])

if uploaded_video:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp.name)
    frame_box = st.empty()
    alert_box = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        detected_classes = set()

        # ---------------- HARNESS MODEL ----------------
        harness_results = harness_model(frame)[0]

        for box in harness_results.boxes:
            cls_id = int(box.cls[0])
            cls_name = harness_model.names[cls_id]

            # Only allow harness class
            if "harness" not in cls_name.lower():
                continue

            detected_classes.add("Harness")

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, "Harness",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ---------------- LANYARD + HOOK MODEL ----------------
        lanyard_results = lanyard_hook_model(frame)[0]

        for box in lanyard_results.boxes:
            cls_id = int(box.cls[0])
            cls_name = lanyard_hook_model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Detect Lanyard
            if "lanyard" in cls_name.lower():
                detected_classes.add("Lanyard")
                color = (255, 0, 0)
                label = "Lanyard"

            # Detect Hook
            elif "hook" in cls_name.lower():
                detected_classes.add("Hook")
                color = (0, 255, 255)
                label = "Hook"

            else:
                continue

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # ---------------- DISPLAY FRAME ----------------
        frame_box.image(annotated, channels="BGR")

        # ---------------- ALERT LOGIC ----------------
        alerts = []
 
        if "Harness" not in detected_classes:
            alerts.append("❌ Harness Missing")

        if "Lanyard" not in detected_classes:
            alerts.append("❌ Lanyard Missing")

        if "Hook" not in detected_classes:
            alerts.append("❌ Hook Missing")

        if alerts:
            alert_box.error(" | ".join(alerts))
        else:
            alert_box.success("✔ Harness, Lanyard & Hook Detected")

    cap.release() 