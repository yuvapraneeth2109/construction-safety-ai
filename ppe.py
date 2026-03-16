import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="PPE Detection", layout="wide")
st.title("🦺 Construction Safety – PPE Detection System")

model = YOLO("models/ppeharness.pt") 

# ❌ Removed no harness
VIOLATIONS = {
    "No helmet": "❌ Helmet Missing",
    "No safety shoes": "❌ Safety Shoes Missing",
    "no gloves": "❌ Gloves Missing"
}

uploaded_video = st.file_uploader(
    "Upload construction site video",
    type=["mp4", "avi", "mov"]
)

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

        results = model(frame)[0]
        annotated = frame.copy()
        alerts = set()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # ❌ Ignore harness classes completely
            if cls_name.lower() in ["harness", "no harness"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name in VIOLATIONS:
                color = (0, 0, 255)
                alerts.add(VIOLATIONS[cls_name])
            else:
                color = (0, 255, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                cls_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        frame_box.image(annotated, channels="BGR")

        if alerts:
            alert_box.warning(" | ".join(alerts))
        else:
            alert_box.success("✔ All safety equipment detected")

    cap.release()  