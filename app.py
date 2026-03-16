import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

from src.model_loader import load_models
from src.inference_engine import run_inference
from src.fusion import global_nms
from src.violation_logic import evaluate_violations
from src.logger import SafetyLogger


st.set_page_config(page_title="Construction Safety Detection", layout="wide")

st.title("🏗 Construction Safety Detection System")


# -----------------------------
# Load Models (Once)
# -----------------------------
if "models" not in st.session_state:
    st.session_state.models = load_models()

if "logger" not in st.session_state:
    st.session_state.logger = SafetyLogger()


uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])


# -----------------------------
# Drawing Function
# -----------------------------
def draw_boxes(frame, detections, unsafe_types):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det["class_name"]
        conf = round(det["confidence"], 2)

        name = label.lower()

        # Default GREEN
        color = (0, 255, 0)

        # 🔴 RED for violations (highest priority)
        if name in unsafe_types:
            color = (0, 0, 255)

        # 🔵 BLUE for lanyard (only if not violation)
        elif "lanyard" in name or "lifeline" in name:
            color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return frame




# -----------------------------
# IMAGE MODE
# -----------------------------
if uploaded_file is not None:

    file_type = uploaded_file.type

    if "image" in file_type:
        image = Image.open(uploaded_file)
        frame = np.array(image)

        detections = run_inference(st.session_state.models, frame)
        filtered = global_nms(detections)

        violations, summary, unsafe_types = evaluate_violations(filtered)

        st.session_state.logger.log(filtered, violations)

        output_frame = draw_boxes(frame.copy(), filtered, unsafe_types)

        st.image(output_frame, channels="BGR", use_container_width=True)

        st.subheader("Detection Summary")
        st.write(summary)

        if violations:
            st.error("Violations Detected:")
            for v in violations:
                st.write(f"- {v}")
        else:
            st.success("No Violations Detected")

        # Debug info (safe)
        st.write("Unsafe Types:", unsafe_types)
        st.write("Detected Classes:", [d["class_name"] for d in filtered])

    # -----------------------------
    # VIDEO MODE
    # -----------------------------
    elif "video" in file_type:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = run_inference(st.session_state.models, frame)
            filtered = global_nms(detections)

            violations, summary, unsafe_types = evaluate_violations(filtered)

            st.session_state.logger.log(filtered, violations)

            output_frame = draw_boxes(frame.copy(), filtered, unsafe_types)

            stframe.image(output_frame, channels="BGR", use_container_width=True)

        cap.release()


# -----------------------------
# DOWNLOAD BUTTON
# -----------------------------
st.subheader("📊 Export Report")

if len(st.session_state.logger.records) > 0:

    filepath = st.session_state.logger.export_excel()

    with open(filepath, "rb") as f:
        st.download_button(
            label="Download Safety Report",
            data=f,
            file_name="safety_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("No records available yet.")

    