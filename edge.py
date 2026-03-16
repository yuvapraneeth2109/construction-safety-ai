import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Construction Safety AI", layout="wide")
st.title("🚧 Site Safety Monitor: Edge & Person Distance")

# --- LOAD MODEL ---
# Using the path from your successful training run
model_path = "models/edgee.pt" 

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Configuration")
input_type = st.sidebar.radio("Select Input Type", ["Image", "Video"])
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)
ppm = st.sidebar.number_input("Pixels per Meter (Calibration)", value=100.0)
danger_zone = st.sidebar.slider("Danger Distance (Meters)", 0.5, 5.0, 2.0)

def process_frame(frame, model, conf, ppm, danger_dist):
    """Detects objects and calculates distance between person and edge."""
    results = model.predict(frame, conf=conf, verbose=False)[0]
    
    persons = []
    edges = []
    
    # Class Mapping: 0:edge, 1:person (based on your updated data.yaml)
    for box in results.boxes:
        coords = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        
        if cls == 1: # Person
            persons.append(coords)
        elif cls == 0: # Edge
            edges.append(coords)
            
    annotated_frame = results.plot()
    
    # Distance Calculation Logic 
    for p in persons:
        # Center of person box
        px, py = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        
        for e in edges:
            # Center of edge box
            ex, ey = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
            
            # Euclidean Distance in Pixels
            dist_px = np.sqrt((px - ex)**2 + (py - ey)**2)
            dist_m = dist_px / ppm
            
            # Visualizing the distance
            color = (0, 0, 255) if dist_m < danger_dist else (0, 255, 0)
            cv2.line(annotated_frame, (int(px), int(py)), (int(ex), int(ey)), color, 2)
            cv2.putText(annotated_frame, f"{dist_m:.2f}m", (int((px+ex)/2), int((py+ey)/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if dist_m < danger_dist:
                cv2.putText(annotated_frame, "!!! DANGER: NEAR EDGE !!!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
    return annotated_frame

# --- MAIN UI LOGIC ---
if model is None:
    st.error("Model file not found. Please check the path.")
else:
    if input_type == "Image":
        img_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            processed_img = process_frame(image, model, conf_threshold, ppm, danger_zone)
            st.image(processed_img, channels="BGR", use_container_width=True)

    else:
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for CPU speed if needed
                frame = cv2.resize(frame, (640, 480))
                processed_frame = process_frame(frame, model, conf_threshold, ppm, danger_zone)
                st_frame.image(processed_frame, channels="BGR", use_container_width=True)
                
            cap.release()
            os.remove(tfile.name)  