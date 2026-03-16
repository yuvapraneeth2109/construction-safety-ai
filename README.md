# Construction Safety Monitoring using Computer Vision

This project implements a computer vision system that automatically detects safety violations at construction sites.  
The goal is to assist site supervisors by continuously monitoring workers and identifying situations where safety equipment such as helmets or reflective vests are missing.

The system processes images or video streams and detects workers in real time, highlighting potential violations.

---
##Sample video

Watch full demo here : https://youtu.be/Ga0TQUKUMHI?si=k1Zj_5BJ0tSqDKXW

## Motivation

Construction sites are high-risk environments where safety compliance is critical.  
Manual supervision alone is often insufficient, especially on large sites or during continuous operations.

This project explores how computer vision can assist safety monitoring by:

- detecting workers in the scene
- identifying required safety equipment
- flagging missing protective gear
- providing visual alerts for violations

The system is intended as an assistive tool rather than a replacement for human supervision.

---

## System Overview

The system processes video frames through a detection pipeline.

1. **Frame Acquisition**  
   Images are captured from a camera feed or uploaded video.

2. **Object Detection**  
   A trained model identifies workers and safety equipment such as helmets and safety vests.

3. **Safety Compliance Check**  
   The system evaluates whether detected workers are wearing the required equipment.

4. **Violation Detection**  
   Missing equipment triggers a violation label.

5. **Visualization**  
   Bounding boxes and labels are drawn on the frame to highlight detected objects.

---

## Architecture

Input Image / Video  
↓  
Frame Preprocessing  
↓  
Object Detection Model  
↓  
Worker + Safety Equipment Detection  
↓  
Safety Rule Evaluation  
↓  
Annotated Output Frame

---

## Technology Stack

**Programming Language**
- Python

**Computer Vision**
- OpenCV

**Object Detection Model**
- YOLO (You Only Look Once)

**Deep Learning Framework**
- PyTorch

**Interface**
- Streamlit

---

## Detection Categories

The system detects the following classes:

- Worker
- Helmet
- Safety Vest
- No Helmet
- No Vest

These labels allow the system to identify both compliant and non-compliant workers.

---

## Model Training

The detection model is trained using labeled images of construction environments.

Training involves:

- annotating objects in images
- training a YOLO detection model
- validating detection accuracy
- exporting trained weights for inference

The trained model is then used for real-time detection.

---

## Running the Application

Install dependencies:
