# Smart Fall Detection & Emergency Alert System

## Overview
A real-time deep learning based system that detects human falls using computer vision and automatically sends emergency email alerts.

This project is designed for elderly care, hospitals, and surveillance systems to improve safety monitoring.

---

## Features
- Real-time human fall detection using camera/video input
- AI-based pose and motion analysis
- Automatic email alert system on fall detection
- Works with both live webcam and recorded video
- Lightweight and easy to deploy

---

## Tech Stack
Python, OpenCV, YOLOv8, PyTorch, NumPy

---

## System Architecture
Camera Input → Frame Processing → YOLO/Pose Detection → Fall Classification → Email Alert System

---

## Demo
A recorded demonstration of the working system is included in this repository.

📁 Demo video file: `Demo.mp4`

To view it:
- Open the repository on GitHub
- Click on `Demo.mp4`
- Download and play it locally

---

## Project Files
- app.py → Main application
- email_alert.py → Email notification system
- model_loader.py → Model loading module
- pose_utils.py → Pose estimation utilities
- best_fall_detector.pth → Trained deep learning model
- yolov8n.pt → YOLOv8 weights

---

## How to Run
```bash
pip install -r requirements.txt
python app.py
