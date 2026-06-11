## Deep Learning Powered Fall Detection & Emergency Alert System

A real-time computer vision system that detects human falls using YOLOv8 and PyTorch, and sends automatic emergency email alerts.

---

## Overview

This project uses deep learning-based pose and motion analysis to detect fall events from live camera or video input. When a fall is detected, the system triggers an automated email alert for safety response.

The system is designed for monitoring in elderly care, hospitals, and surveillance environments.

---

## Features

- Real-time fall detection from webcam or video
- AI-based pose and motion analysis
- Automatic email alert system on fall detection
- Supports both live and recorded input
- Lightweight and easy to run locally

---

## Tech Stack

Python, OpenCV, YOLOv8, PyTorch, NumPy

---

## System Flow

Camera Input → Frame Processing → YOLOv8 Pose Detection → Fall Classification → Email Alert

---

## Demo

A working demo of the system is included in this repository as:

Demo.mp4

Steps to view:
- Open repository
- Click Demo.mp4
- Download and play locally

---

## Project Structure

- app.py → Main execution file
- email_alert.py → Email notification module
- model_loader.py → Model loading logic
- pose_utils.py → Pose estimation utilities
- best_fall_detector.pth → Trained model weights
- yolov8n.pt → YOLOv8 pretrained weights

---

## How to Run

pip install -r requirements.txt

python app.py
