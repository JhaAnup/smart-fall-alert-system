## 🚨 Real-Time AI Fall Detection System

A computer vision based safety system that detects human falls in real time and automatically triggers emergency alerts using YOLOv8 and PyTorch.

---

## 🧠 Problem Statement

Falls are one of the leading causes of injury among elderly individuals and patients in care environments. Manual monitoring is unreliable and slow in real-world scenarios.

---

## ⚡ Solution

This system automates fall detection using deep learning and sends instant email alerts when a fall is detected, improving response time and safety monitoring.

---

## ✨ Features

- Real-time fall detection using webcam or video
- AI-based pose and motion analysis
- Automatic email alert system
- Works with live and recorded video
- Lightweight and deployable

---

## 🧰 Tech Stack

Python, OpenCV, YOLOv8, PyTorch, NumPy

---

## 🏗 System Flow

Camera Input → Frame Processing → YOLOv8 Detection → Fall Classification → Email Alert

---

## 🎥 Demo

A working demonstration of the system is included in this repository.

📁 Demo video: `Demo.mp4`

To view:
1. Open repository
2. Click on `Demo.mp4`
3. Download and play locally

---

## ▶ How to Run

```bash
pip install -r requirements.txt
python app.py
