#Pose Utilities Module
#Handles YOLO person detection and MediaPipe pose extraction

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import torch

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize YOLO
yolo_model = None


def initialize_yolo(model_path='yolov8n.pt'):
    #Initialize YOLO model for person detection
    global yolo_model
    try:
        yolo_model = YOLO(model_path)
        print(f"✅ YOLO model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Error loading YOLO: {e}")
        raise


def detect_person(image):
    #Detect person in image using YOLO
    
    #Returns:
    #bbox: [x1, y1, x2, y2] or None if no person detected means bounding box coordinates
    #cropped_image: Cropped person region or original image 
    
    global yolo_model
    
    if yolo_model is None:
        initialize_yolo()
    
    # Run YOLO detection
    results = yolo_model(image, verbose=False)
    
    # Filter for person class (class 0 in COCO)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls[0]) == 0:  # Person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = image[y1:y2, x1:x2]
                return [x1, y1, x2, y2], cropped
    
    # No person detected, return original image
    return None, image


def extract_pose_landmarks(image):
    #Extract pose landmarks using MediaPipe
    
    #Returns:
    #landmarks: Flattened array of 33 landmarks (x, y, z) = 99 features
    #annotated_image: Image with pose overlay
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        
        # Process
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Draw landmarks on image
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            return np.array(landmarks, dtype=np.float32), annotated_image
        else:
            # No pose detected, return zeros
            return np.zeros(99, dtype=np.float32), image


def process_frame_for_inference(image):
    #Complete pipeline: YOLO detection + Pose extraction
    
    #Returns:
    #pose_features: (99,) numpy array
    #annotated_image: Image with detections drawn
    
    # Detect person (for bounding box only)
    bbox, _ = detect_person(image)
    
    # Extract pose from FULL FRAME (not cropped) to avoid zoom issues
    pose_features, annotated = extract_pose_landmarks(image)
    
    # Draw bounding box on the full annotated image if person detected
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated, "Person", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return pose_features, annotated


class TemporalBuffer:
    #Maintains a sliding window of pose sequences for temporal modeling
    #Required input shape: (30, 99)
    
    def __init__(self, seq_length=30):
        self.seq_length = seq_length
        self.buffer = []
        
    def add_frame(self, pose_features):
        #Add new pose features to buffer
        #pose_features: (99,) numpy array
        self.buffer.append(pose_features)
        
        # Keep only last seq_length frames
        if len(self.buffer) > self.seq_length:
            self.buffer.pop(0)
    
    def get_sequence(self):
        #Get current sequence for model inference
        
        #Returns:
        #tensor: (1, 30, 99) PyTorch tensor or None if buffer not full
        if len(self.buffer) < self.seq_length:
            # Pad with zeros if not enough frames
            padding_needed = self.seq_length - len(self.buffer)
            padded_buffer = [np.zeros(99, dtype=np.float32)] * padding_needed + self.buffer
            sequence = np.stack(padded_buffer)
        else:
            sequence = np.stack(self.buffer[-self.seq_length:])
        
        # Convert to tensor (1, 30, 99)
        return torch.from_numpy(sequence).unsqueeze(0).float()
    
    def is_ready(self):
        #Check if buffer has enough frames for inference
        return len(self.buffer) >= self.seq_length
    
    def reset(self):
        #Clear the buffer
        self.buffer = []


def draw_prediction_on_frame(frame, prediction, confidence):
    #Draw prediction text on frame
    #frame: OpenCV image
    #prediction: "Fall Detected" or "Normal Activity"
    #confidence: Confidence score (0-1)
    
    #Returns:
    #annotated_frame: Frame with prediction overlay
    
    annotated = frame.copy()
    
    # Color based on prediction
    if "Fall" in prediction:
        color = (0, 0, 255)  # Red
        bg_color = (0, 0, 200)
    else:
        color = (0, 255, 0)  # Green
        bg_color = (0, 200, 0)
    
    # Draw semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (400, 100), bg_color, -1)
    cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
    
    # Draw text
    cv2.putText(annotated, prediction, (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(annotated, f"Confidence: {confidence:.2%}", (20, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return annotated
