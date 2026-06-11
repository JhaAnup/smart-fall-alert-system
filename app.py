
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import time
from datetime import datetime
import tempfile
import os

# Import custom modules
from model_loader import load_model, predict
from pose_utils import (
    initialize_yolo, 
    process_frame_for_inference, 
    TemporalBuffer,
    draw_prediction_on_frame
)
from email_alert import email_alert


# Page Configuration
st.set_page_config(
    page_title="Fall Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .status-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .status-fall {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        animation: pulse 1s infinite;
    }
    .status-monitoring {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = 'cpu'
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.7
if 'email_configured' not in st.session_state:
    st.session_state.email_configured = False
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False


def load_system():
    """Initialize the fall detection system"""
    with st.spinner("🔄 Loading Fall Detection System..."):
        try:
            # Load model
            if st.session_state.model is None:
                st.session_state.model = load_model('best_fall_detector.pth', st.session_state.device)
            
            # Initialize YOLO
            initialize_yolo('yolov8n.pt')
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading system: {e}")
            return False


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🚨 Fall Detection System</div>', unsafe_allow_html=True)
    st.markdown("**Deep Learning-Powered Fall Detection using CNN-LSTM Model**")
    
    # Sidebar Navigation
    st.sidebar.title("📋 Navigation")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["🏠 Home", "📷 Image Detection", "🎥 Video Detection", "📹 Live Camera", "✉️ Email Settings"],
        index=0
    )
    
    # Sidebar Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Detection Settings")
    st.session_state.threshold = st.sidebar.slider(
        "Fall Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01,
        help="Higher threshold = more conservative detection"
    )
    
    # System Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    
    if st.sidebar.button("🔄 Load/Reload System"):
        if load_system():
            st.sidebar.success("✅ System Ready")
        else:
            st.sidebar.error("❌ System Error")
    
    model_status = "✅ Loaded" if st.session_state.model is not None else "❌ Not Loaded"
    email_status = "✅ Configured" if st.session_state.email_configured else "❌ Not Configured"
    
    st.sidebar.info(f"**Model:** {model_status}\n\n**Email:** {email_status}")
    
    # Main Content
    if mode == "🏠 Home":
        show_home()
    elif mode == "📷 Image Detection":
        show_image_detection()
    elif mode == "🎥 Video Detection":
        show_video_detection()
    elif mode == "📹 Live Camera":
        show_live_camera()
    elif mode == "✉️ Email Settings":
        show_email_settings()


def show_home():
    """Home page with system information"""
    
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🚨 Fall Detection System</h1>
        <p style="font-size: 1.3rem; color: #666; margin-bottom: 2rem;">
            AI-Powered Safety Monitoring using Deep Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards with Gradients
    st.markdown("""
    <style>
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 1rem;
        opacity: 0.9;
    }
    .feature-card-image {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .feature-card-video {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .feature-card-camera {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
    }
    .stat-item {
        text-align: center;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .tech-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .howto-step {
        background: white;
        padding: 1.5rem;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .howto-number {
        display: inline-block;
        width: 35px;
        height: 35px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 35px;
        font-weight: bold;
        margin-right: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Three Mode Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card feature-card-image">
            <div class="feature-icon">📷</div>
            <div>
                <div class="feature-title">Image Mode</div>
                <div class="feature-desc">Upload and analyze single images with instant fall detection</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card feature-card-video">
            <div class="feature-icon">🎥</div>
            <div>
                <div class="feature-title">Video Mode</div>
                <div class="feature-desc">Process video files frame-by-frame with automatic alerts</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card feature-card-camera">
            <div class="feature-icon">📹</div>
            <div>
                <div class="feature-title">Live Camera</div>
                <div class="feature-desc">Real-time monitoring with instant fall detection and alerts</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-number">99%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">30</div>
            <div class="stat-label">Frames Analyzed</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">< 100ms</div>
            <div class="stat-label">Detection Time</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">24/7</div>
            <div class="stat-label">Monitoring</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How It Works
    st.markdown("<h2 style='text-align: center; margin-top: 3rem; margin-bottom: 2rem;'>🎯 How It Works</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="howto-step">
        <span class="howto-number">1</span>
        <strong>Person Detection</strong> - YOLOv8 detects person in frame with high accuracy
    </div>
    <div class="howto-step">
        <span class="howto-number">2</span>
        <strong>Pose Extraction</strong> - MediaPipe extracts 33 keypoint landmarks from body
    </div>
    <div class="howto-step">
        <span class="howto-number">3</span>
        <strong>Temporal Analysis</strong> - CNN-LSTM analyzes 30-frame sequence for patterns
    </div>
    <div class="howto-step">
        <span class="howto-number">4</span>
        <strong>Fall Classification</strong> - Binary classification determines Fall vs Normal activity
    </div>
    <div class="howto-step">
        <span class="howto-number">5</span>
        <strong>Alert System</strong> - Automatic email notification sent on fall detection
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Technology Stack
    st.markdown("<h2 style='text-align: center; margin-top: 3rem; margin-bottom: 1.5rem;'>⚡ Technology Stack</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <span class="tech-badge">PyTorch</span>
        <span class="tech-badge">YOLOv8</span>
        <span class="tech-badge">MediaPipe</span>
        <span class="tech-badge">CNN-LSTM</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">OpenCV</span>
        <span class="tech-badge">IMVIA Dataset</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Start
    st.markdown("<h2 style='text-align: center; margin-top: 2rem; margin-bottom: 1.5rem;'>🚀 Quick Start</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
            <h3 style="margin-top: 0;">🎬 Get Started</h3>
            <ol style="margin: 0; padding-left: 1.5rem;">
                <li style="margin-bottom: 0.8rem;">Click <strong>🔄 Load/Reload System</strong> in sidebar</li>
                <li style="margin-bottom: 0.8rem;">Configure <strong>Email Settings</strong> (optional)</li>
                <li style="margin-bottom: 0.8rem;">Select your detection mode</li>
                <li style="margin-bottom: 0.8rem;">Upload media or start camera</li>
                <li>View predictions and alerts!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 2rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
            <h3 style="margin-top: 0;">💡 Pro Tips</h3>
            <ul style="margin: 0; padding-left: 1.5rem;">
                <li style="margin-bottom: 0.8rem;"><strong>Lighting:</strong> Ensure good lighting for best results</li>
                <li style="margin-bottom: 0.8rem;"><strong>Angle:</strong> Front-facing camera view works best</li>
                <li style="margin-bottom: 0.8rem;"><strong>Distance:</strong> Keep person clearly visible</li>
                <li style="margin-bottom: 0.8rem;"><strong>Speed:</strong> Adjust frame skip for video processing</li>
                <li><strong>Alerts:</strong> Configure email for automatic notifications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                border-radius: 15px;">
        <h3 style="margin-top: 0; color: #667eea;">Ready to Get Started?</h3>
        <p style="color: #666; margin-bottom: 1.5rem;">
            Choose a mode from the sidebar and start detecting falls with cutting-edge AI technology
        </p>
        <p style="font-size: 0.9rem; color: #999;">
            Deep Learning-Powered Fall Detection • CNN-LSTM Architecture • Real-time Processing
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_image_detection():
    """Image upload and detection mode"""
    st.header("📷 Image Detection Mode")
    
    # Ensure system is loaded
    if st.session_state.model is None:
        st.warning("⚠️ Please load the system first using the sidebar button")
        return
    
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing a person"
    )
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image)
        
        with st.spinner("🔍 Analyzing image..."):
            # Process image
            pose_features, annotated_image = process_frame_for_inference(image_bgr)
            
            # Create temporal buffer with single repeated frame
            # (Since we only have 1 frame, repeat it 30 times)
            pose_sequence = torch.from_numpy(np.tile(pose_features, (30, 1))).unsqueeze(0).float()
            
            # Make prediction
            result = predict(
                st.session_state.model,
                pose_sequence,
                st.session_state.device,
                st.session_state.threshold
            )
            
            # Convert annotated image back to RGB
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("Pose Detection")
            st.image(annotated_rgb)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Detection Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "Fall" in result['prediction']:
                st.markdown(f'<div class="status-fall"><h3>🚨 {result["prediction"]}</h3></div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-normal"><h3>✅ {result["prediction"]}</h3></div>', 
                          unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence Score", f"{result['confidence']:.2%}")
        
        with col3:
            st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
        
        # Probability breakdown
        st.markdown("---")
        st.subheader("📈 Probability Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Normal Activity", f"{result['probabilities'][0]:.2%}")
        with col2:
            st.metric("Fall Detected", f"{result['probabilities'][1]:.2%}")
        
        # Send email alert if fall detected
        if "Fall" in result['prediction'] and st.session_state.email_configured:
            st.markdown("---")
            if st.button("📧 Send Email Alert"):
                with st.spinner("Sending alert..."):
                    success, message = email_alert.send_alert(
                        annotated_image,
                        "Image Upload",
                        result['confidence']
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)


def show_video_detection():
    """Video upload and detection mode"""
    st.header("🎥 Video Detection Mode")
    
    # Ensure system is loaded
    if st.session_state.model is None:
        st.warning("⚠️ Please load the system first using the sidebar button")
        return
    
    uploaded_file = st.file_uploader(
        "Upload a video (MP4, AVI, MOV)",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video file for fall detection"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()  # Close file handle before opening with cv2
        
        # Process video
        cap = cv2.VideoCapture(tfile.name)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        st.info(f"📹 Video Info: {total_frames} frames, {fps} FPS, {duration:.1f} seconds")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("▶️ Process Video", type="primary")
        with col2:
            frame_skip = st.number_input("Frame Skip (for speed)", min_value=1, max_value=10, value=2)
        
        if process_button:
            # Initialize buffer
            temporal_buffer = TemporalBuffer(seq_length=30)
            
            # Placeholders
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_count = 0
            fall_detected_frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for speed
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                pose_features, annotated_frame = process_frame_for_inference(frame)
                temporal_buffer.add_frame(pose_features)
                
                # Make prediction if buffer is ready
                if temporal_buffer.is_ready():
                    pose_sequence = temporal_buffer.get_sequence()
                    result = predict(
                        st.session_state.model,
                        pose_sequence,
                        st.session_state.device,
                        st.session_state.threshold
                    )
                    
                    # Draw prediction on frame
                    annotated_frame = draw_prediction_on_frame(
                        annotated_frame,
                        result['prediction'],
                        result['confidence']
                    )
                    
                    # Check for fall
                    if "Fall" in result['prediction']:
                        fall_detected_frames.append((frame_count, annotated_frame, result))
                        status_placeholder.markdown(
                            '<div class="status-fall">🚨 FALL DETECTED!</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        status_placeholder.markdown(
                            '<div class="status-normal">✅ Normal Activity</div>',
                            unsafe_allow_html=True
                        )
                
                # Display frame
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_rgb, channels="RGB")
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                
                frame_count += 1
            
            cap.release()
            
            # Safely delete temp file (Windows-compatible)
            try:
                os.unlink(tfile.name)
            except PermissionError:
                pass  # File will be cleaned up later by OS
            
            # Show results
            st.success(f"✅ Video processing complete! Processed {frame_count} frames")
            
            if fall_detected_frames:
                st.warning(f"⚠️ {len(fall_detected_frames)} fall(s) detected in video")
                
                # Show first detected fall
                st.subheader("📸 Fall Detection Details")
                frame_num, fall_frame, fall_result = fall_detected_frames[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    fall_rgb = cv2.cvtColor(fall_frame, cv2.COLOR_BGR2RGB)
                    st.image(fall_rgb, caption=f"Frame {frame_num}")
                
                with col2:
                    st.metric("Frame Number", frame_num)
                    st.metric("Confidence", f"{fall_result['confidence']:.2%}")
                    st.metric("Time", f"{frame_num/fps:.1f}s")
                
                # Send email alert
                if st.session_state.email_configured:
                    if st.button("📧 Send Email Alert"):
                        with st.spinner("Sending alert..."):
                            success, message = email_alert.send_alert(
                                fall_frame,
                                "Video Upload",
                                fall_result['confidence']
                            )
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
            else:
                st.success("✅ No falls detected in the video")


def show_live_camera():
    """Live camera detection mode"""
    st.header("📹 Live Camera Mode")
    
    # Ensure system is loaded
    if st.session_state.model is None:
        st.warning("⚠️ Please load the system first using the sidebar button")
        return
    
    st.markdown("""
    **Live monitoring mode with real-time fall detection**
    - Automatically detects falls in real-time
    - Sends email alerts immediately upon detection
    - Press Stop Camera to end monitoring
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start Camera", type="primary", disabled=st.session_state.camera_running):
            st.session_state.camera_running = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_running):
            st.session_state.camera_running = False
            st.rerun()
    
    if st.session_state.camera_running:
        # Camera feed
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera. Please check your camera connection.")
            st.session_state.camera_running = False
            return
        
        # Initialize buffer
        temporal_buffer = TemporalBuffer(seq_length=30)
        
        status_placeholder.markdown(
            '<div class="status-monitoring">🔴 MONITORING...</div>',
            unsafe_allow_html=True
        )
        
        fall_count = 0
        frame_count = 0
        last_alert_time = 0
        alert_cooldown = 10  # seconds between alerts
        
        try:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera")
                    break
                
                # Process every 2nd frame for speed
                if frame_count % 2 == 0:
                    # Process frame
                    pose_features, annotated_frame = process_frame_for_inference(frame)
                    temporal_buffer.add_frame(pose_features)
                    
                    # Make prediction if buffer is ready
                    if temporal_buffer.is_ready():
                        pose_sequence = temporal_buffer.get_sequence()
                        result = predict(
                            st.session_state.model,
                            pose_sequence,
                            st.session_state.device,
                            st.session_state.threshold
                        )
                        
                        # Draw prediction
                        annotated_frame = draw_prediction_on_frame(
                            annotated_frame,
                            result['prediction'],
                            result['confidence']
                        )
                        
                        # Check for fall
                        current_time = time.time()
                        if "Fall" in result['prediction']:
                            fall_count += 1
                            status_placeholder.markdown(
                                '<div class="status-fall">🚨 FALL DETECTED!</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Send email alert (with cooldown)
                            if st.session_state.email_configured and (current_time - last_alert_time) > alert_cooldown:
                                email_alert.send_alert(
                                    annotated_frame,
                                    "Live Camera",
                                    result['confidence']
                                )
                                last_alert_time = current_time
                        else:
                            status_placeholder.markdown(
                                '<div class="status-normal">✅ Normal Activity</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Display metrics
                        col1, col2, col3 = metrics_placeholder.columns(3)
                        col1.metric("Confidence", f"{result['confidence']:.2%}")
                        col2.metric("Falls Detected", fall_count)
                        col3.metric("Frames", frame_count)
                    
                    # Display frame
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_rgb, channels="RGB")
                
                frame_count += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.03)
                
        finally:
            cap.release()
            st.session_state.camera_running = False
    
    else:
        st.info("📷 Camera is not running. Click 'Start Camera' to begin monitoring.")


def show_email_settings():
    """Email configuration page"""
    st.header("✉️ Email Alert Settings")
    
    st.markdown("""
    Configure email settings to receive automatic alerts when a fall is detected.
    
    **⚠️ Important: Gmail App Password Required**
    
    You must use an App Password (not your regular Gmail password):
    1. Enable 2-factor authentication on your Gmail account
    2. Go to: https://myaccount.google.com/apppasswords
    3. Generate an app password for "Mail"
    4. Use the 16-character app password below
    """)
    
    st.markdown("---")
    
    with st.form("email_config_form"):
        sender_email = st.text_input(
            "Sender Email (Gmail)",
            placeholder="your-email@gmail.com",
            help="Your Gmail address"
        )
        
        sender_password = st.text_input(
            "App Password",
            type="password",
            placeholder="16-character app password",
            help="Gmail App Password (not your regular password)"
        )
        
        receiver_email = st.text_input(
            "Receiver Email",
            placeholder="alert-recipient@example.com",
            help="Email address to receive fall detection alerts"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            submit_button = st.form_submit_button("💾 Save Configuration", type="primary")
        
        if submit_button:
            if sender_email and sender_password and receiver_email:
                email_alert.configure(sender_email, sender_password, receiver_email)
                st.session_state.email_configured = True
                st.success("✅ Email configuration saved successfully!")
            else:
                st.error("❌ Please fill in all fields")
    
    # Test email button
    if st.session_state.email_configured:
        st.markdown("---")
        st.subheader("🧪 Test Email Configuration")
        
        if st.button("📧 Send Test Email"):
            with st.spinner("Sending test email..."):
                success, message = email_alert.send_test_email()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        st.info("✅ Email alerts are configured and ready")
    else:
        st.warning("⚠️ Email alerts not configured. Fill in the form above to enable alerts.")


if __name__ == "__main__":
    main()
