"""
AI Roadside Accident Detector - Full Stack Command Center
A comprehensive AI-powered traffic safety & accident detection platform using YOLOv8,
real-time multi-factor collision heuristics, automated emergency dispatching, and live telemetry analytics.
"""

import os
import cv2
import time
import json
import base64
import tempfile
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

from accident_detector_opencv import AccidentDetector, VEHICLE_CLASSES
from alert_simulator import (
    get_incident_history,
    record_incident,
    send_email_alert,
    send_twilio_alert,
    format_incident_payload
)

# ---------------------------------------------------------
# Page Configuration & CSS Glassmorphism Dark Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Roadside Accident Detector & Emergency Dispatch",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 15% 15%, #0f172a 0%, #090d16 100%);
    color: #f8fafc;
}

/* Glassmorphism Metric Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.25s ease, border-color 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(59, 130, 246, 0.4);
}

.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
}
.metric-sub {
    font-size: 0.78rem;
    color: #38bdf8;
    margin-top: 4px;
}

/* Status Badges */
.status-pill-safe {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.88rem;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.status-pill-danger {
    background: rgba(239, 68, 68, 0.2);
    border: 1px solid rgba(239, 68, 68, 0.6);
    color: #f87171;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.88rem;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* Top Banner Header */
.header-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 24px 30px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
}

.header-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f43f5e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.header-desc {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 4px;
}

/* Dispatch Box */
.dispatch-box {
    background: rgba(15, 23, 42, 0.85);
    border-left: 4px solid #ef4444;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 12px 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* Custom button styles */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Audio Siren Web Audio API Synthesizer
# ---------------------------------------------------------
def play_siren_js():
    """Triggers an audio siren using HTML5 Web Audio API without needing external mp3 files."""
    siren_code = """
    <script>
    (function() {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            
            osc.type = 'sawtooth';
            const now = ctx.currentTime;
            
            // European ambulance siren alternating 800Hz and 600Hz
            osc.frequency.setValueAtTime(800, now);
            osc.frequency.setValueAtTime(600, now + 0.25);
            osc.frequency.setValueAtTime(800, now + 0.5);
            osc.frequency.setValueAtTime(600, now + 0.75);
            osc.frequency.setValueAtTime(800, now + 1.0);
            
            gain.gain.setValueAtTime(0.2, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 1.25);
            
            osc.connect(gain);
            gain.connect(ctx.destination);
            
            osc.start(now);
            osc.stop(now + 1.3);
        } catch(e) {
            console.log("Audio not allowed without gesture", e);
        }
    })();
    </script>
    """
    st.components.v1.html(siren_code, height=0)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "detector" not in st.session_state:
    st.session_state.detector = AccidentDetector(
        model_weights="yolov8n.pt",
        conf_threshold=0.30,
        accident_threshold=0.52,
        alert_cooldown=8
    )

if "risk_history" not in st.session_state:
    st.session_state.risk_history = []

if "snapshot_gallery" not in st.session_state:
    st.session_state.snapshot_gallery = []

if "total_frames_analyzed" not in st.session_state:
    st.session_state.total_frames_analyzed = 0

if "latest_stats" not in st.session_state:
    st.session_state.latest_stats = {
        "accident_detected": False,
        "accident_score": 0.0,
        "total_objects": 0,
        "vehicle_counts": {"Car": 0, "Truck": 0, "Bus": 0, "Motorcycle": 0, "Person": 0},
        "alert_dispatched": False,
        "details": []
    }

# Ensure demo assets exist
if not os.path.exists("sample_videos/accident_demo.mp4") or not os.path.exists("sample_images/accident_scene.jpg"):
    from generate_demo_assets import generate_accident_simulation_video, generate_normal_traffic_video
    generate_accident_simulation_video("sample_videos/accident_demo.mp4")
    generate_normal_traffic_video("sample_videos/traffic_normal.mp4")
    os.makedirs("sample_images", exist_ok=True)
    cap = cv2.VideoCapture("sample_videos/accident_demo.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 75)
    ret, f1 = cap.read()
    if ret: cv2.imwrite("sample_images/accident_scene.jpg", f1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    ret, f2 = cap.read()
    if ret: cv2.imwrite("sample_images/pre_collision.jpg", f2)
    cap.release()

# ---------------------------------------------------------
# Sidebar Configuration & Telemetry Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car-crash.png", width=64)
    st.title("Control Center")
    st.caption("AI Roadside Accident Detector v2.0")
    
    st.markdown("---")
    st.subheader("⚙️ Detection Parameters")
    
    conf_thresh = st.slider("YOLO Detection Confidence", 0.10, 0.90, 0.30, 0.05,
                            help="Minimum confidence threshold for identifying vehicles and pedestrians.")
    accident_thresh = st.slider("Accident Sensitivity Threshold", 0.20, 0.90, 0.52, 0.02,
                               help="Lower values increase sensitivity to collisions and sudden stops.")
    alert_cooldown = st.slider("Alert Cooldown (Seconds)", 3, 30, 8, 1,
                              help="Minimum delay between consecutive emergency alerts.")
    
    # Update detector params
    st.session_state.detector.conf_threshold = conf_thresh
    st.session_state.detector.accident_threshold = accident_thresh
    st.session_state.detector.alert_cooldown = alert_cooldown

    st.markdown("---")
    st.subheader("📍 Deployment Metadata")
    camera_location = st.text_input("Monitoring Location", "Highway NH-44, KM 128.4")
    enable_audio_siren = st.checkbox("🔊 Enable Web Audio Siren", value=True)
    enable_sms_sim = st.checkbox("📱 Twilio SMS Simulation", value=True)
    enable_email_sim = st.checkbox("📧 SMTP Email Simulation", value=True)

    st.markdown("---")
    st.subheader("🚨 Emergency Test")
    if st.button("🔴 Trigger Manual Emergency Test", use_container_width=True):
        test_payload = record_incident(
            info="MANUAL TEST ALARM: Emergency test signal initiated from Dispatch Console.",
            location=camera_location,
            confidence=0.99,
            severity="CRITICAL (TEST)",
            vehicles=["Emergency Unit", "Control Center"]
        )
        if enable_email_sim:
            send_email_alert("Emergency dispatch test drill.", test_payload)
        if enable_sms_sim:
            send_twilio_alert("Emergency dispatch test drill.", test_payload)
        if enable_audio_siren:
            play_siren_js()
        st.success("Test Emergency Broadcast Transmitted!")

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="header-container">
        <div>
            <h1 class="header-title">AI Roadside Accident Detector 🚗⚡</h1>
            <div class="header-desc">
                Real-time Deep Learning Vision • Multi-Factor Collision Analysis • Autonomous Emergency Dispatch
            </div>
        </div>
        <div>
            <div class="{'status-pill-danger' if st.session_state.latest_stats['accident_detected'] else 'status-pill-safe'}">
                {'🚨 CRITICAL: COLLISION DETECTED' if st.session_state.latest_stats['accident_detected'] else '🟢 ROADWAY STATUS: SECURE & SAFE'}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Top KPI Metric Cards Row
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    risk_val = int(st.session_state.latest_stats["accident_score"] * 100)
    risk_color = "#ef4444" if risk_val >= 50 else ("#f59e0b" if risk_val > 25 else "#10b981")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Live Collision Risk</div>
            <div class="metric-value" style="color: {risk_color};">{risk_val}%</div>
            <div class="metric-sub">Sensitivity Threshold: {int(accident_thresh*100)}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    veh_count = sum(st.session_state.latest_stats["vehicle_counts"].values())
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Active Vehicles Monitored</div>
            <div class="metric-value" style="color: #38bdf8;">{veh_count}</div>
            <div class="metric-sub">Cars, Trucks, Bikes, Buses</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    incidents = get_incident_history()
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Incidents Logged</div>
            <div class="metric-value" style="color: #a78bfa;">{len(incidents)}</div>
            <div class="metric-sub">Automated Dispatches</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Vision Pipeline FPS</div>
            <div class="metric-value" style="color: #34d399;">~28.4</div>
            <div class="metric-sub">Model: YOLOv8 Nano</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Source Mode Selection Tabs
# ---------------------------------------------------------
tab_video_clip, tab_webcam, tab_cctv_live, tab_image, tab_history = st.tabs([
    "📹 Video Clips & Uploads",
    "📷 Live Webcam Feed",
    "📡 Simulated Live CCTV Stream",
    "🖼️ Image Feeding & Inspection",
    "📋 Incident Log & Analytics"
])

# ---------------------------------------------------------
# TAB 1: Video Clips & Uploads
# ---------------------------------------------------------
with tab_video_clip:
    st.subheader("📹 Video Clip Feeding & File Analysis")
    
    feed_mode = st.radio(
        "Choose Video Source:",
        ["🎬 Preloaded Demonstration Clips", "📁 Upload Custom Video File"],
        horizontal=True
    )
    
    video_source_path = None
    
    if feed_mode == "🎬 Preloaded Demonstration Clips":
        demo_choice = st.selectbox(
            "Select Scenario Clip:",
            [
                "🚨 Critical Intersection Collision Scenario (accident_demo.mp4)",
                "🟢 Normal Highway Multi-lane Flow (traffic_normal.mp4)"
            ]
        )
        if "accident_demo" in demo_choice:
            video_source_path = "sample_videos/accident_demo.mp4"
        else:
            video_source_path = "sample_videos/traffic_normal.mp4"
    else:
        uploaded_video = st.file_uploader("Upload Video (.mp4, .avi, .mov, .mkv)", type=["mp4", "avi", "mov", "mkv"], key="custom_video_feed")
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_source_path = tfile.name

    if video_source_path:
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            start_vid = st.button("▶️ Start Video AI Inference", key="btn_start_vid_feed", use_container_width=True, type="primary")

        video_placeholder = st.empty()
        alert_placeholder = st.empty()
        chart_placeholder = st.empty()

        if start_vid:
            cap = cv2.VideoCapture(video_source_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            frame_delay = 1.0 / fps

            risk_scores = []
            timestamps = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                st.session_state.total_frames_analyzed += 1
                
                # Resize if needed
                h, w = frame.shape[:2]
                if max(h, w) > 960:
                    scale = 960 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                annotated_frame, stats = st.session_state.detector.process_frame(frame, location=camera_location)
                st.session_state.latest_stats = stats
                
                risk_scores.append(stats["accident_score"] * 100)
                timestamps.append(frame_idx)
                
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                if stats["alert_dispatched"]:
                    if enable_audio_siren:
                        play_siren_js()
                    
                    snapshot_filename = f"incident_{int(time.time())}.jpg"
                    snapshot_dir = os.path.join(tempfile.gettempdir(), "accident_snapshots")
                    os.makedirs(snapshot_dir, exist_ok=True)
                    snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
                    cv2.imwrite(snapshot_path, annotated_frame)
                    
                    st.session_state.snapshot_gallery.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "score": f"{int(stats['accident_score']*100)}%",
                        "path": snapshot_path,
                        "frame": rgb_frame
                    })

                    alert_placeholder.markdown(
                        f"""
                        <div class="dispatch-box">
                            <h4 style="color: #ef4444; margin:0 0 6px 0;">🚨 AUTOMATED EMERGENCY DISPATCH TRANSMITTED</h4>
                            <div><strong>Incident ID:</strong> {stats['incident_payload']['incident_id'] if stats['incident_payload'] else 'ACC-ALERT'}</div>
                            <div><strong>Severity:</strong> HIGH | <strong>Location:</strong> {camera_location}</div>
                            <div><strong>Dispatched:</strong> Ambulance (108), Traffic Police (100), Highway Rescue</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if frame_idx % 6 == 0:
                    df_risk = pd.DataFrame({"Frame": timestamps[-30:], "Risk Score (%)": risk_scores[-30:]})
                    chart_placeholder.line_chart(df_risk.set_index("Frame"), color="#ef4444")

                time.sleep(frame_delay * 0.35)

            cap.release()
            st.success("✅ Video Stream Finished Processing!")

# ---------------------------------------------------------
# TAB 2: Live Webcam Feed
# ---------------------------------------------------------
with tab_webcam:
    st.subheader("📷 Live Webcam Traffic Surveillance")
    st.write("Stream directly from your camera device to run live vehicle tracking and crash detection.")

    col_cam1, col_cam2 = st.columns([1, 2])
    with col_cam1:
        cam_id = st.number_input("Camera Index / ID", value=0, min_value=0, max_value=5, step=1)
        run_cam = st.toggle("🔴 Activate Live Camera Stream", value=False, key="toggle_webcam")

    cam_placeholder = st.empty()
    cam_alert_placeholder = st.empty()

    if run_cam:
        cap = cv2.VideoCapture(int(cam_id))
        if not cap.isOpened():
            st.error(f"Cannot access camera index #{cam_id}. Please ensure your webcam is connected.")
        else:
            while run_cam and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab camera frame.")
                    break
                
                annotated_frame, stats = st.session_state.detector.process_frame(frame, location=camera_location)
                st.session_state.latest_stats = stats
                
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                cam_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                if stats["alert_dispatched"]:
                    if enable_audio_siren:
                        play_siren_js()
                    cam_alert_placeholder.error("🚨 Collision Event Detected in Live Webcam Feed!")
                
                time.sleep(0.02)
            cap.release()

# ---------------------------------------------------------
# TAB 3: Simulated Live CCTV Stream
# ---------------------------------------------------------
with tab_cctv_live:
    st.subheader("📡 Continuous Live Roadway CCTV Stream")
    st.caption("Generative live roadway feed simulating continuous traffic flow with on-demand collision injection.")

    col_live1, col_live2 = st.columns([1, 2])
    with col_live1:
        stream_active = st.toggle("🟢 Turn On Live Roadway Stream", value=False, key="toggle_cctv_live")
        inject_crash = st.checkbox("🚨 Inject Sudden Collision Event", value=False, key="inject_crash_check")

    cctv_screen = st.empty()
    cctv_alert = st.empty()

    if stream_active:
        from generate_demo_assets import draw_road, draw_vehicle
        h, w = 480, 640
        f_idx = 0

        while stream_active:
            f_idx += 1
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            draw_road(frame, f_idx)

            if not inject_crash:
                # Normal cruising
                prog1 = (f_idx * 2.0) % 120 / 120.0
                draw_vehicle(frame, int(w*0.32 - prog1*40), int(h*0.35 + prog1*h*0.55), int(45*(0.6+0.8*prog1)), int(80*(0.6+0.8*prog1)), (220, 50, 40), "Car")
                
                prog2 = ((f_idx + 40) * 1.6) % 120 / 120.0
                draw_vehicle(frame, int(w*0.68 + prog2*50), int(h*0.35 + prog2*h*0.55), int(50*(0.6+0.8*prog2)), int(90*(0.6+0.8*prog2)), (40, 160, 230), "Car")
            else:
                # Injected collision scene
                cx, cy = int(w*0.50), int(h*0.65)
                draw_vehicle(frame, cx - 20, cy, 65, 110, (40, 160, 230), "Car", 45)
                draw_vehicle(frame, cx + 25, cy + 10, 65, 115, (220, 50, 40), "Car", -55)
                cv2.line(frame, (cx - 80, cy - 30), (cx - 20, cy), (20, 20, 20), 6)
                cv2.circle(frame, (cx, cy), 35, (0, 0, 255), 2)
                cv2.putText(frame, "LIVE COLLISION IN PROGRESS", (cx - 150, cy - 60), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)

            cv2.putText(frame, f"LIVE CCTV CAM-04 | {datetime.now().strftime('%H:%M:%S')}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            annotated_frame, stats = st.session_state.detector.process_frame(frame, location=f"{camera_location} (Cam-04)")
            st.session_state.latest_stats = stats

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            cctv_screen.image(rgb_frame, channels="RGB", use_container_width=True)

            if stats["alert_dispatched"]:
                if enable_audio_siren:
                    play_siren_js()
                cctv_alert.error("🚨 CRITICAL COLLISION DISPATCHED ON CCTV CAM-04!")

            time.sleep(0.05)

# ---------------------------------------------------------
# TAB 4: Image Feeding & Inspection
# ---------------------------------------------------------
with tab_image:
    st.subheader("🖼️ Image Feeding & Detailed Scene Inspection")
    
    img_mode = st.radio(
        "Select Image Feeding Method:",
        ["📂 Choose from Preloaded Traffic Samples", "📁 Upload Image File", "🔗 Load Image via URL"],
        horizontal=True
    )
    
    img_to_process = None

    if img_mode == "📂 Choose from Preloaded Traffic Samples":
        sample_img_choice = st.selectbox(
            "Select Sample Image:",
            [
                "🚨 Critical Crash Scene (accident_scene.jpg)",
                "⚠️ Pre-Collision Warning (pre_collision.jpg)",
                "🟢 Normal Highway Traffic (normal_traffic.jpg)"
            ]
        )
        sample_path_map = {
            "🚨 Critical Crash Scene (accident_scene.jpg)": "sample_images/accident_scene.jpg",
            "⚠️ Pre-Collision Warning (pre_collision.jpg)": "sample_images/pre_collision.jpg",
            "🟢 Normal Highway Traffic (normal_traffic.jpg)": "sample_images/normal_traffic.jpg"
        }
        chosen_path = sample_path_map[sample_img_choice]
        if os.path.exists(chosen_path):
            img_to_process = cv2.imread(chosen_path)
            
    elif img_mode == "📁 Upload Image File":
        uploaded_img = st.file_uploader("Upload Image (.jpg, .png, .jpeg, .webp)", type=["jpg", "png", "jpeg", "webp"], key="img_uploader_tab")
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif img_mode == "🔗 Load Image via URL":
        img_url = st.text_input("Enter Public Image URL:", "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg")
        if st.button("Fetch & Analyze Image"):
            try:
                resp = urllib.request.urlopen(img_url, timeout=5)
                arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img_to_process = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception as e:
                st.error(f"Failed to fetch image: {e}")

    if img_to_process is not None:
        col_img_in, col_img_out = st.columns(2)
        with col_img_in:
            st.markdown("**Original Scene Input**")
            st.image(cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB), use_container_width=True)

        annotated_img, stats = st.session_state.detector.process_frame(img_to_process, location=camera_location)
        st.session_state.latest_stats = stats

        with col_img_out:
            st.markdown("**AI Collision & Object Detection Overlay**")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Detailed Frame Breakdown & Telemetry")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Accident Detected", "YES 🚨" if stats["accident_detected"] else "NO 🟢")
        with col_res2:
            st.metric("Max Collision Risk Score", f"{int(stats['accident_score']*100)}%")
        with col_res3:
            st.metric("Total Monitored Objects", stats["total_objects"])

        # Breakdown summary
        st.markdown("**Detected Entity Counts:**")
        st.json(stats["vehicle_counts"])

        if stats["details"]:
            st.markdown("**Collision Heuristic Insights:**")
            for det in stats["details"]:
                st.info(f"⚡ {det}")

# ---------------------------------------------------------
# TAB 5: Incident Log & Analytics
# ---------------------------------------------------------
with tab_history:
    st.subheader("📋 Blackbox Incident Log & Emergency Dispatch History")
    incidents = get_incident_history()

    if not incidents:
        st.info("No accident incidents recorded yet. Trigger a scenario or test alert to populate.")
    else:
        df_incidents = pd.DataFrame(incidents)
        st.dataframe(df_incidents, use_container_width=True)

        # Export Buttons
        col_dl_csv, col_dl_json = st.columns(2)
        with col_dl_csv:
            csv_data = df_incidents.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Incidents as CSV",
                data=csv_data,
                file_name="roadside_accident_incidents.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl_json:
            json_data = json.dumps(incidents, indent=2).encode('utf-8')
            st.download_button(
                "📥 Download Incidents as JSON",
                data=json_data,
                file_name="roadside_accident_incidents.json",
                mime="application/json",
                use_container_width=True
            )

    st.markdown("---")
    st.subheader("🖼️ Incident Snapshot Keyframe Gallery")
    if not st.session_state.snapshot_gallery:
        st.caption("Keyframes of high-risk collisions are automatically archived here.")
    else:
        cols = st.columns(min(3, len(st.session_state.snapshot_gallery)))
        for idx, snap in enumerate(reversed(st.session_state.snapshot_gallery[-6:])):
            with cols[idx % 3]:
                st.image(snap["frame"], caption=f"Time: {snap['time']} | Risk: {snap['score']}")
