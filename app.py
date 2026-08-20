"""
AI Roadside Accident Detector - Interactive AI Command Center
A streamlined, high-performance UI/UX for AI-powered Roadside Accident Detection
featuring real-time YOLOv8 video analysis, single-image inspection, and automated emergency dispatching.
"""

import os
import cv2
import time
import json
import base64
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

from accident_detector_opencv import AccidentDetector, VEHICLE_CLASSES
from alert_simulator import (
    get_incident_history,
    record_incident,
    send_email_alert,
    send_twilio_alert
)

# ---------------------------------------------------------
# Page Configuration & Ultra-Modern Interactive Dark Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Roadside Accident Detector",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 10% 10%, #0f172a 0%, #020617 100%);
    color: #f8fafc;
}

/* Top Navbar Banner */
.nav-banner {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 20px 28px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.nav-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f43f5e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.nav-subtitle {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 4px;
}

/* Interactive Status Badges */
.status-pill-safe {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    padding: 7px 16px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.status-pill-danger {
    background: rgba(239, 68, 68, 0.18);
    border: 1px solid rgba(239, 68, 68, 0.6);
    color: #f87171;
    padding: 7px 16px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    animation: danger-pulse 1.5s infinite;
}

@keyframes danger-pulse {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* Interactive Metric Card */
.stat-card {
    background: rgba(30, 41, 59, 0.55);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.35);
}

.stat-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
}
.stat-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}
.stat-sub {
    font-size: 0.75rem;
    color: #38bdf8;
    margin-top: 2px;
}

/* Alert Dispatch Box */
.dispatch-banner {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
    border-left: 5px solid #ef4444;
    border-top: 1px solid rgba(239, 68, 68, 0.3);
    border-right: 1px solid rgba(239, 68, 68, 0.2);
    border-bottom: 1px solid rgba(239, 68, 68, 0.2);
    padding: 16px 20px;
    border-radius: 12px;
    margin: 14px 0;
}

/* Clean Upload Area Highlight */
.upload-card {
    background: rgba(30, 41, 59, 0.35);
    border: 2px dashed rgba(56, 189, 248, 0.3);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    transition: border-color 0.25s ease;
}
.upload-card:hover {
    border-color: rgba(56, 189, 248, 0.7);
}

/* Custom interactive buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Web Audio API Siren Synthesizer
# ---------------------------------------------------------
def trigger_siren_js():
    siren_code = """
    <script>
    (function() {
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            const ctx = new AudioCtx();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sawtooth';
            const now = ctx.currentTime;
            osc.frequency.setValueAtTime(850, now);
            osc.frequency.setValueAtTime(600, now + 0.25);
            osc.frequency.setValueAtTime(850, now + 0.5);
            osc.frequency.setValueAtTime(600, now + 0.75);
            osc.frequency.setValueAtTime(850, now + 1.0);
            gain.gain.setValueAtTime(0.25, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 1.25);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 1.3);
        } catch(e) {}
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
        alert_cooldown=6
    )

if "latest_stats" not in st.session_state:
    st.session_state.latest_stats = {
        "accident_detected": False,
        "accident_score": 0.0,
        "total_objects": 0,
        "vehicle_counts": {"Car": 0, "Truck": 0, "Bus": 0, "Motorcycle": 0, "Person": 0},
        "alert_dispatched": False,
        "details": []
    }

if "snapshots" not in st.session_state:
    st.session_state.snapshots = []

# Ensure sample assets exist
if not os.path.exists("sample_videos/accident_demo.mp4") or not os.path.exists("sample_images/accident_scene.jpg"):
    from generate_demo_assets import generate_accident_simulation_video, generate_normal_traffic_video
    generate_accident_simulation_video("sample_videos/accident_demo.mp4")
    generate_normal_traffic_video("sample_videos/traffic_normal.mp4")
    os.makedirs("sample_images", exist_ok=True)
    cap = cv2.VideoCapture("sample_videos/accident_demo.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 75)
    ret, f1 = cap.read()
    if ret: cv2.imwrite("sample_images/accident_scene.jpg", f1)
    cap.release()

# ---------------------------------------------------------
# Sidebar Controls & Parameters
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car-crash.png", width=64)
    st.title("AI Detection Settings")
    st.caption("AI Roadside Accident Detector v2.5")
    st.markdown("---")

    conf_thresh = st.slider(
        "🎯 YOLO Object Confidence",
        0.10, 0.90, 0.30, 0.05,
        help="Confidence cutoff for identifying vehicles and pedestrians."
    )
    accident_thresh = st.slider(
        "🚨 Accident Sensitivity (IoU)",
        0.20, 0.85, 0.50, 0.02,
        help="Sensitivity for collision overlap and abrupt stoppage detection."
    )
    cooldown = st.slider("⏳ Alert Cooldown (sec)", 2, 20, 6, 1)

    st.session_state.detector.conf_threshold = conf_thresh
    st.session_state.detector.accident_threshold = accident_thresh
    st.session_state.detector.alert_cooldown = cooldown

    st.markdown("---")
    st.subheader("📍 Deployment Info")
    cam_location = st.text_input("Surveillance Location", "Highway NH-44, KM 128.4")
    enable_audio = st.checkbox("🔊 Web Audio Emergency Siren", value=True)
    enable_sms = st.checkbox("📱 Twilio SMS Notification", value=True)
    enable_email = st.checkbox("📧 SMTP Email Notification", value=True)

    st.markdown("---")
    if st.button("🔴 Test Emergency Broadcast", use_container_width=True):
        payload = record_incident(
            info="MANUAL TEST DRILL: Emergency test triggered from operator console.",
            location=cam_location,
            confidence=0.98,
            severity="TEST DRILL",
            vehicles=["Control Center", "Emergency Dispatch"]
        )
        if enable_email:
            send_email_alert("Emergency Test Drill", payload)
        if enable_sms:
            send_twilio_alert("Emergency Test Drill", payload)
        if enable_audio:
            trigger_siren_js()
        st.success("Test Emergency Signal Transmitted!")

# ---------------------------------------------------------
# Top Navigation Header Banner
# ---------------------------------------------------------
is_accident = st.session_state.latest_stats["accident_detected"]
st.markdown(
    f"""
    <div class="nav-banner">
        <div>
            <h1 class="nav-title">AI Roadside Accident Detector 🚗⚡</h1>
            <div class="nav-subtitle">
                Autonomous Vision Collision Detection & Real-Time Emergency Dispatcher
            </div>
        </div>
        <div>
            <div class="{'status-pill-danger' if is_accident else 'status-pill-safe'}">
                {'🚨 CRITICAL: ACCIDENT DETECTED' if is_accident else '🟢 ROADWAY STATUS: SECURE & NORMAL'}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Real-Time Telemetry Stats Row
# ---------------------------------------------------------
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    risk_score = int(st.session_state.latest_stats["accident_score"] * 100)
    risk_color = "#ef4444" if risk_score >= int(accident_thresh*100) else ("#f59e0b" if risk_score > 25 else "#10b981")
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Live Collision Risk</div>
            <div class="stat-val" style="color: {risk_color};">{risk_score}%</div>
            <div class="stat-sub">Threshold: {int(accident_thresh*100)}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_s2:
    veh_total = sum(st.session_state.latest_stats["vehicle_counts"].values())
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Active Entities Tracked</div>
            <div class="stat-val" style="color: #38bdf8;">{veh_total}</div>
            <div class="stat-sub">Cars, Trucks, Bikes, Pedestrians</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_s3:
    incidents = get_incident_history()
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Incidents Logged</div>
            <div class="stat-val" style="color: #a78bfa;">{len(incidents)}</div>
            <div class="stat-sub">Automated Dispatches</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_s4:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Vision Model</div>
            <div class="stat-val" style="color: #34d399;">YOLOv8</div>
            <div class="stat-sub">Real-Time GPU / CPU Engine</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Clean Dedicated Mode: Upload Video vs Upload Image
# ---------------------------------------------------------
app_mode = st.radio(
    "Choose Input Mode:",
    ["📹 Upload & Analyze Video", "🖼️ Upload & Analyze Image", "📋 Incident History & Blackbox"],
    horizontal=True
)

# ---------------------------------------------------------
# MODE 1: Upload Video
# ---------------------------------------------------------
if app_mode == "📹 Upload & Analyze Video":
    st.subheader("📹 Video Surveillance & Collision Analysis")
    
    col_up, col_demo = st.columns([2, 1])
    with col_up:
        uploaded_video = st.file_uploader(
            "Upload Roadway / Dashcam Video (.mp4, .avi, .mov, .mkv)",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_file_uploader"
        )
    with col_demo:
        st.markdown("**Or Test with Sample Clips:**")
        sample_choice = st.selectbox(
            "Select Preloaded Demo:",
            ["🚨 Intersection Crash Simulation (accident_demo.mp4)", "🟢 Normal Traffic Flow (traffic_normal.mp4)"]
        )

    active_video_path = None
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        active_video_path = tfile.name
    elif sample_choice:
        if "accident_demo" in sample_choice:
            active_video_path = "sample_videos/accident_demo.mp4"
        else:
            active_video_path = "sample_videos/traffic_normal.mp4"

    if active_video_path:
        col_run_btn, _ = st.columns([1, 4])
        with col_run_btn:
            start_video = st.button("🚀 Run AI Video Inference", type="primary", use_container_width=True)

        video_frame_placeholder = st.empty()
        alert_box_placeholder = st.empty()
        chart_box_placeholder = st.empty()

        if start_video:
            cap = cv2.VideoCapture(active_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            frame_delay = 1.0 / fps

            risk_timeline = []
            frame_timeline = []
            f_num = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                f_num += 1

                # Resize if high-res
                h, w = frame.shape[:2]
                if max(h, w) > 960:
                    scale = 960 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                annotated_frame, stats = st.session_state.detector.process_frame(frame, location=cam_location)
                st.session_state.latest_stats = stats

                risk_timeline.append(stats["accident_score"] * 100)
                frame_timeline.append(f_num)

                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                if stats["alert_dispatched"]:
                    if enable_audio:
                        trigger_siren_js()
                    
                    st.session_state.snapshots.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "score": f"{int(stats['accident_score']*100)}%",
                        "frame": rgb_frame
                    })

                    alert_box_placeholder.markdown(
                        f"""
                        <div class="dispatch-banner">
                            <h3 style="color: #ef4444; margin:0 0 6px 0;">🚨 CRITICAL COLLISION DETECTED</h3>
                            <div><strong>Incident ID:</strong> {stats['incident_payload']['incident_id'] if stats['incident_payload'] else 'ACC-ALERT'}</div>
                            <div><strong>Location:</strong> {cam_location} | <strong>Severity:</strong> HIGH</div>
                            <div><strong>Dispatched:</strong> Ambulance (108), Traffic Police (100), Highway Rescue</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if f_num % 6 == 0:
                    df_chart = pd.DataFrame({"Frame": frame_timeline[-30:], "Risk Score (%)": risk_timeline[-30:]})
                    chart_box_placeholder.line_chart(df_chart.set_index("Frame"), color="#ef4444")

                time.sleep(frame_delay * 0.35)

            cap.release()
            st.success("✅ Video Analysis Complete!")

# ---------------------------------------------------------
# MODE 2: Upload Image
# ---------------------------------------------------------
elif app_mode == "🖼️ Upload & Analyze Image":
    st.subheader("🖼️ Single Image Inspection & Object Breakdown")

    col_img_up, col_img_sample = st.columns([2, 1])
    with col_img_up:
        uploaded_img = st.file_uploader(
            "Upload Roadway Image (.jpg, .png, .jpeg, .webp)",
            type=["jpg", "png", "jpeg", "webp"],
            key="img_file_uploader"
        )
    with col_img_sample:
        st.markdown("**Or Test with Sample Image:**")
        sample_img = st.selectbox(
            "Choose Sample Image:",
            ["🚨 Critical Crash Scene (accident_scene.jpg)", "🟢 Normal Traffic (normal_traffic.jpg)"]
        )

    img_input = None
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif sample_img:
        sample_path = "sample_images/accident_scene.jpg" if "accident_scene" in sample_img else "sample_images/normal_traffic.jpg"
        if os.path.exists(sample_path):
            img_input = cv2.imread(sample_path)

    if img_input is not None:
        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("### 📸 Original Image")
            st.image(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB), use_container_width=True)

        annotated_img, stats = st.session_state.detector.process_frame(img_input, location=cam_location)
        st.session_state.latest_stats = stats

        with col_after:
            st.markdown("### 🔍 AI Detection & Collision Overlay")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Inspection Breakdown")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Accident Detected", "YES 🚨" if stats["accident_detected"] else "NO 🟢")
        with c2:
            st.metric("Collision Risk Score", f"{int(stats['accident_score']*100)}%")
        with c3:
            st.metric("Total Detected Entities", stats["total_objects"])

        st.markdown("**Monitored Entity Counts:**")
        st.json(stats["vehicle_counts"])

        if stats["details"]:
            st.markdown("**⚡ Collision Analysis Insights:**")
            for d in stats["details"]:
                st.info(f"• {d}")

# ---------------------------------------------------------
# MODE 3: Incident History & Blackbox
# ---------------------------------------------------------
elif app_mode == "📋 Incident History & Blackbox":
    st.subheader("📋 Blackbox Incident Log & Emergency Dispatch History")
    incidents = get_incident_history()

    if not incidents:
        st.info("No accident incidents recorded yet. Trigger a video/image test to populate logs.")
    else:
        df_log = pd.DataFrame(incidents)
        st.dataframe(df_log, use_container_width=True)

        col_c, col_j = st.columns(2)
        with col_c:
            st.download_button(
                "📥 Download Incident Log (CSV)",
                data=df_log.to_csv(index=False).encode('utf-8'),
                file_name="accident_incidents.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_j:
            st.download_button(
                "📥 Download Incident Log (JSON)",
                data=json.dumps(incidents, indent=2).encode('utf-8'),
                file_name="accident_incidents.json",
                mime="application/json",
                use_container_width=True
            )

    if st.session_state.snapshots:
        st.markdown("---")
        st.subheader("🖼️ High-Risk Accident Keyframes")
        cols = st.columns(min(3, len(st.session_state.snapshots)))
        for idx, sn in enumerate(reversed(st.session_state.snapshots[-6:])):
            with cols[idx % 3]:
                st.image(sn["frame"], caption=f"Time: {sn['time']} | Risk: {sn['score']}")
