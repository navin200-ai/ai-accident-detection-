# AI Roadside Accident Detector 🚗⚡

An intelligent, real-time roadside accident detection and emergency dispatch platform powered by YOLOv8 deep learning vision, multi-factor collision heuristics, telemetry analytics, and automated multi-channel alerts.

---

## 🚀 Running Services

The application is fully running and accessible:

- **Frontend & Web Command Center**: [http://localhost:8501](http://localhost:8501) (Streamlit)
- **REST API Backend**: [http://127.0.0.1:8095](http://127.0.0.1:8095) (FastAPI)
- **API Documentation (Swagger UI)**: [http://127.0.0.1:8095/docs](http://127.0.0.1:8095/docs)

---

## 🌟 Key Features

1. **Computer Vision & Collision Engine**:
   - Real-time YOLOv8 vehicle & pedestrian detection (Cars, Trucks, Buses, Motorcycles, Bicycles, Pedestrians).
   - Multi-factor collision scoring: Intersection-over-Union (IoU) overlap ratio, sudden speed drop, pedestrian proximity hazard, and vehicle rollover/obstruction heuristics.
   - Dynamic HUD overlay with real-time risk gauges, color-coded bounding boxes, and collision impact markers.

2. **Full-Featured Web Command Center (`app.py`)**:
   - **🎬 Preloaded Demo Scenarios**: 1-click test between high-speed collision scenarios and smooth traffic flow.
   - **📁 Video File Upload**: Supports MP4, AVI, MOV, and MKV surveillance footage.
   - **📷 Live Webcam Surveillance**: Real-time traffic stream with collision scoring.
   - **🖼️ Single Frame Inspector**: Inspect single images with detailed bounding box breakdowns and telemetry.
   - **🔊 Emergency Siren**: Built-in HTML5 Web Audio API audio siren triggered on collision events.
   - **📋 Incident Blackbox**: Real-time incident logging, keyframe snapshot gallery, and 1-click CSV/JSON export.

3. **Autonomous Emergency Response & Alerting (`alert_simulator.py`)**:
   - Automated dispatch payload generation (Incident ID, GPS location, severity, vehicle breakdown).
   - Automated routing to Emergency Medical Services (EMS 108/911), Traffic Police (100), and Highway Patrol / Towing.
   - SMS alert dispatching (Twilio) and SMTP Email dispatching.

4. **REST API Microservice (`backend_api.py`)**:
   - `GET /api/health` : System health & monitored classes.
   - `POST /api/detect_frame` : Upload frame for YOLO detection, collision metrics, and annotated base64 preview.
   - `GET /api/incidents` : Real-time dispatch and collision history.
   - `POST /api/dispatch_alert` : Manual or programmatic emergency alert broadcast.
   - `GET /api/demo_videos` : List built-in demo video clips.

---

## 🛠️ How to Launch Locally

### 1. Launch Streamlit Web UI:
```bash
streamlit run app.py
```

### 2. Launch FastAPI REST Backend:
```bash
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8095
```

### 3. Run Standalone OpenCV Window:
```bash
python accident_detector_opencv.py sample_videos/accident_demo.mp4
```

### 4. Regenerate Demo Simulation Videos:
```bash
python generate_demo_assets.py
```
