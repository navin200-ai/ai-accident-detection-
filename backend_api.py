"""
AI Roadside Accident Detector - FastAPI REST Backend
Provides RESTful APIs for frame-by-frame inference, emergency dispatching, incident logging, and system health.
"""

import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from accident_detector_opencv import AccidentDetector, VEHICLE_CLASSES
from alert_simulator import get_incident_history, record_incident, send_email_alert, send_twilio_alert

app = FastAPI(
    title="AI Roadside Accident Detector API",
    description="Real-time Deep Learning API for Traffic Anomaly & Collision Detection",
    version="2.0.0"
)

# Enable CORS for cross-origin frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Detector Instance
detector = AccidentDetector(model_weights="yolov8n.pt", conf_threshold=0.30, accident_threshold=0.55)

class ManualDispatchModel(BaseModel):
    location: str = "Highway NH-44, KM 128.4"
    severity: str = "CRITICAL"
    confidence: float = 0.92
    description: str = "High-velocity collision reported at CCTV Sector 4"
    vehicles: List[str] = ["Car", "Truck"]

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI Roadside Accident Detector Backend API",
        "version": "2.0.0",
        "endpoints": [
            "/api/health",
            "/api/detect_frame",
            "/api/incidents",
            "/api/dispatch_alert",
            "/api/demo_videos"
        ]
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model": "YOLOv8 Nano (yolov8n.pt)",
        "model_loaded": detector.model is not None,
        "classes_monitored": list(VEHICLE_CLASSES.values()),
        "incident_count": len(get_incident_history())
    }

@app.get("/api/incidents")
def get_incidents():
    return {
        "count": len(get_incident_history()),
        "incidents": get_incident_history()
    }

@app.post("/api/dispatch_alert")
def manual_dispatch(payload: ManualDispatchModel):
    incident = record_incident(
        info=payload.description,
        location=payload.location,
        confidence=payload.confidence,
        severity=payload.severity,
        vehicles=payload.vehicles
    )
    send_email_alert(payload.description, incident)
    send_twilio_alert(payload.description, incident)
    return {
        "status": "success",
        "message": "Emergency dispatch transmitted to EMS, Traffic Police, and Highway Patrol.",
        "incident": incident
    }

@app.get("/api/demo_videos")
def list_demo_videos():
    demo_dir = "sample_videos"
    if not os.path.exists(demo_dir):
        return {"videos": []}
    files = [f for f in os.listdir(demo_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    return {"videos": files}

@app.post("/api/detect_frame")
async def detect_frame(file: UploadFile = File(...), location: Optional[str] = Form("CCTV Cam-01")):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data uploaded")

        annotated_frame, stats = detector.process_frame(frame, location=location)

        # Encode annotated image to JPEG base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "accident_detected": stats["accident_detected"],
            "accident_score": stats["accident_score"],
            "vehicle_counts": stats["vehicle_counts"],
            "total_objects": stats["total_objects"],
            "details": stats["details"],
            "alert_dispatched": stats["alert_dispatched"],
            "annotated_image_base64": f"data:image/jpeg;base64,{base64_image}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
