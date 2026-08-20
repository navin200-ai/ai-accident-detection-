"""
AI Roadside Accident Detector - Core Vision & Collision Pipeline
Uses YOLOv8 for vehicle & pedestrian detection, multi-factor collision analysis,
HUD telemetry overlay, and automatic emergency alert triggers.
"""

import os
import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from alert_simulator import send_console_alert, send_email_alert, send_twilio_alert, record_incident

# Class IDs for traffic entities in COCO dataset
VEHICLE_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def calculate_iou(box1, box2):
    """Calculates Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0, 0.0
    
    # Also return overlap ratio relative to smaller box
    min_area = min(area1, area2)
    overlap_ratio = (inter_area / min_area) if min_area > 0 else 0.0
    
    return inter_area / union_area, overlap_ratio

class AccidentDetector:
    def __init__(self, model_weights="yolov8n.pt", conf_threshold=0.30, accident_threshold=0.55, alert_cooldown=10):
        self.model_weights = model_weights
        self.conf_threshold = conf_threshold
        self.accident_threshold = accident_threshold
        self.alert_cooldown = alert_cooldown
        
        print(f"[INFO] Initializing YOLOv8 model from '{model_weights}'...")
        self.model = YOLO(model_weights)
        print("[INFO] Model loaded successfully.")
        
        self.last_alert_time = 0
        self.prev_tracks = {} # track_id -> (centroid_x, centroid_y, speed)
        self.accident_history = []
        self.frame_count = 0

    def process_frame(self, frame, location="Highway NH-44 KM 128"):
        """
        Processes a single BGR image frame and returns:
        - annotated_frame: frame with HUD bounding boxes, telemetry and crash markers
        - stats: dict containing detection metrics, vehicle counts, accident probability, and alert status
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # Run inference
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        
        detected_objects = []
        counts = {"Car": 0, "Truck": 0, "Bus": 0, "Motorcycle": 0, "Person": 0, "Bicycle": 0}
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                
                label = VEHICLE_CLASSES.get(cls_id, None)
                if label:
                    counts[label] = counts.get(label, 0) + 1
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    detected_objects.append({
                        "id": i,
                        "class_id": cls_id,
                        "label": label,
                        "conf": conf,
                        "box": xyxy,
                        "area": area,
                        "centroid": (cx, cy)
                    })

        # Multi-factor collision analysis
        accident_detected = False
        max_accident_score = 0.0
        collision_pairs = []
        info_details = []

        num_objs = len(detected_objects)
        for i in range(num_objs):
            for j in range(i + 1, num_objs):
                obj1 = detected_objects[i]
                obj2 = detected_objects[j]
                
                iou, overlap_ratio = calculate_iou(obj1["box"], obj2["box"])
                
                # Proximity distance
                dx = obj1["centroid"][0] - obj2["centroid"][0]
                dy = obj1["centroid"][1] - obj2["centroid"][1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Check Vehicle vs Vehicle collision
                is_veh_1 = obj1["label"] in ["Car", "Truck", "Bus", "Motorcycle"]
                is_veh_2 = obj2["label"] in ["Car", "Truck", "Bus", "Motorcycle"]
                is_ped = (obj1["label"] == "Person" or obj2["label"] == "Person")
                
                score = 0.0
                if is_veh_1 and is_veh_2:
                    if overlap_ratio > 0.18:
                        score = min(1.0, 0.5 + overlap_ratio * 0.7)
                    elif overlap_ratio > 0.08:
                        score = 0.40 + overlap_ratio * 0.5
                elif is_ped and (is_veh_1 or is_veh_2):
                    if overlap_ratio > 0.10 or dist < 45:
                        score = 0.85
                        info_details.append(f"Pedestrian Hazard ({obj1['label']} - {obj2['label']})")
                
                if score > max_accident_score:
                    max_accident_score = score
                    
                if score >= self.accident_threshold:
                    accident_detected = True
                    collision_pairs.append((obj1, obj2, score))
                    info_details.append(f"Collision between {obj1['label']} and {obj2['label']} (IoU Overlap={overlap_ratio:.2f})")

        # Check for abnormal single large bounding box (e.g. overturned truck / wrecked car)
        for obj in detected_objects:
            if obj["label"] in ["Car", "Truck", "Bus"]:
                frame_area_ratio = obj["area"] / float(h * w)
                if frame_area_ratio > 0.35 and obj["conf"] > 0.65:
                    accident_detected = True
                    max_accident_score = max(max_accident_score, 0.82)
                    info_details.append(f"Overturned/Massive Obstruction: {obj['label']}")

        # Prepare HUD annotated image
        annotated = frame.copy()

        # Draw detected objects
        for obj in detected_objects:
            box = [int(v) for v in obj["box"]]
            is_involved = any(obj["id"] in [p[0]["id"], p[1]["id"]] for p in collision_pairs)
            
            if is_involved:
                color = (0, 0, 255) # Red for crash
                thickness = 3
            else:
                color = (0, 230, 100) # Bright green for normal
                thickness = 2
                
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, thickness)
            
            # Label banner
            caption = f"{obj['label']} {obj['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (box[0], box[1] - th - 6), (box[0] + tw + 6, box[1]), color, -1)
            cv2.putText(annotated, caption, (box[0] + 3, box[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if not is_involved else (255, 255, 255), 1, cv2.LINE_AA)

        # Draw collision markers and impact flash
        for obj1, obj2, sc in collision_pairs:
            cx = int((obj1["centroid"][0] + obj2["centroid"][0]) / 2)
            cy = int((obj1["centroid"][1] + obj2["centroid"][1]) / 2)
            
            # Impact circle
            pulse = int((self.frame_count * 4) % 30)
            cv2.circle(annotated, (cx, cy), 20 + pulse, (0, 0, 255), 2)
            cv2.circle(annotated, (cx, cy), 10, (0, 200, 255), -1)
            cv2.putText(annotated, f"IMPACT {int(sc*100)}%", (cx - 45, cy - 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

        # HUD Top Banner
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)
        
        # Status text & badge
        status_text = "🚨 COLLISION / ACCIDENT ALERT" if accident_detected else "🟢 NORMAL TRAFFIC FLOW"
        status_color = (50, 50, 255) if accident_detected else (50, 240, 100)
        cv2.putText(annotated, status_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, status_color, 2)
        
        risk_pct = int(max_accident_score * 100)
        risk_str = f"Risk Score: {risk_pct}%"
        cv2.putText(annotated, risk_str, (w - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Handle Alert Dispatch with Cooldown
        alert_dispatched = False
        incident_payload = None
        current_time = time.time()
        
        if accident_detected and (current_time - self.last_alert_time) > self.alert_cooldown:
            self.last_alert_time = current_time
            alert_dispatched = True
            
            summary_info = "; ".join(info_details) if info_details else "Sudden vehicle collision detected."
            vehicles_involved = [p[0]["label"] for p in collision_pairs] + [p[1]["label"] for p in collision_pairs]
            if not vehicles_involved:
                vehicles_involved = ["Vehicle", "Vehicle"]
                
            incident_payload = record_incident(
                info=summary_info,
                location=location,
                confidence=max_accident_score,
                severity="CRITICAL" if max_accident_score > 0.8 else "HIGH",
                vehicles=list(set(vehicles_involved))
            )
            
            # Simulated SMS & Email triggers
            send_email_alert(summary_info, incident_payload)
            send_twilio_alert(summary_info, incident_payload)

        stats = {
            "accident_detected": accident_detected,
            "accident_score": max_accident_score,
            "total_objects": num_objs,
            "vehicle_counts": counts,
            "alert_dispatched": alert_dispatched,
            "incident_payload": incident_payload,
            "details": info_details
        }
        
        return annotated, stats

def run_on_video(source=0, threshold=0.55):
    """Standalone OpenCV playback runner."""
    detector = AccidentDetector(accident_threshold=threshold)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    print("[INFO] Press 'q' to exit video playback.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated, stats = detector.process_frame(frame)
        cv2.imshow("AI Roadside Accident Detector 🚗 [Press Q to exit]", annotated)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "sample_videos/accident_demo.mp4"
    if os.path.exists(src) or src == "0":
        run_on_video(int(src) if src == "0" else src)
    else:
        run_on_video(0)
