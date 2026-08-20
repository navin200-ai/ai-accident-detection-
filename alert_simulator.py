"""
AI Roadside Accident Detector - Alert & Dispatch Engine
Handles Console alerts, Email (SMTP), Twilio (SMS), Webhooks, and structured Emergency Dispatch.
"""

import time
import json
import smtplib
from email.message import EmailMessage
from datetime import datetime

# Global incident log stored in-memory
INCIDENT_HISTORY = []

def format_incident_payload(info: str, location: str = "Cam-01 Highway NH-44, KM 128.4", severity: str = "HIGH", confidence: float = 0.88, vehicles_involved: list = None):
    return {
        "incident_id": f"ACC-{int(time.time())}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": location,
        "severity": severity,
        "confidence": round(confidence, 3),
        "vehicles_involved": vehicles_involved or ["car", "car"],
        "status": "DISPATCHED",
        "description": info,
        "emergency_services": ["EMS / Ambulance (108)", "Traffic Police (100)", "Highway Patrol / Towing"]
    }

def send_console_alert(info: str, payload: dict = None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=======================================================")
    print(f"🚨 [EMERGENCY ALERT TRIGGERED] {ts}")
    print(f"Details: {info}")
    if payload:
        print(f"Incident ID: {payload.get('incident_id')} | Severity: {payload.get('severity')} | Confidence: {payload.get('confidence')}")
        print(f"Location: {payload.get('location')}")
        print(f"Dispatched: {', '.join(payload.get('emergency_services', []))}")
    print(f"=======================================================\n")

def send_email_alert(info: str, payload: dict = None, smtp_config: dict = None):
    if smtp_config is None or not smtp_config.get("host"):
        print(f"[ALERT - EMAIL SIMULATOR] Dispatch notification formatted for {smtp_config.get('to_addr', 'emergency-dispatch@traffic.gov') if smtp_config else 'emergency-dispatch@traffic.gov'}:")
        print(f"   Subject: 🚨 URGENT: Roadside Accident Detected - {payload.get('incident_id') if payload else 'ACC-ALERT'}")
        print(f"   Body: {info}")
        return True

    msg = EmailMessage()
    body_content = f"""
    *** CRITICAL ROAD SAFETY ALERT ***
    
    An accident event has been detected by the AI Roadside Accident Detector.
    
    Incident Details:
    -----------------
    Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Location: {payload.get('location', 'Unknown') if payload else 'CCTV Roadway'}
    Severity: {payload.get('severity', 'HIGH') if payload else 'HIGH'}
    Confidence: {payload.get('confidence', 'N/A') if payload else 'N/A'}
    Description: {info}
    
    Emergency Units Notified:
    - Ambulance (EMS)
    - Traffic Control Division
    - Highway Patrol
    """
    msg.set_content(body_content)
    msg['Subject'] = f"🚨 URGENT: Roadside Collision Alert - {payload.get('incident_id') if payload else 'LIVE'}"
    msg['From'] = smtp_config.get('from_addr', 'ai-detector@smartcity.gov')
    msg['To'] = smtp_config.get('to_addr', 'dispatch@traffic.gov')

    try:
        with smtplib.SMTP(smtp_config['host'], smtp_config.get('port', 587), timeout=5) as s:
            s.starttls()
            s.login(smtp_config['username'], smtp_config['password'])
            s.send_message(msg)
        print('[ALERT - EMAIL] Email alert successfully sent via SMTP.')
        return True
    except Exception as e:
        print(f'[ALERT - EMAIL ERROR] Failed to send email: {e}')
        return False

def send_twilio_alert(info: str, payload: dict = None, twilio_config: dict = None):
    if twilio_config is None or not twilio_config.get("account_sid"):
        target_phone = twilio_config.get("to_phone", "+1-800-EMERGENCY") if twilio_config else "+1-800-EMERGENCY"
        print(f"[ALERT - TWILIO SMS SIMULATOR] SMS queued for {target_phone}:")
        print(f"   '🚨 AI Collision Alert: {info}. Location: {payload.get('location') if payload else 'Cam-01 NH-44'}. Dispatch ID: {payload.get('incident_id') if payload else 'ACC-99'}'")
        return True

    try:
        from twilio.rest import Client
        client = Client(twilio_config['account_sid'], twilio_config['auth_token'])
        message = client.messages.create(
            body=f"🚨 AI Collision Alert: {info}. Location: {payload.get('location', 'Highway Sector 4')}",
            from_=twilio_config['from_phone'],
            to=twilio_config['to_phone']
        )
        print(f"[ALERT - TWILIO] SMS sent with SID: {message.sid}")
        return True
    except Exception as e:
        print(f"[ALERT - TWILIO ERROR] Twilio dispatch failed: {e}")
        return False

def record_incident(info: str, location: str = "Highway Cam-01", confidence: float = 0.85, severity: str = "HIGH", vehicles: list = None, snapshot_path: str = None):
    payload = format_incident_payload(info, location, severity, confidence, vehicles)
    if snapshot_path:
        payload["snapshot_path"] = snapshot_path
    INCIDENT_HISTORY.insert(0, payload)
    send_console_alert(info, payload)
    return payload

def get_incident_history():
    return INCIDENT_HISTORY
