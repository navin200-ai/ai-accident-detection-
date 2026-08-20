"""
Generates synthetic sample video clips for demoing the AI Roadside Accident Detector.
Creates:
1. sample_videos/accident_demo.mp4 (Simulated intersection/highway collision)
2. sample_videos/traffic_normal.mp4 (Smooth multi-lane highway traffic)
"""

import os
import cv2
import numpy as np

def draw_road(frame, frame_idx):
    h, w = frame.shape[:2]
    # Sky / Horizon
    frame[:int(h*0.25), :] = [180, 150, 110]
    # Grass roadside
    frame[int(h*0.25):, :] = [45, 110, 40]
    
    # Perspective Road polygon
    road_pts = np.array([
        [int(w*0.38), int(h*0.25)],
        [int(w*0.62), int(h*0.25)],
        [int(w*0.95), h],
        [int(w*0.05), h]
    ], np.int32)
    cv2.fillPoly(frame, [road_pts], (60, 60, 65))
    
    # Road shoulder lines
    cv2.polylines(frame, [road_pts], isClosed=False, color=(200, 200, 200), thickness=4)
    
    # Animated Lane Dividers (Dashed lines moving down)
    center_top = (int(w*0.50), int(h*0.25))
    center_bottom = (int(w*0.50), h)
    
    num_dashes = 10
    offset = (frame_idx * 6) % 60
    for i in range(num_dashes):
        progress_1 = ((i * 60 + offset) % (h - int(h*0.25))) / (h - int(h*0.25))
        progress_2 = min(1.0, progress_1 + 0.05)
        
        y1 = int(h*0.25 + progress_1 * (h - h*0.25))
        y2 = int(h*0.25 + progress_2 * (h - h*0.25))
        
        x1 = int(center_top[0] + (center_bottom[0] - center_top[0]) * progress_1)
        x2 = int(center_top[0] + (center_bottom[0] - center_top[0]) * progress_2)
        
        thickness = max(2, int(6 * progress_1))
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness)

def draw_vehicle(frame, x, y, width, height, color, label="Car", angle=0):
    """Draws a vehicle with headlights, windshield, and roof."""
    rect = ((x, y), (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # Shadow
    shadow_box = box + np.array([4, 6])
    cv2.fillPoly(frame, [shadow_box], (30, 30, 30))
    
    # Body
    cv2.fillPoly(frame, [box], color)
    cv2.polylines(frame, [box], True, (20, 20, 20), 2)
    
    # Windshield / Roof detail
    inner_width = int(width * 0.7)
    inner_height = int(height * 0.5)
    inner_rect = ((x, y), (inner_width, inner_height), angle)
    inner_box = np.int32(cv2.boxPoints(inner_rect))
    cv2.fillPoly(frame, [inner_box], (40, 40, 50))
    
    # Headlights
    hl_color = (120, 255, 255)
    cv2.circle(frame, (int(x - width*0.3), int(y - height*0.4)), max(2, int(width*0.08)), hl_color, -1)
    cv2.circle(frame, (int(x + width*0.3), int(y - height*0.4)), max(2, int(width*0.08)), hl_color, -1)

def generate_normal_traffic_video(output_path, num_frames=120, fps=24):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Vehicles moving steadily
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        draw_road(frame, i)
        
        # Car 1 - Left lane
        prog1 = (i * 1.8) / num_frames
        y1 = int(h*0.35 + prog1 * (h*0.55))
        x1 = int(w*0.35 - (prog1 * 50))
        scale1 = 0.6 + 0.8 * prog1
        draw_vehicle(frame, x1, y1, int(45*scale1), int(80*scale1), (220, 50, 40), "Sedan")
        
        # Car 2 - Right lane
        prog2 = ((i + 30) * 1.5) % num_frames / num_frames
        y2 = int(h*0.35 + prog2 * (h*0.55))
        x2 = int(w*0.65 + (prog2 * 60))
        scale2 = 0.6 + 0.8 * prog2
        draw_vehicle(frame, x2, y2, int(50*scale2), int(90*scale2), (40, 160, 230), "SUV")
        
        # Car 3 - Distant truck
        prog3 = ((i + 70) * 1.2) % num_frames / num_frames
        y3 = int(h*0.30 + prog3 * (h*0.45))
        x3 = int(w*0.42)
        scale3 = 0.5 + 0.6 * prog3
        draw_vehicle(frame, x3, y3, int(55*scale3), int(110*scale3), (240, 220, 80), "Truck")
        
        # Overlay timestamp HUD
        cv2.putText(frame, "LIVE TRAFFIC CCTV - SECTOR 12 (NORMAL FLOW)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        out.write(frame)
        
    out.release()
    print(f"[SUCCESS] Created {output_path}")

def generate_accident_simulation_video(output_path, num_frames=140, fps=24):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    crash_frame = 60
    
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        draw_road(frame, i if i < crash_frame else crash_frame)
        
        if i < crash_frame:
            # Pre-crash: Blue car travels down right lane, Red car veers from left into center
            t = i / crash_frame
            # Blue car
            x_blue = int(w * 0.60 - t * 40)
            y_blue = int(h * 0.40 + t * 140)
            angle_blue = int(t * 15)
            scale_blue = 0.8 + t * 0.4
            
            # Red car swerves into blue car
            x_red = int(w * 0.35 + t * 130)
            y_red = int(h * 0.45 + t * 110)
            angle_red = int(t * -35)
            scale_red = 0.8 + t * 0.4
            
            draw_vehicle(frame, x_blue, y_blue, int(50*scale_blue), int(90*scale_blue), (40, 160, 230), "Sedan 1", angle_blue)
            draw_vehicle(frame, x_red, y_red, int(52*scale_red), int(95*scale_red), (220, 50, 40), "Sedan 2", angle_red)
        else:
            # Post-crash: Collision impact, severe overlap, rotation, smoke/debris
            post_t = (i - crash_frame)
            shake_x = np.random.randint(-4, 5) if post_t < 15 else 0
            shake_y = np.random.randint(-4, 5) if post_t < 15 else 0
            
            cx = int(w * 0.52) + shake_x
            cy = int(h * 0.68) + shake_y
            
            # Locked overlapping positions
            draw_vehicle(frame, cx - 20, cy, 65, 110, (40, 160, 230), "Sedan 1", 45 + min(25, post_t * 2))
            draw_vehicle(frame, cx + 25, cy + 10, 65, 115, (220, 50, 40), "Sedan 2", -55 - min(20, post_t * 2))
            
            # Crash skid marks
            cv2.line(frame, (cx - 100, cy - 40), (cx - 20, cy), (20, 20, 20), 6)
            cv2.line(frame, (cx + 80, cy - 60), (cx + 25, cy + 10), (20, 20, 20), 6)
            
            # Impact flash and particles
            if post_t < 12:
                flash_radius = int(30 + post_t * 6)
                cv2.circle(frame, (cx, cy), flash_radius, (0, 200, 255), 3)
                cv2.circle(frame, (cx, cy), max(5, flash_radius - 15), (0, 120, 255), -1)
            
            # Smoke billowing
            for s in range(5):
                smoke_rad = int(15 + ((post_t * 2 + s * 12) % 45))
                smoke_y = cy - int((post_t * 1.5 + s * 10) % 70)
                smoke_x = cx + int(np.sin(s + i*0.2) * 20)
                cv2.circle(frame, (smoke_x, smoke_y), smoke_rad, (160, 160, 160), -1)
                
            # Hazard blinkers
            if (i // 5) % 2 == 0:
                cv2.circle(frame, (cx - 45, cy + 40), 6, (0, 180, 255), -1)
                cv2.circle(frame, (cx + 50, cy + 40), 6, (0, 180, 255), -1)
                
            cv2.putText(frame, "CRITICAL IMPACT DETECTED", (cx - 140, cy - 80), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "LIVE TRAFFIC CCTV - JUNCTION HIGHWAY KM 128", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        out.write(frame)
        
    out.release()
    print(f"[SUCCESS] Created {output_path}")

if __name__ == "__main__":
    generate_normal_traffic_video("sample_videos/traffic_normal.mp4")
    generate_accident_simulation_video("sample_videos/accident_demo.mp4")
