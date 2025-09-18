import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import threading
import time
import base64
from config import COUNT_LINE_POSITION, MIN_WIDTH, MIN_HEIGHT, LINE_OFFSET
from config import LINE_COLOR, CROSS_LINE_COLOR, TEXT_COLOR, MOG_HISTORY, MOG_THRESHOLD
from utils import center_handle, draw_text, draw_count_line

app = Flask(__name__)
CORS(app)

# Global variables for video processing
current_frame = None
vehicle_count = 0
is_processing = False
tracking_data = {
    'tracked_vehicles': {},
    'next_vehicle_id': 0,
    'vehicle_count': 0
}

def generate_demo_frames():
    """Generate synthetic frames for demonstration when no camera is available"""
    global current_frame, is_processing, tracking_data
    
    # Create a simple demo animation
    width, height = 1020, 600
    roi_height = 400
    
    frame_count = 0
    demo_vehicles = []
    
    while is_processing:
        # Create a blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        roi = frame[200:600, :]
        
        # Add some moving rectangles to simulate vehicles
        frame_count += 1
        
        # Create demo vehicles that move across the frame
        if frame_count % 60 == 0:  # Add new vehicle every 60 frames
            demo_vehicles.append({
                'x': 0,
                'y': np.random.randint(50, roi_height - 100),
                'w': np.random.randint(80, 120),
                'h': np.random.randint(40, 60),
                'speed': np.random.randint(3, 8),
                'id': tracking_data['next_vehicle_id'],
                'crossed': False
            })
            tracking_data['next_vehicle_id'] += 1
        
        # Update and draw vehicles
        vehicles_to_remove = []
        for i, vehicle in enumerate(demo_vehicles):
            vehicle['x'] += vehicle['speed']
            
            # Remove vehicles that have moved off screen
            if vehicle['x'] > width:
                vehicles_to_remove.append(i)
                continue
            
            # Draw vehicle rectangle
            cv2.rectangle(roi, (vehicle['x'], vehicle['y']), 
                         (vehicle['x'] + vehicle['w'], vehicle['y'] + vehicle['h']), 
                         (255, 255, 0), 2)
            
            # Draw center point
            cx = vehicle['x'] + vehicle['w'] // 2
            cy = vehicle['y'] + vehicle['h'] // 2
            cv2.circle(roi, (cx, cy), 4, (0, 0, 255), -1)
            
            # Check if vehicle crosses the counting line
            if not vehicle['crossed'] and cy > COUNT_LINE_POSITION:
                vehicle['crossed'] = True
                tracking_data['vehicle_count'] += 1
                cv2.line(roi, (0, COUNT_LINE_POSITION), (roi.shape[1], COUNT_LINE_POSITION), CROSS_LINE_COLOR, 3)
                print(f"Vehicle {vehicle['id']} counted! Total: {tracking_data['vehicle_count']}")
        
        # Remove vehicles that have moved off screen
        for i in reversed(vehicles_to_remove):
            del demo_vehicles[i]
        
        # Draw counting line
        draw_count_line(roi, COUNT_LINE_POSITION, LINE_COLOR)
        
        # Display count
        draw_text(roi, f"Total Vehicles: {tracking_data['vehicle_count']}", 
                 pos=(10, 50), color=TEXT_COLOR, size=1)
        
        # Add demo text
        draw_text(roi, "DEMO MODE - Simulated Vehicle Detection", 
                 pos=(10, roi.shape[0] - 30), color=(255, 255, 255), size=1)
        
        current_frame = frame
        time.sleep(0.033)  # ~30 FPS

def try_camera_capture():
    """Process video from vehicles.mp4 file, fallback to demo mode"""
    global current_frame, is_processing, tracking_data
    
    # Try to open video file
    cap = cv2.VideoCapture('vehicles.mp4')
    
    if not cap.isOpened():
        print("Video file not available, switching to demo mode")
        generate_demo_frames()
        return
    
    print("Video file opened successfully")
    fgbg = cv2.createBackgroundSubtractorMOG2(history=MOG_HISTORY, varThreshold=MOG_THRESHOLD, detectShadows=False)
    
    while is_processing:
        ret, frame = cap.read()
        if not ret:
            print("End of video file reached, restarting video")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video from beginning
            continue
        
        # Process frame similar to original main.py
        frame = cv2.resize(frame, (1020, 600))
        roi = frame[200:600, :]
        
        # Background subtraction
        mask = fgbg.apply(roi)
        _, mask = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw counting line
        draw_count_line(roi, COUNT_LINE_POSITION, LINE_COLOR)
        
        current_frame_detections = []
        
        # Process contours and update tracking
        new_tracked_vehicles = {}
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if w >= MIN_WIDTH and h >= MIN_HEIGHT and area > 300:
                cx, cy = center_handle(x, y, w, h)
                current_frame_detections.append((cx, cy, x, y, w, h))
                cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.circle(roi, (cx, cy), 4, (0, 0, 255), -1)
                
                # Simple vehicle tracking and counting logic for camera mode
                vehicle_id = tracking_data['next_vehicle_id']
                new_tracked_vehicles[vehicle_id] = {
                    'center': (cx, cy),
                    'crossed': False,
                    'prev_y': cy
                }
                tracking_data['next_vehicle_id'] += 1
                
                # Check if vehicle crosses counting line (simplified for camera mode)
                if cy > COUNT_LINE_POSITION - 10 and cy < COUNT_LINE_POSITION + 10:
                    tracking_data['vehicle_count'] += 1
                    cv2.line(roi, (0, COUNT_LINE_POSITION), (roi.shape[1], COUNT_LINE_POSITION), CROSS_LINE_COLOR, 3)
                    print(f"Camera Vehicle {vehicle_id} counted! Total: {tracking_data['vehicle_count']}")
        
        # Display count
        draw_text(roi, f"Total Vehicles: {tracking_data['vehicle_count']}", 
                 pos=(10, 50), color=TEXT_COLOR, size=1)
        
        current_frame = frame
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()

def get_frame():
    """Get current frame as JPEG bytes"""
    global current_frame
    if current_frame is not None:
        ret, buffer = cv2.imencode('.jpg', current_frame)
        if ret:
            return buffer.tobytes()
    return None

def generate_frames():
    """Generator function for video streaming"""
    while True:
        frame_bytes = get_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify({
        'vehicle_count': tracking_data['vehicle_count'],
        'active_tracks': len(tracking_data['tracked_vehicles'])
    })

@app.route('/reset')
def reset():
    global tracking_data
    tracking_data = {
        'tracked_vehicles': {},
        'next_vehicle_id': 0,
        'vehicle_count': 0
    }
    return jsonify({'status': 'reset', 'vehicle_count': 0})

if __name__ == '__main__':
    is_processing = True
    
    # Start video processing in a separate thread
    video_thread = threading.Thread(target=try_camera_capture, daemon=True)
    video_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)