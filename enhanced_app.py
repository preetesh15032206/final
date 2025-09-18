import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
from werkzeug.utils import secure_filename
from config import COUNT_LINE_POSITION, MIN_WIDTH, MIN_HEIGHT, LINE_OFFSET
from config import LINE_COLOR, CROSS_LINE_COLOR, TEXT_COLOR, MOG_HISTORY, MOG_THRESHOLD
from utils import center_handle, draw_text, draw_count_line

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_VIDEOS_FOLDER = 'static/videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_VIDEOS_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global video processing data with thread safety
video_processing_data = {
    'current_video': None,
    'current_frame': None,
    'is_processing': False,
    'processing_video_id': None,
    'vehicle_count': 0,
    'processing_stats': {
        'total_detections': 0,
        'processing_time': 0,
        'fps': 0,
        'frame_count': 0
    }
}

# Individual lane frames for multi-view display
lane_frames = {
    'lane_1': None,
    'lane_2': None,
    'lane_3': None,
    'lane_4': None
}

# Multi-lane traffic management system with analytics
traffic_data = {
    'lanes': {
        'lane_1': {
            'name': 'Main Street North',
            'video_file': 'video1_cctv.mp4',
            'vehicle_count': 0,
            'signal_status': 'GREEN',
            'traffic_density': 0.0,
            'avg_speed': 0.0,
            'is_processing': False,
            'last_update': time.time(),
            'density_history': [],
            'signal_timer': 0,
            'manual_override': False,
            'total_vehicles_handled': 0,
            'peak_vehicles': 0,
            'avg_wait_time': 0,
            'throughput_rate': 0.0
        },
        'lane_2': {
            'name': 'Main Street South', 
            'video_file': 'video2_4k_traffic.mp4',
            'vehicle_count': 0,
            'signal_status': 'RED',
            'traffic_density': 0.0,
            'avg_speed': 0.0,
            'is_processing': False,
            'last_update': time.time(),
            'density_history': [],
            'signal_timer': 0,
            'manual_override': False,
            'total_vehicles_handled': 0,
            'peak_vehicles': 0,
            'avg_wait_time': 0,
            'throughput_rate': 0.0
        },
        'lane_3': {
            'name': 'Cross Street East',
            'video_file': 'video3_safety.mp4', 
            'vehicle_count': 0,
            'signal_status': 'RED',
            'traffic_density': 0.0,
            'avg_speed': 0.0,
            'is_processing': False,
            'last_update': time.time(),
            'density_history': [],
            'signal_timer': 0,
            'manual_override': False,
            'total_vehicles_handled': 0,
            'peak_vehicles': 0,
            'avg_wait_time': 0,
            'throughput_rate': 0.0
        },
        'lane_4': {
            'name': 'Cross Street West',
            'video_file': 'video4_ipcam.mp4',
            'vehicle_count': 0,
            'signal_status': 'RED', 
            'traffic_density': 0.0,
            'avg_speed': 0.0,
            'is_processing': False,
            'last_update': time.time(),
            'density_history': [],
            'signal_timer': 0,
            'manual_override': False,
            'total_vehicles_handled': 0,
            'peak_vehicles': 0,
            'avg_wait_time': 0,
            'throughput_rate': 0.0
        }
    },
    'system_config': {
        'auto_control': True,
        'min_green_time': 15,  # seconds
        'max_green_time': 60,  # seconds
        'yellow_time': 3,      # seconds
        'all_red_time': 2,     # seconds
        'density_threshold_low': 0.3,   # Updated for better thresholds
        'density_threshold_high': 0.7,
        'emergency_override': False
    },
    'analytics': {
        'total_vehicles_system': 0,
        'total_processing_cycles': 0,
        'system_efficiency': 0.0,
        'average_wait_time': 0.0,
        'daily_peak_hour': '00:00',
        'busiest_lane': 'lane_1',
        'start_time': time.time(),
        'hourly_stats': {},
        'lane_performance': {}
    },
    'signal_cycle': {
        'current_phase': 0,
        'phase_start_time': time.time(),
        'phases': [
            {'lanes': ['lane_1'], 'duration': 30},
            {'lanes': ['lane_2'], 'duration': 30}, 
            {'lanes': ['lane_3'], 'duration': 25},
            {'lanes': ['lane_4'], 'duration': 25}
        ]
    }
}

# Simple processing lock
processing_lock = threading.Lock()
traffic_lock = threading.Lock()

# Video database (in production, this would be a real database)
video_database = {
    'videos': [
        {
            'id': 1,
            'title': 'HD CCTV Traffic Footage',
            'filename': 'video1_cctv.mp4',
            'file_path': 'video1_cctv.mp4',
            'serving_url': '/static/videos/video1_cctv.mp4',
            'upload_date': '2025-09-18',
            'size': '45.2 MB',
            'duration': '2:34',
            'detections': 127,
            'views': 89,
            'likes': 23,
            'comments': [],
            'tags': ['cctv', 'traffic', 'hd'],
            'status': 'processed',
            'analytics': {
                'cars': 95,
                'trucks': 20,
                'motorcycles': 8,
                'buses': 4
            }
        },
        {
            'id': 2,
            'title': '4K Traffic Detection Video',
            'filename': 'video2_4k_traffic.mp4',
            'file_path': 'video2_4k_traffic.mp4',
            'serving_url': '/static/videos/video2_4k_traffic.mp4',
            'upload_date': '2025-09-18',
            'size': '128.7 MB',
            'duration': '3:12',
            'detections': 245,
            'views': 156,
            'likes': 42,
            'comments': [],
            'tags': ['4k', 'traffic', 'detection'],
            'status': 'processed',
            'analytics': {
                'cars': 180,
                'trucks': 35,
                'motorcycles': 25,
                'buses': 5
            }
        },
        {
            'id': 3,
            'title': 'Road Safety Analysis',
            'filename': 'video3_safety.mp4',
            'file_path': 'video3_safety.mp4',
            'serving_url': '/static/videos/video3_safety.mp4',
            'upload_date': '2025-09-18',
            'size': '67.8 MB',
            'duration': '4:21',
            'detections': 89,
            'views': 73,
            'likes': 18,
            'comments': [],
            'tags': ['safety', 'analysis', 'traffic'],
            'status': 'processed',
            'analytics': {
                'cars': 65,
                'trucks': 15,
                'motorcycles': 7,
                'buses': 2
            }
        },
        {
            'id': 4,
            'title': 'IP Camera Traffic Monitor',
            'filename': 'video4_ipcam.mp4',
            'file_path': 'video4_ipcam.mp4',
            'serving_url': '/static/videos/video4_ipcam.mp4',
            'upload_date': '2025-09-18',
            'size': '34.1 MB',
            'duration': '1:58',
            'detections': 98,
            'views': 112,
            'likes': 31,
            'comments': [],
            'tags': ['ipcam', 'monitor', 'traffic'],
            'status': 'processed',
            'analytics': {
                'cars': 78,
                'trucks': 12,
                'motorcycles': 6,
                'buses': 2
            }
        }
    ],
    'analytics': {
        'total_uploads': 4,
        'total_views': 430,
        'total_detections': 559,
        'total_likes': 114,
        'processing_hours': 12.5
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_traffic_density(detections_count, roi_area, time_window=5.0):
    """Calculate realistic traffic density based on vehicle detections"""
    if roi_area <= 0 or time_window <= 0 or detections_count < 0:
        return 0.0
    
    # Define realistic traffic thresholds
    light_traffic = 5    # Below 5 vehicles = light traffic
    moderate_traffic = 15  # 5-15 vehicles = moderate traffic
    heavy_traffic = 25   # 15-25 vehicles = heavy traffic
    # Above 25 vehicles = very heavy traffic
    
    # Calculate density on a 0.0-1.0 scale based on realistic thresholds
    if detections_count <= light_traffic:
        density = detections_count / light_traffic * 0.25  # 0.0-0.25 for light traffic
    elif detections_count <= moderate_traffic:
        density = 0.25 + ((detections_count - light_traffic) / (moderate_traffic - light_traffic)) * 0.35  # 0.25-0.6 for moderate
    elif detections_count <= heavy_traffic:
        density = 0.60 + ((detections_count - moderate_traffic) / (heavy_traffic - moderate_traffic)) * 0.25  # 0.6-0.85 for heavy
    else:
        density = 0.85 + min((detections_count - heavy_traffic) / 20, 0.15)  # 0.85-1.0 for very heavy
    
    return round(density, 3)

def analyze_lane_priority(traffic_data):
    """Analyze which lanes need priority based on vehicle count and traffic density"""
    lane_priorities = []
    
    for lane_id, lane_data in traffic_data['lanes'].items():
        if not lane_data['manual_override']:
            priority_score = 0.0
            
            # Primary priority based on actual vehicle count
            vehicle_count = lane_data['vehicle_count']
            density = lane_data['traffic_density']
            
            # Vehicle count weighted scoring (most important factor)
            if vehicle_count >= 25:  # Very heavy traffic
                priority_score += 150
            elif vehicle_count >= 15:  # Heavy traffic
                priority_score += 100
            elif vehicle_count >= 5:   # Moderate traffic
                priority_score += 60
            else:  # Light traffic
                priority_score += vehicle_count * 5
            
            # Density scoring (secondary factor)
            if density > traffic_data['system_config']['density_threshold_high']:
                priority_score += 40
            elif density > traffic_data['system_config']['density_threshold_low']:
                priority_score += 20
            
            # Consider density trend from history
            if len(lane_data['density_history']) >= 3:
                recent_densities = lane_data['density_history'][-3:]
                trend = (recent_densities[-1] - recent_densities[0]) / len(recent_densities)
                priority_score += trend * 30  # Weight trend more heavily
            
            # Consider wait time for fairness
            if lane_data['signal_status'] == 'RED':
                red_duration = time.time() - lane_data['signal_timer']
                if red_duration > 90:  # Been red for over 90 seconds
                    priority_score += 50
                elif red_duration > 45:  # Been red for over 45 seconds  
                    priority_score += 25
            
            lane_priorities.append({
                'lane_id': lane_id,
                'priority_score': priority_score,
                'density': density,
                'vehicle_count': vehicle_count,
                'current_signal': lane_data['signal_status']
            })
    
    # Sort by priority score (highest first)
    lane_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
    return lane_priorities

def intelligent_signal_control(traffic_data):
    """Implement intelligent traffic signal control based on lane comparison"""
    if not traffic_data['system_config']['auto_control']:
        return
    
    current_time = time.time()
    config = traffic_data['system_config']
    
    # Get lane priorities
    lane_priorities = analyze_lane_priority(traffic_data)
    
    if not lane_priorities:
        return
    
    # Find lanes that currently have green signals
    current_green_lanes = [lane_id for lane_id, lane_data in traffic_data['lanes'].items() 
                          if lane_data['signal_status'] == 'GREEN']
    
    # Check if current green lane should continue or switch
    if current_green_lanes:
        current_green_lane = current_green_lanes[0]  # Assume one green at a time
        current_lane_data = traffic_data['lanes'][current_green_lane]
        
        # Check minimum green time
        green_duration = current_time - current_lane_data['signal_timer']
        
        if green_duration >= config['min_green_time']:
            # Check if we should switch to higher priority lane
            highest_priority_lane = lane_priorities[0]['lane_id']
            
            # Switch if:
            # 1. Current lane has low density and another has high density
            # 2. Maximum green time exceeded
            # 3. Much higher priority lane is waiting
            
            should_switch = False
            
            if green_duration >= config['max_green_time']:
                should_switch = True
                print(f"Switching from {current_green_lane}: max green time exceeded")
            
            elif (highest_priority_lane != current_green_lane and 
                  lane_priorities[0]['priority_score'] > 100 and
                  lane_priorities[0]['vehicle_count'] > current_lane_data['vehicle_count'] + 5):
                should_switch = True  
                print(f"Switching from {current_green_lane} to {highest_priority_lane}: vehicle-count priority ({lane_priorities[0]['vehicle_count']} vs {current_lane_data['vehicle_count']})")
            
            if should_switch:
                # Start yellow phase for current green lane
                current_lane_data['signal_status'] = 'YELLOW'
                current_lane_data['signal_timer'] = current_time
                print(f"Lane {current_green_lane} -> YELLOW")
    
    else:
        # No green lanes, check for yellow to red transitions
        yellow_lanes = [lane_id for lane_id, lane_data in traffic_data['lanes'].items() 
                       if lane_data['signal_status'] == 'YELLOW']
        
        if yellow_lanes:
            for lane_id in yellow_lanes:
                lane_data = traffic_data['lanes'][lane_id]
                yellow_duration = current_time - lane_data['signal_timer']
                
                if yellow_duration >= config['yellow_time']:
                    # Switch to red
                    lane_data['signal_status'] = 'RED'
                    lane_data['signal_timer'] = current_time
                    print(f"Lane {lane_id} -> RED")
        
        else:
            # All lanes red, start all-red clearance then give green to highest priority
            all_red_lanes = [lane_id for lane_id, lane_data in traffic_data['lanes'].items() 
                           if lane_data['signal_status'] == 'RED']
            
            if len(all_red_lanes) == len(traffic_data['lanes']):
                # Check if all-red clearance time has passed
                oldest_red_time = min(lane_data['signal_timer'] for lane_data in traffic_data['lanes'].values())
                all_red_duration = current_time - oldest_red_time
                
                if all_red_duration >= config['all_red_time']:
                    # Give green to highest priority lane
                    highest_priority_lane = lane_priorities[0]['lane_id']
                    highest_lane_data = traffic_data['lanes'][highest_priority_lane]
                    
                    highest_lane_data['signal_status'] = 'GREEN'
                    highest_lane_data['signal_timer'] = current_time
                    print(f"Lane {highest_priority_lane} -> GREEN (priority: {lane_priorities[0]['priority_score']:.1f}, vehicles: {lane_priorities[0]['vehicle_count']})")

def process_lane_video(lane_id, video_path):
    """Process video for a specific lane and update traffic data"""
    global traffic_data
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video for lane {lane_id}: {video_path}")
            return False
            
        # Initialize background subtractor for this lane
        fgbg = cv2.createBackgroundSubtractorMOG2(history=MOG_HISTORY, varThreshold=MOG_THRESHOLD, detectShadows=False)
        
        frame_count = 0
        vehicle_detections = []
        start_time = time.time()
        roi_area = 1020 * 400  # Default ROI area (width * height)
        
        with traffic_lock:
            traffic_data['lanes'][lane_id]['is_processing'] = True
        
        while frame_count < 150:  # Process limited frames for real-time performance
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
                
            frame_count += 1
            frame = cv2.resize(frame, (1020, 600))
            roi = frame[200:600, :]
            roi_area = roi.shape[0] * roi.shape[1]
            
            # Background subtraction
            mask = fgbg.apply(roi)
            _, mask = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
            mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=3)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            current_detections = 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                # More sensitive detection parameters
                if w >= MIN_WIDTH//3 and h >= MIN_HEIGHT//3 and area > 100:
                    current_detections += 1
            
            vehicle_detections.append(current_detections)
            
            # Update processing frame for display
            draw_count_line(roi, COUNT_LINE_POSITION, LINE_COLOR)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                # Use same relaxed detection parameters for visualization
                if w >= MIN_WIDTH//3 and h >= MIN_HEIGHT//3 and area > 100:
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 0), 2)
            
            # Add lane identification and signal status
            with traffic_lock:
                lane_data = traffic_data['lanes'][lane_id]
                signal_color = lane_data['signal_status']
                signal_colors = {'GREEN': (0, 255, 0), 'YELLOW': (0, 255, 255), 'RED': (0, 0, 255)}
                
                cv2.rectangle(frame, (10, 10), (300, 80), signal_colors.get(signal_color, (128, 128, 128)), -1)
                cv2.putText(frame, f"{lane_data['name']}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, f"Signal: {signal_color}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, f"Density: {lane_data['traffic_density']:.2f}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, f"Vehicles: {current_detections}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Store frame for this lane
                lane_frames[lane_id] = frame.copy()
        
        processing_time = time.time() - start_time
        avg_detections = np.mean(vehicle_detections) if vehicle_detections else 0
        max_detections = max(vehicle_detections) if vehicle_detections else 0
        traffic_density = calculate_traffic_density(avg_detections, roi_area)
        
        # Update traffic data with analytics
        with traffic_lock:
            lane_data = traffic_data['lanes'][lane_id]
            current_vehicles = int(max_detections)
            
            # Update basic metrics
            lane_data['vehicle_count'] = current_vehicles
            lane_data['traffic_density'] = traffic_density
            lane_data['last_update'] = time.time()
            lane_data['is_processing'] = False
            
            # Update analytics
            lane_data['total_vehicles_handled'] += current_vehicles
            lane_data['peak_vehicles'] = max(lane_data['peak_vehicles'], current_vehicles)
            
            # Calculate throughput rate (vehicles per minute)
            lane_data['throughput_rate'] = (current_vehicles / (processing_time / 60)) if processing_time > 0 else 0
            
            # Update density history (keep last 10 readings)
            lane_data['density_history'].append(traffic_density)
            if len(lane_data['density_history']) > 10:
                lane_data['density_history'].pop(0)
                
            # Update system-wide analytics
            traffic_data['analytics']['total_vehicles_system'] += current_vehicles
            traffic_data['analytics']['total_processing_cycles'] += 1
            
            # Update busiest lane
            current_busiest = traffic_data['analytics']['busiest_lane']
            if (current_vehicles > traffic_data['lanes'][current_busiest]['vehicle_count'] or
                traffic_data['lanes'][lane_id]['total_vehicles_handled'] > 
                traffic_data['lanes'][current_busiest]['total_vehicles_handled']):
                traffic_data['analytics']['busiest_lane'] = lane_id
        
        cap.release()
        print(f"Lane {lane_id} processing complete. Avg detections: {avg_detections:.1f}, Max detections: {max_detections}, Density: {traffic_density:.3f}")
        return True
        
    except Exception as e:
        print(f"Error processing lane {lane_id}: {e}")
        with traffic_lock:
            traffic_data['lanes'][lane_id]['is_processing'] = False
        return False

def traffic_analysis_manager():
    """Main traffic analysis and signal control loop"""
    global traffic_data
    
    print("Traffic Analysis Manager started")
    
    while True:
        try:
            # Process each lane if not already processing
            for lane_id, lane_data in traffic_data['lanes'].items():
                if not lane_data['is_processing']:
                    # Check if video file exists
                    video_path = lane_data['video_file']
                    if os.path.exists(video_path):
                        print(f"Starting processing for {lane_id}: {video_path}")
                        # Start processing in separate thread
                        thread = threading.Thread(
                            target=process_lane_video, 
                            args=(lane_id, video_path), 
                            daemon=True
                        )
                        thread.start()
                    else:
                        print(f"Video file not found for {lane_id}: {video_path}")
                        # Create a placeholder frame for missing video
                        placeholder = create_placeholder_frame(lane_data['name'], lane_data['signal_status'])
                        lane_frames[lane_id] = placeholder
            
            # Run intelligent signal control
            intelligent_signal_control(traffic_data)
            
            # Wait before next analysis cycle
            time.sleep(8)  # Analysis every 8 seconds
            
        except Exception as e:
            print(f"Error in traffic analysis manager: {e}")
            time.sleep(10)

def create_placeholder_frame(lane_name, signal_status):
    """Create a placeholder frame when video is not available"""
    frame = np.zeros((600, 1020, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)  # Dark gray background
    
    # Add lane name
    cv2.putText(frame, lane_name, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, "Camera Offline", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 2)
    
    # Add signal status
    signal_colors = {'GREEN': (0, 255, 0), 'YELLOW': (0, 255, 255), 'RED': (0, 0, 255)}
    color = signal_colors.get(signal_status, (128, 128, 128))
    cv2.circle(frame, (500, 400), 80, color, -1)
    cv2.putText(frame, signal_status, (420, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    return frame

def process_video_for_detection(video_path, video_id):
    """Process video for vehicle detection and update analytics"""
    global video_processing_data
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with processing_lock:
                video_processing_data['is_processing'] = False
                video_processing_data['processing_video_id'] = None
            return False
        
        # Update processing status
        with processing_lock:
            video_processing_data['is_processing'] = True
            video_processing_data['current_video'] = video_path
            video_processing_data['processing_video_id'] = video_id
        
        # Initialize background subtractor
        fgbg = cv2.createBackgroundSubtractorMOG2(history=MOG_HISTORY, varThreshold=MOG_THRESHOLD, detectShadows=False)
        
        vehicle_count = 0
        frame_count = 0
        tracked_vehicles = {}
        next_vehicle_id = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
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
            
            # Process contours
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                if w >= MIN_WIDTH and h >= MIN_HEIGHT and area > 300:
                    cx, cy = center_handle(x, y, w, h)
                    current_frame_detections.append((cx, cy, x, y, w, h))
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.circle(roi, (cx, cy), 4, (0, 0, 255), -1)
            
            # Update tracked vehicles and count
            new_tracked_vehicles = {}
            for detection in current_frame_detections:
                cx, cy, x, y, w, h = detection
                
                # Simple tracking logic
                vehicle_id = None
                min_distance = float('inf')
                for vid, data in tracked_vehicles.items():
                    prev_center = data['center']
                    distance = np.sqrt((cx - prev_center[0])**2 + (cy - prev_center[1])**2)
                    if distance < min_distance and distance < 60:
                        min_distance = distance
                        vehicle_id = vid
                
                if vehicle_id is not None:
                    # Update existing vehicle
                    new_tracked_vehicles[vehicle_id] = {
                        'center': (cx, cy),
                        'crossed': tracked_vehicles[vehicle_id]['crossed'],
                        'prev_y': tracked_vehicles[vehicle_id]['prev_y']
                    }
                else:
                    # New vehicle
                    new_tracked_vehicles[next_vehicle_id] = {
                        'center': (cx, cy),
                        'crossed': False,
                        'prev_y': cy
                    }
                    next_vehicle_id += 1
            
            # Counting logic
            for vehicle_id, data in new_tracked_vehicles.items():
                current_y = data['center'][1]
                prev_y = data['prev_y']
                
                # Count if vehicle crosses the line
                if not data['crossed'] and prev_y < COUNT_LINE_POSITION <= current_y:
                    vehicle_count += 1
                    new_tracked_vehicles[vehicle_id]['crossed'] = True
                    cv2.line(roi, (0, COUNT_LINE_POSITION), (roi.shape[1], COUNT_LINE_POSITION), CROSS_LINE_COLOR, 3)
                
                # Update previous y
                new_tracked_vehicles[vehicle_id]['prev_y'] = current_y
            
            # Update tracked vehicles
            tracked_vehicles = {k: v for k, v in new_tracked_vehicles.items() if v['center'][1] < roi.shape[0] - 10}
            
            # Update current frame for streaming
            draw_text(roi, f"Total Vehicles: {vehicle_count}", pos=(10, 50), color=TEXT_COLOR, size=1)
            video_processing_data['current_frame'] = frame
            video_processing_data['vehicle_count'] = vehicle_count
            
            # Update processing stats
            processing_time = time.time() - start_time
            fps = frame_count / processing_time if processing_time > 0 else 0
            video_processing_data['processing_stats'] = {
                'total_detections': vehicle_count,
                'processing_time': processing_time,
                'fps': fps,
                'frame_count': frame_count
            }
        
        # Update video database with results
        for video in video_database['videos']:
            if video['id'] == video_id:
                video['detections'] = vehicle_count
                video['status'] = 'processed'
                break
        
        cap.release()
        with processing_lock:
            video_processing_data['is_processing'] = False
            video_processing_data['processing_video_id'] = None
        return True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        with processing_lock:
            video_processing_data['is_processing'] = False
            video_processing_data['processing_video_id'] = None
        return False

def get_current_frame():
    """Get current processing frame"""
    if video_processing_data['current_frame'] is not None:
        ret, buffer = cv2.imencode('.jpg', video_processing_data['current_frame'])
        if ret:
            return buffer.tobytes()
    return None

def get_lane_frame(lane_id):
    """Get current frame for specific lane"""
    if lane_id in lane_frames and lane_frames[lane_id] is not None:
        ret, buffer = cv2.imencode('.jpg', lane_frames[lane_id])
        if ret:
            return buffer.tobytes()
    return None

def generate_frames(lane_id=None):
    """Generator for video streaming"""
    while True:
        if lane_id:
            frame_bytes = get_lane_frame(lane_id)
        else:
            frame_bytes = get_current_frame()
        
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.100)  # ~10 FPS for multiple streams

# Routes
@app.route('/')
def dashboard():
    """Main futuristic dashboard"""
    return render_template('futuristic_dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint with lane support"""
    lane_id = request.args.get('lane_id')
    video_id = request.args.get('video_id')
    
    # Handle lane-specific video feeds
    if lane_id and lane_id in traffic_data['lanes']:
        return Response(generate_frames(lane_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Handle specific video ID (original functionality)
    if video_id:
        try:
            video_id = int(video_id)
            video = next((v for v in video_database['videos'] if v['id'] == video_id), None)
            
            if video and os.path.exists(video['file_path']):
                # Check if we're not already processing this or another video
                with processing_lock:
                    if not video_processing_data['is_processing'] or video_processing_data['processing_video_id'] != video_id:
                        video_processing_data['processing_video_id'] = video_id
                        # Start processing in a separate thread
                        thread = threading.Thread(target=process_video_for_detection, args=(video['file_path'], video_id), daemon=True)
                        thread.start()
        except ValueError:
            pass
    
    # Default: return combined or main feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/videos/<filename>')
def serve_video(filename):
    """Serve video files"""
    return send_from_directory('static/videos', filename)

@app.route('/uploads/<filename>')
def serve_uploaded_video(filename):
    """Serve uploaded video files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/videos')
def get_videos():
    """Get all videos in database"""
    return jsonify({
        'videos': video_database['videos'],
        'analytics': video_database['analytics']
    })

@app.route('/api/videos/<int:video_id>')
def get_video(video_id):
    """Get specific video details"""
    video = next((v for v in video_database['videos'] if v['id'] == video_id), None)
    if video:
        return jsonify(video)
    return jsonify({'error': 'Video not found'}), 404

@app.route('/api/videos/<int:video_id>/like', methods=['POST'])
def toggle_like(video_id):
    """Toggle like for a video"""
    video = next((v for v in video_database['videos'] if v['id'] == video_id), None)
    if video:
        json_data = request.json or {}
        action = json_data.get('action', 'toggle')
        if action == 'like':
            video['likes'] += 1
        elif action == 'unlike':
            video['likes'] = max(0, video['likes'] - 1)
        else:  # toggle
            video['likes'] += 1 if video['likes'] < 50 else -1
        
        return jsonify({'success': True, 'likes': video['likes']})
    return jsonify({'error': 'Video not found'}), 404

@app.route('/api/videos/<int:video_id>/comment', methods=['POST'])
def add_comment(video_id):
    """Add comment to video"""
    video = next((v for v in video_database['videos'] if v['id'] == video_id), None)
    if video:
        json_data = request.json or {}
        comment_text = json_data.get('comment', '')
        if comment_text:
            comment = {
                'id': len(video['comments']) + 1,
                'text': comment_text,
                'timestamp': datetime.now().isoformat(),
                'user': 'Anonymous User'
            }
            video['comments'].append(comment)
            return jsonify({'success': True, 'comment': comment})
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/api/videos/<int:video_id>', methods=['DELETE'])
def delete_video(video_id):
    """Delete a video"""
    global video_database
    video_database['videos'] = [v for v in video_database['videos'] if v['id'] != video_id]
    video_database['analytics']['total_uploads'] = len(video_database['videos'])
    return jsonify({'success': True, 'message': 'Video deleted successfully'})

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid conflicts
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # Add to database with proper file paths
            original_filename = file.filename or 'unknown'
            new_video = {
                'id': len(video_database['videos']) + 1,
                'title': original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename,
                'filename': filename,
                'file_path': filepath,
                'serving_url': f'/uploads/{filename}',
                'upload_date': datetime.now().strftime('%Y-%m-%d'),
                'size': f"{os.path.getsize(filepath) / (1024*1024):.1f} MB",
                'duration': '0:00',  # Would need video analysis to get real duration
                'detections': 0,
                'views': 0,
                'likes': 0,
                'comments': [],
                'tags': ['uploaded'],
                'status': 'processing',
                'analytics': {'cars': 0, 'trucks': 0, 'motorcycles': 0, 'buses': 0}
            }
            
            video_database['videos'].append(new_video)
            video_database['analytics']['total_uploads'] += 1
            
            # Start processing the uploaded video
            thread = threading.Thread(target=process_video_for_detection, args=(filepath, new_video['id']), daemon=True)
            thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Video uploaded successfully',
                'video': new_video
            })
            
        except Exception as e:
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/search')
def search_videos():
    """Search videos by query"""
    query = request.args.get('q', '').lower()
    tag_filter = request.args.get('tag', '')
    
    filtered_videos = video_database['videos']
    
    if query:
        filtered_videos = [v for v in filtered_videos 
                          if query in v['title'].lower() or 
                          any(query in tag for tag in v['tags'])]
    
    if tag_filter and tag_filter != 'all':
        filtered_videos = [v for v in filtered_videos if tag_filter in v['tags']]
    
    return jsonify({
        'videos': filtered_videos,
        'count': len(filtered_videos)
    })

@app.route('/api/analytics')
def get_analytics():
    """Get system analytics"""
    # Generate some sample analytics data
    monthly_detections = [127, 245, 189, 298, 156, 234]
    vehicle_distribution = {
        'cars': sum(v['analytics']['cars'] for v in video_database['videos']),
        'trucks': sum(v['analytics']['trucks'] for v in video_database['videos']),
        'motorcycles': sum(v['analytics']['motorcycles'] for v in video_database['videos']),
        'buses': sum(v['analytics']['buses'] for v in video_database['videos'])
    }
    
    return jsonify({
        'monthly_detections': monthly_detections,
        'vehicle_distribution': vehicle_distribution,
        'processing_stats': video_processing_data['processing_stats'],
        'system_stats': video_database['analytics']
    })

@app.route('/api/traffic_data')
def get_traffic_data():
    """Get current traffic data for all lanes"""
    with traffic_lock:
        return jsonify(traffic_data)

@app.route('/api/control_signal', methods=['POST'])
def control_signal():
    """Manual signal control for specific lane"""
    json_data = request.json or {}
    lane_id = json_data.get('lane_id')
    signal_status = json_data.get('signal_status')
    
    if not lane_id or lane_id not in traffic_data['lanes']:
        return jsonify({'error': 'Invalid lane_id'}), 400
    
    if signal_status not in ['RED', 'YELLOW', 'GREEN']:
        return jsonify({'error': 'Invalid signal_status'}), 400
    
    with traffic_lock:
        # Set manual override
        traffic_data['lanes'][lane_id]['manual_override'] = True
        traffic_data['lanes'][lane_id]['signal_status'] = signal_status
        traffic_data['lanes'][lane_id]['signal_timer'] = time.time()
        
        # If setting to GREEN, set others to RED for safety
        if signal_status == 'GREEN':
            for other_lane_id in traffic_data['lanes']:
                if other_lane_id != lane_id:
                    traffic_data['lanes'][other_lane_id]['signal_status'] = 'RED'
                    traffic_data['lanes'][other_lane_id]['signal_timer'] = time.time()
    
    return jsonify({
        'success': True, 
        'message': f'Lane {lane_id} signal set to {signal_status}'
    })

@app.route('/api/emergency_override', methods=['POST'])
def emergency_override():
    """Emergency override - all signals to RED"""
    json_data = request.json or {}
    enable = json_data.get('enable', True)
    
    with traffic_lock:
        traffic_data['system_config']['emergency_override'] = enable
        
        if enable:
            # Set all lanes to RED
            for lane_id in traffic_data['lanes']:
                traffic_data['lanes'][lane_id]['signal_status'] = 'RED'
                traffic_data['lanes'][lane_id]['signal_timer'] = time.time()
                traffic_data['lanes'][lane_id]['manual_override'] = True
            
            return jsonify({
                'success': True,
                'message': 'Emergency override activated - all signals RED'
            })
        else:
            # Clear manual overrides
            for lane_id in traffic_data['lanes']:
                traffic_data['lanes'][lane_id]['manual_override'] = False
            
            return jsonify({
                'success': True,
                'message': 'Emergency override deactivated - returning to auto control'
            })

@app.route('/api/system_config', methods=['GET', 'POST'])
def system_config():
    """Get or update system configuration"""
    if request.method == 'GET':
        with traffic_lock:
            return jsonify(traffic_data['system_config'])
    
    elif request.method == 'POST':
        json_data = request.json or {}
        
        with traffic_lock:
            config = traffic_data['system_config']
            
            # Update allowed configuration parameters
            if 'auto_control' in json_data:
                config['auto_control'] = bool(json_data['auto_control'])
            if 'min_green_time' in json_data:
                config['min_green_time'] = max(5, int(json_data['min_green_time']))
            if 'max_green_time' in json_data:
                config['max_green_time'] = max(15, int(json_data['max_green_time']))
            if 'density_threshold_low' in json_data:
                config['density_threshold_low'] = max(0.1, min(0.5, float(json_data['density_threshold_low'])))
            if 'density_threshold_high' in json_data:
                config['density_threshold_high'] = max(0.5, min(1.0, float(json_data['density_threshold_high'])))
        
        return jsonify({
            'success': True,
            'message': 'System configuration updated',
            'config': traffic_data['system_config']
        })
    
    # Default return for other methods (shouldn't happen but fixes type check)
    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/api/lane_comparison')
def lane_comparison():
    """Get detailed lane comparison analysis"""
    with traffic_lock:
        lanes_data = traffic_data['lanes']
        priorities = analyze_lane_priority(traffic_data)
        
        comparison = {
            'timestamp': time.time(),
            'lanes': {},
            'priorities': priorities,
            'recommendations': []
        }
        
        # Detailed analysis for each lane
        for lane_id, lane_data in lanes_data.items():
            comparison['lanes'][lane_id] = {
                'name': lane_data['name'],
                'vehicle_count': lane_data['vehicle_count'],
                'traffic_density': lane_data['traffic_density'],
                'signal_status': lane_data['signal_status'],
                'avg_speed': lane_data['avg_speed'],
                'density_trend': 'stable',
                'efficiency_score': 0.0
            }
            
            # Calculate density trend
            if len(lane_data['density_history']) >= 3:
                recent = lane_data['density_history'][-3:]
                trend = (recent[-1] - recent[0]) / len(recent)
                if trend > 0.05:
                    comparison['lanes'][lane_id]['density_trend'] = 'increasing'
                elif trend < -0.05:
                    comparison['lanes'][lane_id]['density_trend'] = 'decreasing'
            
            # Calculate efficiency score (0-100)
            density = lane_data['traffic_density']
            signal_status = lane_data['signal_status']
            
            if signal_status == 'GREEN' and density > 0.3:
                efficiency = min(100, density * 150)  # High efficiency when green with traffic
            elif signal_status == 'RED' and density < 0.2:
                efficiency = 90  # Good efficiency when red with low traffic
            elif signal_status == 'GREEN' and density < 0.1:
                efficiency = 20  # Poor efficiency when green with no traffic
            else:
                efficiency = 50  # Average efficiency
            
            comparison['lanes'][lane_id]['efficiency_score'] = round(efficiency, 1)
        
        # Generate recommendations
        if priorities:
            highest_priority = priorities[0]
            if highest_priority['priority_score'] > 75:
                comparison['recommendations'].append(
                    f"Lane {highest_priority['lane_id']} has high priority (density: {highest_priority['density']:.2f}) - consider extending green time"
                )
            
            # Check for inefficient signals
            for lane_id, lane_info in comparison['lanes'].items():
                if lane_info['efficiency_score'] < 30:
                    comparison['recommendations'].append(
                        f"Lane {lane_id} shows low efficiency ({lane_info['efficiency_score']}%) - review signal timing"
                    )
        
        return jsonify(comparison)

@app.route('/api/processing_status')
def get_processing_status():
    """Get current processing status"""
    return jsonify({
        'is_processing': video_processing_data['is_processing'],
        'current_video': video_processing_data['current_video'],
        'vehicle_count': video_processing_data['vehicle_count'],
        'stats': video_processing_data['processing_stats']
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

# Additional API endpoints for the frontend
@app.route('/api/set_auto_control', methods=['POST'])
def set_auto_control():
    """Set automatic traffic control on/off"""
    json_data = request.json or {}
    auto_control = json_data.get('auto_control', True)
    
    with traffic_lock:
        traffic_data['system_config']['auto_control'] = bool(auto_control)
    
    return jsonify({
        'success': True,
        'auto_control': bool(auto_control),
        'message': f'Automatic control {"enabled" if auto_control else "disabled"}'
    })

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """Emergency stop - immediately set all signals to RED"""
    with traffic_lock:
        traffic_data['system_config']['emergency_override'] = True
        
        # Set all lanes to RED immediately
        for lane_id in traffic_data['lanes']:
            traffic_data['lanes'][lane_id]['signal_status'] = 'RED'
            traffic_data['lanes'][lane_id]['signal_timer'] = time.time()
            traffic_data['lanes'][lane_id]['manual_override'] = True
    
    return jsonify({
        'success': True,
        'message': 'Emergency stop activated - all traffic signals set to RED'
    })

@app.route('/api/reset_system', methods=['POST'])
def reset_system():
    """Reset the entire traffic management system"""
    global traffic_data
    
    with traffic_lock:
        # Reset all lane data
        for lane_id in traffic_data['lanes']:
            traffic_data['lanes'][lane_id]['vehicle_count'] = 0
            traffic_data['lanes'][lane_id]['signal_status'] = 'RED'
            traffic_data['lanes'][lane_id]['traffic_density'] = 0.0
            traffic_data['lanes'][lane_id]['avg_speed'] = 0.0
            traffic_data['lanes'][lane_id]['last_update'] = time.time()
            traffic_data['lanes'][lane_id]['density_history'] = []
            traffic_data['lanes'][lane_id]['signal_timer'] = time.time()
            traffic_data['lanes'][lane_id]['manual_override'] = False
        
        # Reset system configuration to defaults
        traffic_data['system_config'] = {
            'auto_control': True,
            'min_green_time': 15,
            'max_green_time': 60,
            'yellow_time': 3,
            'all_red_time': 2,
            'density_threshold_low': 0.2,
            'density_threshold_high': 0.7,
            'emergency_override': False
        }
    
    return jsonify({
        'success': True,
        'message': 'Traffic management system has been reset to default state'
    })

@app.route('/multi_lane_feed/<lane_id>')
def multi_lane_feed(lane_id):
    """Individual lane video feeds for the dashboard"""
    if lane_id not in traffic_data['lanes']:
        return jsonify({'error': 'Invalid lane ID'}), 404
    
    return Response(generate_frames(lane_id), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/comprehensive_analytics')
def comprehensive_analytics():
    """Get comprehensive traffic analytics and performance metrics"""
    with traffic_lock:
        analytics = traffic_data['analytics'].copy()
        
        # Calculate system efficiency
        total_vehicles = analytics['total_vehicles_system']
        total_cycles = analytics['total_processing_cycles']
        system_uptime = time.time() - analytics['start_time']
        
        # Lane-specific analytics
        lane_analytics = {}
        busiest_lane_data = {'lane_id': 'lane_1', 'total_handled': 0}
        total_throughput = 0
        
        for lane_id, lane_data in traffic_data['lanes'].items():
            lane_analytics[lane_id] = {
                'name': lane_data['name'],
                'current_vehicles': lane_data['vehicle_count'],
                'current_density': lane_data['traffic_density'],
                'total_handled': lane_data['total_vehicles_handled'],
                'peak_vehicles': lane_data['peak_vehicles'],
                'throughput_rate': lane_data['throughput_rate'],
                'signal_status': lane_data['signal_status'],
                'efficiency_score': calculate_lane_efficiency(lane_data)
            }
            
            total_throughput += lane_data['throughput_rate']
            
            if lane_data['total_vehicles_handled'] > busiest_lane_data['total_handled']:
                busiest_lane_data = {
                    'lane_id': lane_id,
                    'name': lane_data['name'],
                    'total_handled': lane_data['total_vehicles_handled']
                }
        
        # Calculate overall system efficiency
        system_efficiency = (total_vehicles / max(total_cycles, 1)) * 100 if total_cycles > 0 else 0
        avg_vehicles_per_cycle = total_vehicles / max(total_cycles, 1)
        
        return jsonify({
            'system_overview': {
                'total_vehicles_processed': total_vehicles,
                'system_uptime_hours': round(system_uptime / 3600, 2),
                'total_processing_cycles': total_cycles,
                'system_efficiency_percent': round(system_efficiency, 2),
                'average_vehicles_per_cycle': round(avg_vehicles_per_cycle, 2),
                'total_throughput_rate': round(total_throughput, 2),
                'busiest_lane': busiest_lane_data
            },
            'lane_analytics': lane_analytics,
            'performance_metrics': {
                'lanes_with_high_traffic': len([l for l in lane_analytics.values() if l['current_vehicles'] >= 15]),
                'lanes_with_moderate_traffic': len([l for l in lane_analytics.values() if 5 <= l['current_vehicles'] < 15]),
                'lanes_with_light_traffic': len([l for l in lane_analytics.values() if l['current_vehicles'] < 5]),
                'average_density': round(sum(l['current_density'] for l in lane_analytics.values()) / 4, 3),
                'peak_traffic_lane': max(lane_analytics.keys(), key=lambda k: lane_analytics[k]['peak_vehicles']),
                'most_efficient_lane': max(lane_analytics.keys(), key=lambda k: lane_analytics[k]['efficiency_score'])
            },
            'signal_optimization': {
                'auto_control_enabled': traffic_data['system_config']['auto_control'],
                'emergency_mode': traffic_data['system_config']['emergency_override'],
                'green_time_range': f"{traffic_data['system_config']['min_green_time']}-{traffic_data['system_config']['max_green_time']}s",
                'density_thresholds': {
                    'low': traffic_data['system_config']['density_threshold_low'],
                    'high': traffic_data['system_config']['density_threshold_high']
                }
            }
        })

def calculate_lane_efficiency(lane_data):
    """Calculate efficiency score for a lane (0-100)"""
    density = lane_data['traffic_density']
    vehicles = lane_data['vehicle_count']
    signal_status = lane_data['signal_status']
    throughput = lane_data['throughput_rate']
    
    efficiency = 50  # Base efficiency
    
    # Efficiency based on signal and traffic correlation
    if signal_status == 'GREEN' and vehicles > 10:
        efficiency += 30  # Good: Green light with traffic
    elif signal_status == 'RED' and vehicles < 5:
        efficiency += 20  # Good: Red light with little traffic  
    elif signal_status == 'GREEN' and vehicles < 3:
        efficiency -= 20  # Poor: Green light wasted
    elif signal_status == 'RED' and vehicles > 15:
        efficiency -= 25  # Poor: Heavy traffic waiting
    
    # Throughput factor
    efficiency += min(throughput * 2, 20)
    
    # Density optimization
    if 0.3 <= density <= 0.7:
        efficiency += 10  # Optimal density range
    elif density > 0.8:
        efficiency -= 15  # Congestion penalty
    
    return max(0, min(100, round(efficiency, 1)))

if __name__ == '__main__':
    print("🚀 Starting AI Vision - Futuristic Traffic Analysis System")
    print("📊 Loading video database...")
    print("🎥 Initializing video processing engine...")
    print("🚦 Starting intelligent traffic signal controller...")
    print("✨ Ready for futuristic traffic intelligence!")
    
    # Start traffic analysis manager in background
    traffic_thread = threading.Thread(target=traffic_analysis_manager, daemon=True)
    traffic_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)