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
import sys

# Add the project directory to the system path to allow importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Dummy config and utils files for demonstration purposes
# In a real project, these would be separate modules.

class Config:
    COUNT_LINE_POSITION = 400
    MIN_WIDTH = 80
    MIN_HEIGHT = 80
    LINE_OFFSET = 10
    LINE_COLOR = (0, 255, 255)
    CROSS_LINE_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    MOG_HISTORY = 500
    MOG_THRESHOLD = 16

config = Config()

class Utils:
    def center_handle(x, y, w, h):
        cx = x + w // 2
        cy = y + h // 2
        return cx, cy

    def draw_text(frame, text, pos, color, size):
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

    def draw_count_line(frame, y_pos, color):
        cv2.line(frame, (0, y_pos), (frame.shape[1], y_pos), color, 2)

utils = Utils()

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
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

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
            'name': 'Main Street',
            'video_file': 'video1_cctv.mp4',
            'vehicle_count': 0,
            'signal_status': 'GREEN',
            'traffic_density': 0.0,
            'avg_speed': 0.0,
            'is_processing': False,
            'last_update': time.time(),
            'density_history': [],
            'signal_timer': time.time(),
            'manual_override': False,
            'total_vehicles_handled': 0,
            'peak_vehicles': 0,
            'avg_wait_time': 0,
            'throughput_rate': 0.0
        },
        'lane_2': {
            'name': 'Oak Street',
            'video_file': 'video2_4k_traffic.mp4',
            'vehicle_count': 0,
            'signal_status': 'RED',
            'traffic_density': 0.0,
            'avg_speed': 0.0,
            'is_processing': False,
            'last_update': time.time(),
            'density_history': [],
            'signal_timer': time.time(),
            'manual_override': False,
            'total_vehicles_handled': 0,
            'peak_vehicles': 0,
            'avg_wait_time': 0,
            'throughput_rate': 0.0
        },
        'lane_3': {
            'name': 'Park Ave',
            'video_file': 'video3_safety.mp4',
            'vehicle_count': 0,
            'signal_status': 'RED',
            'traffic_density': 0.0,
            'avg_speed': 0.0,
            'is_processing': False,
            'last_update': time.time(),
            'density_history': [],
            'signal_timer': time.time(),
            'manual_override': False,
            'total_vehicles_handled': 0,
            'peak_vehicles': 0,
            'avg_wait_time': 0,
            'throughput_rate': 0.0
        },
        'lane_4': {
            'name': 'River Rd',
            'video_file': 'video4_ipcam.mp4',
            'vehicle_count': 0,
            'signal_status': 'RED',
            'traffic_density': 0.0,
            'avg_speed': 0.0,
            'is_processing': False,
            'last_update': time.time(),
            'density_history': [],
            'signal_timer': time.time(),
            'manual_override': False,
            'total_vehicles_handled': 0,
            'peak_vehicles': 0,
            'avg_wait_time': 0,
            'throughput_rate': 0.0
        }
    },
    'system_config': {
        'auto_control': True,
        'min_green_time': 15,
        'max_green_time': 60,
        'yellow_time': 3,
        'all_red_time': 2,
        'density_threshold_low': 0.3,
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
    },
    'events': [
        {'type': 'light change', 'timestamp': time.time() - 60},
        {'type': 'manual override', 'timestamp': time.time() - 30},
        {'type': 'traffic spike', 'timestamp': time.time() - 15}
    ]
}

# Simple processing lock
processing_lock = threading.Lock()
traffic_lock = threading.Lock()

# Simple video database placeholder
video_database = {
    'videos': [],
    'analytics': {
        'total_uploads': 0,
        'total_views': 0,
        'total_detections': 0,
        'total_likes': 0,
        'processing_hours': 0
    }
}

# Add some dummy video files for the initial setup
# NOTE: You must have these video files in the `static/videos` directory.
dummy_videos = [
    {'id': 1, 'title': 'Main St Footage', 'filename': 'video1_cctv.mp4', 'file_path': 'static/videos/video1_cctv.mp4'},
    {'id': 2, 'title': 'Oak St Footage', 'filename': 'video2_4k_traffic.mp4', 'file_path': 'static/videos/video2_4k_traffic.mp4'},
    {'id': 3, 'title': 'Park Ave Footage', 'filename': 'video3_safety.mp4', 'file_path': 'static/videos/video3_safety.mp4'},
    {'id': 4, 'title': 'River Rd Footage', 'filename': 'video4_ipcam.mp4', 'file_path': 'static/videos/video4_ipcam.mp4'}
]
video_database['videos'] = dummy_videos


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_traffic_density(detections_count, roi_area):
    if roi_area <= 0 or detections_count < 0:
        return 0.0
    light_traffic = 5
    moderate_traffic = 15
    heavy_traffic = 25
    if detections_count <= light_traffic:
        density = detections_count / light_traffic * 0.25
    elif detections_count <= moderate_traffic:
        density = 0.25 + ((detections_count - light_traffic) / (moderate_traffic - light_traffic)) * 0.35
    elif detections_count <= heavy_traffic:
        density = 0.60 + ((detections_count - moderate_traffic) / (heavy_traffic - moderate_traffic)) * 0.25
    else:
        density = 0.85 + min((detections_count - heavy_traffic) / 20, 0.15)
    return round(density, 3)

def analyze_lane_priority(traffic_data):
    lane_priorities = []
    for lane_id, lane_data in traffic_data['lanes'].items():
        if not lane_data['manual_override']:
            priority_score = 0.0
            vehicle_count = lane_data['vehicle_count']
            density = lane_data['traffic_density']
            if vehicle_count >= 25: priority_score += 150
            elif vehicle_count >= 15: priority_score += 100
            elif vehicle_count >= 5: priority_score += 60
            else: priority_score += vehicle_count * 5
            if density > traffic_data['system_config']['density_threshold_high']: priority_score += 40
            elif density > traffic_data['system_config']['density_threshold_low']: priority_score += 20
            if len(lane_data['density_history']) >= 3:
                recent_densities = lane_data['density_history'][-3:]
                trend = (recent_densities[-1] - recent_densities[0]) / len(recent_densities)
                priority_score += trend * 30
            if lane_data['signal_status'] == 'RED':
                red_duration = time.time() - lane_data['signal_timer']
                if red_duration > 90: priority_score += 50
                elif red_duration > 45: priority_score += 25
            lane_priorities.append({
                'lane_id': lane_id,
                'priority_score': priority_score,
                'density': density,
                'vehicle_count': vehicle_count,
                'current_signal': lane_data['signal_status']
            })
    lane_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
    return lane_priorities

def intelligent_signal_control(traffic_data):
    if not traffic_data['system_config']['auto_control'] or traffic_data['system_config']['emergency_override']:
        return
    current_time = time.time()
    config = traffic_data['system_config']
    lane_priorities = analyze_lane_priority(traffic_data)
    if not lane_priorities: return
    current_green_lanes = [lane_id for lane_id, lane_data in traffic_data['lanes'].items()
                           if lane_data['signal_status'] == 'GREEN']
    if current_green_lanes:
        current_green_lane = current_green_lanes[0]
        current_lane_data = traffic_data['lanes'][current_green_lane]
        green_duration = current_time - current_lane_data['signal_timer']
        should_switch = False
        if green_duration >= config['max_green_time']:
            should_switch = True
        elif green_duration >= config['min_green_time']:
            highest_priority_lane = lane_priorities[0]['lane_id']
            if (highest_priority_lane != current_green_lane and
                lane_priorities[0]['priority_score'] > 100 and
                lane_priorities[0]['vehicle_count'] > current_lane_data['vehicle_count'] + 5):
                should_switch = True
        if should_switch:
            current_lane_data['signal_status'] = 'YELLOW'
            current_lane_data['signal_timer'] = current_time
            add_event('light change')
    else:
        yellow_lanes = [lane_id for lane_id, lane_data in traffic_data['lanes'].items()
                        if lane_data['signal_status'] == 'YELLOW']
        if yellow_lanes:
            for lane_id in yellow_lanes:
                lane_data = traffic_data['lanes'][lane_id]
                yellow_duration = current_time - lane_data['signal_timer']
                if yellow_duration >= config['yellow_time']:
                    lane_data['signal_status'] = 'RED'
                    lane_data['signal_timer'] = current_time
                    add_event('light change')
        else:
            all_red_lanes = [lane_id for lane_id, lane_data in traffic_data['lanes'].items()
                            if lane_data['signal_status'] == 'RED']
            if len(all_red_lanes) == len(traffic_data['lanes']):
                oldest_red_time = min(lane_data['signal_timer'] for lane_data in traffic_data['lanes'].values())
                all_red_duration = current_time - oldest_red_time
                if all_red_duration >= config['all_red_time']:
                    highest_priority_lane = lane_priorities[0]['lane_id']
                    highest_lane_data = traffic_data['lanes'][highest_priority_lane]
                    highest_lane_data['signal_status'] = 'GREEN'
                    highest_lane_data['signal_timer'] = current_time
                    add_event('light change')

def process_lane_video(lane_id, video_path):
    global traffic_data
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        fgbg = cv2.createBackgroundSubtractorMOG2(history=config.MOG_HISTORY, varThreshold=config.MOG_THRESHOLD, detectShadows=False)
        frame_count = 0
        vehicle_detections = []
        start_time = time.time()
        with traffic_lock:
            traffic_data['lanes'][lane_id]['is_processing'] = True
        while frame_count < 150:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_count += 1
            frame = cv2.resize(frame, (1020, 600))
            roi = frame[200:600, :]
            roi_area = roi.shape[0] * roi.shape[1]
            mask = fgbg.apply(roi)
            _, mask = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
            mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=3)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_detections = 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                if w >= config.MIN_WIDTH // 3 and h >= config.MIN_HEIGHT // 3 and area > 100:
                    current_detections += 1
            vehicle_detections.append(current_detections)
            utils.draw_count_line(roi, config.COUNT_LINE_POSITION, config.LINE_COLOR)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                if w >= config.MIN_WIDTH // 3 and h >= config.MIN_HEIGHT // 3 and area > 100:
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 0), 2)
            with traffic_lock:
                lane_data = traffic_data['lanes'][lane_id]
                signal_color = lane_data['signal_status']
                signal_colors = {'GREEN': (0, 255, 0), 'YELLOW': (0, 255, 255), 'RED': (0, 0, 255)}
                cv2.rectangle(frame, (10, 10), (300, 80), signal_colors.get(signal_color, (128, 128, 128)), -1)
                cv2.putText(frame, f"{lane_data['name']}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, f"Signal: {signal_color}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, f"Density: {lane_data['traffic_density']:.2f}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, f"Vehicles: {current_detections}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                lane_frames[lane_id] = frame.copy()
        processing_time = time.time() - start_time
        avg_detections = np.mean(vehicle_detections) if vehicle_detections else 0
        max_detections = max(vehicle_detections) if vehicle_detections else 0
        traffic_density = calculate_traffic_density(avg_detections, roi_area)
        with traffic_lock:
            lane_data = traffic_data['lanes'][lane_id]
            current_vehicles = int(max_detections)
            lane_data['vehicle_count'] = current_vehicles
            lane_data['traffic_density'] = traffic_density
            lane_data['last_update'] = time.time()
            lane_data['is_processing'] = False
            lane_data['total_vehicles_handled'] += current_vehicles
            lane_data['peak_vehicles'] = max(lane_data['peak_vehicles'], current_vehicles)
            lane_data['throughput_rate'] = (current_vehicles / (processing_time / 60)) if processing_time > 0 else 0
            lane_data['density_history'].append(traffic_density)
            if len(lane_data['density_history']) > 10:
                lane_data['density_history'].pop(0)
            traffic_data['analytics']['total_vehicles_system'] += current_vehicles
            traffic_data['analytics']['total_processing_cycles'] += 1
            current_busiest = traffic_data['analytics']['busiest_lane']
            if (current_vehicles > traffic_data['lanes'][current_busiest]['vehicle_count'] or
                traffic_data['lanes'][lane_id]['total_vehicles_handled'] > traffic_data['lanes'][current_busiest]['total_vehicles_handled']):
                traffic_data['analytics']['busiest_lane'] = lane_id
        cap.release()
        return True
    except Exception as e:
        with traffic_lock:
            traffic_data['lanes'][lane_id]['is_processing'] = False
        return False

def traffic_analysis_manager():
    print("Traffic Analysis Manager started")
    while True:
        try:
            for lane_id, lane_data in traffic_data['lanes'].items():
                if not lane_data['is_processing']:
                    video_path = os.path.join(STATIC_VIDEOS_FOLDER, lane_data['video_file'])
                    if os.path.exists(video_path):
                        thread = threading.Thread(target=process_lane_video, args=(lane_id, video_path), daemon=True)
                        thread.start()
                    else:
                        placeholder = create_placeholder_frame(lane_data['name'], lane_data['signal_status'])
                        lane_frames[lane_id] = placeholder
            intelligent_signal_control(traffic_data)
            time.sleep(8)
        except Exception as e:
            time.sleep(10)

def create_placeholder_frame(lane_name, signal_status):
    frame = np.zeros((600, 1020, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)
    cv2.putText(frame, lane_name, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, "Camera Offline", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 2)
    signal_colors = {'GREEN': (0, 255, 0), 'YELLOW': (0, 255, 255), 'RED': (0, 0, 255)}
    color = signal_colors.get(signal_status, (128, 128, 128))
    cv2.circle(frame, (500, 400), 80, color, -1)
    cv2.putText(frame, signal_status, (420, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return frame

def get_lane_frame(lane_id):
    if lane_id in lane_frames and lane_frames[lane_id] is not None:
        ret, buffer = cv2.imencode('.jpg', lane_frames[lane_id])
        if ret:
            return buffer.tobytes()
    return None

def generate_frames(lane_id):
    while True:
        frame_bytes = get_lane_frame(lane_id)
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.100)

def add_event(event_type):
    with traffic_lock:
        traffic_data['events'].append({'type': event_type, 'timestamp': time.time()})
        if len(traffic_data['events']) > 20:
            traffic_data['events'].pop(0)

# --- Routes and API Endpoints ---
@app.route('/')
def dashboard():
    return render_template('futuristic_dashboard.html')

@app.route('/multi_lane_feed/<lane_id>')
def multi_lane_feed(lane_id):
    if lane_id not in traffic_data['lanes']:
        return jsonify({'error': 'Invalid lane ID'}), 404
    return Response(generate_frames(lane_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/traffic_data')
def get_traffic_data():
    with traffic_lock:
        return jsonify(traffic_data)

@app.route('/api/control_signal', methods=['POST'])
def control_signal():
    json_data = request.json or {}
    lane_id = json_data.get('lane_id')
    signal_status = json_data.get('signal_status')
    if not lane_id or lane_id not in traffic_data['lanes']:
        return jsonify({'error': 'Invalid lane_id'}), 400
    if signal_status not in ['RED', 'YELLOW', 'GREEN']:
        return jsonify({'error': 'Invalid signal_status'}), 400
    with traffic_lock:
        traffic_data['lanes'][lane_id]['manual_override'] = True
        traffic_data['lanes'][lane_id]['signal_status'] = signal_status
        traffic_data['lanes'][lane_id]['signal_timer'] = time.time()
        add_event('manual override')
    return jsonify({'success': True, 'message': f'Lane {lane_id} signal set to {signal_status}'})

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    with traffic_lock:
        traffic_data['system_config']['emergency_override'] = True
        for lane_id in traffic_data['lanes']:
            traffic_data['lanes'][lane_id]['signal_status'] = 'RED'
            traffic_data['lanes'][lane_id]['signal_timer'] = time.time()
            traffic_data['lanes'][lane_id]['manual_override'] = True
        add_event('emergency stop')
    return jsonify({'success': True, 'message': 'Emergency stop activated - all traffic signals set to RED'})

@app.route('/api/set_auto_control', methods=['POST'])
def set_auto_control():
    json_data = request.json or {}
    auto_control = json_data.get('auto_control', True)
    with traffic_lock:
        traffic_data['system_config']['auto_control'] = bool(auto_control)
        add_event(f'auto control {"on" if auto_control else "off"}')
        if not auto_control:
            for lane_id in traffic_data['lanes']:
                traffic_data['lanes'][lane_id]['manual_override'] = True
        else:
            for lane_id in traffic_data['lanes']:
                traffic_data['lanes'][lane_id]['manual_override'] = False
    return jsonify({'success': True, 'auto_control': bool(auto_control), 'message': f'Automatic control {"enabled" if auto_control else "disabled"}'})

@app.route('/api/reset_system', methods=['POST'])
def reset_system():
    global traffic_data
    with traffic_lock:
        traffic_data['lanes']['lane_1']['signal_status'] = 'GREEN'
        for lane_id in traffic_data['lanes']:
            lane_data = traffic_data['lanes'][lane_id]
            lane_data['vehicle_count'] = 0
            lane_data['traffic_density'] = 0.0
            lane_data['avg_speed'] = 0.0
            lane_data['last_update'] = time.time()
            lane_data['density_history'] = []
            lane_data['signal_timer'] = time.time()
            lane_data['manual_override'] = False
        traffic_data['system_config'] = {
            'auto_control': True, 'min_green_time': 15, 'max_green_time': 60,
            'yellow_time': 3, 'all_red_time': 2, 'density_threshold_low': 0.3,
            'density_threshold_high': 0.7, 'emergency_override': False
        }
        traffic_data['events'].append({'type': 'system reset', 'timestamp': time.time()})
    return jsonify({'success': True, 'message': 'Traffic management system has been reset to default state'})

@app.route('/api/comprehensive_analytics')
def comprehensive_analytics():
    with traffic_lock:
        lane_analytics = {}
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
        return jsonify(lane_analytics)

def calculate_lane_efficiency(lane_data):
    density = lane_data['traffic_density']
    vehicles = lane_data['vehicle_count']
    signal_status = lane_data['signal_status']
    throughput = lane_data['throughput_rate']
    efficiency = 50
    if signal_status == 'GREEN' and vehicles > 10: efficiency += 30
    elif signal_status == 'RED' and vehicles < 5: efficiency += 20
    elif signal_status == 'GREEN' and vehicles < 3: efficiency -= 20
    elif signal_status == 'RED' and vehicles > 15: efficiency -= 25
    efficiency += min(throughput * 2, 20)
    if 0.3 <= density <= 0.7: efficiency += 10
    elif density > 0.8: efficiency -= 15
    return max(0, min(100, round(efficiency, 1)))

if __name__ == '__main__':
    print("🚀 Starting AI Vision - Futuristic Traffic Analysis System")
    print("📊 Loading video database...")
    print("🎥 Initializing video processing engine...")
    print("🚦 Starting intelligent traffic signal controller...")
    print("✨ Ready for futuristic traffic intelligence!")

    traffic_thread = threading.Thread(target=traffic_analysis_manager, daemon=True)
    traffic_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
