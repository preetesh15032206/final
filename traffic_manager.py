import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import threading
import time
import json
from config import COUNT_LINE_POSITION, MIN_WIDTH, MIN_HEIGHT, LINE_OFFSET
from config import LINE_COLOR, CROSS_LINE_COLOR, TEXT_COLOR, MOG_HISTORY, MOG_THRESHOLD
from utils import center_handle, draw_text, draw_count_line

app = Flask(__name__)
CORS(app)

# Global traffic management data
traffic_data = {
    'lane1': {
        'name': 'Lane 1 - Ambulance Route',
        'video_file': 'lane1_ambulance.mp4',
        'vehicle_count': 0,
        'current_frame': None,
        'signal_status': 'GREEN',  # GREEN, YELLOW, RED
        'traffic_density': 0,
        'last_count_time': time.time(),
        'vehicles_per_minute': 0,
        'last_vehicle_count': 0,
        'manual_override': False
    },
    'lane2': {
        'name': 'Lane 2 - Tilton Traffic',
        'video_file': 'lane2_tilton.mp4', 
        'vehicle_count': 0,
        'current_frame': None,
        'signal_status': 'GREEN',
        'traffic_density': 0,
        'last_count_time': time.time(),
        'vehicles_per_minute': 0,
        'last_vehicle_count': 0,
        'manual_override': False
    },
    'lane3': {
        'name': 'Lane 3 - General Traffic',
        'video_file': 'lane3_traffic.mp4',
        'vehicle_count': 0,
        'current_frame': None,
        'signal_status': 'GREEN', 
        'traffic_density': 0,
        'last_count_time': time.time(),
        'vehicles_per_minute': 0,
        'last_vehicle_count': 0,
        'manual_override': False
    }
}

# Control settings
control_settings = {
    'auto_mode': True,
    'analysis_interval': 30,  # seconds
    'low_traffic_threshold': 5,  # vehicles per minute
    'emergency_override': False,
    'manual_control': {}
}

# Performance settings
performance_settings = {
    'processing_fps': 20,  # Faster processing FPS
    'stream_fps': 24,      # Faster streaming FPS
    'jpeg_quality': 70,    # Balanced quality
    'tile_width': 480,     # Optimized tile size
    'tile_height': 360,    # Optimized tile size
    'single_lane_mode': False,  # Show all lanes or single focused lane
    'focused_lane': 'lane1',
    'performance_mode': False   # Enables fastest settings
}

# Preallocate morphology kernels for better performance
erode_kernel = np.ones((3, 3), np.uint8)
dilate_kernel = np.ones((5, 5), np.uint8)  # Reduced from 7x7 for speed

# Preallocated canvas for frame combining
combined_canvas = None

is_processing = True

def assign_vehicle_id(center, existing_tracks, threshold=60):
    """Assign a unique ID to a new vehicle or match with an existing track."""
    min_distance = float('inf')
    matched_id = None
    for vehicle_id, data in existing_tracks.items():
        prev_center = data['center']
        distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            matched_id = vehicle_id
    return matched_id

def process_lane_video(lane_id):
    """Process video for a specific lane"""
    global traffic_data, is_processing
    
    lane = traffic_data[lane_id]
    cap = cv2.VideoCapture(lane['video_file'])
    
    if not cap.isOpened():
        print(f"Failed to open video for {lane['name']}")
        return
    
    print(f"Processing video for {lane['name']}")
    fgbg = cv2.createBackgroundSubtractorMOG2(history=MOG_HISTORY, varThreshold=MOG_THRESHOLD, detectShadows=False)
    
    tracked_vehicles = {}
    next_vehicle_id = 0
    frame_count = 0
    
    while is_processing:
        ret, frame = cap.read()
        if not ret:
            # Restart video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame_count += 1
        # Use optimized tile size from performance settings
        tile_size = (performance_settings['tile_width'], performance_settings['tile_height'])
        frame = cv2.resize(frame, tile_size)
        roi = frame[80:280, :]  # Adjusted ROI for optimized frame
        
        # Background subtraction with optimized processing
        mask = fgbg.apply(roi)
        _, mask = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)
        
        # Use preallocated kernels and reduced iterations for speed
        mask = cv2.erode(mask, erode_kernel, iterations=1)
        mask = cv2.dilate(mask, dilate_kernel, iterations=2)  # Reduced from 3
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw counting line (adjusted for smaller frame)
        count_line_y = 150  # Adjusted for smaller ROI
        draw_count_line(roi, count_line_y, LINE_COLOR)
        
        current_frame_detections = []
        
        # Process contours
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if w >= MIN_WIDTH//2 and h >= MIN_HEIGHT//2 and area > 150:  # Adjusted for smaller frame
                cx, cy = center_handle(x, y, w, h)
                current_frame_detections.append((cx, cy, x, y, w, h))
                cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.circle(roi, (cx, cy), 3, (0, 0, 255), -1)
        
        # Update tracked vehicles
        new_tracked_vehicles = {}
        for detection in current_frame_detections:
            cx, cy, x, y, w, h = detection
            vehicle_id = assign_vehicle_id((cx, cy), tracked_vehicles)
            
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
            if not data['crossed'] and prev_y < count_line_y <= current_y:
                lane['vehicle_count'] += 1
                new_tracked_vehicles[vehicle_id]['crossed'] = True
                cv2.line(roi, (0, count_line_y), (roi.shape[1], count_line_y), CROSS_LINE_COLOR, 3)
                print(f"Vehicle counted in {lane['name']}! Total: {lane['vehicle_count']}")
            
            # Update previous y
            new_tracked_vehicles[vehicle_id]['prev_y'] = current_y
        
        # Update tracked vehicles
        tracked_vehicles = {k: v for k, v in new_tracked_vehicles.items() 
                           if v['center'][1] < roi.shape[0] - 10}
        
        # Add lane info and signal status
        signal_color = (0, 255, 0) if lane['signal_status'] == 'GREEN' else \
                      (0, 255, 255) if lane['signal_status'] == 'YELLOW' else (0, 0, 255)
        
        draw_text(roi, f"{lane['name']}", pos=(10, 30), color=(255, 255, 255), size=1)
        draw_text(roi, f"Count: {lane['vehicle_count']}", pos=(10, 50), color=TEXT_COLOR, size=1)
        draw_text(roi, f"Signal: {lane['signal_status']}", pos=(10, 70), color=signal_color, size=1)
        draw_text(roi, f"Density: {lane['traffic_density']:.1f}/min", pos=(10, 90), color=(255, 255, 255), size=1)
        
        # Update traffic density every 150 frames (~5 seconds) for more responsive updates
        if frame_count % 150 == 0:
            current_time = time.time()
            time_diff = current_time - lane['last_count_time']
            if time_diff >= 5:  # Calculate rate every 5 seconds
                vehicle_delta = lane['vehicle_count'] - lane['last_vehicle_count']
                lane['vehicles_per_minute'] = (vehicle_delta / time_diff) * 60
                lane['traffic_density'] = lane['vehicles_per_minute']
                lane['last_count_time'] = current_time
                lane['last_vehicle_count'] = lane['vehicle_count']
        
        lane['current_frame'] = frame
        
        # Adaptive FPS control - remove fixed sleep for better performance
        if performance_settings['performance_mode']:
            time.sleep(1.0 / 30)  # Performance mode: 30 FPS
        else:
            time.sleep(1.0 / performance_settings['processing_fps'])  # Configurable FPS
    
    cap.release()

def traffic_analysis_manager():
    """Analyze traffic and control signals automatically"""
    global traffic_data, control_settings
    
    while is_processing:
        time.sleep(control_settings['analysis_interval'])
        
        if not control_settings['auto_mode'] or control_settings['emergency_override']:
            continue
        
        # Calculate average traffic density
        densities = [lane['traffic_density'] for lane in traffic_data.values()]
        avg_density = sum(densities) / len(densities) if densities else 0
        
        print(f"Traffic Analysis - Average density: {avg_density:.2f}")
        
        # Control logic: Stop low traffic lanes, prioritize high traffic
        # Only update signals for lanes not under manual override
        for lane_id, lane in traffic_data.items():
            if lane['manual_override']:
                continue  # Skip lanes under manual control
                
            if lane['traffic_density'] < control_settings['low_traffic_threshold']:
                if lane['traffic_density'] < avg_density * 0.5:  # Less than 50% of average
                    lane['signal_status'] = 'RED'
                    print(f"Setting {lane['name']} to RED - Low traffic ({lane['traffic_density']:.1f})")
                else:
                    lane['signal_status'] = 'YELLOW'
            else:
                lane['signal_status'] = 'GREEN'
                print(f"Setting {lane['name']} to GREEN - Normal traffic ({lane['traffic_density']:.1f})")

def get_combined_frame():
    """Combine all lane frames into one display with optimized performance"""
    global combined_canvas
    
    # Single lane mode for better performance if enabled
    if performance_settings['single_lane_mode']:
        focused_lane = traffic_data.get(performance_settings['focused_lane'])
        if focused_lane and focused_lane['current_frame'] is not None:
            return focused_lane['current_frame']
        return None
    
    # Collect available frames
    frames = []
    for lane_id, lane in traffic_data.items():
        if lane['current_frame'] is not None:
            frames.append(lane['current_frame'])
    
    if not frames:
        return None
    
    # Use optimized frame arrangement with preallocated canvas
    tile_width = performance_settings['tile_width']
    tile_height = performance_settings['tile_height']
    
    # Single frame - just return it
    if len(frames) == 1:
        return frames[0]
    
    # Preallocate canvas for consistent layout
    if len(frames) == 2:
        # Side by side layout
        if combined_canvas is None or combined_canvas.shape != (tile_height, tile_width * 2, 3):
            combined_canvas = np.zeros((tile_height, tile_width * 2, 3), dtype=np.uint8)
        combined_canvas[:, :tile_width] = frames[0]
        combined_canvas[:, tile_width:] = frames[1]
        return combined_canvas
    else:
        # 3 frames - 2 on top, 1 on bottom (stretched)
        if combined_canvas is None or combined_canvas.shape != (tile_height + tile_height//2, tile_width * 2, 3):
            combined_canvas = np.zeros((tile_height + tile_height//2, tile_width * 2, 3), dtype=np.uint8)
        
        # Top row: 2 frames side by side
        combined_canvas[:tile_height, :tile_width] = frames[0]
        combined_canvas[:tile_height, tile_width:] = frames[1]
        
        # Bottom row: stretch frame 3 to full width
        bottom_stretched = cv2.resize(frames[2], (tile_width * 2, tile_height//2))
        combined_canvas[tile_height:, :] = bottom_stretched
        
        return combined_canvas

def generate_frames():
    """Generator for video streaming with optimized performance"""
    import time as timing_module
    last_frame_time = 0
    
    while True:
        current_time = timing_module.time()
        
        # Control streaming FPS - remove the fixed 0.1s sleep!
        target_interval = 1.0 / performance_settings['stream_fps']
        
        if current_time - last_frame_time >= target_interval:
            combined_frame = get_combined_frame()
            if combined_frame is not None:
                # Use configurable JPEG quality for better compression/speed balance
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, performance_settings['jpeg_quality']]
                if performance_settings['performance_mode']:
                    encode_params.extend([cv2.IMWRITE_JPEG_OPTIMIZE, 0])  # Skip optimization for speed
                
                ret, buffer = cv2.imencode('.jpg', combined_frame, encode_params)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            last_frame_time = current_time
        else:
            # Much shorter sleep to maintain responsiveness
            timing_module.sleep(0.01)

@app.route('/')
def dashboard():
    return render_template('traffic_dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/traffic_data')
def get_traffic_data():
    try:
        # Create a copy of traffic_data without the video frames (numpy arrays)
        lanes_data = {}
        for lane_id, lane in traffic_data.items():
            lanes_data[lane_id] = {
                'name': lane['name'],
                'video_file': lane['video_file'],
                'vehicle_count': lane['vehicle_count'],
                'signal_status': lane['signal_status'],
                'traffic_density': lane['traffic_density'],
                'last_count_time': lane['last_count_time'],
                'vehicles_per_minute': lane['vehicles_per_minute']
                # Excluding 'current_frame' as it's a numpy array and not JSON serializable
            }
        
        response_data = {
            'lanes': lanes_data,
            'control_settings': control_settings,
            'timestamp': time.time()
        }
        
        # Add performance stats to response
        response_data['performance'] = {
            'processing_fps': performance_settings['processing_fps'],
            'stream_fps': performance_settings['stream_fps'],
            'jpeg_quality': performance_settings['jpeg_quality'],
            'single_lane_mode': performance_settings['single_lane_mode'],
            'performance_mode': performance_settings['performance_mode']
        }
        
        # Explicitly create JSON response with proper headers
        from flask import Response
        import json
        return Response(
            json.dumps(response_data),
            mimetype='application/json',
            headers={'Cache-Control': 'no-cache'}
        )
    except Exception as e:
        print(f"Error in get_traffic_data: {e}")
        return Response(
            json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )

@app.route('/api/control_signal', methods=['POST'])
def control_signal():
    data = request.get_json()
    lane_id = data.get('lane_id')
    signal = data.get('signal')
    
    if lane_id in traffic_data and signal in ['GREEN', 'YELLOW', 'RED']:
        traffic_data[lane_id]['signal_status'] = signal
        traffic_data[lane_id]['manual_override'] = True  # Mark as manually controlled
        control_settings['manual_control'][lane_id] = signal
        print(f"Manual control: {traffic_data[lane_id]['name']} set to {signal}")
        return jsonify({'success': True, 'message': f'Signal for {lane_id} set to {signal}'})
    
    return jsonify({'success': False, 'message': 'Invalid parameters'})

@app.route('/api/toggle_auto_mode', methods=['POST'])
def toggle_auto_mode():
    control_settings['auto_mode'] = not control_settings['auto_mode']
    return jsonify({
        'success': True, 
        'auto_mode': control_settings['auto_mode'],
        'message': f"Auto mode {'enabled' if control_settings['auto_mode'] else 'disabled'}"
    })

@app.route('/api/emergency_override', methods=['POST'])
def emergency_override():
    control_settings['emergency_override'] = not control_settings['emergency_override']
    if control_settings['emergency_override']:
        # Set all lanes to GREEN for emergency and clear manual overrides
        for lane_id, lane in traffic_data.items():
            lane['signal_status'] = 'GREEN'
            lane['manual_override'] = False
        print("Emergency override activated - all lanes GREEN")
    else:
        # Clear manual overrides and trigger immediate analysis
        for lane_id, lane in traffic_data.items():
            lane['manual_override'] = False
        # Trigger immediate traffic analysis to restore appropriate signals
        threading.Thread(target=restore_auto_signals, daemon=True).start()
        print("Emergency override deactivated - restoring auto control")
    
    return jsonify({
        'success': True,
        'emergency_override': control_settings['emergency_override'],
        'message': f"Emergency override {'activated' if control_settings['emergency_override'] else 'deactivated'}"
    })

def restore_auto_signals():
    """Immediately restore automatic signal control after emergency"""
    time.sleep(1)  # Brief delay to ensure emergency state is cleared
    if not control_settings['emergency_override'] and control_settings['auto_mode']:
        # Run immediate analysis
        densities = [lane['traffic_density'] for lane in traffic_data.values()]
        avg_density = sum(densities) / len(densities) if densities else 0
        
        for lane_id, lane in traffic_data.items():
            if lane['manual_override']:
                continue
                
            if lane['traffic_density'] < control_settings['low_traffic_threshold']:
                if lane['traffic_density'] < avg_density * 0.5:
                    lane['signal_status'] = 'RED'
                else:
                    lane['signal_status'] = 'YELLOW'
            else:
                lane['signal_status'] = 'GREEN'
        print("Auto signals restored")

@app.route('/api/reset_counts', methods=['POST'])
def reset_counts():
    for lane in traffic_data.values():
        lane['vehicle_count'] = 0
        lane['traffic_density'] = 0
        lane['vehicles_per_minute'] = 0
        lane['last_count_time'] = time.time()
    
    return jsonify({'success': True, 'message': 'All counts reset'})

@app.route('/api/performance_settings', methods=['GET', 'POST'])
def handle_performance_settings():
    """Get or update performance settings"""
    global combined_canvas, performance_settings
    
    if request.method == 'GET':
        return jsonify(performance_settings)
    
    # POST: Update settings
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        # Validate and update settings
        valid_keys = {'processing_fps', 'stream_fps', 'jpeg_quality', 'single_lane_mode', 'focused_lane', 'performance_mode'}
        for key, value in data.items():
            if key in valid_keys:
                if key == 'processing_fps' and (1 <= value <= 60):
                    performance_settings[key] = value
                elif key == 'stream_fps' and (5 <= value <= 60):
                    performance_settings[key] = value
                elif key == 'jpeg_quality' and (30 <= value <= 100):
                    performance_settings[key] = value
                elif key in ('single_lane_mode', 'performance_mode'):
                    performance_settings[key] = bool(value)
                elif key == 'focused_lane' and value in traffic_data:
                    performance_settings[key] = value
        
        # Performance mode preset
        if data.get('performance_mode'):
            performance_settings.update({
                'processing_fps': 30,
                'stream_fps': 30,
                'jpeg_quality': 60,
                'single_lane_mode': False
            })
        
        # Reset canvas when settings change
        combined_canvas = None
        
        return jsonify({
            'success': True, 
            'message': 'Performance settings updated',
            'settings': performance_settings
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error updating settings: {str(e)}'})

@app.route('/api/performance_stats', methods=['GET'])
def get_performance_stats():
    """Get real-time performance statistics"""
    try:
        stats = {
            'current_fps': {
                'processing': performance_settings['processing_fps'],
                'streaming': performance_settings['stream_fps']
            },
            'jpeg_quality': performance_settings['jpeg_quality'],
            'single_lane_mode': performance_settings['single_lane_mode'],
            'performance_mode': performance_settings['performance_mode'],
            'active_lanes': len([lane for lane in traffic_data.values() if lane['current_frame'] is not None]),
            'timestamp': time.time()
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    is_processing = True
    
    # Start video processing threads for each lane
    for lane_id in traffic_data.keys():
        thread = threading.Thread(target=process_lane_video, args=(lane_id,), daemon=True)
        thread.start()
    
    # Start traffic analysis manager
    analysis_thread = threading.Thread(target=traffic_analysis_manager, daemon=True)
    analysis_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)