"""
Flask API Server for ESP32 Camera System with YOLOv11 Red Spider Detection
-------------------------------------------------------------------------
This application serves two main purposes:
1. Handle video streams from two ESP32-CAMs
2. Provide relay control API endpoints
3. Process video streams with YOLOv11 to detect red spiders
4. Automatically control relays based on detection results
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('camera_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECTION_THRESHOLD'] = 0.5  # Confidence threshold for detection
app.config['DETECTION_INTERVAL'] = 5  # Detection every 5 seconds

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for state tracking
camera_frames = {1: None, 2: None}
last_frame_time = {1: 0, 2: 0}
detection_results = {1: {"detected": False, "confidence": 0}, 2: {"detected": False, "confidence": 0}}
relay_states = {1: False, 2: False, 3: False, 4: False}
detection_lock = threading.Lock()  # Lock for thread-safe access to detection results

# Load YOLOv11 model
try:
    logging.info("Loading YOLOv11 model...")
    model = torch.hub.load('ultralytics/yolov11', 'custom', path='best.pt')
    model.conf = app.config['DETECTION_THRESHOLD']  # Set confidence threshold
    model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    device = next(model.parameters()).device
    logging.info(f"Model loaded successfully on {device}")
except Exception as e:
    logging.error(f"Error loading YOLOv11 model: {e}")
    model = None


@app.route('/')
def index():
    """Main page with system status and camera feeds"""
    return render_template('index.html')


@app.route('/api/stream/<int:camera_id>', methods=['POST'])
def receive_stream(camera_id):
    """Endpoint to receive video frames from ESP32-CAMs"""
    if camera_id not in [1, 2]:
        return jsonify({"error": "Invalid camera ID"}), 400

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files['image']
        
        # Save the received frame
        timestamp = int(time.time())
        filename = f"camera_{camera_id}_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the image for processing
        frame = cv2.imread(filepath)
        if frame is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Update the global frame for this camera
        with detection_lock:
            camera_frames[camera_id] = frame
            last_frame_time[camera_id] = time.time()
        
        # Clean up old files to prevent disk filling up
        # Keep only the latest file for each camera
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            if f.startswith(f"camera_{camera_id}_") and f != filename:
                try:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
                except:
                    pass
        
        return jsonify({"status": "success"}), 200
    
    except Exception as e:
        logging.error(f"Error processing stream from camera {camera_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/relay/status', methods=['GET'])
def get_relay_status():
    """Endpoint to retrieve current relay status for ESP32"""
    global relay_states
    return jsonify(relay_states)


@app.route('/api/relay/control', methods=['POST'])
def control_relay():
    """Endpoint to manually control relay states"""
    global relay_states
    
    data = request.json
    if not data:
        return jsonify({"error": "Invalid request data"}), 400
    
    # Update relay states based on request
    for relay_id in [1, 2, 3, 4]:
        relay_key = f"relay{relay_id}"
        if relay_key in data:
            relay_states[relay_id] = bool(data[relay_key])
    
    logging.info(f"Relay states updated manually: {relay_states}")
    return jsonify({"status": "success", "relay_states": relay_states}), 200


@app.route('/api/detection/status', methods=['GET'])
def get_detection_status():
    """Endpoint to get current detection status for both cameras"""
    return jsonify(detection_results)


@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    """Endpoint for streaming processed video with detection boxes"""
    if camera_id not in [1, 2]:
        return "Invalid camera ID", 400
    
    def generate():
        while True:
            with detection_lock:
                frame = camera_frames[camera_id]
                if frame is None:
                    # No frame available, return a blank image
                    blank = np.ones((480, 640, 3), dtype=np.uint8) * 200
                    cv2.putText(blank, f"Waiting for Camera {camera_id}...", 
                                (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    _, jpeg = cv2.imencode('.jpg', blank)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                    time.sleep(1)
                    continue
                
                # Draw detection information on the frame
                detection = detection_results[camera_id]
                frame_copy = frame.copy()
                
                if detection["detected"]:
                    # Add a red border for visual alert
                    frame_copy = cv2.rectangle(frame_copy, (0, 0), 
                                              (frame_copy.shape[1], frame_copy.shape[0]), 
                                              (0, 0, 255), 20)
                    
                    # Add text with confidence
                    text = f"RED SPIDER DETECTED ({detection['confidence']:.2f})"
                    cv2.putText(frame_copy, text, (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Add normal status text
                    cv2.putText(frame_copy, "No Detection", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add camera identifier
                cv2.putText(frame_copy, f"Camera {camera_id}", 
                            (frame_copy.shape[1] - 150, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Convert to JPEG
                _, jpeg = cv2.imencode('.jpg', frame_copy)
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            
            time.sleep(0.1)  # Control frame rate
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def detect_red_spiders():
    """Background thread to process frames with YOLOv11 for red spider detection"""
    global detection_results, relay_states
    
    logging.info("Red spider detection thread started")
    
    while True:
        try:
            for camera_id in [1, 2]:
                with detection_lock:
                    frame = camera_frames[camera_id]
                    last_time = last_frame_time[camera_id]
                
                # Skip if no frame available or if the frame is old (>30 seconds)
                if frame is None or time.time() - last_time > 30:
                    continue
                
                # Process with YOLOv11
                if model is not None:
                    # Make a copy to avoid modifying the original
                    frame_copy = frame.copy()
                    
                    # Run detection
                    results = model(frame_copy)
                    
                    # Check if red spiders were detected
                    detected = False
                    max_conf = 0
                    
                    # Parse results
                    if len(results.xyxy[0]) > 0:  # If any detections
                        for detection in results.xyxy[0]:
                            conf = detection[4].item()  # Confidence score
                            cls = int(detection[5].item())  # Class ID
                            
                            # Assuming red spider is class 0
                            if cls == 0 and conf > max_conf:
                                max_conf = conf
                                detected = True
                    
                    # Update detection results
                    with detection_lock:
                        detection_results[camera_id] = {
                            "detected": detected,
                            "confidence": max_conf
                        }
                    
                    # Log detection
                    if detected:
                        logging.info(f"Red spider detected on camera {camera_id} with confidence {max_conf:.2f}")
                    
                    # Update relay states based on detection
                    if detected and max_conf >= app.config['DETECTION_THRESHOLD']:
                        # Turn on relay 1 for camera 1 detection, relay 2 for camera 2
                        relay_states[camera_id] = True
                        
                        # For demonstration, also turn on relay 3 when any detection occurs
                        relay_states[3] = True
                        
                        logging.info(f"Activated relay {camera_id} due to detection")
            
            # Sleep to control detection rate
            time.sleep(app.config['DETECTION_INTERVAL'])
            
        except Exception as e:
            logging.error(f"Error in detection thread: {e}")
            time.sleep(5)  # Wait before retrying on error


@app.route('/templates/index.html')
def serve_template():
    """Serve the index.html template"""
    return render_template('index.html')


if __name__ == '__main__':
    # Start detection thread
    detection_thread = threading.Thread(target=detect_red_spiders, daemon=True)
    detection_thread.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)