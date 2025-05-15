import base64
import io
import logging
import os
import time
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import uvicorn
from PIL import Image
from ultralytics import YOLO  # Import YOLO from ultralytics for YOLOv11

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Red Spider Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for templates and static files
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Create the dashboard HTML template
with open("templates/dashboard.html", "w") as f:
    # Dashboard HTML content remains the same as in the original code
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Red Spider Detection Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .main {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }
        .camera-container {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 560px;
        }
        .camera-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .camera-title {
            font-size: 1.2rem;
            font-weight: bold;
        }
        .camera-feed {
            width: 100%;
            height: 360px;
            background-color: #ddd;
            margin-bottom: 10px;
            position: relative;
            overflow: hidden;
        }
        .camera-feed img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .status-panel {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            width: 100%;
            margin-bottom: 20px;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .status-item:last-child {
            border-bottom: none;
        }
        .status-label {
            font-weight: bold;
        }
        .status-value {
            text-align: right;
        }
        .status-value.detected {
            color: #e74c3c;
            font-weight: bold;
        }
        .status-value.normal {
            color: #2ecc71;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 10px;
        }
        .button {
            padding: 8px 15px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .button:hover {
            background-color: #2980b9;
        }
        .button.danger {
            background-color: #e74c3c;
        }
        .button.danger:hover {
            background-color: #c0392b;
        }
        .detection-alert {
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #e74c3c;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            display: none;
            z-index: 1000;
        }
        @media (max-width: 768px) {
            .camera-container {
                max-width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Red Spider Detection Dashboard</h1>
        </div>
        
        <div class="status-panel">
            <div class="status-item">
                <div class="status-label">Detection Status:</div>
                <div id="detection-status" class="status-value normal">No Detection</div>
            </div>
            <div class="status-item">
                <div class="status-label">Confidence:</div>
                <div id="confidence" class="status-value">0.0</div>
            </div>
            <div class="status-item">
                <div class="status-label">Last Camera:</div>
                <div id="last-camera" class="status-value">None</div>
            </div>
            <div class="status-item">
                <div class="status-label">Last Detection:</div>
                <div id="last-detection" class="status-value">Never</div>
            </div>
            <div class="status-item">
                <div class="status-label">Threshold:</div>
                <div id="threshold" class="status-value">0.5</div>
            </div>
            <div class="controls">
                <button id="reset-button" class="button">Reset Detection</button>
                <input id="threshold-input" type="range" min="0.1" max="0.9" step="0.1" value="0.5">
                <button id="threshold-button" class="button">Set Threshold</button>
            </div>
        </div>
        
        <div class="main">
            <div class="camera-container">
                <div class="camera-header">
                    <div class="camera-title">Camera 1</div>
                    <div class="controls">
                        <button class="button" onclick="toggleView(1)">Toggle Detection View</button>
                    </div>
                </div>
                <div class="camera-feed">
                    <img id="camera1" src="/view/1" alt="Camera 1">
                </div>
            </div>
            
            <div class="camera-container">
                <div class="camera-header">
                    <div class="camera-title">Camera 2</div>
                    <div class="controls">
                        <button class="button" onclick="toggleView(2)">Toggle Detection View</button>
                    </div>
                </div>
                <div class="camera-feed">
                    <img id="camera2" src="/view/2" alt="Camera 2">
                </div>
            </div>
        </div>
    </div>
    
    <div id="detection-alert" class="detection-alert">
        <strong>ALERT:</strong> Red Spider Detected!
    </div>
    
    <script>
        let showingDetection = {
            1: false,
            2: false
        };
        
        function toggleView(cameraId) {
            showingDetection[cameraId] = !showingDetection[cameraId];
            const img = document.getElementById(`camera${cameraId}`);
            
            // Add a timestamp to prevent caching
            const timestamp = new Date().getTime();
            if (showingDetection[cameraId]) {
                img.src = `/view-with-detection/${cameraId}?t=${timestamp}`;
            } else {
                img.src = `/view/${cameraId}?t=${timestamp}`;
            }
        }
        
        function updateStatus() {
            fetch('/check-detection')
                .then(response => response.json())
                .then(data => {
                    const detectionStatus = document.getElementById('detection-status');
                    const confidence = document.getElementById('confidence');
                    const lastCamera = document.getElementById('last-camera');
                    const lastDetection = document.getElementById('last-detection');
                    const detectionAlert = document.getElementById('detection-alert');
                    
                    if (data.spider_detected) {
                        detectionStatus.textContent = 'SPIDER DETECTED';
                        detectionStatus.className = 'status-value detected';
                        confidence.textContent = data.confidence.toFixed(2);
                        lastCamera.textContent = `Camera ${data.camera_id}`;
                        
                        const date = new Date(data.timestamp * 1000);
                        lastDetection.textContent = date.toLocaleTimeString();
                        
                        detectionAlert.style.display = 'block';
                        setTimeout(() => {
                            detectionAlert.style.display = 'none';
                        }, 5000);
                    } else {
                        detectionStatus.textContent = 'No Detection';
                        detectionStatus.className = 'status-value normal';
                        confidence.textContent = '0.0';
                        detectionAlert.style.display = 'none';
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        function refreshImages() {
            const timestamp = new Date().getTime();
            for (let i = 1; i <= 2; i++) {
                const img = document.getElementById(`camera${i}`);
                const currentSrc = img.src.split('?')[0];
                img.src = `${currentSrc}?t=${timestamp}`;
            }
        }
        
        // Reset detection
        document.getElementById('reset-button').addEventListener('click', () => {
            fetch('/reset-detection')
                .then(response => response.json())
                .then(data => {
                    console.log('Detection reset:', data);
                    updateStatus();
                })
                .catch(error => console.error('Error:', error));
        });
        
        // Set threshold
        document.getElementById('threshold-button').addEventListener('click', () => {
            const thresholdValue = document.getElementById('threshold-input').value;
            document.getElementById('threshold').textContent = thresholdValue;
            
            fetch(`/set-threshold?threshold=${thresholdValue}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Threshold set:', data);
                })
                .catch(error => console.error('Error:', error));
        });
        
        // Initial status update
        updateStatus();
        
        // Refresh status and images periodically
        setInterval(updateStatus, 2000);
        setInterval(refreshImages, 5000);
    </script>
</body>
</html>""")

# Models for request/response
class StreamData(BaseModel):
    camera_id: int
    data: str  # Base64 encoded image data

class DetectionResponse(BaseModel):
    spider_detected: bool
    confidence: float = 0.0
    camera_id: Optional[int] = None
    timestamp: float

# Global variables
MODEL_PATH = "best.pt"  # Path to your YOLOv11 model
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
detection_results = {
    "spider_detected": False,
    "confidence": 0.0,
    "camera_id": None,
    "timestamp": time.time()
}

# Define the red spider class name pattern to match
RED_SPIDER_CLASS_PATTERN = "redspider"  # This will be used to check if a class name contains "redspider"

# Load YOLOv11 model
try:
    # Use YOLO class from ultralytics package for YOLOv11
    model = YOLO(MODEL_PATH)
    logger.info(f"YOLOv11 model loaded successfully from {MODEL_PATH}")
    
    # Log available class names from the model to help with debugging
    if hasattr(model, 'names'):
        logger.info(f"Model class names: {model.names}")
except Exception as e:
    logger.error(f"Failed to load YOLOv11 model: {e}")
    model = None

# Frame buffer for each camera
frames = {}
processed_frames = {}  # To store frames with detection boxes

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Red Spider Detection API with YOLOv11")
    if model is None:
        logger.warning("WARNING: YOLOv11 model not loaded. Detections will not work.")

@app.get("/")
async def root():
    return {"message": "Red Spider Detection API", "status": "running", "model": "YOLOv11"}

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """
    Web dashboard for viewing camera streams and detection status
    """
    return templates.TemplateResponse("dashboard.html", {"request": request})

def is_redspider_class(class_name):
    """
    Helper function to check if a class name is a red spider, 
    handling the specific format "redspider - v2 2023-02-24 6-12pm"
    """
    return RED_SPIDER_CLASS_PATTERN in class_name.lower()

@app.post("/stream-video")
async def receive_stream(data: StreamData):
    """
    Endpoint to receive video stream data from ESP32
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(data.data)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error(f"Failed to decode image from camera {data.camera_id}")
            return {"status": "error", "message": "Failed to decode image"}
        
        # Store frame
        frames[data.camera_id] = img
        
        # Run detection
        if model is not None:
            # Make a copy for processed frame
            processed_img = img.copy()
            
            # YOLOv11 prediction
            results = model.predict(img, conf=CONFIDENCE_THRESHOLD)
            
            # Check if red spiders are detected
            spider_detected = False
            highest_conf = 0.0
            
            # Process detection results
            for result in results:
                # Log the complete results for debugging
                logger.debug(f"Detection result: {result}")
                
                # Draw boxes for visualization on processed image
                for i, conf in enumerate(result.boxes.conf):
                    cls_id = int(result.boxes.cls[i].item())
                    class_name = result.names[cls_id]
                    confidence = conf.item()
                    
                    # Log each detection for debugging
                    logger.debug(f"Detection: Class {cls_id}, Name '{class_name}', Confidence {confidence:.4f}")
                    
                    # Check if this is a red spider detection using our helper function
                    if is_redspider_class(class_name):
                        # Get bounding box coordinates
                        box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = box
                        
                        # Draw rectangle and label on processed image - simplified display name
                        cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(processed_img, f"Red Spider: {confidence:.2f}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Update detection status
                        if confidence > highest_conf:
                            spider_detected = True
                            highest_conf = confidence
            
            # Store processed frame
            processed_frames[data.camera_id] = processed_img
            
            if spider_detected:
                # Update global detection status
                global detection_results
                detection_results = {
                    "spider_detected": True,
                    "confidence": float(highest_conf),
                    "camera_id": data.camera_id,
                    "timestamp": time.time()
                }
                logger.info(f"Red spider detected! Camera: {data.camera_id}, Confidence: {highest_conf:.2f}")
            else:
                # Reset detection after 5 seconds if no new detections
                if detection_results["spider_detected"] and (time.time() - detection_results["timestamp"]) > 5:
                    detection_results = {
                        "spider_detected": False,
                        "confidence": 0.0,
                        "camera_id": None,
                        "timestamp": time.time()
                    }
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing stream: {e}")
        if isinstance(e, ValueError) and "no detections" in str(e).lower():
            # Handle the "no detections" case gracefully
            return {"status": "success", "detections": "none"}
        else:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-detection")
async def check_detection():
    """
    Endpoint for ESP32 to check if red spiders are detected
    """
    return detection_results

@app.get("/view/{camera_id}")
async def view_camera(camera_id: int):
    """
    Endpoint to view camera stream in browser
    """
    try:
        camera_id = int(camera_id)
        if camera_id not in frames:
            # Return a placeholder image if camera not found
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Camera {camera_id} not connected", (100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, jpeg = cv2.imencode('.jpg', placeholder)
            return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")
        
        img = frames[camera_id]
        _, jpeg = cv2.imencode('.jpg', img)
        return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid camera ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/view-with-detection/{camera_id}")
async def view_camera_with_detection(camera_id: int):
    """
    Endpoint to view camera stream with detection boxes in browser
    """
    try:
        camera_id = int(camera_id)
        
        # Check if we have a processed frame for this camera
        if camera_id in processed_frames:
            img = processed_frames[camera_id]
        elif camera_id in frames:
            # If no processed frame but we have the original, use that
            img = frames[camera_id].copy()
        else:
            # Return a placeholder if camera not found
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Camera {camera_id} not connected", (100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, jpeg = cv2.imencode('.jpg', placeholder)
            return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")
        
        _, jpeg = cv2.imencode('.jpg', img)
        return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid camera ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-info")
async def debug_info():
    """
    Endpoint to get system debugging information
    """
    model_classes = None
    if model is not None and hasattr(model, 'names'):
        model_classes = model.names
        
    return {
        "model_loaded": model is not None,
        "model_type": "YOLOv11",
        "model_classes": model_classes,
        "cameras_connected": list(frames.keys()),
        "detection_status": detection_results,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

@app.post("/set-threshold")
async def set_threshold(threshold: float):
    """
    Endpoint to adjust the detection confidence threshold
    """
    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
    
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = threshold
    
    return {"status": "success", "new_threshold": CONFIDENCE_THRESHOLD}

@app.get("/reset-detection")
async def reset_detection():
    """
    Endpoint to manually reset detection status
    """
    global detection_results
    detection_results = {
        "spider_detected": False,
        "confidence": 0.0,
        "camera_id": None,
        "timestamp": time.time()
    }
    return {"status": "detection reset"}

@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded model
    """
    if model is None:
        return {"status": "error", "message": "No model loaded"}
    
    try:
        # Get model information
        model_info = {
            "model_type": "YOLOv11",
            "model_path": MODEL_PATH,
            "classes": model.names if hasattr(model, 'names') else "Unknown",
            "red_spider_pattern": RED_SPIDER_CLASS_PATTERN
        }
        return model_info
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/live-stream/{camera_id}")
async def live_stream(camera_id: int):
    """
    Stream MJPEG video for continuous live view (for advanced browsers)
    """
    async def generate_frames():
        while True:
            try:
                if int(camera_id) in frames:
                    img = frames[int(camera_id)]
                    _, jpeg = cv2.imencode('.jpg', img)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                else:
                    # Generate placeholder
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"Camera {camera_id} not connected", (100, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, jpeg = cv2.imencode('.jpg', placeholder)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                logger.error(f"Error in live stream: {e}")
                
            await asyncio.sleep(0.1)  # 10 FPS
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import asyncio
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)