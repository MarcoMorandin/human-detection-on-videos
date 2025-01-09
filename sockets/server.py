import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from datetime import datetime
import time
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from Detector import HumanDetector

app = Flask(__name__)
socketio = SocketIO(app, debug=True, cors_allowed_origins='*')

@app.route('/')
def index():
    return render_template('index.html')

def video_stream():
    cap = cv2.VideoCapture("../in.avi")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return  # Handle this error gracefully
    
    humanDetector = None
    first_frame = True

    while cap.isOpened():
        if first_frame:
            first_frame = False
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read the first frame.")
                
            humanDetector = HumanDetector(frame)
            continue
        
        ret, frame = cap.read()
        if not ret:
            print("End of video or read error. Restarting video stream.")
            video_stream()
            continue
            
        try:
            frame_edited, boxes = humanDetector.detect_humans(frame)
            _, encoded_frame = cv2.imencode('.jpg', frame_edited)
            frame_data = encoded_frame.tobytes()
            socketio.emit('frame', frame_data)
        except Exception as e:
            print(f"Error during detection: {e}")
        time.sleep(1/int(15))
    
    
    cap.release()

@socketio.on('connect')
def handle_connect():
    print(f"Client connected at {datetime.now()}")
    socketio.start_background_task(video_stream)

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected at {datetime.now()}")

if __name__ == '__main__':
    socketio.run(app, allow_unsafe_werkzeug=True, debug=True)