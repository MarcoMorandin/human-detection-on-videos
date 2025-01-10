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
from OpticalFlowDetector import OpticalFlowTracking
from threading import Event

thread_event = Event()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/')
def index():
    return render_template('index.html')

def video_stream(video = "in.avi"):
    global thread_event
    cap = cv2.VideoCapture("../" + video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    humanDetector = None
    first_frame = True
    old_frame=None
    while cap.isOpened():
        if thread_event.is_set():
            print("Stopping current video stream.")
            break
        
        if first_frame:
            first_frame = False
            ret, old_frame = cap.read()
            if not ret:
                print("Error: Could not read the first frame.")
            humanDetector_opticalflow= OpticalFlowTracking()
            humanDetector = HumanDetector(old_frame)
            continue
        
        ret, frame = cap.read()
        if not ret:
            print("End of video or read error. Restarting video stream.")
            video_stream(video)
            continue
            
        try:
            boxes_of = humanDetector_opticalflow.detect_humans(old_frame, frame)
            frame_edited, boxes = humanDetector.detect_humans(frame, boxes_of, overlap_boxes_treshold=0.7)
            socketio.emit('n-boxes', len(boxes))
            _, encoded_frame = cv2.imencode('.jpg', frame_edited)
            frame_data = encoded_frame.tobytes()
            socketio.emit('frame', frame_data)
        except Exception as e:
            print(f"Error during detection: {e}")
        old_frame=frame
        time.sleep(1/fps)
    
    
    cap.release()

@socketio.on('connect')
def handle_connect():
    global thread_event
    print(f"Client connected at {datetime.now()}")
    thread_event.set()
    thread_event.clear()
    socketio.start_background_task(video_stream)
    
@socketio.on('message')
def handle_message(data):
    global thread_event
    print(f"Message from client: {data}")
    if not thread_event.is_set():
        thread_event.set()
        time.sleep(1/10)
    thread_event.clear()    
    socketio.start_background_task(video_stream, data)

@socketio.on('disconnect')
def handle_disconnect():
    global thread_event
    print(f"Client disconnected at {datetime.now()}")
    thread_event.set()

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)