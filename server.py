import cv2
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from datetime import datetime
import time
from Detector import HumanDetector
from OpticalFlowDetector import OpticalFlowTracking
from threading import Event

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

clients = {}

@app.route('/')
def index():
    return render_template('index.html')

def video_stream(sid):
    if(sid not in clients):
        print(f"[{sid}] Error: Client not found.")
        return
    if("video" not in clients[sid]):
        cap = cv2.VideoCapture("in.avi")    
    else:
        cap = cv2.VideoCapture(clients[sid]["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print(f"[{sid}] Error: Could not open video file.")
        return
    
    humanDetector = None
    first_frame = True
    old_frame=None

    while cap.isOpened():
        if clients[sid]["stop_event"].is_set():
            print(f"[{sid}] Stopping current video stream.")
            cap.release()
            return
        
        if first_frame:
            first_frame = False
            ret, old_frame = cap.read()
            if not ret:
                print(f"[{sid}] Error: Could not read the first frame.")
            humanDetector_opticalflow= OpticalFlowTracking()
            humanDetector = HumanDetector(old_frame)
            continue
        
        ret, frame = cap.read()
        if not ret:
            print(f"[{sid}] End of video or read error. Restarting video stream.")
            cap.release()
            return    
        try:
            if clients[sid]["stop_event"].is_set():
                cap.release()
                return
            
            boxes_of = humanDetector_opticalflow.detect_humans(old_frame, frame)
            frame_edited, boxes = humanDetector.detect_humans(frame, boxes_of, overlap_boxes_treshold=0.7)
            socketio.emit('n-boxes', len(boxes), to=sid)
            _, encoded_frame = cv2.imencode('.jpg', frame_edited)
            frame_data = encoded_frame.tobytes()
            socketio.emit('frame', frame_data, to=sid)
        except Exception as e:
            print(f"[{sid}] Error during detection: {e}")
        old_frame=frame
        time.sleep(1/fps)
    
    
    cap.release()

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"[{sid}] Client connected at {datetime.now()}")
    clients[sid] = {
        "thread": None,
        "stop_event": Event(),
        "video": "video/in.avi"
    }
    
    clients[sid]["thread"] = socketio.start_background_task(video_stream, sid)
    
@socketio.on('message')
def handle_message(data):
    sid = request.sid
    video = data
    print(f"[{sid}] Message from client: {data}")

    if clients[sid]["thread"]:
        clients[sid]["stop_event"].set()
        clients[sid]["thread"].join()

    clients[sid]["stop_event"].clear()
    clients[sid]["video"] = "video/"+video
    clients[sid]["thread"] = socketio.start_background_task(video_stream, sid)

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"[{sid}] Client disconnected at {datetime.now()}")

    if sid in clients:
        clients[sid]["stop_event"].set()
        if clients[sid]["thread"]:
            clients[sid]["thread"].join()
        del clients[sid]


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)