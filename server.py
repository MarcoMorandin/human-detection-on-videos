import cv2
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from datetime import datetime
import time
from Detector import HumanDetector
from OpticalFlowDetector import OpticalFlowTracking
from threading import Event

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Dictionary to store connected client information
clients = {}

@app.route('/')
def index():
    # Render the main page of the web app
    return render_template('index.html')

def video_stream(sid):
    # Check if client exists
    if(sid not in clients):
        print(f"[{sid}] Error: Client not found.")
        return
    
    # Open video file (only for simulation purposes, in real life this would be a video stream)
    # if the client has not specified a video file, use the default one
    if("video" not in clients[sid]):
        cap = cv2.VideoCapture("in.avi")    
    else:
        cap = cv2.VideoCapture(clients[sid]["video"])
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Check if the video could be opened otherwise exit
    if not cap.isOpened():
        print(f"[{sid}] Error: Could not open video file.")
        return
    
    humanDetector = None
    first_frame = True
    old_frame=None

    # Read the video frame by frame until the end of the video
    while cap.isOpened():
        # Check if the client has requested to stop the video stream
        if clients[sid]["stop_event"].is_set():
            print(f"[{sid}] Stopping current video stream.")
            cap.release()
            return
        
        # Read the first frame and initialize the human detector (the first frame is used as a background)
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
            # Check if the client has requested to stop the video stream while processing
            if clients[sid]["stop_event"].is_set():
                cap.release()
                return
            
            # Get the bounding boxes of humans in the current frame detected by the optical flow
            boxes_of = humanDetector_opticalflow.detect_humans(old_frame, frame)
            # Get the frame with the bounding boxes drawned and the bounding boxes
            frame_edited, boxes = humanDetector.detect_humans(frame, boxes_of, overlap_boxes_treshold=0.7)
            # Send the number of bounding boxes to the client
            socketio.emit('n-boxes', len(boxes), to=sid)
            # Encode the frame and send it to the client
            _, encoded_frame = cv2.imencode('.jpg', frame_edited)
            frame_data = encoded_frame.tobytes()
            socketio.emit('frame', frame_data, to=sid)
        except Exception as e:
            print(f"[{sid}] Error during detection: {e}")
        old_frame=frame
        time.sleep(1/fps)
    
    
    cap.release()

# SocketIO event handlers for client connections
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"[{sid}] Client connected at {datetime.now()}")
    
    # Add client to the dictionary
    clients[sid] = {
        "thread": None,
        "stop_event": Event(),
        "video": "video/in.avi"
    }
    
    # Start the video stream for the client
    clients[sid]["thread"] = socketio.start_background_task(video_stream, sid)
    
# SocketIO event handlers for client messages
@socketio.on('message')
def handle_message(data):
    sid = request.sid
    video = data
    print(f"[{sid}] Message from client: {data}")

    # Stop the current video stream if it is running
    if clients[sid]["thread"]:
        clients[sid]["stop_event"].set()
        clients[sid]["thread"].join()

    # Start the video stream with the new video file
    clients[sid]["stop_event"].clear()
    clients[sid]["video"] = "video/"+video
    clients[sid]["thread"] = socketio.start_background_task(video_stream, sid)

# SocketIO event handlers for client disconnections
@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"[{sid}] Client disconnected at {datetime.now()}")

    # Clean up client information
    if sid in clients:
        clients[sid]["stop_event"].set()
        if clients[sid]["thread"]:
            clients[sid]["thread"].join()
        del clients[sid]


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)