# Human Detection Video Stream System

A real-time human detection system that processes video streams using a combination of optical flow tracking, image contouring and HOG-based human detection, served through a web interface.

## Features

- Real-time human detection in video streams
- Combination of multiple detection techniques:
  - Optical flow tracking for motion detection
  - HOG-based human detection
  - Background subtraction
  - Edge detection
- Web-based interface for viewing detection results
- Support for multiple concurrent clients
- Real-time detection count display

## Technical Architecture

The system consists of three main components:

1. **Human Detector**: Uses HOG (Histogram of Oriented Gradients) descriptors and background subtraction for human detection
2. **Optical Flow Tracker**: Implements Farneback optical flow algorithm combined with background subtraction and edge detection
3. **Web Server**: Flask-based server with SocketIO for real-time video streaming

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MarcoMorandin/human-detection-on-videos
cd human-detection-on-videos
```
2. **Create a virtual environment**
```bash
python -m venv .venv
```
3. **Activate the virtual environment**
   * Unix based OS
   ```bash
   source .venv/bin/activate
   ```
   * Windows
   ```bash
   .\\.venv\bin\activate
   ```
4. **Install required packages:**
```bash
pip install -r requirements.txt
```

## Usage
1. Start the server:
```bash
python -m server
```

2. Access the web interface:
- Open a web browser and navigate to `http://127.0.0.1:5000`
- The video stream will automatically start with human detection overlay
- The number of detected humans will be displayed in real-time
- It is possible to view multiple cameras

## Docker usage
> [!WARNING]
> The system uses a significant amount of CPU and memory, so it is recommended to increase the resources allocated to Docker in order to ensure smooth system operation.

1. **Clone the repository:**
```bash
git clone https://github.com/MarcoMorandin/human-detection-on-videos
cd human-detection-on-videos
```
1. **Run docker compose**
```bash
docker compose up --build
```
1. Access the web interface:
- Open a web browser and navigate to `http://127.0.0.1:8080`
  
## Result example:
Follows a short example of the obtained result:
![Example of the obtained result](result_example.gif)

