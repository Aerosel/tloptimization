# tloptimization

This is a project that aims to optimize traffic light switching by detecting pedestrians and vehicles in real-time from video streams. The goal is to reduce unnecessary waiting times for pedestrians when no cars are present, and vice versa, by automating traffic light state changes based on object detection.

## Features

- Real-time pedestrian and vehicle detection from HLS video streams
- Uses YOLOv8n (lightweight model) for efficient CPU-based inference
- Visualizes detections with bounding boxes in a live video window
- Outputs FPS and cumulative detection counters to the console
- Handles connection interruptions with automatic reconnection

## Project Structure

- `detect_vehicles_pedestrians.py`: Main script for object detection from HLS stream
- Others: Additional files for full system integration (tbd)

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Linux/Mac: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`

3. Install dependencies:
   ```bash
   pip install ultralytics opencv-python
   ```

## Usage

Run the detection script:
```bash
python detect_vehicles_pedestrians.py
```

This will:
- Connect to the specified HLS stream URL
- Perform real-time detection of persons, cars, motorcycles, buses, and trucks within specified ROI (pedestrians only in ROI, vehicles frame-wide)
- Display a video window with bounding boxes around detected objects (red boxes for out-of-ROI pedestrians)
- Visualize pedestrian ROI (green rectangle), vehicle ROI (blue rectangle), traffic light (bottom-right), mouse coordinates (top-right), dwell times
- Display FPS, detection counters, and individual dwell times (in seconds) for active objects in ROIs
- Automatically reconnect if the stream is interrupted
- Manual traffic light control: Press 'r' for red light, 'g' for green light, 'q' to quit

## Configuration

The script is configured with the following defaults:
- Stream URL: `https://flussonic2.powernet.com.ru:444/user83475/tracks-v1/mono.m3u8?token=dont-panic-and-carry-a-towel`
- Detection classes: person (0), car (2), motorcycle (3), bus (5), truck (7)
- Model: yolov8n.pt (auto-downloaded on first run)
- Pedestrian Detection ROI: (820, 7, 500, 400) - Only detections within this region are counted as pedestrians.
- Vehicle Detection ROI: (0, 0, 1280, 720) - Only detections within this region are counted as vehicles (default full frame).

To modify, edit the constants at the top of `detect_vehicles_pedestrians.py`.

Additional configuration:
- MAX_MISSING_TIME = 1.0  # Seconds to keep object tracks if temporarily undetected

## Requirements

- Python 3.6+
- CPU with reasonable performance (optimized for low-end hardware)
- Internet connection for HLS stream access

## Future Integration

This script is a component of the full tloptimization system. It will be integrated with traffic light control logic to adjust signals based on detection results, prioritizing pedestrian flow when no vehicles are detected and vice versa.
