import cv2
import time
from ultralytics import YOLO

# Define the classes to detect
TARGET_CLASSES = [0, 2, 3, 5, 7]  # person(0), car(2), truck(7), bus(5), motorcycle(3)
CLASS_NAMES = {
    0: 'person',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Stream URL
STREAM_URL = 'https://flussonic2.powernet.com.ru:444/user83475/tracks-v1/mono.m3u8?token=dont-panic-and-carry-a-towel'

# Regions of Interest (x, y, w, h)
PEDESTRIAN_ROI = (820, 7, 500, 400)  # Green ROI for pedestrians
VEHICLE_ROI = (289, 219, 1600, 600)  # Full frame for vehicles, adjust if needed

# Tracking parameters
MAX_MISSING_TIME = 1.0  # Seconds to keep track if object temporarily disappears

# Mouse position
mouse_x, mouse_y = -1, -1

# Traffic light state
traffic_light_state = 'red'

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def main():
    global traffic_light_state
    # Load YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Create window for mouse callback
    cv2.namedWindow('Detection')
    cv2.setMouseCallback('Detection', mouse_callback)

    # Open video capture
    cap = cv2.VideoCapture(STREAM_URL)

    prev_time = time.time()
    frame_count = 0
    total_person_frames = 0  # Frames with at least one pedestrian in ROI
    total_vehicle_frames = 0  # Frames with at least one vehicle in ROI
    pedestrian_tracks = []  # List of active pedestrian tracks: [{'bbox': (x,y,w,h), 'time': 0, 'id': int, 'consecutive': 0}]
    vehicle_tracks = []  # For vehicles
    track_id_counter = 0  # global ID counter

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Frame not read, attempting to reconnect...")
                cap.release()
                time.sleep(5)  # Wait before reconnecting
                cap = cv2.VideoCapture(STREAM_URL)
                continue

            # Get frame dimensions
            height, width = frame.shape[:2]

            # Perform detection
            results = model(frame, device='cpu', classes=TARGET_CLASSES, verbose=False)

            # Process detections
            frame_counters = {cls: 0 for cls in CLASS_NAMES.values()}
            annotated_frame = frame.copy()

            # Draw ROIs
            # Pedestrian ROI (green)
            x_roi_p, y_roi_p, w_roi_p, h_roi_p = PEDESTRIAN_ROI
            cv2.rectangle(annotated_frame, (x_roi_p, y_roi_p), (x_roi_p + w_roi_p, y_roi_p + h_roi_p), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Pedestrian ROI", (x_roi_p, y_roi_p - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Vehicle ROI (blue)
            x_roi_v, y_roi_v, w_roi_v, h_roi_v = VEHICLE_ROI
            if VEHICLE_ROI != (0, 0, width, height):  # Only draw if not full frame
                cv2.rectangle(annotated_frame, (x_roi_v, y_roi_v), (x_roi_v + w_roi_v, y_roi_v + h_roi_v), (255, 0, 0), 2)
                cv2.putText(annotated_frame, "Vehicle ROI", (x_roi_v, y_roi_v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    if cls_id in CLASS_NAMES:
                        cls_name = CLASS_NAMES[cls_id]

                        # Determine ROI
                        is_pedestrian = cls_id == 0
                        is_vehicle = cls_id in [2, 3, 5, 7]
                        if is_pedestrian:
                            x_roi, y_roi, w_roi, h_roi = PEDESTRIAN_ROI
                        elif is_vehicle:
                            x_roi, y_roi, w_roi, h_roi = VEHICLE_ROI
                        else:
                            continue  # Skip other classes if any

                        center_x = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                        center_y = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                        in_roi = (x_roi <= center_x <= x_roi + w_roi and y_roi <= center_y <= y_roi + h_roi)

                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if in_roi:
                            frame_counters[cls_name] += 1
                            color = (255, 0, 0)  # Blue for valid detections
                        else:
                            color = (0, 0, 255)  # Red for out-of-ROI detections (not counted)

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color[2], 1)  # Text color matching box

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # Collect current detections in ROI
            current_pedestrian_dets = []
            current_vehicle_dets = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    x1, y1, x2, y2 = box.xyxy[0]
                    w = x2 - x1
                    h = y2 - y1
                    bbox = (x1.item(), y1.item(), w.item(), h.item())
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    if cls_id == 0:
                        x_roi, y_roi, w_roi, h_roi = PEDESTRIAN_ROI
                        if x_roi <= center_x <= x_roi + w_roi and y_roi <= center_y <= y_roi + h_roi:
                            current_pedestrian_dets.append(bbox)
                    elif cls_id in [2, 3, 5, 7]:
                        x_roi, y_roi, w_roi, h_roi = VEHICLE_ROI
                        if x_roi <= center_x <= x_roi + w_roi and y_roi <= center_y <= y_roi + h_roi:
                            current_vehicle_dets.append(bbox)

            # Update pedestrian tracks
            matched = [False] * len(pedestrian_tracks)
            for det in current_pedestrian_dets:
                best_iou = 0
                best_idx = -1
                for i, track in enumerate(pedestrian_tracks):
                    if not matched[i]:
                        iou = calculate_iou(track['bbox'], det)
                        if iou > best_iou and iou > 0.3:
                            best_iou = iou
                            best_idx = i
                if best_idx != -1:
                    pedestrian_tracks[best_idx]['bbox'] = det
                    pedestrian_tracks[best_idx]['last_seen'] = current_time
                    matched[best_idx] = True
                else:
                    new_track = {'bbox': det, 'start_time': current_time, 'last_seen': current_time}
                    pedestrian_tracks.append(new_track)

            now = current_time
            pedestrian_tracks = [t for t in pedestrian_tracks if now - t['last_seen'] < MAX_MISSING_TIME]  # keep tracks if object disappears temporarily

            # Update vehicle tracks
            matched = [False] * len(vehicle_tracks)
            for det in current_vehicle_dets:
                best_iou = 0
                best_idx = -1
                for i, track in enumerate(vehicle_tracks):
                    if not matched[i]:
                        iou = calculate_iou(track['bbox'], det)
                        if iou > best_iou and iou > 0.4:
                            best_iou = iou
                            best_idx = i
                if best_idx != -1:
                    vehicle_tracks[best_idx]['bbox'] = det
                    vehicle_tracks[best_idx]['last_seen'] = current_time
                    matched[best_idx] = True
                else:
                    new_track = {'bbox': det, 'start_time': current_time, 'last_seen': current_time}
                    vehicle_tracks.append(new_track)

            vehicle_tracks = [t for t in vehicle_tracks if now - t['last_seen'] < MAX_MISSING_TIME]

            # Update counters and totals
            num_persons = len(current_pedestrian_dets)
            num_vehicles = len(current_vehicle_dets)
            if num_persons > 0:
                total_person_frames += 1
            if num_vehicles > 0:
                total_vehicle_frames += 1

            # Overlay text
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Persons: {num_persons} | Cars: {num_vehicles}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if mouse_x >= 0 and mouse_y >= 0:
                cv2.putText(annotated_frame, f"Mouse: ({mouse_x}, {mouse_y})", (width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw traffic light
            light_x = width - 100
            light_y = height - 200
            cv2.rectangle(annotated_frame, (light_x, light_y), (light_x + 50, light_y + 100), (0, 0, 0), -1)
            # Red light
            color_red = (0, 0, 255) if traffic_light_state == 'red' else (100, 100, 100)
            cv2.circle(annotated_frame, (light_x + 25, light_y + 25), 15, color_red, -1)
            # Green light
            color_green = (0, 255, 0) if traffic_light_state == 'green' else (100, 100, 100)
            cv2.circle(annotated_frame, (light_x + 25, light_y + 75), 15, color_green, -1)
            cv2.putText(annotated_frame, "Traffic Light", (light_x - 50, light_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Dwell times display
            if pedestrian_tracks:
                pedestrian_dwell_times = [current_time - t['start_time'] for t in pedestrian_tracks]
                times_str = ", ".join(f"{dt:.1f}s" for dt in sorted(pedestrian_dwell_times)[:5])
                cv2.putText(annotated_frame, f"Ped Dwell: [{times_str}] ({len(pedestrian_tracks)})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if vehicle_tracks:
                vehicle_dwell_times = [current_time - t['start_time'] for t in vehicle_tracks]
                v_times_str = ", ".join(f"{dt:.1f}s" for dt in sorted(vehicle_dwell_times)[:5])
                cv2.putText(annotated_frame, f"Veh Dwell: [{v_times_str}] ({len(vehicle_tracks)})", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            cv2.imshow('Detection', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                traffic_light_state = 'green'
            elif key == ord('r'):
                traffic_light_state = 'red'

            frame_count += 1

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
