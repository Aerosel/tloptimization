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

# Region of Interest for pedestrians (x, y, w, h)
PEDESTRIAN_ROI = (820, 7, 500, 400)  # Adjust these coordinates as needed

# Mouse position
mouse_x, mouse_y = -1, -1

# Traffic light state
traffic_light_state = 'red'

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

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

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Frame not read, attempting to reconnect...")
                cap.release()
                time.sleep(5)  # Wait before reconnecting
                cap = cv2.VideoCapture(STREAM_URL)
                continue

            # Perform detection
            results = model(frame, device='cpu', classes=TARGET_CLASSES, verbose=False)

            # Process detections
            frame_counters = {cls: 0 for cls in CLASS_NAMES.values()}
            annotated_frame = frame.copy()

            # Draw pedestrian ROI
            x_roi, y_roi, w_roi, h_roi = PEDESTRIAN_ROI
            cv2.rectangle(annotated_frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Pedestrian ROI", (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    if cls_id in CLASS_NAMES:
                        cls_name = CLASS_NAMES[cls_id]

                        # Check ROI for pedestrians
                        if cls_id == 0:  # person
                            center_x = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                            center_y = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                            if not (x_roi <= center_x <= x_roi + w_roi and y_roi <= center_y <= y_roi + h_roi):
                                # Still draw the box but don't count
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red outline for out-of-ROI pedestrians
                                cv2.putText(annotated_frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                                continue  # Skip counting

                        frame_counters[cls_name] += 1

                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Get frame dimensions
            height, width = annotated_frame.shape[:2]

            # Overlay text
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Persons: {frame_counters['person']} | Cars: {frame_counters['car']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
