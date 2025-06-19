import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import time

# Load the trained YOLOv8 model
model = YOLO("weights/best.pt")  # Make sure best.pt is placed inside 'weights' folder

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Define the classes to detect (example: 0 = Backboard, 2 = SmallBackboard)
class_ids_to_detect = [0, 2]

# Optional: Set font for OpenCV text
font = cv2.FONT_HERSHEY_SIMPLEX

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert RealSense frame to OpenCV format
        color_image = np.asanyarray(color_frame.get_data())

        # Time inference start
        start_time = time.time()

        # Run YOLOv8 inference
        results = model.predict(color_image, imgsz=640, conf=0.5)

        # Calculate FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-6)

        # Extract results
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # Annotate detected objects
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            if int(class_id) in class_ids_to_detect:
                class_name = model.names[int(class_id)]
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box and label
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(color_image, label, (x1, y1 - 10), font, 0.9, (0, 255, 0), 2)

        # Display FPS on frame
        cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30), font, 1, (0, 255, 0), 2)

        # Show the result
        cv2.imshow("YOLOv8 RealSense Inference", color_image)

        # Print detections
        print(f"FPS: {fps:.2f}")
        print(f"Detected Objects: {results[0].boxes}")

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
