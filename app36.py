import numpy as np
import cv2
import argparse
from datetime import datetime
import time
import os
import onnxruntime as ort

# Configuration
MODEL_PATH = 'best_ir8.onnx'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Crack and Pothole Detection with Camera Input')
parser.add_argument('--source', type=str, default='camera',
                    help='Source: "camera" for live camera, "video" for video file')
parser.add_argument('--video-path', type=str, default='waki.mp4',
                    help='Path to video file when source is "video"')
parser.add_argument('--camera-id', type=int, default=0,
                    help='Camera ID (default: 0)')
parser.add_argument('--output', type=str, default='output_annotated_video_optimized.mp4',
                    help='Output video file path')
parser.add_argument('--save-video', action='store_true',
                    help='Save processed video to file')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to use: "cpu" or "cuda"')
args = parser.parse_args()

# Set device based on arguments
USE_CUDA = args.device.lower() == 'cuda'
DEVICE = 'cuda' if USE_CUDA else 'cpu'

# Set source based on arguments
USE_CAMERA = args.source == 'camera'
SOURCE_VIDEO_PATH = args.video_path
TARGET_VIDEO_PATH = args.output
CAMERA_ID = args.camera_id

# Initialize video source and get video info
if USE_CAMERA:
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Cannot open camera {}".format(CAMERA_ID))
        exit()

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default fallback for camera

    video_info = {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': None
    }
    print("Camera {} opened: {}x{} @ {} FPS".format(CAMERA_ID, width, height, fps))
else:
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Cannot open video file: {}".format(SOURCE_VIDEO_PATH))
        exit()
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_info = {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames
    }
    cap.release()

line_y_position = int(video_info['height'] * 0.55)  # 55% from top

# Load ONNX model
try:
    # Set providers for ONNX Runtime
    providers = ['CUDAExecutionProvider'] if USE_CUDA else ['CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print("ONNX model loaded successfully on {}".format(DEVICE))
except Exception as e:
    print("Error loading ONNX model: {}".format(e))
    exit()

# Get input and output info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_names = [output.name for output in session.get_outputs()]

# Class names (replace with your actual class names)
CLASS_NAMES = ['crack', 'pothole']  # Adjust according to your model

# Detection tracking variables
pothole_count = 0
crack_count = 0
detection_log = []
processed_trackers = set()

# Simple centroid tracker implementation
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return []

        # Compute centroids for current rects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If we are currently not tracking any objects, register each of them
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
            return []

        # Compute the distance between each pair of object centroids and input centroids
        objectIDs = list(self.objects.keys())
        objectCentroids = np.array(list(self.objects.values()))
        D = np.linalg.norm(objectCentroids[:, np.newaxis] - inputCentroids, axis=2)

        # Find the smallest distances between each pair of centroids
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        # Keep track of which centroids have already been matched
        usedRows = set()
        usedCols = set()

        # Loop over the combination of the (row, column) index tuples
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = inputCentroids[col]
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        # Compute both the row and column index we have NOT yet used
        unusedRows = set(range(D.shape[0])).difference(usedRows)
        unusedCols = set(range(D.shape[1])).difference(usedCols)

        # If the number of object centroids is greater than the number of input centroids, check for disappeared objects
        if D.shape[0] >= D.shape[1]:
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
        else:
            for col in unusedCols:
                self.register(inputCentroids[col])

        # Return the list of tracked objects with their IDs
        tracked_objects = []
        for objectID, centroid in self.objects.items():
            tracked_objects.append((objectID, centroid))

        return tracked_objects

# Create centroid tracker instance
centroid_tracker = CentroidTracker()

def get_timestamp(frame_index, fps):
    """Convert frame index to timestamp in MM:SS format"""
    if frame_index is None or fps == 0:
        return "00:00"
    seconds = frame_index / fps
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return "{:02d}:{:02d}".format(minutes, seconds)

def preprocess_image(image, input_shape):
    """Preprocess image for ONNX model input"""
    # Resize image to match input shape
    resized = cv2.resize(image, (input_shape[3], input_shape[2]))
    
    # Normalize image
    resized = resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    resized = np.expand_dims(resized, axis=0)
    
    # Change HWC to CHW
    resized = np.transpose(resized, (0, 3, 1, 2))
    
    return resized

def postprocess_output(output, image_shape, conf_threshold=0.4):
    """Postprocess ONNX model output"""
    # Convert output to numpy array
    output0 = np.asarray(output[0])
    
    # If output is 1D, reshape to 2D with 6 columns (YOLO format)
    if output0.ndim == 1:
        output0 = output0.reshape(-1, 6)
    
    # Extract boxes, scores, and class IDs
    boxes = output0[:, :4]
    scores = output0[:, 4]
    class_ids = output0[:, 5].astype(int)
    
    # Filter by confidence
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    # Scale boxes to original image size
    input_height, input_width = image_shape[0], image_shape[1]
    output_height, output_width = image_shape[2], image_shape[3]
    
    x_scale = input_width / output_width
    y_scale = input_height / output_height
    
    scaled_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1 * x_scale)
        y1 = int(y1 * y_scale)
        x2 = int(x2 * x_scale)
        y2 = int(y2 * y_scale)
        scaled_boxes.append([x1, y1, x2, y2])
    
    return scaled_boxes, scores, class_ids

def draw_detections(image, detections, tracker_ids=None, confidences=None, class_ids=None):
    """Draw bounding boxes and labels on image"""
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections[i]
        
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Create label
        label = ""
        if class_ids is not None and i < len(class_ids):
            class_name = CLASS_NAMES[class_ids[i]]
            label = class_name
        
        if tracker_ids is not None and i < len(tracker_ids):
            tracker_id = tracker_ids[i]
            label = "ID: {} {}".format(tracker_id, label)
            
        if confidences is not None and i < len(confidences):
            label += " {:.2f}".format(confidences[i])
        
        # Draw label
        cv2.putText(image, label, (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def process_frame(frame: np.ndarray, frame_index: int = None) -> np.ndarray:
    """Process a single frame for crack and pothole detection"""
    global pothole_count, crack_count, detection_log, processed_trackers

    start_time = time.time()

    # Preprocess image
    input_tensor = preprocess_image(frame, input_shape)
    
    # Run inference
    outputs = session.run(output_names, {input_name: input_tensor})
    
    # Postprocess output
    detections, confidences, class_ids = postprocess_output(outputs, input_shape)
    
    # Filter for crack and pothole only
    filtered_indices = []
    for i, class_id in enumerate(class_ids):
        if class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
            if class_name in ['crack', 'pothole']:
                filtered_indices.append(i)
    
    if filtered_indices:
        filtered_detections = [detections[i] for i in filtered_indices]
        filtered_confidences = [confidences[i] for i in filtered_indices]
        filtered_class_ids = [class_ids[i] for i in filtered_indices]
    else:
        filtered_detections = []
        filtered_confidences = []
        filtered_class_ids = []
    
    # Update tracker with filtered detections
    tracked_objects = centroid_tracker.update(filtered_detections)
    
    # Create tracker IDs and associate with detections
    tracker_ids = []
    for object_id, centroid in tracked_objects:
        # Find the detection closest to this centroid
        min_dist = float('inf')
        closest_idx = -1
        
        for i, detection in enumerate(filtered_detections):
            cX = int((detection[0] + detection[2]) / 2.0)
            cY = int((detection[1] + detection[3]) / 2.0)
            dist = np.sqrt((cX - centroid[0])**2 + (cY - centroid[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        if closest_idx != -1 and min_dist < 50:  # Threshold for matching
            tracker_ids.append(object_id)
        else:
            tracker_ids.append(-1)  # No match
    
    # Draw detections
    annotated_frame = frame.copy()
    if filtered_detections:
        annotated_frame = draw_detections(
            annotated_frame, 
            filtered_detections, 
            tracker_ids, 
            filtered_confidences,
            filtered_class_ids
        )
    
    # Check for objects below the line
    timestamp = get_timestamp(frame_index, video_info['fps']) if frame_index else "LIVE"
    
    for i, tracker_id in enumerate(tracker_ids):
        if tracker_id != -1 and tracker_id not in processed_trackers:
            detection = filtered_detections[i]
            center_y = int((detection[1] + detection[3]) / 2)
            
            if center_y >= line_y_position:  # Object is below the line
                class_id = filtered_class_ids[i]
                class_name = CLASS_NAMES[class_id]
                confidence = filtered_confidences[i]
                
                if class_name == 'pothole':
                    pothole_count += 1
                    detection_log.append({
                        'type': 'pothole',
                        'tracker_id': int(tracker_id),
                        'timestamp': timestamp,
                        'frame': frame_index if frame_index else 0,
                        'position': center_y,
                        'confidence': confidence,
                        'detection_type': 'detected_below_line'
                    })
                    print("🕳️ POTHOLE #{0} (ID:#{1}) DETECTED at {2} (Y: {3}, Conf: {4:.2f})".format(
                        pothole_count, tracker_id, timestamp, center_y, confidence))
                
                elif class_name == 'crack':
                    crack_count += 1
                    detection_log.append({
                        'type': 'crack',
                        'tracker_id': int(tracker_id),
                        'timestamp': timestamp,
                        'frame': frame_index if frame_index else 0,
                        'position': center_y,
                        'confidence': confidence,
                        'detection_type': 'detected_below_line'
                    })
                    print("〰️ CRACK #{0} (ID:#{1}) DETECTED at {2} (Y: {3}, Conf: {4:.2f})".format(
                        crack_count, tracker_id, timestamp, center_y, confidence))
                
                processed_trackers.add(tracker_id)
    
    # Add text overlay with statistics
    processing_time = time.time() - start_time
    
    # Frame info
    if frame_index and video_info['total_frames']:
        frame_text = 'Frame: {0}/{1}'.format(frame_index, video_info['total_frames'])
    else:
        frame_text = 'Frame: {0}'.format(frame_index if frame_index else "LIVE")
    
    cv2.putText(annotated_frame, frame_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, 'Potholes: {0}'.format(pothole_count), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, 'Cracks: {0}'.format(crack_count), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, 'Time: {0}'.format(timestamp), (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, 'Process: {0:.3f}s'.format(processing_time), (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, 'Device: {0}'.format(DEVICE.upper()), (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw detection line
    annotated_frame = cv2.line(annotated_frame,
                             (0, line_y_position),
                             (video_info['width'], line_y_position),
                             (255, 255, 0), 2)
    
    return annotated_frame

def process_camera_feed():
    """Process live camera feed"""
    global cap

    print("Processing camera feed from Camera {}".format(CAMERA_ID))
    print("Model: {}".format(MODEL_PATH))
    print("Line position: Y={} (55% from top)".format(line_y_position))
    print("Resolution: {}x{}".format(video_info['width'], video_info['height']))
    print("FPS: {}".format(video_info['fps']))
    print("Using CUDA: {}".format(USE_CUDA))
    print("Press 'q' to quit, 's' to save current frame")
    print("-" * 60)

    # Setup video writer if saving
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_info['fps'],
                                     (video_info['width'], video_info['height']))
        print("Saving video to: {}".format(TARGET_VIDEO_PATH))

    frame_count = 0
    start_processing_time = time.time()

    try:
        cap = cv2.VideoCapture(CAMERA_ID)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                break

            # Process frame
            processed_frame = process_frame(frame, frame_count)

            # Save frame if recording
            if video_writer:
                video_writer.write(processed_frame)

            # Display frame
            cv2.imshow('Crack and Pothole Detection - Live Feed', processed_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = "screenshot_{}.jpg".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
                cv2.imwrite(screenshot_path, processed_frame)
                print("Screenshot saved: {}".format(screenshot_path))

            frame_count += 1

    except KeyboardInterrupt:
        print("\nCamera processing interrupted by user")

    finally:
        # Cleanup
        if video_writer:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

    processing_duration = time.time() - start_processing_time
    return processing_duration, frame_count

def process_video_file():
    """Process video file"""
    print("Processing video: {}".format(SOURCE_VIDEO_PATH))
    print("Model: {}".format(MODEL_PATH))
    print("Line position: Y={} (55% from top)".format(line_y_position))
    print("Output: {}".format(TARGET_VIDEO_PATH))
    print("Total frames: {}".format(video_info['total_frames']))
    print("FPS: {}".format(video_info['fps']))
    print("Resolution: {}x{}".format(video_info['width'], video_info['height']))
    print("Using CUDA: {}".format(USE_CUDA))
    print("-" * 60)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_info['fps'],
                                 (video_info['width'], video_info['height']))

    frame_count = 0
    start_processing_time = time.time()

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = process_frame(frame, frame_count)

            # Save frame
            video_writer.write(processed_frame)

            frame_count += 1

            # Print progress
            if frame_count % 100 == 0:
                print("Processed {} frames".format(frame_count))

    except KeyboardInterrupt:
        print("\nVideo processing interrupted by user")

    finally:
        # Cleanup
        video_writer.release()
        cap.release()

    processing_duration = time.time() - start_processing_time
    return processing_duration, frame_count

# Main execution
if __name__ == "__main__":
    if USE_CAMERA:
        # Process camera feed
        processing_duration, total_frames = process_camera_feed()
        source_text = "Camera {}".format(CAMERA_ID)
    else:
        # Process video file
        processing_duration, total_frames = process_video_file()
        source_text = SOURCE_VIDEO_PATH

    # Print final summary
    print("\n" + "="*60)
    print("🔍 FINAL DETECTION SUMMARY")
    print("="*60)
    print("📹 Source: {}".format(source_text))
    print("🕳️  Total Potholes Detected: {}".format(pothole_count))
    print("〰️  Total Cracks Detected: {}".format(crack_count))
    print("📊 Total Detections: {}".format(pothole_count + crack_count))
    print("⏱️  Processing Time: {:.2f} seconds".format(processing_duration))

    if not USE_CAMERA:
        print("🎬 Video Duration: {:.2f} seconds".format(video_info['total_frames']/video_info['fps']))
        print("🚀 Processing Speed: {:.2f} FPS".format(video_info['total_frames']/processing_duration))
    else:
        print("🎬 Total Frames Processed: {}".format(total_frames))
        print("🚀 Processing Speed: {:.2f} FPS".format(total_frames/processing_duration))

    if detection_log:
        print("\n📋 Detailed Detection Log:")
        print("-" * 60)
        pothole_logs = [log for log in detection_log if log['type'] == 'pothole']
        crack_logs = [log for log in detection_log if log['type'] == 'crack']

        if pothole_logs:
            print("\n🕳️  Potholes ({}):".format(len(pothole_logs)))
            for i, log in enumerate(pothole_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print("   {}. Tracker ID #{0} at {1} (Frame {2}, Y: {3}, Conf: {4:.2f}) [{5}]".format(
                    i, log['tracker_id'], log['timestamp'], log['frame'], 
                    log['position'], log['confidence'], detection_type.replace('_', ' ').title()))

        if crack_logs:
            print("\n〰️  Cracks ({}):".format(len(crack_logs)))
            for i, log in enumerate(crack_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print("   {}. Tracker ID #{0} at {1} (Frame {2}, Y: {3}, Conf: {4:.2f}) [{5}]".format(
                    i, log['tracker_id'], log['timestamp'], log['frame'], 
                    log['position'], log['confidence'], detection_type.replace('_', ' ').title()))

    print("="*60)

    # Save detection report to file
    report_content = """
OPTIMIZED CRACK AND POTHOLE DETECTION REPORT
Source: {0}
Model: {1}
Processing Date: {2}
Line Position: Y={3} (55% from top)
Using CUDA: {4}

SUMMARY:
- Total Potholes Detected: {5}
- Total Cracks Detected: {6}
- Total Detections: {7}
- Processing Time: {8:.2f} seconds
- Processing Speed: {9:.2f} FPS

DETAILED LOG:
""".format(
        source_text, MODEL_PATH, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        line_y_position, USE_CUDA, pothole_count, crack_count, 
        pothole_count + crack_count, processing_duration, total_frames/processing_duration
    )

    if detection_log:
        for log in detection_log:
            detection_type = log.get('detection_type', 'unknown')
            report_content += "- {0} (Tracker #{1}) at {2} (Frame {3}, Y: {4}, Confidence: {5:.2f}) [{6}]\n".format(
                log['type'].upper(), log['tracker_id'], log['timestamp'], 
                log['frame'], log['position'], log['confidence'], 
                detection_type.replace('_', ' ').title())

    with open("detection_report_optimized.txt", "w") as f:
        f.write(report_content)

    print("📄 Detection report saved to: detection_report_optimized.txt")
    print("🎥 Annotated video saved to: {}".format(TARGET_VIDEO_PATH))