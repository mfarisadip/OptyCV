import numpy as np
import supervision as sv
from ultralytics import YOLO
from datetime import datetime
import time
import cv2
import argparse

# Configuration
MODEL_PATH = 'weights/YOLOV8n640IR8.onnx'

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
args = parser.parse_args()

# Set source based on arguments
USE_CAMERA = args.source == 'camera'
SOURCE_VIDEO_PATH = args.video_path
TARGET_VIDEO_PATH = args.output
CAMERA_ID = args.camera_id

# Initialize video source and get video info
if USE_CAMERA:
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_ID}")
        exit()

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default fallback for camera

    video_info = sv.VideoInfo(width=width, height=height, fps=fps, total_frames=None)
    print(f"Camera {CAMERA_ID} opened: {width}x{height} @ {fps} FPS")
else:
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

line_y_position = int(video_info.height * 0.55)  # 55% from top

# Create horizontal line zone
LINE_START = sv.Point(0, line_y_position)
LINE_END = sv.Point(video_info.width, line_y_position)

# Load YOLO model
model = YOLO(MODEL_PATH)

# create BYTETracker instance for better object tracking
byte_tracker = sv.ByteTrack()

# LineZone untuk visualisasi garis (tanpa counter)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# create annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Detection tracking variables
pothole_count = 0
crack_count = 0
detection_log = []
processed_trackers = set()

def get_timestamp(frame_index, fps):
    """Convert frame index to timestamp in MM:SS format"""
    if frame_index is None or fps == 0:
        return "00:00"
    seconds = frame_index / fps
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def filter_detections(detections):
    """Filter detections to only include crack and pothole classes"""
    filtered_indices = []
    for i, class_id in enumerate(detections.class_id):
        class_name = model.names[class_id].lower()
        if class_name in ['crack', 'pothole']:
            filtered_indices.append(i)

    if filtered_indices:
        return detections[filtered_indices]
    else:
        return sv.Detections.empty()

def process_frame(frame: np.ndarray, frame_index: int = None) -> np.ndarray:
    """Process a single frame for crack and pothole detection"""
    global pothole_count, crack_count, detection_log, processed_trackers

    start_time = time.time()

    # model prediction on single frame
    results = model(frame, conf=0.4, iou=0.3, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    # filter for crack and pothole only
    detections = filter_detections(detections)

    # track detections
    detections = byte_tracker.update_with_detections(detections)

    # create labels for annotations
    labels = []
    for i in range(len(detections)):
        if detections.tracker_id is not None and i < len(detections.tracker_id):
            tracker_id = detections.tracker_id[i]
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            class_name = model.names[class_id]
            labels.append(f"#{tracker_id} {class_name} {confidence:0.2f}")
        else:
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            class_name = model.names[class_id]
            labels.append(f"{class_name} {confidence:0.2f}")

    # annotate frame with traces, boxes, and labels
    annotated_frame = trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # check for objects below the line
    timestamp = get_timestamp(frame_index, video_info.fps) if frame_index else "LIVE"
    if detections.tracker_id is not None:
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id not in processed_trackers:
                bbox = detections.xyxy[i]
                center_y = int((bbox[1] + bbox[3]) / 2)

                if center_y >= line_y_position:  # Object is below the line
                    class_id = detections.class_id[i]
                    class_name = model.names[class_id].lower()
                    confidence = detections.confidence[i]

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
                        print(f"🕳️ POTHOLE #{pothole_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y: {center_y}, Conf: {confidence:.2f})")

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
                        print(f"〰️ CRACK #{crack_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y: {center_y}, Conf: {confidence:.2f})")

                    processed_trackers.add(tracker_id)

    # add text overlay with statistics
    processing_time = time.time() - start_time

    # Frame info
    if frame_index and video_info.total_frames:
        frame_text = f'Frame: {frame_index}/{video_info.total_frames}'
    else:
        frame_text = f'Frame: {frame_index if frame_index else "LIVE"}'

    cv2.putText(annotated_frame, frame_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'Potholes: {pothole_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f'Cracks: {crack_count}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f'Time: {timestamp}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f'Process: {processing_time:.3f}s', (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw detection line
    annotated_frame = cv2.line(annotated_frame,
                             (LINE_START.x, LINE_START.y),
                             (LINE_END.x, LINE_END.y),
                             (255, 255, 0), 2)
    return annotated_frame

# define callback function to be used in video processing
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    return process_frame(frame, index)

def process_camera_feed():
    """Process live camera feed"""
    global cap

    print(f"Processing camera feed from Camera {CAMERA_ID}")
    print(f"Model: {MODEL_PATH}")
    print(f"Line position: Y={line_y_position} (55% from top)")
    print(f"Resolution: {video_info.width}x{video_info.height}")
    print(f"FPS: {video_info.fps}")
    print("Press 'q' to quit, 's' to save current frame")
    print("-" * 60)

    # Setup video writer if saving
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_info.fps,
                                     (video_info.width, video_info.height))
        print(f"Saving video to: {TARGET_VIDEO_PATH}")

    frame_count = 0
    start_processing_time = time.time()

    try:
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
                screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, processed_frame)
                print(f"Screenshot saved: {screenshot_path}")

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

# Main execution
if __name__ == "__main__":
    import cv2

    if USE_CAMERA:
        # Process camera feed
        processing_duration, total_frames = process_camera_feed()
        source_text = f"Camera {CAMERA_ID}"
    else:
        # Process video file (original functionality)
        print(f"Processing video: {SOURCE_VIDEO_PATH}")
        print(f"Model: {MODEL_PATH}")
        print(f"Line position: Y={line_y_position} (55% from top)")
        print(f"Output: {TARGET_VIDEO_PATH}")
        print(f"Total frames: {video_info.total_frames}")
        print(f"FPS: {video_info.fps}")
        print(f"Resolution: {video_info.width}x{video_info.height}")
        print("-" * 60)

        # process the whole video
        start_processing_time = time.time()

        sv.process_video(
            source_path=SOURCE_VIDEO_PATH,
            target_path=TARGET_VIDEO_PATH,
            callback=callback
        )

        processing_duration = time.time() - start_processing_time
        total_frames = video_info.total_frames
        source_text = SOURCE_VIDEO_PATH

    # Print final summary
    print("\n" + "="*60)
    print("🔍 FINAL DETECTION SUMMARY")
    print("="*60)
    print(f"📹 Source: {source_text}")
    print(f"🕳️  Total Potholes Detected: {pothole_count}")
    print(f"〰️  Total Cracks Detected: {crack_count}")
    print(f"📊 Total Detections: {pothole_count + crack_count}")
    print(f"⏱️  Processing Time: {processing_duration:.2f} seconds")

    if not USE_CAMERA:
        print(f"🎬 Video Duration: {video_info.total_frames/video_info.fps:.2f} seconds")
        print(f"🚀 Processing Speed: {video_info.total_frames/processing_duration:.2f} FPS")
    else:
        print(f"🎬 Total Frames Processed: {total_frames}")
        print(f"🚀 Processing Speed: {total_frames/processing_duration:.2f} FPS")

    if detection_log:
        print("\n📋 Detailed Detection Log:")
        print("-" * 60)
        pothole_logs = [log for log in detection_log if log['type'] == 'pothole']
        crack_logs = [log for log in detection_log if log['type'] == 'crack']

        if pothole_logs:
            print(f"\n🕳️  Potholes ({len(pothole_logs)}):")
            for i, log in enumerate(pothole_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print(f"   {i}. Tracker ID #{log['tracker_id']} at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Conf: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]")

        if crack_logs:
            print(f"\n〰️  Cracks ({len(crack_logs)}):")
            for i, log in enumerate(crack_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print(f"   {i}. Tracker ID #{log['tracker_id']} at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Conf: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]")

    print("="*60)

    # Save detection report to file
    report_content = f"""
OPTIMIZED CRACK AND POTHOLE DETECTION REPORT
Source: {source_text}
Model: {MODEL_PATH}
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Line Position: Y={line_y_position} (55% from top)

SUMMARY:
- Total Potholes Detected: {pothole_count}
- Total Cracks Detected: {crack_count}
- Total Detections: {pothole_count + crack_count}
- Processing Time: {processing_duration:.2f} seconds
- Processing Speed: {total_frames/processing_duration:.2f} FPS

DETAILED LOG:
"""

    if detection_log:
        for log in detection_log:
            detection_type = log.get('detection_type', 'unknown')
            report_content += f"- {log['type'].upper()} (Tracker #{log['tracker_id']}) at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Confidence: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]\n"

    with open("detection_report_optimized.txt", "w") as f:
        f.write(report_content)

    print(f"📄 Detection report saved to: detection_report_optimized.txt")
    print(f"🎥 Annotated video saved to: {TARGET_VIDEO_PATH}")
