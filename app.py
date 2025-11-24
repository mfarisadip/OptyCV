#!/usr/bin/env python3
"""
Optimized Crack and Pothole Detection
Usage:
    python app.py --source video --video-path pole.mp4 --device cuda --cuda-device 0 --save-video
"""
import argparse
from datetime import datetime
import time
import os
import sys

import cv2
import numpy as np
import torch

# supervision & ultralytics
import supervision as sv
from ultralytics import YOLO

# ---------------------------
# Configuration & Arguments
# ---------------------------
parser = argparse.ArgumentParser(description='Crack and Pothole Detection with Camera or Video input')
parser.add_argument('--source', type=str, default='camera',
                    help='Source: "camera" for live camera, "video" for video file')
parser.add_argument('--video-path', type=str, default='waki.mp4',
                    help='Path to video file when source is "video"')
parser.add_argument('--camera-id', type=int, default=0,
                    help='Camera ID (default: 0)')
parser.add_argument('--output', type=str, default=None,
                    help='Output video file path (auto-generated if not specified)')
parser.add_argument('--save-video', action='store_true',
                    help='Save processed video to file')
parser.add_argument('--device', type=str, default='auto',
                    help='Device to use: "cuda", "cpu", or "auto" (default: auto)')
parser.add_argument('--cuda-device', type=int, default=0,
                    help='CUDA device ID when using CUDA (default: 0)')
parser.add_argument('--model', type=str, default='weights/YOLOV8n320IR8.onnx',
                    help='Model path (can be .onnx or .pt). For ONNX use device arg at predict-time.')
parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
parser.add_argument('--iou', type=float, default=0.3, help='NMS IoU threshold')
args = parser.parse_args()

MODEL_PATH = args.model
CONF_THRESHOLD = args.conf
IOU_THRESHOLD = args.iou

# Parse resolution
try:
    width, height = map(int, args.resolution.split('x'))
except:
    width, height = 640, 360
    print("⚠️  Invalid resolution format, using 640x360")

# ---------------------------
# CUDA / Device Utilities
# ---------------------------
def check_cuda_availability():
    """Return tuple (is_available, device_count, current_device_index, device_name)"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current = torch.cuda.current_device()
        try:
            name = torch.cuda.get_device_name(current)
        except Exception:
            name = f"cuda:{current}"
        return True, device_count, current, name
    else:
        return False, 0, None, None

CUDA_AVAILABLE, CUDA_DEVICE_COUNT, CUDA_CURRENT_DEVICE, CUDA_DEVICE_NAME = check_cuda_availability()

def determine_device_str_and_arg(requested: str, cuda_idx: int = 0):
    """
    Returns:
      DEVICE_STR: string used in logs like 'cuda:0' or 'cpu'
      device_arg: int (cuda index) or 'cpu' - this is passed to ultralytics ONNX inference as device
    """
    req = requested.lower()
    if req == 'cuda':
        if CUDA_AVAILABLE and cuda_idx < CUDA_DEVICE_COUNT:
            return f'cuda:{cuda_idx}', int(cuda_idx)
        else:
            print("⚠️ CUDA requested but not available or invalid index. Falling back to CPU.")
            return 'cpu', 'cpu'
    elif req == 'cpu':
        return 'cpu', 'cpu'
    elif req == 'auto':
        if CUDA_AVAILABLE:
            return f'cuda:{cuda_idx}', int(cuda_idx)
        else:
            return 'cpu', 'cpu'
    else:
        # unknown string -> try auto fallback
        print(f"⚠️ Unknown device '{requested}'. Using auto detection.")
        return ('cuda:0', 0) if CUDA_AVAILABLE else ('cpu', 'cpu')

DEVICE_STR, device_arg = determine_device_str_and_arg(args.device, args.cuda_device)

# ---------------------------
# Output Folder and Paths Setup
# ---------------------------
def setup_output_paths(source_path, save_video_flag):
    """Create output folder and generate dynamic output paths"""
    # Create output folder if it doesn't exist
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created output folder: {output_folder}")

    # Generate base filename from source
    if args.source == 'camera':
        base_name = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        # Extract filename without extension
        base_name = os.path.splitext(os.path.basename(source_path))[0]

    # Generate output paths
    video_path = os.path.join(output_folder, f"{base_name}_annotated.mp4") if save_video_flag else None
    report_path = os.path.join(output_folder, f"{base_name}_report.txt")

    return base_name, video_path, report_path

# ---------------------------
# Video Source Initialization
# ---------------------------
USE_CAMERA = args.source == 'camera'
SOURCE_VIDEO_PATH = args.video_path

# Setup output paths
BASE_NAME, TARGET_VIDEO_PATH, REPORT_PATH = setup_output_paths(SOURCE_VIDEO_PATH, args.save_video)

CAMERA_ID = args.camera_id

def initialize_jetson_camera():
    """Initialize camera with Jetson optimizations"""
    print(f"🎥 Initializing Jetson camera {CAMERA_ID}...")

    # Try multiple camera initialization methods

    # Method 1: GStreamer pipeline (best for Jetson)
    gst_pipeline = (
        f"nvarguscamerasrc sensor-id={CAMERA_ID} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )

    # Method 2: Simpler GStreamer pipeline
    gst_simple = (
        f"nvarguscamerasrc sensor-id={CAMERA_ID} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height} ! "
        "nvvidconv ! video/x-raw, format=BGR ! appsink"
    )

    # Method 3: Fallback to standard VideoCapture
    cap_methods = [
        (gst_pipeline, cv2.CAP_GSTREAMER, "GStreamer Pipeline"),
        (gst_simple, cv2.CAP_GSTREAMER, "Simple GStreamer"),
        (CAMERA_ID, cv2.CAP_ANY, "Standard Camera"),
    ]

    for pipeline, cap_type, method_name in cap_methods:
        try:
            cap = cv2.VideoCapture(pipeline, cap_type)
            if cap.isOpened():
                print(f"✅ Camera initialized using {method_name}")

                # Test if we can actually get frames
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"📸 Successfully captured test frame: {test_frame.shape}")
                    return cap
                else:
                    print(f"⚠️  {method_name}: Cannot capture frames, trying next method...")
                    cap.release()
            else:
                print(f"⚠️  {method_name}: Cannot open camera, trying next method...")

        except Exception as e:
            print(f"⚠️  {method_name} error: {e}")
            if 'cap' in locals():
                cap.release()

    print("❌ All camera initialization methods failed!")
    print("🔧 Troubleshooting tips:")
    print("   - Check camera connection")
    print("   - Try different camera ID: --camera-id 1")
    print("   - Check if camera is being used by another application")
    return None

if USE_CAMERA:
    cap = initialize_jetson_camera()
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_ID}")
        exit(1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30
    video_info = sv.VideoInfo(width=width, height=height, fps=fps, total_frames=None)
    print(f"Camera {CAMERA_ID} opened: {width}x{height} @ {fps} FPS")
else:
    if not os.path.exists(SOURCE_VIDEO_PATH):
        print(f"Error: video file '{SOURCE_VIDEO_PATH}' not found.")
        exit(1)
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"Video loaded: {SOURCE_VIDEO_PATH} -> {video_info.width}x{video_info.height} @ {video_info.fps} FPS, total {video_info.total_frames} frames")

# line position (55% from top)
line_y_position = int(video_info.height * 0.55)
LINE_START = sv.Point(0, line_y_position)
LINE_END = sv.Point(video_info.width, line_y_position)

# ---------------------------
# Load YOLO model (ONNX or PT)
# ---------------------------
print(f"🔄 Loading YOLO model from {MODEL_PATH}...")
print(f"🖥️  Target runtime device string: {DEVICE_STR}  | device_arg (for inference) = {device_arg}")
# Force task='detect' to avoid task-guessing
model = YOLO(MODEL_PATH, task='detect')

# Important: DO NOT call model.to(...) for exported formats (ONNX/TensorRT).
# If model is a .pt and you want PyTorch operations, you could use model.to('cuda:0'),
# but for ONNX we pass device at inference time.

print("✅ Model loaded. Will use runtime device when calling model(..., device=device_arg).")

# ---------------------------
# Tracking & Annotators
# ---------------------------
byte_tracker = sv.ByteTrack()          # ByteTrack wrapper from supervision
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# ---------------------------
# Detection Tracking Variables
# ---------------------------
pothole_count = 0
alligator_cracking_count = 0
lateral_cracking_count = 0
longitudinal_cracking_count = 0
rutting_count = 0
crack_count = 0  # generic crack count fallback
detection_log = []
processed_trackers = set()

def get_timestamp(frame_index, fps):
    """Convert frame index to timestamp in MM:SS format"""
    if frame_index is None or fps == 0 or fps is None:
        return "00:00"
    seconds = frame_index / fps
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def filter_detections(detections):
    """Filter detections to only include crack and pothole related classes"""
    # Return an sv.Detections object containing only desired classes
    if detections is None or len(detections) == 0:
        return sv.Detections.empty()

    # Valid classes that should be detected
    valid_classes = [
        'pothole',
        'alligator cracking',
        'lateral cracking',
        'longitudinal cracking',
        'rutting',
        'crack'  # fallback for generic crack class
    ]

    filtered_indices = []
    for i, class_id in enumerate(detections.class_id):
        # use model.names to map class id to name
        # ensure class_id int
        try:
            class_name = model.names[int(class_id)].lower()
        except Exception:
            class_name = str(class_id).lower()

        # Check if class_name is in our valid classes list
        if class_name in valid_classes:
            filtered_indices.append(i)

    if filtered_indices:
        return detections[filtered_indices]
    else:
        return sv.Detections.empty()

def process_frame(frame: np.ndarray, frame_index: int = None) -> np.ndarray:
    """Process a single frame for crack and pothole detection"""
    global pothole_count, crack_count, alligator_cracking_count, lateral_cracking_count, longitudinal_cracking_count, rutting_count, detection_log, processed_trackers

    start_time = time.time()

    # Run inference - pass device_arg for ONNX / exported models
    # model(...) returns list of Result objects; results[0] is for this frame
    try:
        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False, device=device_arg)
    except Exception as e:
        # Fallback: try without device_arg (some runtimes accept 'cpu'/'cuda:0')
        print(f"⚠️ Inference call with device_arg failed: {e}. Retrying without explicit device...")
        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    # Convert to supervision detections
    detections = sv.Detections.from_ultralytics(results[0])

    # Filter to desired classes
    detections = filter_detections(detections)

    # Track detections
    detections = byte_tracker.update_with_detections(detections)

    # Build labels
    labels = []
    for i in range(len(detections)):
        if hasattr(detections, 'tracker_id') and detections.tracker_id is not None and i < len(detections.tracker_id):
            tracker_id = detections.tracker_id[i]
            class_id = int(detections.class_id[i])
            confidence = float(detections.confidence[i])
            class_name = model.names[class_id] if class_id in model.names else str(class_id)
            labels.append(f"#{tracker_id} {class_name} {confidence:0.2f}")
        else:
            class_id = int(detections.class_id[i])
            confidence = float(detections.confidence[i])
            class_name = model.names[class_id] if class_id in model.names else str(class_id)
            labels.append(f"{class_name} {confidence:0.2f}")

    # Annotate traces, boxes, labels
    annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Timestamp for logs
    timestamp = get_timestamp(frame_index, video_info.fps) if frame_index is not None else "LIVE"

    # Check objects below the line and log new trackers
    if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
            if tracker_id not in processed_trackers:
                bbox = detections.xyxy[i]
                center_y = int((bbox[1] + bbox[3]) / 2)
                # check if object center is below line
                if center_y >= line_y_position:
                    class_id = int(detections.class_id[i])
                    class_name = model.names[class_id].lower() if class_id in model.names else "unknown"
                    confidence = float(detections.confidence[i])

                    # Handle different types of detections
                    if class_name == 'pothole':
                        pothole_count += 1
                        detection_log.append({
                            'type': 'pothole',
                            'tracker_id': int(tracker_id),
                            'timestamp': timestamp,
                            'frame': frame_index if frame_index is not None else 0,
                            'position': int(center_y),
                            'confidence': confidence,
                            'detection_type': 'detected_below_line'
                        })
                        print(f"🕳️ POTHOLE #{pothole_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Conf:{confidence:.2f})")

                    elif class_name == 'alligator cracking':
                        alligator_cracking_count += 1
                        detection_log.append({
                            'type': 'alligator_cracking',
                            'tracker_id': int(tracker_id),
                            'timestamp': timestamp,
                            'frame': frame_index if frame_index is not None else 0,
                            'position': int(center_y),
                            'confidence': confidence,
                            'detection_type': 'detected_below_line'
                        })
                        print(f"🐊 ALLIGATOR CRACKING #{alligator_cracking_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Conf:{confidence:.2f})")

                    elif class_name == 'lateral cracking':
                        lateral_cracking_count += 1
                        detection_log.append({
                            'type': 'lateral_cracking',
                            'tracker_id': int(tracker_id),
                            'timestamp': timestamp,
                            'frame': frame_index if frame_index is not None else 0,
                            'position': int(center_y),
                            'confidence': confidence,
                            'detection_type': 'detected_below_line'
                        })
                        print(f"↔️ LATERAL CRACKING #{lateral_cracking_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Conf:{confidence:.2f})")

                    elif class_name == 'longitudinal cracking':
                        longitudinal_cracking_count += 1
                        detection_log.append({
                            'type': 'longitudinal_cracking',
                            'tracker_id': int(tracker_id),
                            'timestamp': timestamp,
                            'frame': frame_index if frame_index is not None else 0,
                            'position': int(center_y),
                            'confidence': confidence,
                            'detection_type': 'detected_below_line'
                        })
                        print(f"↕️ LONGITUDINAL CRACKING #{longitudinal_cracking_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Conf:{confidence:.2f})")

                    elif class_name == 'rutting':
                        rutting_count += 1
                        detection_log.append({
                            'type': 'rutting',
                            'tracker_id': int(tracker_id),
                            'timestamp': timestamp,
                            'frame': frame_index if frame_index is not None else 0,
                            'position': int(center_y),
                            'confidence': confidence,
                            'detection_type': 'detected_below_line'
                        })
                        print(f"🛤️  RUTTING #{rutting_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Conf:{confidence:.2f})")

                    elif class_name == 'crack':
                        crack_count += 1
                        detection_log.append({
                            'type': 'crack',
                            'tracker_id': int(tracker_id),
                            'timestamp': timestamp,
                            'frame': frame_index if frame_index is not None else 0,
                            'position': int(center_y),
                            'confidence': confidence,
                            'detection_type': 'detected_below_line'
                        })
                        print(f"〰️ CRACK #{crack_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Conf:{confidence:.2f})")

                    processed_trackers.add(tracker_id)

    # Overlay stats text
    processing_time = time.time() - start_time
    if frame_index is not None and video_info.total_frames:
        frame_text = f'Frame: {frame_index}/{video_info.total_frames}'
    else:
        frame_text = f'Frame: {frame_index if frame_index is not None else "LIVE"}'

        # Calculate total cracks (all crack types combined)
    total_cracks = alligator_cracking_count + lateral_cracking_count + longitudinal_cracking_count + rutting_count + crack_count

    cv2.putText(annotated_frame, frame_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'Potholes: {pothole_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f'Cracks: {total_cracks}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
    cv2.putText(annotated_frame, f'Time: {timestamp}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f'Process: {processing_time:.3f}s', (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    device_text = f'Device: {DEVICE_STR.upper()}'
    device_color = (0, 255, 0) if DEVICE_STR.startswith('cuda') else (0, 165, 255)
    cv2.putText(annotated_frame, device_text, (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, device_color, 2)

    # Draw detection line
    annotated_frame = cv2.line(annotated_frame,
                               (LINE_START.x, LINE_START.y),
                               (LINE_END.x, LINE_END.y),
                               (255, 255, 0), 2)

    return annotated_frame

# callback wrapper for sv.process_video (if used)
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    return process_frame(frame, index)

# ---------------------------
# Processing Loop
# ---------------------------
def process_camera_feed():
    global cap
    print(f"Processing camera feed from Camera {CAMERA_ID}")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE_STR} (device_arg={device_arg})")
    print(f"Line position: Y={line_y_position} (55% from top)")
    print(f"Resolution: {video_info.width}x{video_info.height}")
    print(f"FPS: {video_info.fps}")
    print("Press 'q' to quit, 's' to save current frame")
    print("-" * 60)

    video_writer = None
    if args.save_video and TARGET_VIDEO_PATH:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_info.fps,
                                       (video_info.width, video_info.height))
        print(f"📹 Saving video to: {TARGET_VIDEO_PATH}")

    frame_count = 0
    start_processing_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                break

            processed_frame = process_frame(frame, frame_count)

            if video_writer:
                video_writer.write(processed_frame)

            cv2.imshow('Crack and Pothole Detection - Live Feed', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, processed_frame)
                print(f"Screenshot saved: {screenshot_path}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nCamera processing interrupted by user")

    finally:
        if video_writer:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

    return time.time() - start_processing_time, frame_count

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    if USE_CAMERA:
        processing_duration, total_frames = process_camera_feed()
        source_text = f"Camera {CAMERA_ID}"
    else:
        # Process video file using supervision.process_video wrapper (uses callback)
        print(f"🎥 Processing video: {SOURCE_VIDEO_PATH}")
        print(f"📁 Output folder: output/")
        if TARGET_VIDEO_PATH:
            print(f"📹 Output video: {TARGET_VIDEO_PATH}")
        print(f"📄 Output report: {REPORT_PATH}")
        print(f"🤖 Model: {MODEL_PATH}")
        print(f"🖥️  Device: {DEVICE_STR} (device_arg={device_arg})")
        print(f"📍 Line position: Y={line_y_position} (55% from top)")
        print(f"🎞️  Total frames: {video_info.total_frames}")
        print(f"⚡ FPS: {video_info.fps}")
        print(f"📐 Resolution: {video_info.width}x{video_info.height}")
        print("-" * 60)

        start_processing_time = time.time()
        # supervision wrapper will call callback(frame, index)
        sv.process_video(
            source_path=SOURCE_VIDEO_PATH,
            target_path=TARGET_VIDEO_PATH,
            callback=callback
        )
        processing_duration = time.time() - start_processing_time
        total_frames = video_info.total_frames
        source_text = SOURCE_VIDEO_PATH

    # Summary
    print("\n" + "="*60)
    print("🔍 FINAL DETECTION SUMMARY")
    print("="*60)
    print(f"📹 Source: {source_text}")
    print(f"🖥️  Device: {DEVICE_STR.upper()} (device_arg={device_arg})")
    print(f"🕳️  Total Potholes Detected: {pothole_count}")
    print(f"🐊  Total Alligator Cracking Detected: {alligator_cracking_count}")
    print(f"↔️  Total Lateral Cracking Detected: {lateral_cracking_count}")
    print(f"↕️  Total Longitudinal Cracking Detected: {longitudinal_cracking_count}")
    print(f"🛤️  Total Rutting Detected: {rutting_count}")
    print(f"〰️  Total Other Cracks Detected: {crack_count}")
    total_detections = pothole_count + alligator_cracking_count + lateral_cracking_count + longitudinal_cracking_count + rutting_count + crack_count
    print(f"📊 Total Detections: {total_detections}")
    print(f"⏱️  Processing Time: {processing_duration:.2f} seconds")

    if not USE_CAMERA:
        if video_info.fps and video_info.total_frames:
            print(f"🎬 Video Duration: {video_info.total_frames/video_info.fps:.2f} seconds")
            print(f"🚀 Processing Speed: {video_info.total_frames/processing_duration:.2f} FPS")
    else:
        print(f"🎬 Total Frames Processed: {total_frames}")
        print(f"🚀 Processing Speed: {total_frames/processing_duration:.2f} FPS")

    if detection_log:
        print("\n📋 Detailed Detection Log:")
        print("-" * 60)
        pothole_logs = [log for log in detection_log if log['type'] == 'pothole']
        alligator_logs = [log for log in detection_log if log['type'] == 'alligator_cracking']
        lateral_logs = [log for log in detection_log if log['type'] == 'lateral_cracking']
        longitudinal_logs = [log for log in detection_log if log['type'] == 'longitudinal_cracking']
        rutting_logs = [log for log in detection_log if log['type'] == 'rutting']
        crack_logs = [log for log in detection_log if log['type'] == 'crack']

        if pothole_logs:
            print(f"\n🕳️  Potholes ({len(pothole_logs)}):")
            for i, log in enumerate(pothole_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print(f"   {i}. Tracker ID #{log['tracker_id']} at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Conf: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]")

        if alligator_logs:
            print(f"\n🐊  Alligator Cracking ({len(alligator_logs)}):")
            for i, log in enumerate(alligator_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print(f"   {i}. Tracker ID #{log['tracker_id']} at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Conf: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]")

        if lateral_logs:
            print(f"\n↔️  Lateral Cracking ({len(lateral_logs)}):")
            for i, log in enumerate(lateral_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print(f"   {i}. Tracker ID #{log['tracker_id']} at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Conf: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]")

        if longitudinal_logs:
            print(f"\n↕️  Longitudinal Cracking ({len(longitudinal_logs)}):")
            for i, log in enumerate(longitudinal_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print(f"   {i}. Tracker ID #{log['tracker_id']} at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Conf: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]")

        if rutting_logs:
            print(f"\n🛤️  Rutting ({len(rutting_logs)}):")
            for i, log in enumerate(rutting_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print(f"   {i}. Tracker ID #{log['tracker_id']} at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Conf: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]")

        if crack_logs:
            print(f"\n〰️  Other Cracks ({len(crack_logs)}):")
            for i, log in enumerate(crack_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print(f"   {i}. Tracker ID #{log['tracker_id']} at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Conf: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]")

    print("="*60)

    # Save report
    cuda_device_info = f"CUDA Device: {CUDA_DEVICE_NAME} (ID: {args.cuda_device})" if DEVICE_STR.startswith('cuda') else "Device: CPU"
    report_content = f"""
OPTIMIZED CRACK AND POTHOLE DETECTION REPORT
Source: {source_text}
Model: {MODEL_PATH}
{cuda_device_info}
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Line Position: Y={line_y_position} (55% from top)

SUMMARY:
- Total Potholes Detected: {pothole_count}
- Total Alligator Cracking Detected: {alligator_cracking_count}
- Total Lateral Cracking Detected: {lateral_cracking_count}
- Total Longitudinal Cracking Detected: {longitudinal_cracking_count}
- Total Rutting Detected: {rutting_count}
- Total Other Cracks Detected: {crack_count}
- Total Detections: {pothole_count + alligator_cracking_count + lateral_cracking_count + longitudinal_cracking_count + rutting_count + crack_count}
- Processing Time: {processing_duration:.2f} seconds
- Processing Speed: {(total_frames/processing_duration):.2f} FPS if frames processed >0
- Hardware Acceleration: {DEVICE_STR.upper()}

DETAILED LOG:
"""
    if detection_log:
        for log in detection_log:
            detection_type = log.get('detection_type', 'unknown')
            report_content += f"- {log['type'].upper()} (Tracker #{log['tracker_id']}) at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Confidence: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]\n"

    with open(REPORT_PATH, "w") as f:
        f.write(report_content)

    print(f"📄 Detection report saved to: {REPORT_PATH}")
    if TARGET_VIDEO_PATH and not USE_CAMERA:
        print(f"🎥 Annotated video saved to: {TARGET_VIDEO_PATH}")
