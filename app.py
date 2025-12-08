#!/usr/bin/env python3
"""
Optimized Crack and Pothole Detection
Usage:
    python app.py --source video --video-path pole.mp4 --device cuda --cuda-device 0 --save-video
    python app.py --source camera --camera-id 0 --camera-type usb --device cuda --cuda-device 0 --vehicle-token YOUR_TOKEN
"""
import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/")
import argparse
from datetime import datetime
import time
import os
from threading import Thread
import json
import asyncio
import serial
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

# Imageio for power-off safe video recording
try:
    import imageio
    IMAGEIO_AVAILABLE = True
    print("✅ ImageIO available - Using power-off safe video recording")
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("⚠️ ImageIO not available - Install with: pip install imageio[ffmpeg]")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  'requests' library not available. API features will be disabled.")

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
parser.add_argument('--camera-type', type=str, default='auto',
                    choices=['auto', 'jetson', 'usb', 'standard'],
                    help='Camera type: auto (detect), jetson (nvarguscamerasrc), usb (v4l2src), standard (opencv)')
parser.add_argument('--output', type=str, default=None,
                    help='Output video file path (auto-generated if not specified)')
parser.add_argument('--save-video', action='store_true',
                    help='Save processed video to file')
parser.add_argument('--headless', action='store_true',
                    help='Run in headless mode (no GUI display)')
parser.add_argument('--device', type=str, default='auto',
                    help='Device to use: "cuda", "cpu", or "auto" (default: auto)')
parser.add_argument('--cuda-device', type=int, default=0,
                    help='CUDA device ID when using CUDA (default: 0)')
parser.add_argument('--model', type=str, default='weights/YOLOV8n320IR8.onnx',
                    help='Model path (can be .onnx or .pt). For ONNX use device arg at predict-time.')
parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
parser.add_argument('--iou', type=float, default=0.3, help='NMS IoU threshold')
parser.add_argument('--top-line', type=float, default=0.43,
                    help='Top line position as percentage from top (default: 0.43 = 43%)')
parser.add_argument('--bottom-line', type=float, default=0.70,
                    help='Bottom line position as percentage from top (default: 0.70 = 70%)')
parser.add_argument('--vehicle-token', type=str, default=None,
                    help='Vehicle token for API authentication. If provided, API submission is automatically enabled.')
parser.add_argument('--clean-video', action='store_true',
                    help='Save clean original video without annotations (faster performance)')
parser.add_argument('--save-screenshots', action='store_true',
                    help='Auto-save screenshots when detections occur (works in both GUI and headless mode)')
args = parser.parse_args()

MODEL_PATH = args.model
CONF_THRESHOLD = args.conf
IOU_THRESHOLD = args.iou

# ---------------------------
# API Configuration
# ---------------------------
API_BASE_URL = "https://airoad.roots.web.id"
VEHICLE_TOKEN = args.vehicle_token
API_ENABLED = VEHICLE_TOKEN is not None and REQUESTS_AVAILABLE

if VEHICLE_TOKEN and not REQUESTS_AVAILABLE:
    print("⚠️  Vehicle token provided but 'requests' library is not available.")
    print("    Install with: pip install requests")
    print("    API features will be disabled.")
elif API_ENABLED:
    print(f"✅ API enabled: {API_BASE_URL}")
    print(f"   Vehicle token: {VEHICLE_TOKEN[:10]}..." if len(VEHICLE_TOKEN) > 10 else f"   Vehicle token: {VEHICLE_TOKEN}")

# ---------------------------
# GPS Manager Class
# ---------------------------
class GPSManager:
    """GPS Manager for SIMCOM7600X with async operation and fallback ports"""

    def __init__(self):
        self.serial_port = None
        self.is_running = False
        self.gps_thread = None
        self.current_position = {'latitude': 0.0, 'longitude': 0.0}
        self.last_update = None
        self.lock = threading.Lock()
        self.ports_to_try = ["/dev/ttyUSB3", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB4", "/dev/ttyUSB5"]

    def send_at(self, command, expected_back, timeout):
        """Send AT command to GPS module with connection validation"""
        try:
            if not self.serial_port or not self.serial_port.is_open:
                return False, ""

            rec_buff = ''
            self.serial_port.write((command + '\r\n').encode())
            time.sleep(timeout)

            if self.serial_port.inWaiting():
                time.sleep(0.01)
                rec_buff = self.serial_port.read(self.serial_port.inWaiting())

                decoded_response = rec_buff.decode()
                if expected_back in decoded_response:
                    return True, decoded_response
                else:
                    return False, decoded_response
            return False, ""

        except Exception as e:
            # Don't print error here - let the caller handle it
            # This prevents spamming errors during recovery attempts
            error_msg = str(e)
            if "Errno 5" in error_msg or "Input/output error" in error_msg:
                # Silent handling for connection errors - recovery logic will handle these
                return False, ""
            else:
                # Still log other types of errors
                print(f"⚠️ GPS AT command error: {e}")
                return False, ""

    def parse_gps_data(self, gps_data):
        """Parse GPS data string from +CGPSINFO response"""
        try:
            print(f"🔍 Parsing GPS data: {repr(gps_data)}")

            # Extract data after +CGPSINFO:
            if "+CGPSINFO: " in gps_data:
                gps_data = gps_data.split("+CGPSINFO: ")[1].strip()

            # Remove any trailing "OK" and whitespace
            gps_data = gps_data.replace("OK", "").strip()
            print(f"🔍 Extracted GPS data: {repr(gps_data)}")

            # Split by comma and clean
            parts = [p.strip() for p in gps_data.split(',')]
            print(f"🔍 Split parts: {parts}")

            if len(parts) < 4:  # Need at least lat, lat_dir, lon, lon_dir
                print("❌ Invalid: Missing GPS data parts")
                return None

            # Handle two possible formats:
            # Format 1: ['0614.180641,S', '10648.682916,E', '031225', ...]
            # Format 2: ['0614.180641', 'S', '10648.682916', 'E', '031225', ...]

            if ',' in parts[0]:  # Format 1: lat+dir and lon+dir are combined
                lat_str = parts[0]
                lon_str = parts[1]
            else:  # Format 2: lat, dir, lon, dir are separate
                lat_str = parts[0] + parts[1]  # Combine lat + direction
                lon_str = parts[2] + parts[3]  # Combine lon + direction

            print(f"🔍 Combined Lat: {lat_str}, Lon: {lon_str}")

            if not lat_str or not lon_str:
                print("❌ Invalid: Missing lat/lon strings")
                return None

            # Parse latitude
            lat = None
            if lat_str.endswith('S'):
                lat_str = lat_str[:-1]
                if len(lat_str) >= 4:
                    deg = float(lat_str[:2])
                    minutes = float(lat_str[2:])
                    lat = -(deg + minutes / 60)
            elif lat_str.endswith('N'):
                lat_str = lat_str[:-1]
                if len(lat_str) >= 4:
                    deg = float(lat_str[:2])
                    minutes = float(lat_str[2:])
                    lat = deg + minutes / 60
            else:
                return None

            # Parse longitude
            lon = None
            if lon_str.endswith('W'):
                lon_str = lon_str[:-1]
                if len(lon_str) >= 4:
                    deg = float(lon_str[:3])
                    minutes = float(lon_str[3:])
                    lon = -(deg + minutes / 60)
            elif lon_str.endswith('E'):
                lon_str = lon_str[:-1]
                if len(lon_str) >= 4:
                    deg = float(lon_str[:3])
                    minutes = float(lon_str[3:])
                    lon = deg + minutes / 60
            else:
                return None

            if lat is not None and lon is not None:
                result = {
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': float(parts[4]) if len(parts) > 4 and parts[4] else 0,
                    'speed': float(parts[5]) if len(parts) > 5 and parts[5] else 0,
                    'date': parts[2] if len(parts) > 2 else '',
                    'time': parts[3] if len(parts) > 3 else '',
                    'raw': gps_data
                }
                print(f"✅ Successfully parsed GPS: {result['latitude']:.6f}, {result['longitude']:.6f}")
                return result
            else:
                print("❌ Failed to parse lat/lon values")
                return None

        except Exception as e:
            print(f"⚠️ GPS parsing error: {e}")
            return None

    def connect_to_port(self, port):
        """Try to connect to a specific USB port"""
        try:
            ser = serial.Serial(port, 115200, timeout=1)
            if ser.is_open:
                # Test basic communication
                ser.write(b'AT\r\n')
                time.sleep(0.5)
                if ser.inWaiting():
                    response = ser.read(ser.inWaiting()).decode()
                    if 'OK' in response:
                        print(f"✅ GPS connected to {port}")
                        return ser
                ser.close()
        except Exception as e:
            print(f"⚠️ Failed to connect to {port}: {e}")
        return None

    def initialize_gps(self):
        """Initialize GPS connection with fallback to multiple ports and retry mechanism"""
        print("🛰️ Initializing GPS...")

        max_attempts = 10
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            print(f"🔄 GPS connection attempt {attempt}/{max_attempts}")

            for port in self.ports_to_try:
                print(f"🔍 Trying GPS on {port}...")
                self.serial_port = self.connect_to_port(port)

                if self.serial_port:
                    # Reset and start GPS
                    self.send_at('AT+CGPS=0', 'OK', 1)
                    time.sleep(0.5)
                    success, _ = self.send_at('AT+CGPS=1', 'OK', 1)

                    if success:
                        print(f"✅ GPS initialized on {port}")
                        return True
                    else:
                        self.serial_port.close()
                        self.serial_port = None

            if attempt < max_attempts:
                print(f"⚠️ GPS connection failed, retrying in 2 seconds...")
                time.sleep(2)

        print(f"❌ GPS connection failed after {max_attempts} attempts!")
        print("❌ ERROR: USB GPS not found or not connected!")
        print("Please check:")
        print("1. GPS device is connected to USB port")
        print("2. USB device is detected by system")
        print("3. USB serial ports are available")
        return False

    def gps_update_loop(self):
        """Background thread for GPS updates with auto-recovery"""
        print("📍 GPS update thread started")
        consecutive_errors = 0
        max_consecutive_errors = 3
        retry_delay = 2.0
        max_retry_delay = 30.0

        while self.is_running:
            try:
                if not self.serial_port or not self.serial_port.is_open:
                    print("🔌 GPS connection lost, attempting recovery...")
                    self.recover_gps_connection()
                    consecutive_errors = 0
                    continue

                success, response = self.send_at('AT+CGPSINFO', '+CGPSINFO: ', 1)

                if success and response:
                    print(f"📍 GPS: {response}")
                    parsed_data = self.parse_gps_data(response)

                    if parsed_data:
                        with self.lock:
                            self.current_position = {
                                'latitude': parsed_data['latitude'],
                                'longitude': parsed_data['longitude']
                            }
                            self.last_update = datetime.now()
                        print(f"📍 GPS: {parsed_data['latitude']:.6f}, {parsed_data['longitude']:.6f}")
                        consecutive_errors = 0  # Reset error counter on success
                        retry_delay = 2.0  # Reset retry delay
                    else:
                        print("📍 GPS: No fix yet")
                        consecutive_errors = 0  # "No fix" is not a connection error
                else:
                    consecutive_errors += 1
                    print(f"⚠️ GPS command failed (attempt {consecutive_errors}/{max_consecutive_errors})")

                # Check if we need recovery due to consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    print(f"🚨 GPS connection lost after {consecutive_errors} consecutive errors")
                    self.recover_gps_connection()
                    consecutive_errors = 0

                time.sleep(retry_delay)

            except Exception as e:
                consecutive_errors += 1
                error_msg = str(e)

                # Check for specific connection errors
                if "Errno 5" in error_msg or "Input/output error" in error_msg or "write failed" in error_msg:
                    print(f"🚨 GPS USB connection error: {e}")
                    print("🔄 Attempting GPS connection recovery...")
                    self.recover_gps_connection()
                    consecutive_errors = 0
                else:
                    print(f"⚠️ GPS update error: {e}")

                # Exponential backoff for retry delay
                retry_delay = min(retry_delay * 1.5, max_retry_delay)
                print(f"⏱️ Retrying GPS in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)

        print("📍 GPS update thread stopped")

    def recover_gps_connection(self):
        """Recover GPS connection after disconnection or error"""
        print("🔄 Starting GPS connection recovery...")

        # Close existing connection
        if self.serial_port and self.serial_port.is_open:
            try:
                self.send_at('AT+CGPS=0', 'OK', 1)
                self.serial_port.close()
            except:
                pass
            self.serial_port = None

        # Wait a moment before reconnection
        time.sleep(1.0)

        # Try to reinitialize GPS on all available ports (but don't call initialize_gps to avoid double scanning)
        recovery_success = False
        for attempt in range(3):  # Try 3 times
            print(f"🔄 GPS recovery attempt {attempt + 1}/3")

            # Try each port once per recovery attempt
            for port in self.ports_to_try:
                print(f"🔍 Trying GPS on {port}...")
                self.serial_port = self.connect_to_port(port)

                if self.serial_port:
                    # Reset and start GPS
                    self.send_at('AT+CGPS=0', 'OK', 1)
                    time.sleep(0.5)
                    success, _ = self.send_at('AT+CGPS=1', 'OK', 1)

                    if success:
                        print(f"✅ GPS recovered on {port}")
                        recovery_success = True
                        break
                    else:
                        print(f"❌ GPS init failed on {port}")
                        if self.serial_port:
                            self.serial_port.close()
                            self.serial_port = None

                if recovery_success:
                    break  # Found working port, exit port loop

            if recovery_success:
                break  # Recovery successful, exit attempt loop

            if attempt < 2:  # Don't sleep on last attempt
                time.sleep(2.0)  # Wait between attempts

        if not recovery_success:
            print("❌ GPS recovery failed after 3 attempts")
            print("📡 GPS will be unavailable until USB connection is restored")
            # Reset position to default
            with self.lock:
                self.current_position = {'latitude': 0.0, 'longitude': 0.0}
                self.last_update = None

    def start(self):
        """Start GPS monitoring"""
        if not self.is_running:
            if self.initialize_gps():
                self.is_running = True
                self.gps_thread = threading.Thread(target=self.gps_update_loop, daemon=True)
                self.gps_thread.start()
                return True
        return False

    def get_position(self):
        """Get current GPS position (thread-safe)"""
        with self.lock:
            return {
                'latitude': self.current_position['latitude'],
                'longitude': self.current_position['longitude'],
                'last_update': self.last_update
            }

    def stop(self):
        """Stop GPS monitoring gracefully"""
        print("🛑 Stopping GPS...")
        self.is_running = False

        if self.gps_thread and self.gps_thread.is_alive():
            self.gps_thread.join(timeout=3)

        if self.serial_port:
            try:
                self.send_at('AT+CGPS=0', 'OK', 1)
                self.serial_port.close()
            except:
                pass
            self.serial_port = None

        print("✅ GPS stopped")


# Initialize GPS Manager
gps_manager = GPSManager()

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
CAMERA_TYPE = args.camera_type

def initialize_camera():
    """Initialize camera based on camera type with optimizations"""
    print(f"🎥 Initializing camera (Type: {CAMERA_TYPE}, ID: {CAMERA_ID})...")

    cap_methods = []

    if CAMERA_TYPE == 'auto' or CAMERA_TYPE == 'jetson':
        # Check if Jetson hardware is available before adding Jetson methods
        jetson_available = False
        try:
            # Check for nvargus-daemon service
            import subprocess
            result = subprocess.run(['systemctl', 'is-active', '--quiet', 'nvargus-daemon'],
                                   capture_output=True, text=True)
            if result.returncode == 0:
                jetson_available = True
                print("🤖 Jetson hardware detected")
            else:
                print("⚠️ Jetson hardware not detected, skipping Jetson methods")
        except:
            print("⚠️ Could not detect Jetson hardware, skipping Jetson methods")

        if jetson_available:
            print("🤖 Jetson hardware detected - checking for CSI cameras...")

            # First, try to detect if CSI cameras are actually available
            try:
                # Try to list available cameras on Jetson
                import subprocess
                result = subprocess.run(['v4l2-ctl', '--list-devices'],
                                       capture_output=True, text=True)
                has_csi = 'platform: tegra' in result.stdout.lower()

                if has_csi:
                    print("📹 CSI cameras detected, adding Jetson methods")
                    # Jetson/PCIe Camera Methods (nvarguscamerasrc)
                    gst_jetson_pipeline = (
                        f"nvarguscamerasrc sensor-id={CAMERA_ID} ! "
                        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
                        f"format=NV12, framerate=30/1 ! "
                        "nvvidconv ! video/x-raw, format=BGRx ! "
                        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
                    )

                    gst_jetson_simple = (
                        f"nvarguscamerasrc sensor-id={CAMERA_ID} ! "
                        f"video/x-raw(memory:NVMM), width={width}, height={height} ! "
                        "nvvidconv ! video/x-raw, format=BGR ! appsink"
                    )

                    cap_methods.extend([
                        (gst_jetson_pipeline, cv2.CAP_GSTREAMER, "Jetson GStreamer Pipeline"),
                        (gst_jetson_simple, cv2.CAP_GSTREAMER, "Jetson Simple GStreamer"),
                    ])
                else:
                    print("⚠️ No CSI cameras detected on Jetson, skipping Jetson methods")
                    print("🔄 Will try USB/Standard camera methods instead")
            except Exception as e:
                print(f"⚠️ Could not check cameras: {e}")
                print("🔄 Will try USB/Standard camera methods instead")
        else:
            print("🔄 Skipping Jetson camera methods (no Jetson hardware)")

    if CAMERA_TYPE == 'auto' or CAMERA_TYPE == 'usb':
        # USB Camera Methods (v4l2src)
        gst_usb_pipeline = (
            f"v4l2src device=/dev/video{CAMERA_ID} ! "
            f"video/x-raw, width={width}, height={height}, "
            f"framerate=30/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )

        gst_usb_simple = (
            f"v4l2src device=/dev/video{CAMERA_ID} ! "
            f"video/x-raw, width={width}, height={height} ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )

        # USB Camera dengan MJPEG (lebih stabil untuk USB 2.0)
        gst_usb_mjpeg = (
            f"v4l2src device=/dev/video{CAMERA_ID} ! "
            f"image/jpeg, width={width}, height={height}, "
            f"framerate=30/1 ! "
            "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )

        # USB Camera dengan YUYV (fallback)
        gst_usb_yuyv = (
            f"v4l2src device=/dev/video{CAMERA_ID} ! "
            f"video/x-raw, format=YUY2, width={width}, height={height}, "
            f"framerate=30/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )

        cap_methods.extend([
            (gst_usb_mjpeg, cv2.CAP_GSTREAMER, "USB Camera MJPEG"),
            (gst_usb_pipeline, cv2.CAP_GSTREAMER, "USB Camera RAW"),
            (gst_usb_yuyv, cv2.CAP_GSTREAMER, "USB Camera YUYV"),
            (gst_usb_simple, cv2.CAP_GSTREAMER, "USB Camera Simple"),
        ])

    if CAMERA_TYPE == 'auto' or CAMERA_TYPE == 'standard':
        # Standard OpenCV Method (fallback universal)
        cap_methods.append((CAMERA_ID, cv2.CAP_ANY, "Standard OpenCV Camera"))

    # Filter methods based on camera type
    if CAMERA_TYPE != 'auto':
        if CAMERA_TYPE == 'jetson':
            cap_methods = [m for m in cap_methods if 'Jetson' in m[2]]
        elif CAMERA_TYPE == 'usb':
            cap_methods = [m for m in cap_methods if 'USB' in m[2]]
        elif CAMERA_TYPE == 'standard':
            cap_methods = [m for m in cap_methods if 'Standard' in m[2]]

    print(f"🔧 Trying {len(cap_methods)} initialization methods...")

    for pipeline, cap_type, method_name in cap_methods:
        try:
            print(f"📹 Attempting {method_name}...")
            cap = cv2.VideoCapture(pipeline, cap_type)

            if cap.isOpened():
                print(f"✅ Camera opened using {method_name}")

                # Set additional properties
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Try MJPG

                # Test if we can actually get frames
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)

                    print(f"📸 Successfully captured test frame: {test_frame.shape}")
                    print(f"📺 Camera properties: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")

                    return cap
                else:
                    print(f"⚠️  {method_name}: Cannot capture frames, trying next method...")
                    cap.release()
            else:
                print(f"⚠️  {method_name}: Cannot open camera, trying next method...")

        except Exception as e:
            print(f"⚠️  {method_name} error: {e}")
            if 'cap' in locals():
                try:
                    cap.release()
                except:
                    pass

    print("❌ All camera initialization methods failed!")
    print("🔧 Troubleshooting tips:")

    if CAMERA_TYPE in ['auto', 'jetson']:
        print("   - For Jetson: Check if camera is connected to CSI port")
        print("   - Run: sudo systemctl restart nvargus-daemon")

    if CAMERA_TYPE in ['auto', 'usb']:
        print("   - For USB: Check if camera is connected to USB port")
        print("   - Check device permissions: ls -l /dev/video*")
        print("   - Try different camera ID: --camera-id 1, --camera-id 2")
        print("   - Check if camera is being used by another application")
        print("   - Run: v4l2-ctl --list-devices")

    print("   - Try specific camera type: --camera-type usb")
    print("   - Try standard mode: --camera-type standard")

    return None

if USE_CAMERA:
    cap = initialize_camera()
    if cap is None:
        print(f"Error: Cannot initialize camera {CAMERA_ID}")
        exit(1)

    # Get native camera resolution for video writer
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30

    # Video writer uses NATIVE camera resolution for quality output
    video_info = sv.VideoInfo(width=actual_width, height=actual_height, fps=fps, total_frames=None)

    # Detection resolution (smaller for performance)
    detection_width, detection_height = 640, 360

    print(f"Camera {CAMERA_ID} opened: {actual_width}x{actual_height} @ {fps} FPS")
    print(f"🎥 Video Output: {actual_width}x{actual_height} (Native Resolution)")
    print(f"🔍 Detection Resolution: {detection_width}x{detection_height} (Fast Processing)")
else:
    if not os.path.exists(SOURCE_VIDEO_PATH):
        print(f"Error: video file '{SOURCE_VIDEO_PATH}' not found.")
        exit(1)
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"Video loaded: {SOURCE_VIDEO_PATH} -> {video_info.width}x{video_info.height} @ {video_info.fps} FPS, total {video_info.total_frames} frames")
    # Set detection resolution untuk video mode
    detection_width, detection_height = video_info.width, video_info.height

# Dual line positions (adjustable) - use actual dimensions
TOP_LINE_PERCENTAGE = args.top_line
BOTTOM_LINE_PERCENTAGE = args.bottom_line

if USE_CAMERA:
    top_line_y = int(actual_height * TOP_LINE_PERCENTAGE)
    bottom_line_y = int(actual_height * BOTTOM_LINE_PERCENTAGE)
    # Calculate detection line positions (for zone checking on resized frames)
    detection_top_line_y = int(detection_height * TOP_LINE_PERCENTAGE)
    detection_bottom_line_y = int(detection_height * BOTTOM_LINE_PERCENTAGE)
    print(f"🔍 Detection Resolution: {detection_width}x{detection_height}")
    print(f"📍 Detection Zone: Y={detection_top_line_y} to Y={detection_bottom_line_y} (detection coords)")
    print(f"📍 Annotation Zone: Y={top_line_y} to Y={bottom_line_y} (actual coords)")
else:
    # Video mode: use video resolution
    top_line_y = int(video_info.height * TOP_LINE_PERCENTAGE)
    bottom_line_y = int(video_info.height * BOTTOM_LINE_PERCENTAGE)
    detection_top_line_y = top_line_y
    detection_bottom_line_y = bottom_line_y

# Keep backward compatibility with single line
line_y_position = int((top_line_y + bottom_line_y) / 2)  # Middle line for backward compatibility
LINE_START = sv.Point(0, line_y_position)
LINE_END = sv.Point(video_info.width if not USE_CAMERA else actual_width, line_y_position)

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

# ---------------------------
# API Submission Function
# ---------------------------
def send_detection_to_api(detection_type, class_name, confidence, timestamp, frame_index, video_time, encoded_image, latitude=0.0, longitude=0.0):
    """
    Send detection data to API endpoint in a non-blocking way.
    This function is called in a separate thread to avoid blocking the main detection loop.
    """
    if not API_ENABLED:
        return
    
    try:
        # Prepare form data
        form_data = {
            'detection_type': detection_type,
            'class': class_name,
            'latitude': str(latitude),
            'longitude': str(longitude),
            'confidence': str(confidence),
            'timestamp': datetime.now().isoformat() + 'Z',  # ISO 8601 format
            'sequence': str(frame_index),
            'video_time': video_time,
            'frame_index': str(frame_index)
        }
        
        # Prepare image file
        files = {}
        if encoded_image is not None:
            files = {
                'image': ('detection.jpg', encoded_image, 'image/jpeg')
            }

        # Prepare headers
        headers = {
            'X-Vehicle-Token': VEHICLE_TOKEN
        }
        
        # Send POST request with timeout
        url = f"{API_BASE_URL}/api/vehicle-entries"
        response = requests.post(url, data=form_data, files=files, headers=headers, timeout=10)
        
        if response.status_code in [200, 201]:
            print(f"   ✅ API: Detection sent successfully (ID: {frame_index})")
        else:
            print(f"   ⚠️  API: Server returned {response.status_code} for detection {frame_index}")
            print(f"       Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"   ⚠️  API: Timeout sending detection {frame_index}")
    except requests.exceptions.ConnectionError:
        print(f"   ⚠️  API: Connection error (server may be down)")
    except Exception as e:
        print(f"   ⚠️  API: Error sending detection {frame_index}: {e}")

def send_detection_async(detection_type, class_name, confidence, timestamp, frame_index, video_time, encoded_image, latitude=0.0, longitude=0.0):
    """
    Wrapper to send detection to API in a separate thread (non-blocking).
    """
    if API_ENABLED:
        # encoded_image is bytes, so no need to copy
        thread = Thread(
            target=send_detection_to_api,
            args=(detection_type, class_name, confidence, timestamp, frame_index, video_time, encoded_image, latitude, longitude),
            daemon=True
        )
        thread.start()

def save_detection_screenshot(detection_type, detection_count, annotated_frame):
    """
    Save screenshot of detection if --save-screenshots is enabled.
    """
    if args.save_screenshots and annotated_frame is not None:
        screenshot_folder = "output/screenshots"
        if not os.path.exists(screenshot_folder):
            os.makedirs(screenshot_folder)
        screenshot_path = os.path.join(screenshot_folder, f"{detection_type}_{detection_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(screenshot_path, annotated_frame)
        print(f"   📸 Screenshot saved: {screenshot_path}")


def process_frame(frame: np.ndarray, frame_index: int = None, return_annotated: bool = True) -> np.ndarray:
    """Process a single frame for crack and pothole detection with dual-resolution optimization

    Args:
        frame: Input frame to process
        frame_index: Frame index for logging
        return_annotated: If True, returns annotated frame for display; if False, returns None for performance
    """
    global pothole_count, crack_count, alligator_cracking_count, lateral_cracking_count, longitudinal_cracking_count, rutting_count, detection_log, processed_trackers

    start_time = time.time()

    # Dual-resolution optimization: detect on small frame, annotate on original frame
    if USE_CAMERA and (frame.shape[1] != detection_width or frame.shape[0] != detection_height):
        # Resize frame for detection (faster processing)
        detection_frame = cv2.resize(frame, (detection_width, detection_height))
    else:
        # Use original frame for detection (video mode or already correct size)
        detection_frame = frame

    # Run inference on optimized frame size
    try:
        results = model(detection_frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False, device=device_arg)
    except Exception as e:
        # Fallback: try without device_arg (some runtimes accept 'cpu'/'cuda:0')
        print(f"⚠️ Inference call with device_arg failed: {e}. Retrying without explicit device...")
        results = model(detection_frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    # Convert to supervision detections
    detections = sv.Detections.from_ultralytics(results[0])

    # Scale detections back to original frame size if we used resized detection frame
    if USE_CAMERA and (frame.shape[1] != detection_width or frame.shape[0] != detection_height):
        scale_x = frame.shape[1] / detection_width
        scale_y = frame.shape[0] / detection_height

        if len(detections.xyxy) > 0:
            detections.xyxy = detections.xyxy * np.array([scale_x, scale_y, scale_x, scale_y])

    # Filter to desired classes
    detections = filter_detections(detections)
    
    # Store total detections before zone filtering
    total_detections = len(detections)
    
    # Filter detections to ONLY show objects INSIDE the zone (between lines)
    zone_indices = []
    for i in range(len(detections)):
        bbox = detections.xyxy[i]
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        # For dual-resolution mode, use correct line position
        if USE_CAMERA:
            # Convert center_y to detection coordinates for line checking
            detection_center_y = int(center_y * (detection_height / frame.shape[0]))
            if detection_top_line_y <= detection_center_y <= detection_bottom_line_y:
                zone_indices.append(i)
        else:
            # Video mode: use original coordinates
            if top_line_y <= center_y <= bottom_line_y:
                zone_indices.append(i)
    
    # Keep only detections inside zone
    if zone_indices:
        detections = detections[zone_indices]
    else:
        detections = sv.Detections.empty()
    
    inside_zone_count = len(detections)

    # Track detections (only those in zone)
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

    # Timestamp for logs
    timestamp = get_timestamp(frame_index, video_info.fps) if frame_index is not None else "LIVE"

    # Get GPS position for detection API calls
    gps_position = gps_manager.get_position()
    gps_latitude = gps_position['latitude']
    gps_longitude = gps_position['longitude']

    # Create annotated frame for API (always needed when API is enabled)
    annotated_frame = None
    encoded_jpg_bytes = None

    if API_ENABLED or return_annotated:
        # Always create annotated frame for API, optionally for display
        annotated_frame = trace_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Optimization: Lazy encoding of image for API
        # Only encode if API is enabled and we have detections
        if API_ENABLED and len(detections) > 0:
            success, buf = cv2.imencode('.jpg', annotated_frame)
            if success:
                encoded_jpg_bytes = buf.tobytes()

    # Check objects below the line and log new trackers
    if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
            if tracker_id not in processed_trackers:
                bbox = detections.xyxy[i]
                center_y = int((bbox[1] + bbox[3]) / 2)

                # For dual-resolution mode, use correct line position
                if USE_CAMERA:
                    # Convert center_y to detection coordinates for zone checking
                    detection_center_y = int(center_y * (detection_height / frame.shape[0]))
                    in_zone = detection_top_line_y <= detection_center_y <= detection_bottom_line_y
                else:
                    # Video mode: use original coordinates
                    in_zone = top_line_y <= center_y <= bottom_line_y

                # Only process objects that are inside the zone (between top and bottom lines)
                if in_zone:
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
                        print(f"🕳️ POTHOLE #{pothole_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Zone:{top_line_y}-{bottom_line_y}, Conf:{confidence:.2f})")
                        save_detection_screenshot('pothole', pothole_count, annotated_frame)
                        
                        # Send to API
                        send_detection_async('pothole', 'pothole', confidence, timestamp,
                                           frame_index if frame_index is not None else 0, timestamp, encoded_jpg_bytes, gps_latitude, gps_longitude)

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
                        print(f"🐊 ALLIGATOR CRACKING #{alligator_cracking_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Zone:{top_line_y}-{bottom_line_y}, Conf:{confidence:.2f})")
                        save_detection_screenshot('alligator_cracking', alligator_cracking_count, annotated_frame)
                        
                        
                        # Send to API
                        send_detection_async('alligator_cracking', 'alligator cracking', confidence, timestamp,
                                           frame_index if frame_index is not None else 0, timestamp, encoded_jpg_bytes, gps_latitude, gps_longitude)

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
                        print(f"↔️ LATERAL CRACKING #{lateral_cracking_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Zone:{top_line_y}-{bottom_line_y}, Conf:{confidence:.2f})")
                        save_detection_screenshot('lateral_cracking', lateral_cracking_count, annotated_frame)
                        
                        
                        # Send to API
                        send_detection_async('lateral_cracking', 'lateral cracking', confidence, timestamp,
                                           frame_index if frame_index is not None else 0, timestamp, encoded_jpg_bytes, gps_latitude, gps_longitude)

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
                        print(f"↕️ LONGITUDINAL CRACKING #{longitudinal_cracking_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Zone:{top_line_y}-{bottom_line_y}, Conf:{confidence:.2f})")
                        save_detection_screenshot('longitudinal_cracking', longitudinal_cracking_count, annotated_frame)
                        
                        
                        # Send to API
                        send_detection_async('longitudinal_cracking', 'longitudinal cracking', confidence, timestamp,
                                           frame_index if frame_index is not None else 0, timestamp, encoded_jpg_bytes, gps_latitude, gps_longitude)

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
                        print(f"🛤️  RUTTING #{rutting_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Zone:{top_line_y}-{bottom_line_y}, Conf:{confidence:.2f})")
                        save_detection_screenshot('rutting', rutting_count, annotated_frame)
                        
                        
                        # Send to API
                        send_detection_async('rutting', 'rutting', confidence, timestamp,
                                           frame_index if frame_index is not None else 0, timestamp, encoded_jpg_bytes, gps_latitude, gps_longitude)

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
                        print(f"〰️ CRACK #{crack_count} (ID:#{tracker_id}) DETECTED at {timestamp} (Y:{center_y}, Zone:{top_line_y}-{bottom_line_y}, Conf:{confidence:.2f})")
                        save_detection_screenshot('crack', crack_count, annotated_frame)
                        
                        
                        # Send to API
                        send_detection_async('crack', 'crack', confidence, timestamp,
                                           frame_index if frame_index is not None else 0, timestamp, encoded_jpg_bytes, gps_latitude, gps_longitude)

                    processed_trackers.add(tracker_id)

    # Overlay stats text only if we have an annotated frame
    processing_time = time.time() - start_time
    if frame_index is not None and video_info.total_frames:
        frame_text = f'Frame: {frame_index}/{video_info.total_frames}'
    else:
        frame_text = f'Frame: {frame_index if frame_index is not None else "LIVE"}'

        # Calculate total cracks (all crack types combined)
    total_cracks = alligator_cracking_count + lateral_cracking_count + longitudinal_cracking_count + rutting_count + crack_count

    # Only overlay text if annotated_frame exists and return_annotated is True
    if annotated_frame is not None and return_annotated:
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

        # GPS coordinates display
        if gps_position['latitude'] != 0.0 or gps_position['longitude'] != 0.0:
            gps_text = f'GPS: {gps_position["latitude"]:.6f}, {gps_position["longitude"]:.6f}'
            gps_color = (0, 255, 0)
        else:
            gps_text = 'GPS: No Signal'
            gps_color = (0, 0, 255)  # Red for no signal

        cv2.putText(annotated_frame, gps_text, (10, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gps_color, 2)
        
        # Add zone statistics
        cv2.putText(annotated_frame, f'Total Detections: {total_detections}', (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Inside Zone: {inside_zone_count}', (10, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw detection line
        # Get frame dimensions for drawing lines
        frame_height, frame_width = annotated_frame.shape[:2]
        
        # Draw TOP line (yellow)
        cv2.line(annotated_frame, (0, top_line_y), (frame_width, top_line_y), (0, 255, 255), 3)
        cv2.putText(annotated_frame, 'TOP LINE', (frame_width - 150, top_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw BOTTOM line (cyan)
        cv2.line(annotated_frame, (0, bottom_line_y), (frame_width, bottom_line_y), (255, 255, 0), 3)
        cv2.putText(annotated_frame, 'BOTTOM LINE', (frame_width - 200, bottom_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return annotated_frame
    else:
        # Performance mode: don't return annotated frame
        return None

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

    if args.headless:
        print("🖥️  Running in HEADLESS mode (no GUI)")
        print("📡 API & GPS integration enabled")
        print("Press Ctrl+C to stop")
    else:
        print("🖥️  Running in GUI mode")
        print("Press 'q' to quit, 's' to save screenshot")

    print("-" * 60)

    # Video Configuration - Power-off Safe with ImageIO
    video_writer = None
    video_writer_type = None  # 'imageio' or 'opencv'

    if args.save_video and TARGET_VIDEO_PATH:
        # Check if file exists and warn
        if os.path.exists(TARGET_VIDEO_PATH):
            file_size = os.path.getsize(TARGET_VIDEO_PATH)
            print(f"⚠️  Video file exists: {TARGET_VIDEO_PATH}")
            print(f"📊 Current size: {file_size / (1024*1024):.1f} MB")
            print("🔄 Will overwrite existing file")

        # Try ImageIO first (power-off safe)
        imageio_available = IMAGEIO_AVAILABLE  # Local copy
        if imageio_available:
            try:
                # Use MKV container for better power-off recovery
                video_path_mkv = TARGET_VIDEO_PATH.replace('.mp4', '.mkv')
                video_writer = imageio.get_writer(
                    video_path_mkv,
                    fps=video_info.fps,
                    codec='libx264',
                    format='ffmpeg',
                    pixelformat='yuv420p',
                    quality=8  # Good balance of quality vs file size
                )
                video_writer_type = 'imageio'
                print(f"🎬 ImageIO Writer Initialized: {video_path_mkv}")
                print("🚀 Power-off Safe: FFmpeg + MKV container")
                print("⚡ Real-time: Each frame written immediately")
                print("🔋 100% Safe from power-off corruption")
            except Exception as e:
                print(f"⚠️ ImageIO initialization failed: {e}")
                print("🔄 Falling back to OpenCV VideoWriter")
                imageio_available = False

        # Fallback to OpenCV if ImageIO fails
        if not imageio_available or video_writer is None:
            # Use AVI container as fallback
            actual_video_path = TARGET_VIDEO_PATH.replace('.mp4', '.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(actual_video_path, fourcc, video_info.fps,
                                           (video_info.width, video_info.height))
            video_writer_type = 'opencv'
            print(f"🎬 OpenCV Writer Initialized: {actual_video_path}")
            print("⚠️ Note: May corrupt on power-off (fallback mode)")

        if args.clean_video:
            print("🚀 Clean Mode: Original frames (no annotations)")
        else:
            print("🎬 Annotated Mode: Frames with detection boxes")
    else:
        print("🔍 Detection Mode: API + Display only (no video saving)")

    frame_count = 0
    start_processing_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                break

            # Dual Pipeline:
            # 1. Process frame for detection & API (clean video mode = faster, annotated mode = slower)
            # 2. Write original frame to video (clean recording) or processed frame (annotated mode)
            if args.clean_video:
                # Clean video mode: Process for API only (no annotation overhead)
                process_frame(frame, frame_count, return_annotated=False)
            else:
                # Annotated video mode: Process for API + display
                processed_frame = process_frame(frame, frame_count, return_annotated=True)

            # Power-off safe video writing with ImageIO or OpenCV
            if video_writer:
                try:
                    if video_writer_type == 'imageio':
                        # ImageIO: Convert BGR to RGB and append
                        if args.clean_video:
                            # Write ORIGINAL frame (no annotations)
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            # Write ANNOTATED frame
                            if 'processed_frame' in locals() and processed_frame is not None:
                                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            else:
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        video_writer.append_data(rgb_frame)  # Immediate write to FFmpeg
                    else:
                        # OpenCV: Traditional write with flush
                        if args.clean_video:
                            video_writer.write(frame)
                        else:
                            if 'processed_frame' in locals() and processed_frame is not None:
                                video_writer.write(processed_frame)
                            else:
                                video_writer.write(frame)
                        video_writer.flush()  # Force write to disk

                    # Log every 100 frames
                    if frame_count % 100 == 0:
                        writer_type = "ImageIO (Power-off Safe)" if video_writer_type == 'imageio' else "OpenCV"
                        print(f"💾 Frame {frame_count} written ({writer_type})")

                except Exception as e:
                    print(f"⚠️ Error writing frame {frame_count}: {e}")
                    # Continue processing even if video write fails

            # GUI vs Headless mode
            if not args.headless:
                # GUI Mode: Always show annotated frame for display
                if args.clean_video:
                    # Need to process again for display (with annotations)
                    display_frame = process_frame(frame, frame_count, return_annotated=True)
                else:
                    # Use already processed frame
                    display_frame = processed_frame if 'processed_frame' in locals() and processed_frame is not None else frame

                if display_frame is not None:
                    cv2.imshow('Crack and Pothole Detection - Live Feed', display_frame)
                else:
                    # Fallback: show original frame if processing failed
                    cv2.imshow('Crack and Pothole Detection - Live Feed', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_folder = "output/screenshots"
                    if not os.path.exists(screenshot_folder):
                        os.makedirs(screenshot_folder)
                    screenshot_path = os.path.join(screenshot_folder, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    
                    # Save the annotated frame if available, otherwise save raw frame
                    frame_to_save = processed_frame if 'processed_frame' in locals() and processed_frame is not None else frame
                    cv2.imwrite(screenshot_path, frame_to_save)
                    print(f"📸 Screenshot saved: {screenshot_path}")
            else:
                # Headless Mode: No display, just process for API
                pass

            frame_count += 1

    except KeyboardInterrupt:
        print("\nCamera processing interrupted by user")

    finally:
        if video_writer:
            if video_writer_type == 'imageio':
                try:
                    video_writer.close()  # ImageIO close
                    print("✅ ImageIO writer closed - Video file complete")
                except Exception as e:
                    print(f"⚠️ Error closing ImageIO writer: {e}")
            else:
                video_writer.release()  # OpenCV VideoWriter release
                print("✅ OpenCV writer closed")
        cap.release()

        # Only destroy windows in GUI mode
        if not args.headless:
            cv2.destroyAllWindows()

    return time.time() - start_processing_time, frame_count

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Start GPS monitoring (async, won't block main process) - OPTIONAL
    gps_started = gps_manager.start()
    if gps_started:
        print("🛰️ GPS monitoring started")
    else:
        print("⚠️ GPS initialization failed - continuing without GPS")
        print("📍 GPS data will not be available for this session")


    try:
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

    except KeyboardInterrupt:
        print("\n⚠️ Application interrupted by user")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
    finally:
        # Graceful GPS shutdown
        print("🛑 Shutting down GPS...")
        gps_manager.stop()
        print("✅ GPS shutdown complete")
