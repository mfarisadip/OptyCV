#!/usr/bin/env python3
"""
Configuration constants for OptyCV Detection Project
"""
import os
from datetime import datetime

# Folders
OUTPUT_FOLDER = "outputs"
CAPTURES_FOLDER = os.path.join(OUTPUT_FOLDER, "images")

# File naming patterns
PROJECT_START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_project_paths(video_path):
    """Generate project-specific paths based on video filename"""
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    return {
        "json_filename": f"{video_filename}-{PROJECT_START_TIME}_detections.json",
        "captures_folder": CAPTURES_FOLDER,
        "frame_prefix": f"{video_filename}-{PROJECT_START_TIME}"
    }

# Ensure output folders exist
def ensure_output_folders():
    """Create output folders if they don't exist"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CAPTURES_FOLDER, exist_ok=True)

# Detection settings - supports both simple and advanced crack types
DETECTION_CLASSES = [
    'pothole',
    'crack',  # fallback for generic crack class
    'alligator cracking',
    'lateral cracking',
    'longitudinal cracking',
    'rutting'
]

# Crack type mapping for categorization
CRACK_TYPE_MAPPING = {
    'crack': 'crack',
    'alligator cracking': 'alligator_cracking',
    'lateral cracking': 'lateral_cracking',
    'longitudinal cracking': 'longitudinal_cracking',
    'rutting': 'rutting'
}
LINE_POSITION_PERCENTAGE = 0.55  # 55% from top

# JSON structure template
DETECTION_JSON_TEMPLATE = {
    "video_info": {
        "filename": "",
        "total_frames": 0,
        "duration": "",
        "resolution": "",
        "fps": 0,
        "processing_time": 0
    },
    "project_info": {
        "start_time": PROJECT_START_TIME,
        "model_path": "",
        "device": "",
        "confidence_threshold": 0,
        "iou_threshold": 0
    },
    "detection_summary": {
        "total_detections": 0,
        "potholes": 0,
        "cracks": {
            "crack": 0,
            "alligator_cracking": 0,
            "lateral_cracking": 0,
            "longitudinal_cracking": 0,
            "rutting": 0
        },
        "total_cracks": 0
    },
    "detections": []
}