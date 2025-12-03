#!/usr/bin/env python3
"""
Modular Detection Exporter for OptyCV
Handles JSON data management and frame image saving
"""
import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import (ensure_output_folders, get_project_paths, DETECTION_JSON_TEMPLATE,
                   DETECTION_CLASSES, CRACK_TYPE_MAPPING)


class DetectionExporter:
    """Modular class to handle detection data export and frame saving"""

    def __init__(self, video_path: str, model_path: str, device: str,
                 conf_threshold: float, iou_threshold: float):
        """Initialize exporter with project configuration"""

        # Ensure output folders exist
        ensure_output_folders()

        # Get project paths
        paths = get_project_paths(video_path)
        self.json_filename = paths["json_filename"]
        self.captures_folder = paths["captures_folder"]
        self.frame_prefix = paths["frame_prefix"]

        # Initialize detection data structure
        self.detection_data = DETECTION_JSON_TEMPLATE.copy()
        self.detection_data["project_info"]["model_path"] = model_path
        self.detection_data["project_info"]["device"] = device
        self.detection_data["project_info"]["confidence_threshold"] = conf_threshold
        self.detection_data["project_info"]["iou_threshold"] = iou_threshold

        # Update video info when available
        self.video_filename = os.path.basename(video_path)
        self.detection_data["video_info"]["filename"] = self.video_filename

        # Detection counters
        self.detection_counter = 0
        self.processed_trackers = set()

        # FPS tracking
        self.fps_samples = []
        self.min_fps = None
        self.max_fps = None
        self.avg_fps = None

        print(f"📁 Detection Exporter initialized")
        print(f"   JSON Output: {self.json_filename}")
        print(f"   Captures Folder: {self.captures_folder}")

    def update_video_info(self, total_frames: int, fps: float, resolution: str = ""):
        """Update video information once video is loaded"""
        self.detection_data["video_info"]["total_frames"] = total_frames
        self.detection_data["video_info"]["fps"] = fps
        if resolution:
            self.detection_data["video_info"]["resolution"] = resolution

        # Calculate duration
        if fps > 0 and total_frames > 0:
            duration_seconds = total_frames / fps
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            self.detection_data["video_info"]["duration"] = f"{minutes:02d}:{seconds:02d}"

    def get_timestamp_from_frame(self, frame_index: int, fps: float) -> str:
        """Convert frame index to video timestamp"""
        if frame_index is None or fps <= 0:
            return "00:00"
        seconds = frame_index / fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def get_server_timestamp(self) -> str:
        """Get current server timestamp in ISO format"""
        return datetime.now().isoformat() + "Z"

    def save_detection_frame(self, annotated_frame: np.ndarray, frame_index: int,
                           detection_data: Dict[str, Any]) -> Optional[str]:
        """Save annotated detection frame with bounding boxes and return filename"""
        try:
            self.detection_counter += 1
            frame_filename = f"{self.frame_prefix}_frame_{frame_index}_detection_{self.detection_counter}.jpg"
            frame_path = os.path.join(self.captures_folder, frame_filename)

            # Save annotated frame (with bounding boxes, labels, etc.)
            success = cv2.imwrite(frame_path, annotated_frame)
            if success:
                return frame_filename
            else:
                print(f"⚠️ Failed to save frame: {frame_path}")
                return None

        except Exception as e:
            print(f"⚠️ Error saving frame: {e}")
            return None

    def add_detection(self, annotated_frame: np.ndarray, frame_index: int,
                     tracker_id: int, class_name: str, confidence: float,
                     bbox: List[float], fps: float, line_y_position: int,
                     video_info: Any = None, gps_info: str = ""):
        """Add a detection to the export data"""

        # Check if this tracker was already processed
        if tracker_id in self.processed_trackers:
            return

        # Check if object is below the line
        bbox_center_y = int((bbox[1] + bbox[3]) / 2)
        if bbox_center_y < line_y_position:
            return

        # Mark tracker as processed
        self.processed_trackers.add(tracker_id)

        # Save annotated detection frame (with bounding boxes and labels)
        frame_filename = self.save_detection_frame(annotated_frame, frame_index, {
            "tracker_id": tracker_id,
            "class_name": class_name
        })

        # Create detection record
        detection_record = {
            "detection_id": self.detection_counter,
            "tracker_id": f"#{tracker_id}",
            "timestamp": self.get_server_timestamp(),
            "video_time": self.get_timestamp_from_frame(frame_index, fps),
            "frame_index": frame_index,
            "class": class_name.lower(),
            "confidence": round(confidence, 3),
            "bbox": [int(coord) for coord in bbox],  # Convert to integers
            "center_point": [int((bbox[0] + bbox[2]) / 2), bbox_center_y],
            "frame_image": frame_filename if frame_filename else "",
            "detection_type": "detected_below_line",
            "line_position": line_y_position,
            "gps_info": gps_info
        }

        # Add to detections array
        self.detection_data["detections"].append(detection_record)

        # Update summary counters
        self.detection_data["detection_summary"]["total_detections"] += 1

        class_name_lower = class_name.lower()
        if class_name_lower == 'pothole':
            self.detection_data["detection_summary"]["potholes"] += 1
        elif class_name_lower in CRACK_TYPE_MAPPING:
            crack_type = CRACK_TYPE_MAPPING[class_name_lower]
            self.detection_data["detection_summary"]["cracks"][crack_type] += 1
            self.detection_data["detection_summary"]["total_cracks"] += 1

        # Get emoji for logging
        emoji_map = {
            'pothole': '🕳️',
            'crack': '〰️',
            'alligator cracking': '🐊',
            'lateral cracking': '↔️',
            'longitudinal cracking': '↕️',
            'rutting': '🛤️'
        }
        emoji = emoji_map.get(class_name_lower, '🔍')

        # Log detection
        print(f"{emoji} {class_name.upper()} #{tracker_id} detected at {detection_record['video_time']} "
              f"(Frame {frame_index}, Y:{bbox_center_y}, Conf:{confidence:.2f})")

    def add_fps_sample(self, fps: float):
        """Add a FPS sample for tracking"""
        self.fps_samples.append(fps)

    def calculate_fps_stats(self):
        """Calculate min, max, and average FPS from samples"""
        if self.fps_samples:
            self.min_fps = min(self.fps_samples)
            self.max_fps = max(self.fps_samples)
            self.avg_fps = sum(self.fps_samples) / len(self.fps_samples)

            # Store in video info for JSON export
            self.detection_data["video_info"]["performance"] = {
                "min_fps": round(self.min_fps, 2),
                "max_fps": round(self.max_fps, 2),
                "avg_fps": round(self.avg_fps, 2),
                "total_samples": len(self.fps_samples)
            }

    def finalize_processing(self, processing_time: float):
        """Finalize the export data with processing information"""
        self.detection_data["video_info"]["processing_time"] = round(processing_time, 2)
        # Calculate FPS stats when finalizing
        self.calculate_fps_stats()

    def save_json(self) -> str:
        """Save detection data to JSON file and return file path"""
        try:
            json_path = os.path.join("outputs", self.json_filename)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.detection_data, f, indent=2, ensure_ascii=False)

            print(f"📄 Detection data saved to: {json_path}")
            return json_path

        except Exception as e:
            print(f"❌ Error saving JSON file: {e}")
            return ""

    def get_detection_summary(self) -> Dict[str, Any]:
        """Get current detection summary"""
        cracks_summary = self.detection_data["detection_summary"]["cracks"]

        # Get FPS stats if available
        fps_stats = {}
        if hasattr(self, 'min_fps') and self.min_fps is not None:
            fps_stats = {
                "min_fps": round(self.min_fps, 2),
                "max_fps": round(self.max_fps, 2),
                "avg_fps": round(self.avg_fps, 2)
            }

        return {
            "total_detections": self.detection_data["detection_summary"]["total_detections"],
            "potholes": self.detection_data["detection_summary"]["potholes"],
            "total_cracks": self.detection_data["detection_summary"]["total_cracks"],
            "crack_breakdown": cracks_summary,
            "crack": cracks_summary.get("crack", 0),
            "alligator_cracking": cracks_summary.get("alligator_cracking", 0),
            "lateral_cracking": cracks_summary.get("lateral_cracking", 0),
            "longitudinal_cracking": cracks_summary.get("longitudinal_cracking", 0),
            "rutting": cracks_summary.get("rutting", 0),
            "frames_saved": self.detection_counter,
            **fps_stats  # Add FPS stats if available
        }

    def export_summary_report(self) -> str:
        """Export a text summary report"""
        summary = self.get_detection_summary()

        report = f"""
DETECTION EXPORT SUMMARY
{'='*50}
Video: {self.video_filename}
Model: {self.detection_data['project_info']['model_path']}
Device: {self.detection_data['project_info']['device']}
Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DETECTION SUMMARY:
- Total Detections: {summary['total_detections']}
- Potholes: {summary['potholes']}
- Total Cracks: {summary['total_cracks']}
  - Generic Cracks: {summary['crack']}
  - Alligator Cracking: {summary['alligator_cracking']}
  - Lateral Cracking: {summary['lateral_cracking']}
  - Longitudinal Cracking: {summary['longitudinal_cracking']}
  - Rutting: {summary['rutting']}
- Frames Saved: {summary['frames_saved']}

PERFORMANCE:
- Processing Time: {self.detection_data['video_info'].get('processing_time', 'N/A')} seconds
- Min FPS: {summary.get('min_fps', 'N/A')}
- Max FPS: {summary.get('max_fps', 'N/A')}
- Average FPS: {summary.get('avg_fps', 'N/A')}

OUTPUT FILES:
- JSON Data: {self.json_filename}
- Captures Folder: {self.captures_folder}

{'='*50}
"""

        report_path = os.path.join("outputs", f"{os.path.splitext(self.json_filename)[0]}_summary.txt")
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"📋 Summary report saved to: {report_path}")
        return report_path