#!/usr/bin/env python3
"""
Video processor with face tracking and effects.
Uses OpenCV for face detection and tracking.
"""

import cv2
import numpy as np
from pathlib import Path
import json


class VideoProcessor:
    """Process videos with effects including face tracking."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_face(self, frame):
        """Detect face in frame and return center coordinates."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Get the largest face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            # Return center of face
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y), (x, y, w, h)

        return None, None

    def smooth_tracking(self, current_pos, prev_pos, smoothing_factor=0.7):
        """Apply smoothing to face tracking."""
        if prev_pos is None:
            return current_pos

        new_x = int(smoothing_factor * prev_pos[0] + (1 - smoothing_factor) * current_pos[0])
        new_y = int(smoothing_factor * prev_pos[1] + (1 - smoothing_factor) * current_pos[1])

        return (new_x, new_y)

    def apply_face_tracking(self, output_path: str, zoom_level: float = 1.5,
                           smoothing: str = 'medium'):
        """
        Apply face tracking with pan and scan.

        Args:
            output_path: Where to save the output video
            zoom_level: How much to zoom (1.2 - 2.5)
            smoothing: Smoothing level ('low', 'medium', 'high')
        """
        # Set smoothing factor
        smoothing_factors = {
            'low': 0.5,
            'medium': 0.7,
            'high': 0.9
        }
        smoothing_factor = smoothing_factors.get(smoothing, 0.7)

        # Target output dimensions (9:16 for shorts)
        target_width = 1080
        target_height = 1920

        # Calculate crop size based on zoom level
        crop_width = int(self.width / zoom_level)
        crop_height = int(self.height / zoom_level)

        # Ensure crop dimensions maintain aspect ratio
        aspect_ratio = target_width / target_height
        if crop_width / crop_height != aspect_ratio:
            crop_height = int(crop_width / aspect_ratio)

        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps,
                             (target_width, target_height))

        prev_center = None
        frame_count = 0

        print(f"Processing {self.total_frames} frames...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{self.total_frames}")

            # Detect face
            face_center, face_box = self.detect_face(frame)

            if face_center:
                # Smooth the movement
                prev_center = self.smooth_tracking(face_center, prev_center, smoothing_factor)
                target_x, target_y = prev_center
            else:
                # No face detected, use previous position or center
                if prev_center:
                    target_x, target_y = prev_center
                else:
                    target_x, target_y = self.width // 2, self.height // 2

            # Calculate crop boundaries centered on face
            x1 = max(0, min(target_x - crop_width // 2, self.width - crop_width))
            y1 = max(0, min(target_y - crop_height // 2, self.height - crop_height))
            x2 = x1 + crop_width
            y2 = y1 + crop_height

            # Crop the frame
            cropped = frame[y1:y2, x1:x2]

            # Resize to target dimensions
            resized = cv2.resize(cropped, (target_width, target_height),
                               interpolation=cv2.INTER_LINEAR)

            out.write(resized)

        # Release resources
        self.cap.release()
        out.release()

        print(f"Video saved to {output_path}")

    def apply_effects(self, output_path: str, settings: dict):
        """
        Apply various video effects.

        Args:
            output_path: Where to save output
            settings: Dictionary of effect settings
        """
        trim_settings = settings.get('trim', {})
        effects = settings.get('effects', {})

        # Calculate start/end frames
        start_frame = 0
        end_frame = self.total_frames

        if trim_settings.get('start'):
            start_time = self._parse_time(trim_settings['start'])
            start_frame = int(start_time * self.fps)

        if trim_settings.get('end'):
            end_time = self._parse_time(trim_settings['end'])
            end_frame = int(end_time * self.fps)

        # Check if face tracking is enabled
        if effects.get('faceTracking'):
            print("Applying face tracking...")
            self.apply_face_tracking(
                output_path,
                zoom_level=float(effects.get('faceZoomLevel', 1.5)),
                smoothing=effects.get('faceSmoothing', 'medium')
            )
        else:
            # Basic processing without face tracking
            self._basic_process(output_path, start_frame, end_frame, effects)

    def _parse_time(self, time_str: str) -> float:
        """Parse time string (MM:SS or HH:MM:SS) to seconds."""
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
        return 0

    def _basic_process(self, output_path: str, start_frame: int,
                      end_frame: int, effects: dict):
        """Basic video processing without face tracking."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps,
                             (self.width, self.height))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_num in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Apply effects
            if effects.get('mirror'):
                frame = cv2.flip(frame, 1)

            if effects.get('grayscale'):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if effects.get('sepia'):
                frame = self._apply_sepia(frame)

            out.write(frame)

        self.cap.release()
        out.release()

    def _apply_sepia(self, frame):
        """Apply sepia filter to frame."""
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: video_processor.py <input_video> <output_video>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]

    processor = VideoProcessor(input_video)

    # Example: Apply face tracking with default settings
    settings = {
        'effects': {
            'faceTracking': True,
            'faceZoomLevel': 1.5,
            'faceSmoothing': 'medium'
        }
    }

    processor.apply_effects(output_video, settings)
