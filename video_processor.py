#!/usr/bin/env python3
"""
Video processor with face tracking and effects.
Uses ffmpeg for processing to preserve audio.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import subprocess
import tempfile


class VideoProcessor:
    """Process videos with effects including face tracking."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def apply_effects(self, output_path: str, settings: dict, cancel_flag=None):
        """
        Apply time-based effects using ffmpeg segmentation.
        """
        import subprocess
        import tempfile
        import os

        trim_settings = settings.get('trim', {})
        effect_markers = settings.get('effect_markers', [])
        global_effects = settings.get('effects', {})

        # Parse trim settings
        start_time = '0'
        end_time = None

        if trim_settings.get('start'):
            start_time = trim_settings['start']
        if trim_settings.get('end'):
            end_time = trim_settings['end']

        # If no effect markers, create intermediate file for global effects
        if not effect_markers:
            intermediate_output = output_path
            if global_effects.get('faceTracking'):
                # Use intermediate file for global effects processing
                intermediate_output = os.path.join(tempfile.mkdtemp(), 'intermediate.mp4')
            self._copy_video_with_trim(intermediate_output, start_time, end_time, settings, cancel_flag)
        else:
            # Process timeline effect markers
            # Sort markers by start time
            sorted_markers = sorted(effect_markers, key=lambda m: m['start_time'])

            # Create segments directory
            segments_dir = tempfile.mkdtemp()
            segment_files = []

            current_time = 0  # in seconds
            video_duration = self.total_frames / self.fps

            try:
                # Create segments based on effect markers
                for i, marker in enumerate(sorted_markers):
                    effect_type = marker['type']
                    start = marker['start_time']
                    end = marker['end_time']

                    # Add pre-effect segment (no effects)
                    if start > current_time + 0.1:  # Only if there's meaningful time gap
                        pre_end = start
                        pre_segment = os.path.join(segments_dir, f'pre_{i}.mp4')
                        self._create_segment(current_time, pre_end, pre_segment, settings, cancel_flag)
                        segment_files.append(pre_segment)

                    # Add effect segment
                    effect_segment = os.path.join(segments_dir, f'effect_{i}.mp4')
                    self._create_effect_segment(start, end, effect_segment, effect_type, settings, cancel_flag)
                    segment_files.append(effect_segment)

                    current_time = end

                # Add post-effects segment (no effects)
                if current_time < video_duration - 0.1:
                    post_segment = os.path.join(segments_dir, 'post_final.mp4')
                    self._copy_video_with_trim(post_segment, self._format_time(current_time), end_time, settings, cancel_flag)
                    segment_files.append(post_segment)

                # Concatenate all segments to intermediate file
                intermediate_output = os.path.join(segments_dir, 'concatenated.mp4')
                self._concatenate_segments(segment_files, intermediate_output, settings, cancel_flag)

            finally:
                # Clean up segments directory
                import shutil
                shutil.rmtree(os.path.dirname(intermediate_output), ignore_errors=True)

        # Apply global effects to the final output
        if global_effects.get('faceTracking'):
            self._apply_face_tracking(intermediate_output, output_path, global_effects, cancel_flag)
        elif intermediate_output != output_path:
            # Just move intermediate to final output if no global effects
            import shutil
            shutil.move(intermediate_output, output_path)

    def _format_time(self, seconds):
        """Format seconds to HH:MM:SS."""
        h = int(seconds / 3600)
        m = int((seconds % 3600) / 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _run_with_cancel(self, cmd, cancel_flag=None):
        """Run subprocess with cancellation support."""
        import time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if cancel_flag:
            # Poll while process is running
            while process.poll() is None:
                if cancel_flag():
                    process.terminate()
                    time.sleep(0.5)
                    if process.poll() is None:
                        process.kill()
                    raise subprocess.CalledProcessError(process.returncode, cmd, "Cancelled by user")
                time.sleep(0.1)

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, stderr)

        return stdout

    def _copy_video_with_trim(self, output_path, start_time, end_time, settings, cancel_flag=None):
        """Copy video with optional trim, no effects."""
        ffmpeg_cmd = ['ffmpeg', '-i', str(self.video_path)]

        if start_time != '0':
            ffmpeg_cmd.extend(['-ss', start_time])
        if end_time:
            ffmpeg_cmd.extend(['-t', end_time])

        ffmpeg_cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-y',
            str(output_path)
        ])

        if cancel_flag:
            self._run_with_cancel(ffmpeg_cmd, cancel_flag)
        else:
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

    def _create_segment(self, start_sec, end_sec, output_path, settings, cancel_flag=None):
        """Create a video segment with no effects."""
        self._copy_video_with_trim(output_path, self._format_time(start_sec), self._format_time(end_sec), settings, cancel_flag)

    def _create_effect_segment(self, start_sec, end_sec, output_path, effect_type, settings, cancel_flag=None):
        """Create a video segment with specific effect applied."""
        effect_filters = {
            'mirror': 'hwdir=1',
            'grayscale': 'format=gray',
            'sepia': 'sepia',
            'blur': 'gblur=5',
            'zoom': 'scale=1.2:1080:1920'
        }

        ffmpeg_cmd = ['ffmpeg', '-i', str(self.video_path),
                        '-ss', self._format_time(start_sec),
                        '-t', self._format_time(end_sec - start_sec)]

        if effect_filters.get(effect_type):
            ffmpeg_cmd.extend(['-vf', effect_filters[effect_type]])

        ffmpeg_cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-y',
            str(output_path)
        ])

        if cancel_flag:
            self._run_with_cancel(ffmpeg_cmd, cancel_flag)
        else:
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

    def _concatenate_segments(self, segment_files, output_path, settings, cancel_flag=None):
        """Concatenate video segments using ffmpeg concat demuxer."""
        # Create concat file
        concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)

        for segment in segment_files:
            concat_file.write(f"file '{segment}'\n")

        concat_file.close()

        # Build ffmpeg concat command
        ffmpeg_cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', concat_file.name,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-y',
            str(output_path)
        ]

        if cancel_flag:
            self._run_with_cancel(ffmpeg_cmd, cancel_flag)
        else:
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

        # Clean up concat file
        os.unlink(concat_file.name)

    def _apply_face_tracking(self, input_path: str, output_path: str, effects: dict, cancel_flag=None):
        """Apply face tracking with zoom and pan."""
        import tempfile
        import shutil

        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Open video
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get zoom level and smoothing
        zoom_level = float(effects.get('faceZoomLevel', 1.5))
        smoothing = effects.get('faceSmoothing', 'medium')

        # Smoothing factor (lower = more smoothing)
        if smoothing == 'low':
            smooth_factor = 0.1
        elif smoothing == 'high':
            smooth_factor = 0.5
        else:  # medium
            smooth_factor = 0.3

        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        temp_output = os.path.join(temp_dir, 'output.mp4')

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        # Calculate target size (zoomed in)
        target_width = int(width / zoom_level)
        target_height = int(height / zoom_level)

        # Initialize smoothed position (center)
        smooth_x = (width - target_width) // 2
        smooth_y = (height - target_height) // 2

        frame_count = 0

        try:
            while True:
                if cancel_flag and cancel_flag():
                    raise Exception("Cancelled by user")

                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    # Use the largest face
                    face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = face

                    # Calculate center of face
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2

                    # Calculate crop position to center the face
                    target_x = int(face_center_x - target_width // 2)
                    target_y = int(face_center_y - target_height // 3)

                    # Clamp to bounds
                    target_x = max(0, min(target_x, width - target_width))
                    target_y = max(0, min(target_y, height - target_height))
                else:
                    # No face detected, stay in center
                    target_x = (width - target_width) // 2
                    target_y = (height - target_height) // 2

                # Smooth the movement
                smooth_x = int(smooth_x + (target_x - smooth_x) * smooth_factor)
                smooth_y = int(smooth_y + (target_y - smooth_y) * smooth_factor)

                # Extract and resize the region
                cropped = frame[smooth_y:smooth_y + target_height, smooth_x:smooth_x + target_width]
                resized = cv2.resize(cropped, (width, height))

                # Write frame
                out.write(resized)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")

        finally:
            cap.release()
            out.release()

        # Use ffmpeg to add audio to the output
        ffmpeg_cmd = [
            'ffmpeg', '-i', temp_output, '-i', input_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0', '-map', '1:a:0?',
            '-movflags', '+faststart', '-y',
            output_path
        ]

        if cancel_flag:
            self._run_with_cancel(ffmpeg_cmd, cancel_flag)
        else:
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Clean up input if it was a temporary file
        if input_path != output_path and os.path.exists(input_path):
            os.remove(input_path)

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

