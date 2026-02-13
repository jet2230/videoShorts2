#!/usr/bin/env python3
"""
Video processor with face tracking and effects.
Uses ffmpeg for processing to preserve audio.
"""

import os

# Disable hardware acceleration for FFmpeg backend to fix AV1 decoding issues
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;none"

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
        self.cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
            
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(str(video_path))
            if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def apply_effects(self, output_path: str, settings: dict, cancel_flag=None, log_callback=None):
        """
        Apply time-based effects using ffmpeg segmentation.
        """
        import subprocess
        import tempfile
        import os

        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        trim_settings = settings.get('trim', {})
        effect_markers = settings.get('effect_markers', [])
        global_effects = settings.get('effects', {})

        log("Starting video processing...")

        # Parse trim settings
        start_time = '0'
        end_time = None

        if trim_settings.get('start'):
            start_time = trim_settings['start']
        if trim_settings.get('end'):
            end_time = trim_settings['end']

        intermediate_output = None
        segments_dir = None
        temp_dir_for_cleanup = None

        try:
            # If no effect markers, create intermediate file for global effects
            if not effect_markers:
                print("Processing video (no timeline effects)...")
                intermediate_output = output_path
                if global_effects.get('faceTracking'):
                    # Use intermediate file for global effects processing
                    temp_dir_for_cleanup = tempfile.mkdtemp()
                    intermediate_output = os.path.join(temp_dir_for_cleanup, 'intermediate.mp4')
                self._copy_video_with_trim(intermediate_output, start_time, end_time, settings, cancel_flag, log_callback)
                log("Video trim/copy complete")
            else:
                # Process timeline effect markers
                # Sort markers by start time
                sorted_markers = sorted(effect_markers, key=lambda m: m['start_time'])
                print(f"Processing {len(sorted_markers)} timeline effect segments...")

                # Create segments directory
                segments_dir = tempfile.mkdtemp()
                segment_files = []

                current_time = 0  # in seconds
                video_duration = self.total_frames / self.fps

                # Create segments based on effect markers
                for i, marker in enumerate(sorted_markers):
                    effect_type = marker['type']
                    start = marker['start_time']
                    end = marker['end_time']

                    print(f"  Creating segment {i+1}/{len(sorted_markers)}: {effect_type} ({start}-{end})")

                    # Add pre-effect segment (no effects)
                    if start > current_time + 0.1:  # Only if there's meaningful time gap
                        pre_end = start
                        pre_segment = os.path.join(segments_dir, f'pre_{i}.mp4')
                        self._create_segment(current_time, pre_end, pre_segment, settings, cancel_flag, log_callback)
                        segment_files.append(pre_segment)

                    # Add effect segment
                    effect_segment = os.path.join(segments_dir, f'effect_{i}.mp4')
                    self._create_effect_segment(start, end, effect_segment, effect_type, settings, cancel_flag, log_callback)
                    segment_files.append(effect_segment)

                    current_time = end

                # Add post-effects segment (no effects)
                if current_time < video_duration - 0.1:
                    log("  Creating final segment...")
                    post_segment = os.path.join(segments_dir, 'post_final.mp4')
                    self._copy_video_with_trim(post_segment, self._format_time(current_time), end_time, settings, cancel_flag, log_callback)
                    segment_files.append(post_segment)

                # If face tracking is enabled, concatenate to a temp file outside segments_dir
                # Otherwise, concatenate directly to intermediate output
                log("Concatenating segments...")
                if global_effects.get('faceTracking'):
                    # Concatenate to a temporary intermediate file for face tracking
                    temp_concat = os.path.join(segments_dir, 'temp_concat.mp4')
                    self._concatenate_segments(segment_files, temp_concat, settings, cancel_flag, log_callback)

                    # Move to a temp location outside segments_dir for face tracking
                    temp_dir_for_cleanup = tempfile.mkdtemp()
                    intermediate_output = os.path.join(temp_dir_for_cleanup, 'intermediate.mp4')
                    import shutil
                    shutil.move(temp_concat, intermediate_output)
                else:
                    # No face tracking, concatenate directly to output
                    intermediate_output = output_path
                    self._concatenate_segments(segment_files, intermediate_output, settings, cancel_flag, log_callback)
                log("Segment concatenation complete")

            # Apply global effects to the final output
            if global_effects.get('faceTracking'):
                self._apply_face_tracking(intermediate_output, output_path, global_effects, cancel_flag, log_callback)
            elif intermediate_output != output_path:
                # Just move intermediate to final output if no global effects
                import shutil
                shutil.move(intermediate_output, output_path)

            log("Video processing complete!")

        finally:
            # Clean up temp directories
            import shutil
            if segments_dir:
                shutil.rmtree(segments_dir, ignore_errors=True)
            if temp_dir_for_cleanup:
                shutil.rmtree(temp_dir_for_cleanup, ignore_errors=True)

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

    def _copy_video_with_trim(self, output_path, start_time, end_time, settings, cancel_flag=None, log_callback=None):
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

    def _create_segment(self, start_sec, end_sec, output_path, settings, cancel_flag=None, log_callback=None):
        """Create a video segment with no effects."""
        self._copy_video_with_trim(output_path, self._format_time(start_sec), self._format_time(end_sec), settings, cancel_flag)

    def _create_effect_segment(self, start_sec, end_sec, output_path, effect_type, settings, cancel_flag=None, log_callback=None):
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

    def _concatenate_segments(self, segment_files, output_path, settings, cancel_flag=None, log_callback=None):
        """Concatenate video segments using ffmpeg concat demuxer."""
        import os

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

    def _apply_face_tracking(self, input_path: str, output_path: str, effects: dict, cancel_flag=None, log_callback=None):
        """Apply face tracking with zoom and pan."""
        import os
        import tempfile
        import shutil
        import time

        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        log(f"Starting face tracking processing...")

        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Open video
        cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
        if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
            
        if not cap.isOpened():
            cap = cv2.VideoCapture(input_path)
            if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)

        # Verify readability
        ret, test_frame = cap.read()
        if not ret:
            log("OpenCV failed to read video directly. Attempting automatic transcode fallback for face tracking...")
            cap.release()
            
            # Create a temporary proxy video (H.264 is very compatible)
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                proxy_path = tmp.name
            
            transcode_cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '20',
                '-c:a', 'copy',
                proxy_path
            ]
            
            log(f"Running fallback transcode for face tracking...")
            ts_result = subprocess.run(transcode_cmd, capture_output=True, text=True)
            
            if ts_result.returncode == 0:
                input_path = proxy_path # Use proxy for reading
                cap = cv2.VideoCapture(input_path)
                log("Transcode successful. Proceeding with face tracking.")
            else:
                log(f"Transcode failed: {ts_result.stderr}")
                raise Exception(f"Could not read video with OpenCV even after transcode: {ts_result.stderr}")

        # Reset capture to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        log(f"Video info: {total_frames} frames at {fps}fps, resolution: {width}x{height}")

        # Get zoom level and smoothing
        zoom_level = float(effects.get('faceZoomLevel', 1.5))
        smoothing = effects.get('faceSmoothing', 'medium')

        # Smoothing factor (much lower = much more smoothing)
        if smoothing == 'low':
            smooth_factor = 0.05  # Smooth
        elif smoothing == 'high':
            smooth_factor = 0.25  # More responsive, faster movement
        else:  # medium
            smooth_factor = 0.12  # Balanced

        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        temp_output = os.path.join(temp_dir, 'output.mp4')

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        # Calculate target size (zoomed in)
        target_width = int(width / zoom_level)
        target_height = int(height / zoom_level)

        # Initialize smoothed position (center) - use float for precision
        smooth_x = float(width - target_width) / 2.0
        smooth_y = float(height - target_height) / 2.0

        # Track previous valid positions for stability
        prev_valid_x = smooth_x
        prev_valid_y = smooth_y
        no_face_count = 0

        frame_count = 0
        last_progress_print = 0
        progress_interval = max(1, total_frames // 100)  # Print 100 times total
        start_time = time.time()

        try:
            while True:
                if cancel_flag and cancel_flag():
                    raise Exception("Cancelled by user")

                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces with more conservative parameters for stability
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

                target_x = smooth_x
                target_y = smooth_y

                if len(faces) > 0:
                    # Use the largest face
                    face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = face

                    # Calculate center of face
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2

                    # Calculate crop position to center the face
                    new_target_x = float(face_center_x - target_width // 2)
                    new_target_y = float(face_center_y - target_height // 3)

                    # Validate the new position - reject large jumps
                    max_jump = max(target_width, target_height) * 0.3
                    if abs(new_target_x - prev_valid_x) < max_jump and abs(new_target_y - prev_valid_y) < max_jump:
                        target_x = new_target_x
                        target_y = new_target_y
                        prev_valid_x = target_x
                        prev_valid_y = target_y
                        no_face_count = 0
                    else:
                        # Jump too large, use previous valid position
                        target_x = prev_valid_x
                        target_y = prev_valid_y
                else:
                    # No face detected, gradually return to center over time
                    no_face_count += 1
                    center_x = float(width - target_width) / 2.0
                    center_y = float(height - target_height) / 2.0

                    # Very slowly move to center if no face for a while
                    if no_face_count > 30:
                        target_x = center_x
                        target_y = center_y

                # Clamp to bounds
                target_x = max(0, min(target_x, width - target_width))
                target_y = max(0, min(target_y, height - target_height))

                # Smooth the movement with float precision
                smooth_x = smooth_x + (target_x - smooth_x) * smooth_factor
                smooth_y = smooth_y + (target_y - smooth_y) * smooth_factor

                # Convert to int for array slicing
                smooth_x_int = int(smooth_x)
                smooth_y_int = int(smooth_y)

                # Extract and resize the region
                cropped = frame[smooth_y_int:smooth_y_int + target_height, smooth_x_int:smooth_x_int + target_width]
                resized = cv2.resize(cropped, (width, height))

                # Write frame
                out.write(resized)

                frame_count += 1

                # Print progress more frequently with time estimates
                if frame_count - last_progress_print >= progress_interval:
                    percent = int((frame_count / total_frames) * 100)
                    elapsed = time.time() - start_time
                    fps_rate = frame_count / elapsed if elapsed > 0 else 0
                    remaining_frames = total_frames - frame_count
                    eta_seconds = remaining_frames / fps_rate if fps_rate > 0 else 0
                    eta_minutes = int(eta_seconds / 60)
                    eta_secs = int(eta_seconds % 60)

                    # Calculate completion time
                    from datetime import datetime, timedelta
                    completion_time = datetime.now() + timedelta(seconds=eta_seconds)
                    completion_str = completion_time.strftime("%H:%M:%S")

                    log(f"Processed {frame_count}/{total_frames} frames ({percent}%) | ETA: {eta_minutes}m {eta_secs}s | Complete by: {completion_str}")
                    last_progress_print = frame_count

            # Final progress update
            elapsed = time.time() - start_time
            log(f"Processed {total_frames}/{total_frames} frames (100%) | Total time: {int(elapsed)}s")

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

