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
        Apply time-based effects using a single-pass FFmpeg complex filter.
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

        log("Starting optimized video processing...")

        # Parse trim settings
        start_time = trim_settings.get('start', '0')
        end_time = trim_settings.get('end')

        # If we have face tracking, we MUST do a second pass because it's content-aware.
        # Otherwise, we can do everything in one pass.
        intermediate_output = output_path
        temp_dir_for_cleanup = None
        if global_effects.get('faceTracking'):
            temp_dir_for_cleanup = tempfile.mkdtemp()
            intermediate_output = os.path.join(temp_dir_for_cleanup, 'intermediate.mp4')

        # Build filter complex
        vf_filters = []
        
        # 1. Base scaling to ensure consistency (optional, but good for stability)
        # vf_filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

        # 2. Add timeline-based filters from markers
        for marker in effect_markers:
            etype = marker['type']
            s = marker['start_time']
            e = marker['end_time']
            enable = f"between(t,{s},{e})"
            
            if etype == 'mirror':
                vf_filters.append(f"hflip=enable='{enable}'")
            elif etype == 'grayscale':
                vf_filters.append(f"hue=s=0:enable='{enable}'")
            elif etype == 'sepia':
                vf_filters.append(f"colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131:enable='{enable}'")
            elif etype == 'blur':
                vf_filters.append(f"gblur=sigma=5:enable='{enable}'")
            elif etype == 'glitch':
                # RGB Shift + jitter
                vf_filters.append(f"rgbashift=rh=4:gh=-4:enable='{enable}'")
            elif etype == 'vhs':
                # Noise + Color bleed + subtle blur
                vf_filters.append(f"noise=alls=15:allf=t+u,hue=s=0.4,gblur=sigma=1:enable='{enable}'")
            elif etype == 'neon':
                # Edge detect + neon colors
                vf_filters.append(f"edgedetect=low=0.1:high=0.4,hue=h=120:s=2:enable='{enable}'")
            elif etype == 'vignette':
                vf_filters.append(f"vignette='PI/4':enable='{enable}'")
            elif etype == 'cinematic':
                # Add black bars top and bottom
                vf_filters.append(f"drawbox=y=0:h=ih*0.1:t=fill:c=black:enable='{enable}'")
                vf_filters.append(f"drawbox=y=ih*0.9:h=ih*0.1:t=fill:c=black:enable='{enable}'")
            elif etype == 'vibrance':
                vf_filters.append(f"vibrance=intensity=0.8:enable='{enable}'")
            elif etype == 'shake':
                # Random crop/pan for shake effect
                vf_filters.append(f"crop=w=iw-40:h=ih-40:x='20+20*sin(2*PI*t*10)':y='20+20*cos(2*PI*t*13)':enable='{enable}',scale={self.width}:{self.height}")
            elif etype == 'pixelate':
                # Pixelate using scale down and scale up with neighbor flags
                # This is tricky with 'enable' in a single filter string, so we use a sub-filter or boxblur as fallback
                # For single pass with 'enable', boxblur is safer
                vf_filters.append(f"boxblur=20:enable='{enable}'")
            elif etype == 'zoom':
                # Zoom is tricky with 'enable'. We use a scale/crop chain.
                # Since scale doesn't support 'enable', we use the 'zoompan' filter if possible, 
                # or just fallback to the segmented approach for zoom specifically if needed.
                # Actually, we can use the 'boxblur' trick or 'overlay' trick but for a single pass 
                # let's use a slightly simplified approach or stick to the others.
                # For now, we'll implement zoom using zoompan which is single-pass compatible.
                vf_filters.append(f"zoompan=z='if({enable},1.2,1)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={self.width}x{self.height}:enable='{enable}'")

        # Build FFmpeg command
        cmd = ['ffmpeg', '-y']
        
        # Fast input seeking for trim
        if start_time != '0':
            cmd.extend(['-ss', start_time])
        
        cmd.extend(['-i', str(self.video_path)])
        
        if end_time:
            cmd.extend(['-t', end_time])

        if vf_filters:
            cmd.extend(['-vf', ",".join(vf_filters)])

        cmd.extend([
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart',
            str(intermediate_output)
        ])

        log(f"Running single-pass effects processing...")
        if cancel_flag:
            self._run_with_cancel(cmd, cancel_flag)
        else:
            subprocess.run(cmd, capture_output=True, check=True)

        # Apply global effects (Face Tracking) if needed
        if global_effects.get('faceTracking'):
            self._apply_face_tracking(intermediate_output, output_path, global_effects, cancel_flag, log_callback)

        log("Video processing complete!")
        
        if temp_dir_for_cleanup:
            import shutil
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
            'mirror': 'hflip',
            'grayscale': 'format=gray',
            'sepia': 'colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131',
            'blur': 'gblur=sigma=5',
            'zoom': 'scale=1.2*iw:-1,crop=iw/1.2:ih/1.2'
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
        """Apply face tracking with zoom and pan - Optimized version."""
        import os
        import tempfile
        import shutil
        import time

        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        log(f"Starting optimized face tracking processing...")

        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Open video
        cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
        if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
            
        if not cap.isOpened():
            cap = cv2.VideoCapture(input_path)

        # Verify readability
        ret, test_frame = cap.read()
        if not ret:
            log("OpenCV failed to read video directly. Attempting automatic transcode fallback...")
            cap.release()
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                proxy_path = tmp.name
            transcode_cmd = ['ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20', '-c:a', 'copy', proxy_path]
            subprocess.run(transcode_cmd, capture_output=True)
            input_path = proxy_path
            cap = cv2.VideoCapture(input_path)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        log(f"Video info: {total_frames} frames at {fps}fps, {width}x{height}")

        zoom_level = float(effects.get('faceZoomLevel', 1.5))
        smoothing = effects.get('faceSmoothing', 'medium')
        smooth_factor = {'low': 0.05, 'high': 0.25}.get(smoothing, 0.12)

        # OPTIMIZATION: Setup direct FFmpeg pipe instead of cv2.VideoWriter + final pass
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', # Stdin
            '-i', input_path, # For audio
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0', '-map', '1:a:0?',
            '-movflags', '+faststart',
            output_path
        ]
        
        # Start FFmpeg process
        pipe_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        target_width = int(width / zoom_level)
        target_height = int(height / zoom_level)
        smooth_x = float(width - target_width) / 2.0
        smooth_y = float(height - target_height) / 2.0
        prev_valid_x, prev_valid_y = smooth_x, smooth_y
        
        # Detection optimizations
        detect_interval = 5 # Detect every 5 frames
        detect_scale = 0.4 # Downscale for detection
        last_faces = []
        no_face_count = 0
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                if cancel_flag and cancel_flag():
                    raise Exception("Cancelled by user")

                ret, frame = cap.read()
                if not ret: break

                target_x, target_y = smooth_x, smooth_y

                # Only detect periodically
                if frame_count % detect_interval == 0:
                    # OPTIMIZATION: Convert and downscale only for detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    small_gray = cv2.resize(gray, (0,0), fx=detect_scale, fy=detect_scale)
                    
                    faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(50*detect_scale), int(50*detect_scale)))
                    last_faces = faces
                else:
                    faces = last_faces

                if len(faces) > 0:
                    # Map back to original scale
                    face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = [int(v / detect_scale) for f in [face] for v in f]

                    face_center_x, face_center_y = x + w // 2, y + h // 2
                    new_target_x = float(face_center_x - target_width // 2)
                    new_target_y = float(face_center_y - target_height // 3)

                    max_jump = max(target_width, target_height) * 0.4
                    if abs(new_target_x - prev_valid_x) < max_jump and abs(new_target_y - prev_valid_y) < max_jump:
                        target_x, target_y = new_target_x, new_target_y
                        prev_valid_x, prev_valid_y = target_x, target_y
                        no_face_count = 0
                else:
                    no_face_count += 1
                    if no_face_count > 30:
                        target_x, target_y = (width - target_width) / 2.0, (height - target_height) / 2.0

                # Clamp and smooth
                target_x = max(0, min(target_x, width - target_width))
                target_y = max(0, min(target_y, height - target_height))
                smooth_x += (target_x - smooth_x) * smooth_factor
                smooth_y += (target_y - smooth_y) * smooth_factor

                # Extract and resize
                sx, sy = int(smooth_x), int(smooth_y)
                cropped = frame[sy:sy + target_height, sx:sx + target_width]
                resized = cv2.resize(cropped, (width, height))

                # Write to pipe
                pipe_proc.stdin.write(resized.tobytes())

                frame_count += 1
                if frame_count % 50 == 0:
                    percent = int((frame_count / total_frames) * 100)
                    log(f"Processing... {percent}% ({frame_count}/{total_frames})")

            # Finish pipe
            pipe_proc.stdin.close()
            pipe_proc.wait()
            log(f"Face tracking complete! Total time: {int(time.time() - start_time)}s")

        finally:
            cap.release()
            if pipe_proc.poll() is None:
                pipe_proc.terminate()

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

