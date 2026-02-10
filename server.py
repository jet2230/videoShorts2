#!/usr/bin/env python3
"""
Flask server for the YouTube Shorts Creator web GUI.
"""

from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import subprocess
import threading
import queue
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List
import configparser
from datetime import datetime
import json

from shorts_creator import YouTubeShortsCreator, load_settings

# Setup logging to file
log_handler = logging.FileHandler('server.log')
log_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)

# Get Flask's logger and add file handler
flask_logger = logging.getLogger('werkzeug')
flask_logger.setLevel(logging.INFO)
flask_logger.addHandler(log_handler)

# Also log our app messages
app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_logger.addHandler(log_handler)

# Keep console output too
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)
flask_logger.addHandler(console_handler)
app_logger.addHandler(console_handler)

app = Flask(__name__, static_folder='static')
CORS(app)

# Required for SharedArrayBuffer (ffmpeg.wasm support)
@app.after_request
def set_headers(response):
    """Set security headers required for SharedArrayBuffer (ffmpeg.wasm)."""
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    return response

# Global state
creator = YouTubeShortsCreator()
settings = load_settings()

# Background task management
tasks: Dict[str, Dict] = {}
task_counter = 0
task_lock = threading.Lock()

# Canvas karaoke export progress tracking
canvas_karaoke_progress: Dict[str, Dict] = {}
canvas_karaoke_lock = threading.Lock()


class OutputCapture:
    """Context manager to capture stdout and stderr."""
    def __init__(self):
        self.buffer = []
        self.lock = threading.Lock()

    def write(self, text):
        # Capture all non-empty text
        if text and text.strip():
            timestamp = datetime.now().strftime('%H:%M:%S')
            with self.lock:
                self.buffer.append(f"{timestamp} - {text.rstrip()}")

    def flush(self):
        pass

    def get_output(self):
        with self.lock:
            return '\n'.join(self.buffer)


def run_task_with_callback(task_id: str, func, *args, **kwargs):
    """Run a function in a background thread with callback support."""
    # Initialize log field
    with task_lock:
        if task_id in tasks:
            tasks[task_id]['log'] = ''

    # Create log callback for this task
    def progress_callback(msg):
        try:
            msg_str = str(msg) if not isinstance(msg, str) else msg
            # Don't print here - OutputCaptureWithCallback will capture stdout and cause recursion!
            with task_lock:
                if task_id in tasks and 'log' in tasks[task_id]:
                    current_log = tasks[task_id]['log']
                    if not isinstance(current_log, str):
                        current_log = str(current_log) if current_log else ''
                    tasks[task_id]['log'] = current_log + msg_str + '\n'
        except Exception as e:
            # Write directly to original stdout to avoid recursion
            sys.__stdout__.write(f"Callback error: {e}\n")

    # Create cancel flag checker
    def check_cancelled():
        with task_lock:
            return tasks.get(task_id, {}).get('cancelled', False)

    # Add callback and cancel checker to kwargs
    kwargs['progress_callback'] = progress_callback
    kwargs['cancel_check'] = check_cancelled

    print(f"[DEBUG] run_task_with_callback: task_id={task_id}, progress_callback set, calling run_task")

    # Run using the original run_task function
    run_task(task_id, func, *args, **kwargs)


def run_task(task_id: str, func, *args, **kwargs):
    """Run a function in a background thread and update task status."""
    # Set up output capture to also send to callback
    class OutputCaptureWithCallback:
        def __init__(self, callback=None, original_stdout=None):
            self.callback = callback
            self.original_stdout = original_stdout

        def write(self, text):
            try:
                # Call callback FIRST for real-time updates
                if text is not None and text.strip():
                    text_str = str(text) if not isinstance(text, str) else text
                    if self.callback:
                        self.callback(text_str.rstrip())

                # Then forward to original stdout (log file)
                if self.original_stdout:
                    self.original_stdout.write(text)
            except:
                pass  # Ignore write errors

        def flush(self):
            try:
                if self.callback:
                    # Force callback to flush any pending data
                    pass
                if self.original_stdout:
                    self.original_stdout.flush()
            except:
                pass

    # Get progress_callback from kwargs if available, but don't pop it
    progress_callback = kwargs.get('progress_callback', None)
    print(f"[DEBUG] run_task: task_id={task_id}, progress_callback={progress_callback is not None}")

    # Redirect stdout and stderr FIRST (save originals)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Set up output capture with callback and original stdout
    capture = OutputCaptureWithCallback(progress_callback, original_stdout)

    sys.stdout = capture
    sys.stderr = capture

    try:
        with task_lock:
            tasks[task_id]['status'] = 'running'
            if 'log' not in tasks[task_id]:
                tasks[task_id]['log'] = ''

        result = func(*args, **kwargs)

        with task_lock:
            tasks[task_id]['status'] = 'completed'
            tasks[task_id]['result'] = result
    except Exception as e:
        import traceback
        with task_lock:
            # Check if this is a cancellation
            if 'Cancelled by user' in str(e):
                tasks[task_id]['status'] = 'cancelled'
                # Add clear cancellation message to log
                current_log = tasks[task_id].get('log', '')
                if not isinstance(current_log, str):
                    current_log = ''
                tasks[task_id]['log'] = current_log + f'\n✓ Process cancelled by user'
            else:
                tasks[task_id]['status'] = 'failed'
                tasks[task_id]['error'] = str(e)
                # Add error to log (but filter out whisper/yt-dlp termination errors from cancel)
                error_msg = str(e)
                if 'Terminated' not in error_msg and 'killed' not in error_msg.lower():
                    current_log = tasks[task_id].get('log', '')
                    if not isinstance(current_log, str):
                        current_log = ''
                    tasks[task_id]['log'] = current_log + f'\nERROR: {error_msg}'
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@app.route('/')
def index():
    """Serve the HTML frontend."""
    return send_from_directory('.', 'index.html')


@app.route('/edit.html')
def edit_page():
    """Serve the editor page."""
    return send_from_directory('.', 'edit.html')


@app.route('/adjust.html')
def adjust_page():
    """Serve the adjust theme page."""
    return send_from_directory('.', 'adjust.html')


@app.route('/videos/<path:filepath>')
def serve_video(filepath):
    """Serve video files from the videos directory."""
    base_dir = Path(settings.get('video', 'output_dir'))
    # Flask automatically decodes URL-encoded paths
    video_path = base_dir / filepath

    # Debug logging
    print(f"Video request: {filepath}", file=sys.stderr)
    print(f"Full path: {video_path}", file=sys.stderr)
    print(f"Exists: {video_path.exists()}", file=sys.stderr)

    if not video_path.exists():
        return jsonify({'error': f'File not found: {filepath}'}), 404

    # Use send_file with conditional support for range requests
    return send_file(str(video_path), mimetype='video/mp4', conditional=True)


@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings."""
    return jsonify({
        'whisper': {
            'model': settings.get('whisper', 'model'),
            'language': settings.get('whisper', 'language'),
        },
        'video': {
            'output_dir': settings.get('video', 'output_dir'),
            'aspect_ratio': settings.get('video', 'aspect_ratio'),
        },
        'theme': {
            'ai_enabled': settings.get('theme', 'ai_enabled') == 'true',
            'ai_model': settings.get('theme', 'ai_model'),
        }
    })


@app.route('/api/settings', methods=['POST'])
def save_settings():
    """Save settings."""
    data = request.json

    # Update settings
    if 'whisper' in data:
        settings.set('whisper', 'model', data['whisper'].get('model', 'small'))
        settings.set('whisper', 'language', data['whisper'].get('language', 'en'))

    if 'video' in data:
        settings.set('video', 'output_dir', data['video'].get('output_dir', 'videos'))
        settings.set('video', 'aspect_ratio', data['video'].get('aspect_ratio', '9:16'))

    if 'theme' in data:
        settings.set('theme', 'ai_enabled', 'true' if data['theme'].get('ai_enabled', False) else 'false')
        settings.set('theme', 'ai_model', data['theme'].get('ai_model', 'llama3'))

    # Save to file
    with open('settings.ini', 'w') as f:
        settings.write(f)

    return jsonify({'success': True})


@app.route('/api/folders', methods=['GET'])
def list_folders():
    """List all video folders."""
    base_dir = Path(settings.get('video', 'output_dir'))
    folders = []

    for folder in sorted(base_dir.iterdir()):
        if folder.is_dir():
            # Check if it has a video file
            video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
            if video_files:
                # Extract folder number
                match = folder.name.split('_')[0]
                if match.isdigit():
                    # Check for themes.md
                    themes_file = folder / 'themes.md'
                    has_themes = themes_file.exists()

                    # Check for shorts folder
                    shorts_dir = folder / 'shorts'
                    shorts_count = len(list(shorts_dir.glob('*.mp4'))) if shorts_dir.exists() else 0

                    folders.append({
                        'number': match,
                        'name': folder.name,
                        'path': str(folder),
                        'video_file': video_files[0].name,
                        'has_themes': has_themes,
                        'shorts_count': shorts_count
                    })

    return jsonify(folders)


@app.route('/api/folder/<folder_number>/themes', methods=['GET'])
def get_themes(folder_number: str):
    """Get themes for a specific folder."""
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None

    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    themes_file = folder / 'themes.md'
    if not themes_file.exists():
        return jsonify({'error': 'No themes file found'}), 404

    themes = creator.parse_themes_file(themes_file)

    # Check for existing shorts
    shorts_dir = folder / 'shorts'
    existing_shorts = {}
    if shorts_dir.exists():
        import re
        for short_file in shorts_dir.glob('theme_*.mp4'):
            # Extract theme number from filename: theme_001_title.mp4
            match = re.match(r'theme_(\d+)_', short_file.name)
            if match:
                theme_num = int(match.group(1))
                existing_shorts[theme_num] = {
                    'filename': short_file.name,
                    'path': str(short_file),
                    'size': short_file.stat().st_size
                }

    # Add short status to each theme
    for theme in themes:
        theme_num = theme['number']
        if theme_num in existing_shorts:
            theme['short_created'] = True
            theme['short_info'] = existing_shorts[theme_num]
        else:
            theme['short_created'] = False

    # Get video info
    video_info_file = folder / 'video info.txt'
    video_title = folder.name.split('_', 1)[1] if '_' in folder.name else folder.name
    video_filename = None

    # First, find the actual video file in the folder (universal fix for any special characters)
    video_files = []
    for ext in ['*.mp4', '*.mkv', '*.webm', '*.mov', '*.avi']:
        video_files.extend(folder.glob(ext))

    # Filter out videos in shorts/edited_shorts subdirectories
    main_video_files = [f for f in video_files
                        if 'shorts' not in f.parent.name
                        and f.is_file()]

    if main_video_files:
        # Use the first (main) video file found
        video_filename = main_video_files[0].name

    # Get title from video info.txt if available (for display only)
    if video_info_file.exists():
        with open(video_info_file, 'r') as f:
            for line in f:
                if line.startswith('Title:'):
                    video_title = line.split(':', 1)[1].strip()
                    break

    # Check for adjusted theme files and add adjusted values as separate fields
    # Keep original values from themes.md intact
    for theme in themes:
        adjust_file = folder / 'shorts' / f"theme_{theme['number']:03d}_adjust.md"
        if adjust_file.exists():
            try:
                with open(adjust_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Parse adjusted values from the file
                    import re
                    title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?:\n\n|\n\*)', content)
                    time_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})', content)

                    # Add adjusted values as separate fields, don't override original
                    if title_match:
                        theme['adjusted_title'] = title_match.group(1).strip()
                    if time_match:
                        theme['adjusted_start'] = time_match.group(1)
                        theme['adjusted_end'] = time_match.group(2)
                    theme['adjusted'] = True
            except Exception as e:
                print(f"Error reading adjust file {adjust_file}: {e}")

    # If no video file found, try to get from Video Path in info file
    if not video_filename and video_info_file.exists():
        with open(video_info_file, 'r') as f:
            for line in f:
                if line.startswith('Video Path:'):
                    path = line.split(':', 1)[1].strip()
                    potential_file = Path(path)
                    if potential_file.exists():
                        video_filename = potential_file.name
                    break

    return jsonify({
        'folder': folder.name,
        'title': video_title,
        'video_filename': video_filename,
        'themes': themes
    })


@app.route('/api/update-theme', methods=['POST'])
def update_theme():
    """Update theme details in themes.md file."""
    data = request.json
    folder_number = data.get('folder')
    theme_number = int(data.get('theme'))
    new_title = data.get('title')
    new_start = data.get('start')
    new_end = data.get('end')

    if not all([folder_number, theme_number, new_title, new_start, new_end]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Parse the themes file
    themes_file = folder / 'themes.md'
    if not themes_file.exists():
        return jsonify({'error': 'Themes file not found'}), 404

    themes = creator.parse_themes_file(themes_file)

    # Find and update the theme
    theme_found = False
    for theme in themes:
        if theme['number'] == theme_number:
            theme['title'] = new_title
            theme['start'] = new_start
            theme['end'] = new_end
            theme_found = True
            break

    if not theme_found:
        return jsonify({'error': 'Theme not found'}), 404

    # DO NOT modify themes.md - only create/update the adjust file

    # Save adjusted theme details to separate file in shorts folder
    shorts_dir = folder / 'shorts'
    shorts_dir.mkdir(exist_ok=True)
    adjust_file = shorts_dir / f'theme_{theme_number:03d}_adjust.md'

    # Calculate duration for the adjusted theme
    start_secs = creator.parse_timestamp_to_seconds(new_start)
    end_secs = creator.parse_timestamp_to_seconds(new_end)
    duration_secs = end_secs - start_secs
    minutes = int(duration_secs // 60)
    seconds = int(duration_secs % 60)
    duration_str = f"{minutes}m {seconds}s"

    # Read existing adjust.md file if it exists (to preserve Position field)
    existing_position = None
    if adjust_file.exists():
        with open(adjust_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract existing subtitle_position line
            import re
            position_match = re.search(r'\*\*subtitle_position:\*\*\s*(top|middle|bottom)', content)
            if position_match:
                existing_position = position_match.group(1)

    # Write adjust file with all fields including position
    with open(adjust_file, 'w', encoding='utf-8') as f:
        f.write(f"# Theme {theme_number}\n\n")
        f.write(f"**Title:** {new_title}\n\n")
        f.write(f"**Time Range:** {new_start} - {new_end} ({duration_str})\n")
        if existing_position:
            f.write(f"**subtitle_position:** {existing_position}\n")
        f.write(f"\n**Folder:** {folder.name}\n")
        f.write(f"**Last Modified:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return jsonify({'success': True, 'message': 'Theme updated successfully'})


@app.route('/api/reset-theme', methods=['POST'])
def reset_theme():
    """Reset theme time adjustment but preserve Position setting."""
    import re
    data = request.json
    folder_number = data.get('folder')
    theme_number = int(data.get('theme'))

    if not all([folder_number, theme_number]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Check adjust file and preserve Position
    shorts_dir = folder / 'shorts'
    adjust_file = shorts_dir / f'theme_{theme_number:03d}_adjust.md'

    existing_position = None
    if adjust_file.exists():
        with open(adjust_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract existing subtitle_position line
            position_match = re.search(r'\*\*subtitle_position:\*\*\s*(top|middle|bottom)', content)
            if position_match:
                existing_position = position_match.group(1)

    # Get original theme data from themes.md
    themes_file = folder / 'themes.md'
    if not themes_file.exists():
        return jsonify({'error': 'Themes file not found'}), 404

    themes = creator.parse_themes_file(themes_file)
    theme_found = False
    theme_title = None
    theme_start = None
    theme_end = None

    for theme in themes:
        if theme['number'] == theme_number:
            theme_title = theme.get('title', '')
            theme_start = theme['start']
            theme_end = theme['end']
            theme_found = True
            break

    if not theme_found:
        return jsonify({'error': 'Theme not found'}), 404

    # Rebuild adjust.md with original time range but preserve Position
    start_secs = creator.parse_timestamp_to_seconds(theme_start)
    end_secs = creator.parse_timestamp_to_seconds(theme_end)
    duration_secs = end_secs - start_secs
    minutes = int(duration_secs // 60)
    seconds = int(duration_secs % 60)
    duration_str = f"{minutes}m {seconds}s"

    with open(adjust_file, 'w', encoding='utf-8') as f:
        f.write(f"# Theme {theme_number}\n\n")
        f.write(f"**Title:** {theme_title}\n\n")
        f.write(f"**Time Range:** {theme_start} - {theme_end} ({duration_str})\n")
        if existing_position:
            f.write(f"**subtitle_position:** {existing_position}\n")
        f.write(f"\n**Folder:** {folder.name}\n")
        f.write(f"**Last Modified:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return jsonify({'success': True, 'message': 'Theme reset successfully'})


@app.route('/api/subtitles/<folder_number>.vtt', methods=['GET'])
def get_vtt_subtitles(folder_number: str):
    """Convert SRT to VTT and return as WebVTT format for browser native subtitles."""
    import re

    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None

    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return "Folder not found", 404

    # Find SRT file
    srt_files = list(folder.glob('*.srt'))
    if not srt_files:
        return "SRT file not found", 404

    srt_file = srt_files[0]

    # Check for offset parameter (for clip preview)
    offset = request.args.get('offset', default='0')
    try:
        offset_seconds = float(offset)
    except:
        offset_seconds = 0.0

    # Read SRT and convert to VTT
    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    # Convert SRT to VTT
    vtt_content = "WEBVTT\n\n"

    lines = srt_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip sequence numbers and empty lines
        if line.isdigit() or not line:
            i += 1
            continue

        # Look for timestamp lines (format: 00:00:00,000 --> 00:00:05,000)
        if '-->' in line:
            # Convert SRT timestamp to VTT format (commas to periods)
            timestamp = line.replace(',', '.')

            # Apply offset if specified
            if offset_seconds != 0:
                # Parse the timestamp
                match = re.match(r'(\d+):(\d+):([\d.]+)\s*-->\s*(\d+):(\d+):([\d.]+)', timestamp)
                if match:
                    h1, m1, s1, h2, m2, s2 = match.groups()
                    start_sec = int(h1) * 3600 + int(m1) * 60 + float(s1)
                    end_sec = int(h2) * 3600 + int(m2) * 60 + float(s2)

                    # Apply offset
                    start_sec -= offset_seconds
                    end_sec -= offset_seconds

                    # Only include if still visible after offset
                    if end_sec > 0:
                        start_sec = max(0, start_sec)
                        # Convert back to timestamp format
                        def sec_to_vtt(s):
                            h = int(s // 3600)
                            m = int((s % 3600) // 60)
                            sec = s % 60
                            return f"{h:02d}:{m:02d}:{sec:06.3f}"

                        timestamp = f"{sec_to_vtt(start_sec)} --> {sec_to_vtt(end_sec)}"
                    else:
                        # Skip this subtitle, it's before the clip start
                        i += 1
                        continue

            vtt_content += timestamp + '\n'

            # Add subtitle text lines until next empty line or sequence number
            i += 1
            while i < len(lines):
                text_line = lines[i].strip()
                if not text_line or text_line.isdigit():
                    break
                vtt_content += text_line + '\n'
                i += 1

            vtt_content += '\n'
        else:
            i += 1

    # Return with cache-control headers to prevent browser caching
    return vtt_content, 200, {
        'Content-Type': 'text/vtt; charset=utf-8',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }


@app.route('/api/theme-subtitles/<folder_number>/<theme_number>', methods=['GET'])
def get_theme_subtitles(folder_number: str, theme_number: str):
    """Get adjusted subtitles for a specific theme, or fall back to original filtered by theme time range."""
    import re

    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None

    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return "Folder not found", 404

    # Get theme start/end time from themes file
    themes_file = folder / 'themes.md'
    theme_start_sec = None
    theme_end_sec = None

    # First check if there's an adjust file with the theme times
    adjust_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'
    if adjust_file.exists():
        with open(adjust_file, 'r', encoding='utf-8') as f:
            content = f.read()
            time_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})', content)
            if time_match:
                theme_start_sec = creator.parse_timestamp_to_seconds(time_match.group(1))
                theme_end_sec = creator.parse_timestamp_to_seconds(time_match.group(2))
    else:
        # Parse themes file to get theme time range
        themes = creator.parse_themes_file(themes_file)
        for theme in themes:
            if theme['number'] == int(theme_number):
                theme_start_sec = creator.parse_timestamp_to_seconds(theme['start'])
                theme_end_sec = creator.parse_timestamp_to_seconds(theme['end'])
                break

    if theme_start_sec is None or theme_end_sec is None:
        return jsonify({'error': 'Theme time range not found'}), 404

    # Always use the original SRT for loading (filter based on current theme time)
    # The adjusted SRT is only for saving custom edits
    srt_files = list(folder.glob('*.srt'))
    if not srt_files:
        return "SRT file not found", 404

    srt_file = srt_files[0]

    # Check if adjusted subtitles exist (for info only)
    shorts_dir = folder / 'shorts'
    shorts_dir.mkdir(exist_ok=True)
    adjusted_srt = shorts_dir / f'theme_{int(theme_number):03d}_adjust.srt'
    has_adjusted = adjusted_srt.exists()

    # Read and parse SRT file
    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    # Parse SRT into JSON format, filtering by theme time range
    cues = []
    filtered_cues = []
    lines = srt_content.strip().split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Sequence number
        if line.isdigit():
            seq_num = int(line)
            i += 1

            # Timestamp line
            if i < len(lines) and '-->' in lines[i]:
                timestamp_line = lines[i].strip()
                # Parse timestamps: 00:00:00,000 --> 00:00:05,000
                match = re.match(r'(\d{2}:\d{2}:\d{2}),(\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}),(\d{3})', timestamp_line)
                if match:
                    # Convert to seconds for filtering
                    start_h, start_m, start_s = map(int, match.group(1).split(':'))
                    start_millis = int(match.group(2))
                    cue_start_sec = start_h * 3600 + start_m * 60 + start_s + start_millis / 1000

                    end_h, end_m, end_s = map(int, match.group(3).split(':'))
                    end_millis = int(match.group(4))
                    cue_end_sec = end_h * 3600 + end_m * 60 + end_s + end_millis / 1000

                    # Filter: only include cues that overlap with theme time range
                    if cue_end_sec > theme_start_sec and cue_start_sec < theme_end_sec:
                        # Get subtitle text (may be multiple lines)
                        i += 1
                        text_lines = []
                        while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                            text_lines.append(lines[i].strip())
                            i += 1

                        filtered_cues.append({
                            'sequence': seq_num,
                            'start': f"{match.group(1)}.{match.group(2)}",
                            'end': f"{match.group(3)}.{match.group(4)}",
                            'text': '\n'.join(text_lines)
                        })
                        continue
            else:
                i += 1
        else:
            i += 1

    return jsonify({
        'cues': filtered_cues,
        'is_adjusted': has_adjusted
    })


@app.route('/api/all-subtitles/<folder_number>', methods=['GET'])
def get_all_subtitles(folder_number: str):
    """Get ALL subtitles from the original SRT file with sequence numbers (no theme filtering)."""
    import re

    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None

    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return "Folder not found", 404

    # Find the SRT file (could have any name, not just folder_number.srt)
    srt_files = list(folder.glob('*.srt'))
    if not srt_files:
        return jsonify({'error': 'SRT file not found'}), 404

    srt_file = srt_files[0]  # Use the first (and likely only) SRT file
    print(f"[DEBUG] Using SRT file: {srt_file.name}")

    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    # Parse SRT into JSON format (no time filtering)
    all_cues = []
    lines = srt_content.strip().split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Sequence number
        if line.isdigit():
            seq_num = int(line)
            i += 1

            # Timestamp line
            if i < len(lines) and '-->' in lines[i]:
                timestamp_line = lines[i].strip()
                # Parse timestamps: 00:00:00,000 --> 00:00:05,000
                match = re.match(r'(\d{2}:\d{2}:\d{2}),(\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}),(\d{3})', timestamp_line)
                if match:
                    # Convert to VTT format (period instead of comma)
                    start_vtt = f"{match.group(1)}.{match.group(2)}"
                    end_vtt = f"{match.group(3)}.{match.group(4)}"

                    # Get subtitle text (may be multiple lines)
                    i += 1
                    text_lines = []
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                        text_lines.append(lines[i].strip())
                        i += 1

                    all_cues.append({
                        'sequence': seq_num,
                        'start': start_vtt,
                        'end': end_vtt,
                        'text': '\n'.join(text_lines)
                    })
                    continue
            else:
                i += 1
        else:
            i += 1

    return jsonify({
        'cues': all_cues,
        'total': len(all_cues)
    })


@app.route('/api/save-subtitle-formatting', methods=['POST'])
def save_subtitle_formatting():
    """Save subtitle formatting metadata for a theme."""
    data = request.json
    folder_number = data.get('folder')
    theme_number = data.get('theme')
    formatting = data.get('formatting', {})  # {start_time: {bold, italic, color, size, text}}

    print(f"[DEBUG] Saving subtitle formatting: folder={folder_number}, theme={theme_number}")
    print(f"[DEBUG] Formatting keys: {list(formatting.keys())}")
    print(f"[DEBUG] Formatting data: {formatting}")

    if not all([folder_number, theme_number]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Save formatting metadata to JSON file
    shorts_dir = folder / 'shorts'
    shorts_dir.mkdir(exist_ok=True)
    formatting_file = shorts_dir / f'theme_{int(theme_number):03d}_formatting.json'

    print(f"[DEBUG] Writing to: {formatting_file}")

    with open(formatting_file, 'w', encoding='utf-8') as f:
        json.dump(formatting, f, indent=2)

    print(f"[DEBUG] Successfully wrote formatting file")

    return jsonify({
        'success': True,
        'message': f'Saved subtitle formatting for theme {theme_number}'
    })


@app.route('/api/subtitle-formatting/<folder_number>/<theme_number>', methods=['GET'])
def get_subtitle_formatting(folder_number: str, theme_number: str):
    """Get subtitle formatting metadata for a theme."""

    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Check if formatting file exists
    formatting_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_formatting.json'

    if formatting_file.exists():
        with open(formatting_file, 'r', encoding='utf-8') as f:
            formatting = json.load(f)
        return jsonify({'formatting': formatting})
    else:
        return jsonify({'formatting': {}})


@app.route('/api/save-global-position', methods=['POST'])
def save_global_position():
    """Save global subtitle position for a theme to the adjust.md file."""
    import re
    data = request.json
    folder_number = data.get('folder')
    theme_number = data.get('theme')
    position = data.get('position', 'bottom')
    # Custom position data (optional)
    custom_left = data.get('left')
    custom_top = data.get('top')
    h_align = data.get('h_align', 'center')
    v_align = data.get('v_align', 'middle')

    if not all([folder_number, theme_number]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Path to adjust.md file
    adjust_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'
    shorts_dir = folder / 'shorts'
    shorts_dir.mkdir(exist_ok=True)

    # Read existing content or create new
    existing_title = None
    existing_time_range = None
    existing_folder = None

    if adjust_file.exists():
        with open(adjust_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract existing fields
            title_match = re.search(r'\*\*Title:\*\*\s*(.+)', content)
            time_match = re.search(r'\*\*Time Range:\*\*\s*(.+)', content)
            folder_match = re.search(r'\*\*Folder:\*\*\s*(.+)', content)
            if title_match:
                existing_title = title_match.group(1).strip()
            if time_match:
                existing_time_range = time_match.group(1).strip()
            if folder_match:
                existing_folder = folder_match.group(1).strip()
    else:
        # No existing file, fetch theme data from themes.md
        themes_file = folder / 'themes.md'
        if themes_file.exists():
            themes = creator.parse_themes_file(themes_file)
            for theme in themes:
                if theme['number'] == int(theme_number):
                    existing_title = theme.get('title', 'Theme Title')
                    # Calculate duration
                    start_secs = creator.parse_timestamp_to_seconds(theme['start'])
                    end_secs = creator.parse_timestamp_to_seconds(theme['end'])
                    duration_secs = end_secs - start_secs
                    minutes = int(duration_secs // 60)
                    seconds = int(duration_secs % 60)
                    duration_str = f"{minutes}m {seconds}s"
                    existing_time_range = f"{theme['start']} - {theme['end']} ({duration_str})"
                    break
        # Set defaults if not found in themes.md
        if not existing_title:
            existing_title = "Theme Title"
        if not existing_time_range:
            existing_time_range = "--:--:-- - --:--:--"
        existing_folder = folder.name

    # Rebuild file with updated position
    with open(adjust_file, 'w', encoding='utf-8') as f:
        f.write(f"# Theme {int(theme_number)}\n\n")
        if existing_title:
            f.write(f"**Title:** {existing_title}\n\n")
        f.write(f"**Time Range:** {existing_time_range}\n")

        # Write position - either preset or custom with coordinates
        if custom_left is not None and custom_top is not None:
            # Custom position with X/Y coordinates
            f.write(f"**subtitle_position:** custom\n")
            f.write(f"**subtitle_left:** {custom_left}\n")
            f.write(f"**subtitle_top:** {custom_top}\n")
            f.write(f"**subtitle_h_align:** {h_align}\n")
            f.write(f"**subtitle_v_align:** {v_align}\n")
        else:
            # Preset position
            f.write(f"**subtitle_position:** {position}\n")

        if existing_folder:
            f.write(f"\n**Folder:** {existing_folder}\n")
        f.write(f"**Last Modified:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"[DEBUG] Saved global position: folder={folder_number}, theme={theme_number}, position={position}, custom={custom_left}")

    return jsonify({
        'success': True,
        'message': f'Saved global position for theme {theme_number}',
        'position': position
    })


@app.route('/api/get-global-position', methods=['GET'])
def get_global_position():
    """Get global subtitle position for a theme from the adjust.md file."""
    import re
    folder_number = request.args.get('folder')
    theme_number = request.args.get('theme')

    if not all([folder_number, theme_number]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Path to adjust.md file
    adjust_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'

    # Read position from file
    if adjust_file.exists():
        with open(adjust_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if it's a custom position with coordinates
        position_match = re.search(r'\*\*subtitle_position:\*\*\s*custom', content)
        if position_match:
            # Read custom coordinates
            left_match = re.search(r'\*\*subtitle_left:\*\*\s*(\d+)', content)
            top_match = re.search(r'\*\*subtitle_top:\*\*\s*(\d+)', content)
            h_align_match = re.search(r'\*\*subtitle_h_align:\*\*\s*(left|center|right)', content)
            v_align_match = re.search(r'\*\*subtitle_v_align:\*\*\s*(top|middle|bottom)', content)

            result = {
                'position': 'custom',
                'left': int(left_match.group(1)) if left_match else None,
                'top': int(top_match.group(1)) if top_match else None,
                'h_align': h_align_match.group(1) if h_align_match else 'center',
                'v_align': v_align_match.group(1) if v_align_match else 'middle'
            }
            print(f"[DEBUG] Found global custom position: folder={folder_number}, theme={theme_number}, result={result}")
            return jsonify(result)

        # Check for preset position
        position_match = re.search(r'\*\*subtitle_position:\*\*\s*(top|middle|bottom)', content)
        if position_match:
            position = position_match.group(1)
            print(f"[DEBUG] Found global position: folder={folder_number}, theme={theme_number}, position={position}")
            return jsonify({'position': position})

    # No position found, return default
    print(f"[DEBUG] No global position found: folder={folder_number}, theme={theme_number}, using default='bottom'")
    return jsonify({'position': 'bottom'})


@app.route('/api/subtitles/<folder_number>/<theme_number>.vtt', methods=['GET'])
def get_theme_vtt_subtitles(folder_number: str, theme_number: str):
    """Get adjusted theme subtitles as VTT for preview video.

    Query parameters:
        start: Override theme start time (seconds)
        end: Override theme end time (seconds)
    """
    import re

    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None

    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return "Folder not found", 404

    # Check for query parameter overrides (from timeline dragging)
    start_override = request.args.get('start', type=float)
    end_override = request.args.get('end', type=float)

    # Debug logging
    app.logger.info(f"VTT request: folder={folder_number}, theme={theme_number}, start_override={start_override}, end_override={end_override}")

    # Get theme start/end time
    themes_file = folder / 'themes.md'
    theme_start_sec = None
    theme_end_sec = None

    # Use query parameters if provided, otherwise read from adjust/themes file
    if start_override is not None and end_override is not None:
        theme_start_sec = start_override
        theme_end_sec = end_override
        app.logger.info(f"Using query parameters: theme_start_sec={theme_start_sec}, theme_end_sec={theme_end_sec}")
    else:
        adjust_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'
        if adjust_file.exists():
            with open(adjust_file, 'r', encoding='utf-8') as f:
                content = f.read()
                time_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})', content)
                if time_match:
                    theme_start_sec = creator.parse_timestamp_to_seconds(time_match.group(1))
                    theme_end_sec = creator.parse_timestamp_to_seconds(time_match.group(2))
        else:
            themes = creator.parse_themes_file(themes_file)
            for theme in themes:
                if theme['number'] == int(theme_number):
                    theme_start_sec = creator.parse_timestamp_to_seconds(theme['start'])
                    theme_end_sec = creator.parse_timestamp_to_seconds(theme['end'])
                    break

    if theme_start_sec is None or theme_end_sec is None:
        return "Theme time range not found", 404

    # For preview, ALWAYS use the original SRT (not the trimmed/adjusted one)
    # The preview video plays from theme start in the original video, so it needs original timestamps
    srt_files = list(folder.glob('*.srt'))
    if not srt_files:
        return "SRT file not found", 404
    srt_file = srt_files[0]

    # Read SRT and convert to VTT
    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    # Convert SRT to VTT
    vtt_content = "WEBVTT\n\n"
    lines = srt_content.strip().split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and sequence numbers
        if not line or line.isdigit():
            i += 1
            continue

        # Look for timestamp lines
        if '-->' in line:
            # Convert SRT timestamp to VTT format (commas to periods)
            timestamp = line.replace(',', '.')

            # Parse timestamps to check if within theme range
            match = re.match(r'(\d+):(\d+):([\d.]+)\s*-->\s*(\d+):(\d+):([\d.]+)', timestamp)
            if match:
                h1, m1, s1 = int(match.group(1)), int(match.group(2)), float(match.group(3))
                h2, m2, s2 = int(match.group(4)), int(match.group(5)), float(match.group(6))
                cue_start_sec = h1 * 3600 + m1 * 60 + s1
                cue_end_sec = h2 * 3600 + m2 * 60 + s2

                # Only include if within theme range
                if cue_end_sec > theme_start_sec and cue_start_sec < theme_end_sec:
                    # Keep original timestamps for preview (preview video seeks to theme start)
                    vtt_content += timestamp + '\n'

                    # Add subtitle text lines
                    i += 1
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                        vtt_content += lines[i].strip() + '\n'
                        i += 1

                    vtt_content += '\n'
                    continue

        i += 1

    # Return with cache-control headers to prevent browser caching
    return vtt_content, 200, {
        'Content-Type': 'text/vtt; charset=utf-8',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }


@app.route('/api/shorts', methods=['GET'])
def list_shorts():
    """List all shorts from all folders."""
    base_dir = Path(settings.get('video', 'output_dir'))
    shorts = []

    for folder in sorted(base_dir.iterdir()):
        if folder.is_dir():
            shorts_dir = folder / 'shorts'
            if shorts_dir.exists():
                for short_file in sorted(shorts_dir.glob('theme_*.mp4')):
                    # Extract theme number from filename
                    import re
                    match = re.match(r'theme_(\d+)_', short_file.name)
                    theme_num = int(match.group(1)) if match else 0

                    # Get file size
                    size_bytes = short_file.stat().st_size
                    size_mb = round(size_bytes / (1024 * 1024), 2)

                    shorts.append({
                        'filename': short_file.name,
                        'folder': folder.name,
                        'folder_number': folder.name.split('_')[0],
                        'theme_number': theme_num,
                        'path': str(short_file.relative_to(base_dir)),
                        'url_path': str(short_file.relative_to(base_dir)),
                        'size': size_mb,
                        'size_bytes': size_bytes
                    })

    return jsonify(shorts)


@app.route('/api/process', methods=['POST'])
def process_video():
    """Process a new video (URL or local file)."""
    global task_counter

    data = request.json
    url = data.get('url', '').strip()
    local_file = data.get('local_file', '').strip()
    model = data.get('model', settings.get('whisper', 'model'))
    resolution = data.get('resolution', 'best')

    if not url and not local_file:
        return jsonify({'error': 'Either URL or local file must be provided'}), 400

    with task_lock:
        task_counter += 1
        task_id = f"task_{task_counter}"
        tasks[task_id] = {
            'type': 'process',
            'status': 'pending',
            'url': url if url else local_file,
            'log': '',
            'cancelled': False
        }

    # Start background task - pass task_id directly (daemon=True to allow Ctrl+C)
    thread = threading.Thread(
        target=run_task_with_callback,
        args=(task_id, _process_video, url, local_file, model, resolution),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


def _process_video(url: str, local_file: str, model: str, resolution: str = 'best', progress_callback=None, cancel_check=None):
    """Process video in background."""
    # Initialize AI generator
    ai_generator = None
    try:
        from ai_theme_generator import AIThemeGenerator
        ai_generator = AIThemeGenerator()
        if not ai_generator.is_available():
            ai_generator = None
    except ImportError:
        pass

    # Check for cancellation
    if cancel_check and cancel_check():
        raise Exception('Cancelled by user')

    if url:
        video_info = creator.download_video(url, resolution=resolution, progress_callback=progress_callback)
    else:
        video_info = creator.process_local_video(local_file, progress_callback=progress_callback)

    if cancel_check and cancel_check():
        raise Exception('Cancelled by user')

    creator.create_video_info(video_info, progress_callback=progress_callback)

    if cancel_check and cancel_check():
        raise Exception('Cancelled by user')

    creator.generate_subtitles(video_info, model_size=model, progress_callback=progress_callback)

    if cancel_check and cancel_check():
        raise Exception('Cancelled by user')

    creator.generate_themes(video_info, ai_generator=ai_generator, model_size=model, progress_callback=progress_callback)

    return {
        'folder': video_info['folder'],
        'folder_number': video_info['folder_number'],
        'title': video_info['title']
    }


@app.route('/api/regenerate-themes', methods=['POST'])
def regenerate_themes():
    """Regenerate themes for an existing video."""
    global task_counter

    data = request.json
    folder_number = data.get('folder_number', '').strip()
    model = data.get('model', settings.get('whisper', 'model'))

    if not folder_number:
        return jsonify({'error': 'folder_number is required'}), 400

    with task_lock:
        task_counter += 1
        task_id = f"task_{task_counter}"
        tasks[task_id] = {
            'type': 'regenerate',
            'status': 'pending',
            'folder_number': folder_number,
            'log': ''
        }

    thread = threading.Thread(
        target=run_task_with_callback,
        args=(task_id, _regenerate_themes, folder_number, model),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


def _regenerate_themes(folder_number: str, model: str, progress_callback=None):
    """Regenerate themes in background."""
    base_dir = Path(settings.get('video', 'output_dir'))
    matching_folders = list(base_dir.glob(f"{folder_number.zfill(3)}_*"))

    if not matching_folders:
        raise ValueError(f"Folder {folder_number} not found")

    folder = matching_folders[0]
    video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
    if not video_files:
        raise ValueError(f"No video file found in {folder}")

    video_path = video_files[0]

    # Check if subtitle file exists
    srt_file = folder / f"{video_path.stem}.srt"
    if not srt_file.exists():
        raise ValueError(f"No subtitle file found: {srt_file.name}")

    video_info = {
        'title': folder.name.split('_', 1)[1] if '_' in folder.name else folder.name,
        'url': 'Existing video',
        'folder': str(folder),
        'folder_number': folder_number,
        'video_path': str(video_path),
        'is_local': True
    }

    # Initialize AI generator
    ai_generator = None
    try:
        from ai_theme_generator import AIThemeGenerator
        ai_generator = AIThemeGenerator()
        if not ai_generator.is_available():
            ai_generator = None
    except ImportError:
        pass

    creator.generate_themes(video_info, ai_generator=ai_generator, model_size=model, progress_callback=progress_callback)

    return {
        'folder': str(folder),
        'folder_number': folder_number,
        'title': video_info['title']
    }


@app.route('/api/create-shorts', methods=['POST'])
def create_shorts():
    """Create shorts for selected themes."""
    global task_counter

    data = request.json
    folder_number = data.get('folder_number', '').strip()
    themes = data.get('themes', [])  # List of theme numbers, or 'all'

    if not folder_number:
        return jsonify({'error': 'folder_number is required'}), 400
    if not themes:
        return jsonify({'error': 'themes is required'}), 400

    with task_lock:
        task_counter += 1
        task_id = f"task_{task_counter}"
        tasks[task_id] = {
            'type': 'create_shorts',
            'status': 'pending',
            'folder_number': folder_number,
            'themes': themes
        }

    thread = threading.Thread(
        target=run_task_with_callback,
        args=(task_id, _create_shorts, folder_number, themes),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


def _create_shorts(folder_number: str, themes: List, progress_callback=None, cancel_check=None):
    """Create shorts in background."""
    print(f"[DEBUG] _create_shorts: progress_callback={progress_callback is not None}, cancel_check={cancel_check is not None}")
    # Check for cancellation at start
    if cancel_check and cancel_check():
        raise Exception('Cancelled by user')

    theme_str = 'all' if themes == 'all' else ','.join(map(str, themes))
    creator.create_shorts(folder_number, theme_str, progress_callback=progress_callback, cancel_check=cancel_check)

    base_dir = Path(settings.get('video', 'output_dir'))
    folder = creator.get_video_folder_by_number(folder_number)
    shorts_dir = folder / 'shorts'
    shorts_count = len(list(shorts_dir.glob('*.mp4'))) if shorts_dir.exists() else 0

    return {
        'folder': str(folder),
        'folder_number': folder_number,
        'shorts_created': shorts_count
    }


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id: str):
    """Get status of a background task."""
    with task_lock:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
        return jsonify(tasks[task_id])


@app.route('/api/task/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id: str):
    """Cancel a background task by terminating subprocesses and cleaning up files."""
    import subprocess
    import signal
    import shutil
    import time

    print(f"[CANCEL] Cancel requested for task: {task_id}")
    url = ''
    with task_lock:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404

        if tasks[task_id]['status'] == 'running':
            # Set cancelled flag and get URL
            tasks[task_id]['cancelled'] = True
            tasks[task_id]['status'] = 'cancelled'
            url = tasks[task_id].get('url', '')

    # Try to kill yt-dlp, whisper, and ffmpeg processes
    try:
        # Find and kill yt-dlp processes
        result = subprocess.run(['pgrep', '-f', 'yt-dlp'], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"Killed yt-dlp process {pid}")
                    except ProcessLookupError:
                        pass

        # Find and kill whisper processes
        result = subprocess.run(['pgrep', '-f', 'whisper'], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"Killed whisper process {pid}")
                    except ProcessLookupError:
                        pass

        # Also try pkill as fallback
        subprocess.run(['pkill', '-f', 'yt-dlp'], capture_output=True)
        subprocess.run(['pkill', '-f', 'whisper'], capture_output=True)
        subprocess.run(['pkill', '-f', 'ffmpeg'], capture_output=True)

    except Exception as e:
        print(f"Error during cancellation: {e}")

    # Wait a moment for processes to terminate
    time.sleep(0.5)

    # Clean up partial downloads
    output_dir = Path(settings.get('video', 'output_dir'))
    if url and output_dir.exists():
        try:
            # Look for folders that were recently created
            folders = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for folder in folders:
                if folder.is_dir():
                    # Check if folder has video file but no subtitles/themes yet (partial download)
                    video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
                    srt_files = list(folder.glob('*.srt'))
                    vtt_files = list(folder.glob('*.vtt'))
                    themes_file = folder / 'themes.md'

                    # If has video but no subtitles/themes, it's a partial download
                    if len(video_files) > 0 and len(srt_files) == 0 and len(vtt_files) == 0 and not themes_file.exists():
                        # This looks like a partial download, remove it
                        try:
                            shutil.rmtree(folder)
                            print(f"Cleaned up partial download: {folder}")

                            # Update log to show cleanup
                            with task_lock:
                                if task_id in tasks and 'log' in tasks[task_id]:
                                    current_log = tasks[task_id].get('log', '')
                                    if not isinstance(current_log, str):
                                        current_log = str(current_log) if current_log else ''
                                    tasks[task_id]['log'] = current_log + f'\n✓ Removed partial files: {folder.name}'
                            break  # Only remove the most recent one
                        except Exception as e:
                            print(f"Failed to cleanup {folder}: {e}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    # Clean up partial shorts for create_shorts tasks
    print(f"[CANCEL] Checking cleanup for task {task_id}")
    with task_lock:
        task_type = tasks.get(task_id, {}).get('type', 'unknown')
        print(f"[CANCEL] Task type: {task_type}")
        if task_id in tasks and tasks[task_id].get('type') == 'create_shorts':
            folder_number = tasks[task_id].get('folder_number', '')
            print(f"[CANCEL] Folder number: {folder_number}")
            if folder_number:
                folder = creator.get_video_folder_by_number(folder_number)
                if folder:
                    shorts_dir = folder / 'shorts'
                    if shorts_dir.exists():
                        try:
                            print(f"[CANCEL] Cleaning up shorts directory: {shorts_dir}")

                            # List all files first for debugging
                            all_files = list(shorts_dir.glob('*'))
                            print(f"[CANCEL] Found {len(all_files)} files in shorts directory")
                            for f in all_files:
                                print(f"[CANCEL]   - {f.name}")

                            # Remove ALL mp4 files (both complete and partial)
                            mp4_count = 0
                            for video_file in shorts_dir.glob('*.mp4'):
                                try:
                                    video_file.unlink()
                                    mp4_count += 1
                                    print(f"[CANCEL] Removed short: {video_file.name}")
                                except Exception as e:
                                    print(f"[CANCEL] Failed to remove {video_file}: {e}")

                            # Remove ALL trimmed SRT files (theme_XXX.srt)
                            srt_count = 0
                            for srt_file in shorts_dir.glob('*.srt'):
                                try:
                                    srt_file.unlink()
                                    srt_count += 1
                                    print(f"[CANCEL] Removed SRT: {srt_file.name}")
                                except Exception as e:
                                    print(f"[CANCEL] Failed to remove {srt_file}: {e}")

                            print(f"[CANCEL] Removed {mp4_count} MP4 files and {srt_count} SRT files")

                            # Update log to show cleanup
                            if task_id in tasks and 'log' in tasks[task_id]:
                                current_log = tasks[task_id].get('log', '')
                                if not isinstance(current_log, str):
                                    current_log = str(current_log) if current_log else ''
                                tasks[task_id]['log'] = current_log + f'\n✓ Cleaned up {mp4_count} short(s) and {srt_count} subtitle file(s)'

                        except Exception as e:
                            print(f"[CANCEL] Error cleaning up shorts: {e}")
                            import traceback
                            traceback.print_exc()

    return jsonify({'success': True})


# Track edit processes for cancellation
edit_processes: Dict[str, Dict] = {}
edit_counter = 0


@app.route('/api/process-edit', methods=['POST'])
def process_video_edit():
    """Process video with effects including face tracking."""
    global edit_counter

    data = request.json
    video_path = data.get('video_path')
    edit_settings = data.get('settings', {})

    if not video_path:
        return jsonify({'error': 'video_path is required'}), 400

    # Construct full video path
    base_dir = Path(settings.get('video', 'output_dir'))
    input_video = base_dir / video_path

    if not input_video.exists():
        return jsonify({'error': f'Video not found: {video_path}'}), 404

    # Use edited_shorts folder for output
    output_dir = input_video.parent / 'edited_shorts'
    output_dir.mkdir(exist_ok=True)

    # Generate output path
    output_filename = f"edited_{input_video.stem}.mp4"
    output_video = output_dir / output_filename

    with task_lock:
        edit_counter += 1
        edit_id = f"edit_{edit_counter}"
        edit_processes[edit_id] = {
            'status': 'pending',
            'output_path': str(output_video),
            'cancelled': False,
            'log': ''
        }

    # Start background task (daemon=True to allow Ctrl+C)
    thread = threading.Thread(
        target=run_edit_task,
        args=(edit_id, str(input_video), str(output_video), edit_settings),
        daemon=True
    )
    thread.start()

    return jsonify({'edit_id': edit_id, 'status': 'started'})


def run_edit_task(edit_id: str, input_video: str, output_video: str, edit_settings: dict):
    """Run video edit in background thread."""
    from video_processor import VideoProcessor
    import os
    import time

    # Function to log directly to status
    def log_message(msg):
        with task_lock:
            if edit_id in edit_processes:
                current_log = edit_processes[edit_id].get('log', '')
                edit_processes[edit_id]['log'] = current_log + msg + '\n'

    try:
        with task_lock:
            edit_processes[edit_id]['status'] = 'running'
            edit_processes[edit_id]['log'] = ''  # Initialize log

        log_message(f"Processing video: {input_video}")
        log_message(f"Settings: {edit_settings}")

        # Process video with ffmpeg (preserves audio)
        processor = VideoProcessor(input_video)
        processor.apply_effects(output_video, edit_settings, cancel_flag=lambda: edit_processes.get(edit_id, {}).get('cancelled', False), log_callback=log_message)

        with task_lock:
            edit_processes[edit_id]['status'] = 'completed'
            edit_processes[edit_id]['success'] = True

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log_message(f"Processing error: {error_details}")

        # Clean up partial output file on error or cancellation
        if os.path.exists(output_video):
            try:
                os.remove(output_video)
                log_message(f"Cleaned up partial output file: {output_video}")
            except Exception as cleanup_error:
                log_message(f"Failed to clean up output file: {cleanup_error}")

        with task_lock:
            # Check if this was a cancellation
            if edit_processes.get(edit_id, {}).get('cancelled'):
                edit_processes[edit_id]['status'] = 'cancelled'
            else:
                edit_processes[edit_id]['status'] = 'failed'
                edit_processes[edit_id]['error'] = str(e)
    finally:
        # Final log update is done by log_message so no need to do it here
        pass


@app.route('/api/process-edit/<edit_id>/cancel', methods=['POST'])
def cancel_edit(edit_id: str):
    """Cancel a running video edit."""
    with task_lock:
        if edit_id not in edit_processes:
            return jsonify({'error': 'Edit not found'}), 404

        edit_processes[edit_id]['cancelled'] = True
        edit_processes[edit_id]['status'] = 'cancelled'

    return jsonify({'success': True})


@app.route('/api/process-edit/<edit_id>/status', methods=['GET'])
def get_edit_status(edit_id: str):
    """Get status of a video edit."""
    with task_lock:
        if edit_id not in edit_processes:
            return jsonify({'error': 'Edit not found'}), 404

        return jsonify(edit_processes[edit_id])


# === Karaoke Highlighting API Endpoints ===

@app.route('/api/check-word-timestamps', methods=['GET'])
def check_word_timestamps():
    """Check if word timestamps JSON file exists for a video."""
    folder_number = request.args.get('folder')

    if not folder_number:
        return jsonify({'error': 'folder is required'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Check for word timestamps file
    word_timestamps_file = None
    for file in folder.glob('*_word_timestamps.json'):
        word_timestamps_file = file
        break

    exists = word_timestamps_file is not None and word_timestamps_file.exists()

    # Get video duration for estimation
    video_duration = None
    if exists:
        try:
            with open(word_timestamps_file, 'r') as f:
                data = json.load(f)
                video_duration = data.get('duration')
        except:
            pass

    return jsonify({
        'exists': exists,
        'video_duration': format_duration(video_duration) if video_duration else None
    })


@app.route('/api/create-word-timestamps', methods=['POST'])
def create_word_timestamps():
    """Create word timestamps by re-transcribing the video with word_timestamps=True."""
    data = request.json
    folder_number = data.get('folder')

    if not folder_number:
        return jsonify({'error': 'folder is required'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Find the video file
    video_file = None
    for ext in ['.mp4', '.mkv', '.webm', '.mov', '.avi']:
        potential = folder / f"{folder.name}{ext}"
        if potential.exists():
            video_file = potential
            break

    if not video_file:
        return jsonify({'error': 'No video file found'}), 404

    # Create a task for word timestamp generation
    global task_counter
    with task_lock:
        task_counter += 1
        task_id = f"task_{task_counter}"
        tasks[task_id] = {
            'type': 'create_word_timestamps',
            'status': 'pending',
            'folder_number': folder_number,
            'video_path': str(video_file)
        }

    # Run in background thread
    thread = threading.Thread(
        target=run_task_with_callback,
        args=(task_id, _create_word_timestamps, folder_number, str(video_file)),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


def _create_word_timestamps(folder_number: str, video_path: str, progress_callback=None, cancel_check=None):
    """Create word timestamps in background."""
    if cancel_check and cancel_check():
        raise Exception('Cancelled by user')

    # Get video info
    video_info = {'video_path': video_path, 'folder': Path(video_path).parent}

    # Run transcription with word timestamps
    srt_path = creator.generate_subtitles(
        video_info,
        model_size='base',  # Use base model for speed
        language=None,
        task='transcribe',
        progress_callback=progress_callback
    )

    # Verify word timestamps were created
    folder = Path(video_path).parent
    word_timestamps_files = list(folder.glob('*_word_timestamps.json'))

    if not word_timestamps_files:
        raise Exception('Word timestamps file was not created')

    return {
        'word_timestamps_file': str(word_timestamps_files[0]),
        'srt_file': srt_path
    }


@app.route('/api/save-karaoke-setting', methods=['POST'])
def save_karaoke_setting():
    """Save karaoke setting to adjust.md file."""
    import re
    data = request.json
    folder_number = data.get('folder')
    theme_number = data.get('theme')
    enabled = data.get('enabled', True)

    if not all([folder_number, theme_number]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Path to adjust.md file
    adjust_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'

    # Read existing content or create new
    if adjust_file.exists():
        with open(adjust_file, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = f'# Theme {theme_number} Settings\n\n'

    # Update or add karaoke setting
    karaoke_pattern = r'\*\*karaoke_highlighting:\*\*\s*(true|false)'
    karaoke_line = f'**karaoke_highlighting:** {str(enabled).lower()}'

    if re.search(karaoke_pattern, content):
        content = re.sub(karaoke_pattern, karaoke_line, content)
    else:
        content += f'\n{karaoke_line}\n'

    # Write back
    with open(adjust_file, 'w', encoding='utf-8') as f:
        f.write(content)

    return jsonify({'success': True, 'enabled': enabled})


@app.route('/api/get-karaoke-setting', methods=['GET'])
def get_karaoke_setting():
    """Get karaoke setting from adjust.md file."""
    import re
    folder_number = request.args.get('folder')
    theme_number = request.args.get('theme')

    if not all([folder_number, theme_number]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Path to adjust.md file
    adjust_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'

    # Read setting from file
    if adjust_file.exists():
        with open(adjust_file, 'r', encoding='utf-8') as f:
            content = f.read()

        karaoke_match = re.search(r'\*\*karaoke_highlighting:\*\*\s*(true|false)', content)
        if karaoke_match:
            enabled = karaoke_match.group(1) == 'true'
            return jsonify({'enabled': enabled})

    # Default: enabled
    return jsonify({'enabled': True})


@app.route('/api/save-highlight-style', methods=['POST'])
def save_highlight_style():
    """Save highlight style settings to JSON file."""
    import re
    data = request.json
    folder_number = data.get('folder')
    style_data = {
        'preset': data.get('preset', 'yellow-glow'),
        'textColor': data.get('textColor', '#ffff00'),
        'glowColor': data.get('glowColor', '#ffff00'),
        'glowBlur': data.get('glowBlur', '10'),
        'fontWeight': data.get('fontWeight', 'bold'),
        # Karaoke mode settings
        'karaoke_mode': data.get('karaokeMode', 'normal'),
        'font_size_scale': data.get('fontSizeScale', 1.0),
        'past_color': data.get('pastColor', None)
    }

    if not folder_number:
        return jsonify({'error': 'Missing required fields'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Path to highlight style file (saved in parent folder, applies to all themes)
    style_file = folder / 'highlight_style.json'

    # Save as JSON
    with open(style_file, 'w', encoding='utf-8') as f:
        json.dump(style_data, f, indent=2)

    return jsonify({'success': True, 'style': style_data})


@app.route('/api/get-highlight-style', methods=['GET'])
def get_highlight_style():
    """Get highlight style settings from highlight_style.json file."""
    folder_number = request.args.get('folder')

    if not folder_number:
        return jsonify({'error': 'Missing required fields'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Path to highlight style file
    style_file = folder / 'highlight_style.json'

    # Read style from file
    if style_file.exists():
        with open(style_file, 'r', encoding='utf-8') as f:
            style_data = json.load(f)
            return jsonify({'style': style_data})

    # Default: return empty style (will use defaults)
    return jsonify({'style': None})


@app.route('/api/regenerate-ass', methods=['POST'])
def regenerate_ass():
    """Regenerate ASS files with karaoke on/off."""
    data = request.json
    folder_number = data.get('folder')
    theme_number = data.get('theme')
    karaoke = data.get('karaoke', True)

    if not all([folder_number, theme_number]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Find SRT and word timestamps files
    # Use the theme-specific SRT from the shorts folder, not the main video SRT
    srt_file = folder / 'shorts' / f'theme_{int(theme_number):03d}.srt'
    if not srt_file.exists():
        return jsonify({'error': 'Theme SRT file not found'}), 404

    # Word timestamps are in the parent folder (from the main video transcription)
    # Find any word timestamps file in the parent folder
    word_timestamps = None
    for file in folder.glob('*_word_timestamps.json'):
        try:
            with open(file, 'r') as f:
                word_data = json.load(f)
                word_timestamps = word_data.get('words', [])
                print(f"    Loaded {len(word_timestamps)} word timestamps from {file.name}")
                break
        except Exception as e:
            print(f"Warning: Could not load word timestamps from {file.name}: {e}")

    # Check for formatting JSON
    formatting_json_path = folder / 'shorts' / f'theme_{int(theme_number):03d}_formatting.json'

    # Check for adjust.md
    adjust_md_path = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'

    # Load highlight style settings if karaoke is enabled
    karaoke_style = None
    if karaoke and word_timestamps is not None:
        style_file = folder / 'highlight_style.json'
        if style_file.exists():
            try:
                with open(style_file, 'r', encoding='utf-8') as f:
                    style_data = json.load(f)
                    # Convert JSON format to karaoke_style dict
                    karaoke_style = {
                        'mode': style_data.get('karaoke_mode', 'normal'),
                        'font_size_scale': style_data.get('font_size_scale', 1.0),
                        'past_color': style_data.get('past_color', None),
                        'textColor': style_data.get('textColor', '#ffff00')
                    }
            except Exception as e:
                print(f"Warning: Could not load highlight style: {e}")

    try:
        # Create ASS formatter
        from ass_formatter import ASSFormatter
        formatter = ASSFormatter(settings)

        # Generate ASS output path
        ass_output_path = folder / 'shorts' / f'theme_{int(theme_number):03d}.ass'

        # Create ASS file
        success = formatter.create_ass_file(
            srt_file,
            formatting_json_path,
            ass_output_path,
            adjust_md_path,
            use_karaoke=karaoke and word_timestamps is not None,
            karaoke_style=karaoke_style
        )

        if success:
            return jsonify({
                'success': True,
                'ass_file': str(ass_output_path),
                'karaoke': karaoke and word_timestamps is not None
            })
        else:
            return jsonify({'error': 'Failed to create ASS file'}), 500
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in regenerate_ass: {e}\n{error_trace}")
        return jsonify({'error': f'Failed to create ASS: {str(e)}'}), 500


def format_duration(seconds):
    """Format seconds to human-readable duration."""
    if not seconds:
        return None

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f'{hours}h {minutes}m'
    elif minutes > 0:
        return f'{minutes}m {secs}s'
    else:
        return f'{secs}s'


@app.route('/api/word-timestamps/<folder_number>', methods=['GET'])
def get_word_timestamps_api(folder_number):
    """Get word timestamps JSON for a video."""
    if not folder_number:
        return jsonify({'error': 'folder is required'}), 400

    # Find the folder
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    # Find word timestamps file
    word_timestamps_file = None
    for file in folder.glob('*_word_timestamps.json'):
        word_timestamps_file = file
        break

    if not word_timestamps_file or not word_timestamps_file.exists():
        return jsonify({'words': []})

    # Read and return word timestamps
    try:
        with open(word_timestamps_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/encode-canvas-karaoke', methods=['POST'])
def encode_canvas_karaoke():
    """Generate and encode canvas karaoke video server-side (fast)."""
    try:
        data = request.get_json()
        folder_number = data.get('folder')
        theme_number = data.get('theme')
        karaoke_settings = data.get('settings', {})

        if not folder_number or not theme_number:
            return jsonify({'error': 'Missing folder or theme number'}), 400

        # Get folder path
        base_dir = Path(settings.get('video', 'output_dir'))
        folder = None

        for f in base_dir.iterdir():
            if f.is_dir() and f.name.startswith(f"{folder_number}_"):
                folder = f
                break

        if not folder:
            return jsonify({'error': 'Folder not found'}), 404

        # Get video file
        video_files = list(folder.glob('*.mp4'))
        if not video_files:
            return jsonify({'error': 'No video file found'}), 404
        video_file = video_files[0]

        # Get theme timing from themes.md
        themes_file = folder / 'themes.md'
        if not themes_file.exists():
            return jsonify({'error': 'themes.md not found'}), 404

        with open(themes_file, 'r', encoding='utf-8') as f:
            themes_content = f.read()

        # Parse time range
        import re
        time_pattern = r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})'
        time_match = re.search(time_pattern, themes_content)

        if not time_match:
            return jsonify({'error': 'Could not parse theme time range from themes.md'}), 400

        def parse_time_str(time_str):
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s

        start_time = parse_time_str(time_match.group(1))
        end_time = parse_time_str(time_match.group(2))

        # Get word timestamps
        word_timestamps_file = None
        for file in folder.glob('*_word_timestamps.json'):
            word_timestamps_file = file
            break

        if not word_timestamps_file or not word_timestamps_file.exists():
            return jsonify({'error': 'Word timestamps not found'}), 404

        # Get theme SRT
        srt_file = folder / 'shorts' / f'theme_{int(theme_number):03d}.srt'
        if not srt_file.exists():
            srt_file = folder / 'adjust.srt'

        if not srt_file.exists():
            return jsonify({'error': 'Subtitle file not found'}), 404

        # Output path
        output_path = folder / 'shorts' / f'theme_{theme_number}_canvas_karaoke.mp4'
        output_path.parent.mkdir(exist_ok=True)

        # Create job ID
        job_id = f"{folder_number}_{theme_number}"

        # Initialize progress tracking
        with canvas_karaoke_lock:
            canvas_karaoke_progress[job_id] = {
                'progress': 0,
                'stage': 'starting',
                'message': 'Initializing...',
                'complete': False,
                'error': None,
                'output_path': str(output_path)
            }

        # Progress callback
        def progress_callback(progress, stage, message):
            with canvas_karaoke_lock:
                if job_id in canvas_karaoke_progress:
                    canvas_karaoke_progress[job_id].update({
                        'progress': progress,
                        'stage': stage,
                        'message': message,
                        'complete': (stage == 'complete'),
                        'error': (message if stage == 'error' else None)
                    })

        # Background rendering function
        def render_in_background():
            try:
                app_logger.info(f"Starting server-side canvas karaoke export for theme {theme_number}")

                # Import the renderer
                import canvas_karaoke_exporter

                # Render video with progress callback
                success = canvas_karaoke_exporter.render_canvas_karaoke_video(
                    str(video_file),
                    str(word_timestamps_file),
                    str(srt_file),
                    str(output_path),
                    start_time,
                    end_time,
                    {
                        'fontSize': karaoke_settings.get('fontSize', 48) * 2,  # Double for 1080x1920
                        'fontName': karaoke_settings.get('fontName', 'Arial'),
                        'textColor': karaoke_settings.get('textColor', '#ffff00'),
                        'pastColor': karaoke_settings.get('pastColor', '#808080'),
                        'mode': karaoke_settings.get('mode', 'normal')
                    },
                    progress_callback
                )

                if not success:
                    progress_callback(-1, 'error', 'Rendering failed')

            except Exception as e:
                import traceback
                app_logger.error(f"Export error: {traceback.format_exc()}")
                progress_callback(-1, 'error', str(e))

        # Start background thread
        thread = threading.Thread(target=render_in_background, daemon=True)
        thread.start()

        # Return immediately with job ID
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Export started'
        })

    except Exception as e:
        import traceback
        app_logger.error(f"Export error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/canvas-karaoke-progress/<job_id>')
def canvas_karaoke_progress_endpoint(job_id):
    """Get progress for a canvas karaoke export job."""
    with canvas_karaoke_lock:
        if job_id not in canvas_karaoke_progress:
            return jsonify({'error': 'Job not found'}), 404

        progress_info = canvas_karaoke_progress[job_id].copy()

        # Clean up old completed jobs (older than 5 minutes)
        if progress_info.get('complete') or progress_info.get('error'):
            # Keep it for now so client can get final status
            pass

        return jsonify(progress_info)


@app.route('/api/download-canvas-karaoke/<folder>/<theme>')
def download_canvas_karaoke(folder, theme):
    """Download the canvas karaoke video."""
    try:
        base_dir = Path(settings.get('video', 'output_dir'))
        folder = None

        for f in base_dir.iterdir():
            if f.is_dir() and f.name.startswith(f"{folder}_"):
                folder = f
                break

        if not folder:
            return jsonify({'error': 'Folder not found'}), 404

        video_path = folder / 'shorts' / f'theme_{theme}_canvas_karaoke.mp4'

        if not video_path.exists():
            return jsonify({'error': 'Video not found'}), 404

        return send_file(str(video_path), as_attachment=True, download_name=video_path.name)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-canvas-karaoke', methods=['POST'])
def export_canvas_karaoke():
    """Export canvas karaoke video using FFmpeg on server (fast processing)."""
    try:
        data = request.get_json()
        folder_number = data.get('folder')
        theme_number = data.get('theme')
        karaoke_settings = data.get('settings', {})

        if not folder_number or not theme_number:
            return jsonify({'error': 'Missing folder or theme number'}), 400

        # Get folder path (same pattern as other endpoints)
        base_dir = Path(settings.get('video', 'output_dir'))
        folder = None

        for f in base_dir.iterdir():
            if f.is_dir() and f.name.startswith(f"{folder_number}_"):
                folder = f
                break

        if not folder:
            return jsonify({'error': 'Folder not found'}), 404

        # Get video file
        video_files = list(folder.glob('*.mp4'))
        if not video_files:
            return jsonify({'error': 'No video file found'}), 404
        video_file = video_files[0]

        # Get theme timing from themes.md file
        themes_file = folder / 'themes.md'
        if not themes_file.exists():
            return jsonify({'error': 'themes.md not found'}), 404

        with open(themes_file, 'r', encoding='utf-8') as f:
            themes_content = f.read()

        # Parse time range for the theme (format: "00:00:48 - 00:02:00")
        import re
        time_pattern = r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})'

        # Find the time range for the specific theme
        # Look for the theme section first
        theme_section_pattern = rf'### Theme {theme_number}:.*?\*\*Time Range:\*\*\s*(\d{{2}}:\d{{2}}:\d{{2}})\s*-\s*(\d{{2}}:\d{{2}}:\d{{2}})'

        time_match = re.search(theme_section_pattern, themes_content, re.DOTALL)

        if not time_match:
            # Fallback: try to find any time range in the file
            time_match = re.search(time_pattern, themes_content)

        if not time_match:
            return jsonify({'error': 'Could not parse theme time range from themes.md'}), 400

        start_time_str = time_match.group(1)
        end_time_str = time_match.group(2)

        # Convert "HH:MM:SS" to seconds
        def parse_time_str(time_str):
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s

        start_time = parse_time_str(start_time_str)
        end_time = parse_time_str(end_time_str)

        # Output path
        output_path = folder / 'shorts' / f'theme_{theme_number}_canvas_karaoke.mp4'
        output_path.parent.mkdir(exist_ok=True)

        # Get word timestamps
        word_timestamps_file = None
        for file in folder.glob('*_word_timestamps.json'):
            word_timestamps_file = file
            break

        if not word_timestamps_file or not word_timestamps_file.exists():
            return jsonify({'error': 'Word timestamps not found'}), 404

        with open(word_timestamps_file, 'r', encoding='utf-8') as f:
            word_timestamps_data = json.load(f)
            words = word_timestamps_data.get('words', [])

        # Get theme subtitles for the text
        srt_file = folder / 'shorts' / f'theme_{int(theme_number):03d}.srt'
        if not srt_file.exists():
            # Fall back to adjust.srt
            srt_file = folder / 'adjust.srt'

        if not srt_file.exists():
            return jsonify({'error': 'Subtitle file not found'}), 404

        # Use existing ASS file with karaoke effects
        # The regenerate-ass endpoint already creates these: theme_XXX_normal.ass, theme_XXX_cumulative.ass
        if karaoke_settings.get('mode') == 'cumulative':
            ass_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_cumulative.ass'
        else:
            ass_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_normal.ass'

        if not ass_file.exists():
            # Fallback to any theme ASS file
            ass_files = list(folder.glob('shorts/theme_*.ass'))
            if not ass_files:
                return jsonify({'error': 'No ASS subtitle file found. Please regenerate ASS files first.'}), 404
            ass_file = ass_files[0]

        # Use FFmpeg to burn subtitles into video with 9:16 aspect ratio
        # For 16:9 to 9:16: scale height to 1920, then crop width to 1080, then apply subtitles
        video_filter = f'scale=-1:1920,crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2,ass={ass_file}'

        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-ss', str(start_time),  # Start time (input seeking)
            '-i', str(video_file),  # Input video
            '-t', str(end_time - start_time),  # Duration
            '-vf', video_filter,
            '-c:v', 'libx264',  # Video codec
            '-preset', 'medium',  # Encoding preset
            '-crf', '23',  # Quality
            '-c:a', 'aac',  # Audio codec
            '-b:a', '128k',  # Audio bitrate
            '-movflags', '+faststart',  # Fast start for web
            str(output_path)  # Output file
        ]

        app_logger.info(f"Running FFmpeg command for canvas karaoke export: {' '.join(ffmpeg_cmd)}")

        app_logger.info(f"Running FFmpeg command for canvas karaoke export: {' '.join(ffmpeg_cmd)}")

        # Run FFmpeg
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            app_logger.error(f"FFmpeg error: {result.stderr}")
            return jsonify({'error': f'FFmpeg failed: {result.stderr}'}), 500

        return jsonify({
            'success': True,
            'output_path': str(output_path),
            'message': 'Video exported successfully'
        })

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Export timed out (5 minutes)'}), 500
    except Exception as e:
        import traceback
        app_logger.error(f"Export error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download-video/<folder>/<theme>/<type>')
def download_video(folder, theme, type):
    """Download exported video."""
    try:
        folder = get_video_folder_by_number(folder)
        if not folder:
            return jsonify({'error': 'Folder not found'}), 404

        if type == 'canvas_karaoke':
            video_path = folder / 'shorts' / f'theme_{theme}_canvas_karaoke.mp4'
        else:
            return jsonify({'error': 'Invalid type'}), 400

        if not video_path.exists():
            return jsonify({'error': 'Video not found'}), 404

        return send_file(str(video_path), as_attachment=True, download_name=video_path.name)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import signal

    print("=" * 60)
    print("YouTube Shorts Creator - Web Server")
    print("=" * 60)
    print(f"Server running at: http://localhost:5000")
    print(f"Video directory: {settings.get('video', 'output_dir')}")
    print("=" * 60)

    # Log server startup
    app_logger.info("=" * 60)
    app_logger.info("YouTube Shorts Creator - Web Server STARTED")
    app_logger.info(f"Server running at: http://localhost:5000")
    app_logger.info(f"Video directory: {settings.get('video', 'output_dir')}")
    app_logger.info("=" * 60)

    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\033[93mShutdown requested. Cleaning up...\033[0m")
        app_logger.warning("Shutdown requested. Cleaning up...")
        # Cancel all running tasks
        with task_lock:
            for task_id, task in tasks.items():
                if task.get('status') == 'running':
                    task['cancelled'] = True
        print("\033[92mServer shut down cleanly.\033[0m")
        app_logger.info("Server shut down cleanly")
        import sys
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)

    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=True)
