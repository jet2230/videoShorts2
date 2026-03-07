#!/usr/bin/env python3
"""
Flask server for the YouTube Shorts Creator web GUI.
"""

from flask import Flask, jsonify, request, send_from_directory, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import subprocess
import threading
import time
import queue
import os
import sys
import re
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Disable hardware acceleration for FFmpeg backend to fix AV1 decoding issues
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;none"

from typing import Dict, List
import configparser
from datetime import datetime
import json
import urllib.request
import urllib.parse
import urllib.error

from shorts_creator import YouTubeShortsCreator, load_settings

# Setup logging to file with rotation
# Max size set to 10MB to ensure we capture enough history
log_handler = RotatingFileHandler('server.log', maxBytes=10*1024*1024, backupCount=5)
log_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)

# Get Flask's logger and add file handler
flask_logger = logging.getLogger('werkzeug')
flask_logger.setLevel(logging.WARNING) # Silence request logs
flask_logger.addHandler(log_handler)

# Also log our app messages
app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_logger.addHandler(log_handler)

# Ensure canvas_karaoke_exporter logger also logs to file
exporter_logger = logging.getLogger('canvas_karaoke_exporter')
exporter_logger.setLevel(logging.INFO)
exporter_logger.addHandler(log_handler)

# Ensure subtitle_renderer logger also logs to file
subtitle_logger = logging.getLogger('subtitle_renderer')
subtitle_logger.setLevel(logging.INFO)
subtitle_logger.addHandler(log_handler)

# Keep console output too
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)
# flask_logger.addHandler(console_handler)
# app_logger.addHandler(console_handler)
# exporter_logger.addHandler(console_handler)

app = Flask(__name__, static_folder='static')
CORS(app)

# Use a local .tmp directory instead of /tmp (tmpfs) to save RAM
GLOBAL_TEMP_DIR = Path('videos/.tmp')
GLOBAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)

@app.before_request
def log_request_info():
    if "/status" not in request.path and "/progress" not in request.path:
        app_logger.info(f"INCOMING REQUEST: {request.method} {request.path}")

# Required for SharedArrayBuffer (ffmpeg.wasm support)
@app.after_request
def set_headers(response):
    """Set security headers required for SharedArrayBuffer (ffmpeg.wasm)."""
    # Log every request for debugging except high-frequency polling
    if "/status" not in request.path and "/progress" not in request.path:
        app_logger.info(f"[REQ] {request.method} {request.path} -> {response.status_code}")
    
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Content-Range'
    return response

@app.route('/favicon.ico')
def favicon():
    """Return an empty response for favicon to prevent 404s."""
    return '', 204

@app.route('/api/font/<name>')
def serve_font(name):
    """Serve font files from media/fonts."""
    fonts_dir = Path('media/fonts')
    if not fonts_dir.exists():
        return "Fonts directory not found", 404
        
    # Try common extensions
    for ext in ['.ttf', '.otf', '.woff', '.woff2']:
        font_path = fonts_dir / (name + ext)
        if font_path.exists():
            return send_file(str(font_path))
            
    return "Font not found", 404

# Utility functions for background tasks
def to_float(val, default=0.0):
    try:
        if val is None or str(val).strip() == "": return default
        return float(val)
    except: return default

def to_int(val, default=0):
    try:
        if val is None or str(val).strip() == "": return default
        return int(float(val))
    except: return default

# Global state
creator = YouTubeShortsCreator()
settings = load_settings()
task_counter = 0

import multiprocessing

# Set start method to 'spawn' for CUDA compatibility in subprocesses
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Global process manager for shared state
task_manager = None
tasks = {}
task_processes = {}
task_lock = threading.RLock()

def parse_timestamp(s):
    """Parses timestamps in HH:MM:SS.mmm, MM:SS.mmm, or float string format."""
    if s is None:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not s:
        return 0.0
    try:
        # Check if it's just a number
        return float(s)
    except ValueError:
        # HH:MM:SS or HH:MM:SS.mmm
        s = s.replace(',', '.')
        parts = s.split(':')
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        elif len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
        else:
            raise ValueError(f"Unknown timestamp format: {s}")

def format_srt_time(seconds):
    """Formats seconds to HH:MM:SS,mmm string."""
    milliseconds = int((seconds % 1) * 1000)
    total_seconds = int(seconds)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
# Canvas karaoke progress tracking
canvas_karaoke_progress = {}
canvas_karaoke_lock = threading.RLock()

def ensure_manager():
    """Ensure that the multiprocessing manager and shared dictionaries are initialized."""
    global task_manager, tasks, canvas_karaoke_progress
    if task_manager is None:
        app_logger.info("Initializing multiprocessing manager and shared state...")
        task_manager = multiprocessing.Manager()
        
        # Convert existing tasks to managed dict if it's a plain dict
        if not hasattr(tasks, 'shm'): # Rough check for Manager.dict
            old_tasks = dict(tasks)
            tasks = task_manager.dict()
            for k, v in old_tasks.items():
                tasks[k] = v
                
        # Convert existing progress to managed dict if it's a plain dict
        if not hasattr(canvas_karaoke_progress, 'shm'):
            old_progress = dict(canvas_karaoke_progress)
            canvas_karaoke_progress = task_manager.dict()
            for k, v in old_progress.items():
                canvas_karaoke_progress[k] = v
    return task_manager

def run_task_with_callback(task_id: str, func, *args, **kwargs):
    """Run a function in a background process with shared state support."""
    global tasks, task_processes
    ensure_manager()

    # Ensure task entry exists in the shared dictionary
    with task_lock:
        if task_id not in tasks:
            tasks[task_id] = task_manager.dict({
                'type': kwargs.get('task_type', 'unknown'),
                'status': 'pending',
                'log': '',
                'progress': 0,
                'cancelled': False
            })

    # Start independent process
    p = multiprocessing.Process(
        target=_process_wrapper,
        args=(task_id, func, tasks, args, kwargs)
    )
    p.start()
    
    with task_lock:
        task_processes[task_id] = p

def _process_wrapper(task_id, func, shared_tasks, args, kwargs):
    """Internal wrapper that runs inside the child process."""
    import sys

    import traceback
    
    # Define a callback that updates the SHARED dictionary
    def process_callback(msg):
        msg_str = str(msg)
        # Log to the real system stdout so we can see it in terminal
        sys.__stdout__.write(f"[{task_id}] {msg_str}\n")
        sys.__stdout__.flush()
        
        # ALSO log to server.log via app_logger
        app_logger.info(f"[{task_id}] {msg_str}")
        
        try:
            # We must pull, update, and push back the sub-dictionary
            # because manager.dict() doesn't track nested changes automatically
            task_data = dict(shared_tasks[task_id])
            
            # Simple percentage parsing
            match = re.search(r'Progress:\s*(\d+)%', msg_str)
            if match:
                task_data['progress'] = int(match.group(1))
            
            task_data['log'] = task_data.get('log', '') + msg_str + '\n'
            task_data['status'] = 'processing'
            
            # Write back to shared memory
            shared_tasks[task_id] = task_data
        except Exception as e:
            sys.__stdout__.write(f"Callback error in process {task_id}: {e}\n")

    # Add special kwargs for the function to use
    kwargs['progress_callback'] = process_callback
    
    def check_cancelled():
        return shared_tasks.get(task_id, {}).get('cancelled', False)
    kwargs['cancel_check'] = check_cancelled

    try:
        # Actually run the heavy task (Whisper, FFmpeg, etc)
        result = func(*args, **kwargs)
        
        # Mark as complete in shared memory
        task_data = dict(shared_tasks[task_id])
        task_data['status'] = 'complete'
        task_data['progress'] = 100
        task_data['result'] = str(result)
        shared_tasks[task_id] = task_data
        
    except Exception as e:
        task_data = dict(shared_tasks[task_id])
        if "Cancelled by user" in str(e) or task_data.get('cancelled'):
            task_data['status'] = 'cancelled'
            task_data['log'] = task_data.get('log', '') + "\n✓ Process stopped by user."
        else:
            task_data['status'] = 'failed'
            task_data['error'] = str(e)
            task_data['log'] = task_data.get('log', '') + f"\nERROR: {e}\n{traceback.format_exc()}"
        
        shared_tasks[task_id] = task_data
        sys.__stdout__.write(f"Process {task_id} failed or cancelled: {e}\n")
        sys.__stdout__.flush()



def get_available_fonts():
    """Returns a list of available font names by scanning media/fonts and system fonts."""
    fonts = ['Arial', 'Calibri', 'Times New Roman', 'Courier New', 'Georgia', 'Verdana', 'Impact', 'Comic Sans MS']
    
    # Scan media/fonts directory
    fonts_dir = Path('media/fonts')
    if fonts_dir.exists():
        for font_file in fonts_dir.glob('*.*'):
            if font_file.suffix.lower() in ['.ttf', '.otf', '.woff', '.woff2']:
                # Use filename without extension as font name
                fonts.append(font_file.stem)
            
    # Remove duplicates and sort
    return sorted(list(set(fonts)))

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


@app.route('/finalcut.html')
def finalcut_page():
    """Serve the final cut editor page."""
    return send_from_directory('.', 'finalcut.html')


@app.route('/videos/<path:filepath>')
def serve_video(filepath):
    """Serve video files from the videos directory."""
    base_dir = Path(settings.get('video', 'output_dir'))
    # Flask automatically decodes URL-encoded paths
    video_path = base_dir / filepath

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
        },
        'subtitle': {
            'font_name': settings.get('subtitle', 'font_name'),
            'font_size': settings.get('subtitle', 'font_size'),
            'max_subtitle_lines': int(settings.get('subtitle', 'max_subtitle_lines', fallback=2)),
            'available_fonts': get_available_fonts()
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
        
        # New theme settings
        if 'theme_cap' in data['theme']:
            settings.set('theme', 'theme_cap', str(data['theme']['theme_cap']))
        if 'target_themes_per_window' in data['theme']:
            settings.set('theme', 'target_themes_per_window', str(data['theme']['target_themes_per_window']))
        if 'min_duration' in data['theme']:
            settings.set('theme', 'min_duration', str(data['theme']['min_duration']))
        if 'max_duration' in data['theme']:
            settings.set('theme', 'max_duration', str(data['theme']['max_duration']))

    if 'subtitle' in data:
        if 'font_name' in data['subtitle']:
            settings.set('subtitle', 'font_name', data['subtitle']['font_name'])
        if 'font_size' in data['subtitle']:
            settings.set('subtitle', 'font_size', str(data['subtitle']['font_size']))
        if 'max_subtitle_lines' in data['subtitle']:
            settings.set('subtitle', 'max_subtitle_lines', str(data['subtitle']['max_subtitle_lines']))

    # Save to file with comments manually to preserve them
    try:
        with open('settings.ini', 'w', encoding='utf-8') as f:
            f.write("; YouTube Shorts Creator Settings\n\n")
            
            f.write("[whisper]\n")
            f.write(f"; Whisper model size: tiny, base, small, medium, large\n")
            f.write(f"model = {settings.get('whisper', 'model')}\n")
            f.write(f"; Default language code or 'auto' for detection\n")
            f.write(f"language = {settings.get('whisper', 'language')}\n")
            f.write(f"; Task type: transcribe or translate\n")
            f.write(f"task = {settings.get('whisper', 'task')}\n\n")
            
            f.write("[video]\n")
            f.write(f"; Directory where processed content is stored\n")
            f.write(f"output_dir = {settings.get('video', 'output_dir')}\n")
            f.write(f"; Output video shape (Shorts MUST be 9:16)\n")
            f.write(f"aspect_ratio = {settings.get('video', 'aspect_ratio')}\n")
            f.write(f"resolution_width = {settings.get('video', 'resolution_width')}\n")
            f.write(f"resolution_height = {settings.get('video', 'resolution_height')}\n")
            f.write(f"codec = {settings.get('video', 'codec')}\n")
            f.write(f"preset = {settings.get('video', 'preset')}\n")
            f.write(f"crf = {settings.get('video', 'crf')}\n\n")
            
            f.write("[subtitle]\n")
            f.write(f"max_subtitle_lines = {settings.get('subtitle', 'max_subtitle_lines')}\n")
            f.write(f"font_name = {settings.get('subtitle', 'font_name')}\n")
            f.write(f"font_size = {settings.get('subtitle', 'font_size')}\n")
            f.write(f"primary_colour = {settings.get('subtitle', 'primary_colour')}\n")
            f.write(f"back_colour = {settings.get('subtitle', 'back_colour')}\n")
            f.write(f"outline_colour = {settings.get('subtitle', 'outline_colour')}\n")
            f.write(f"alignment = {settings.get('subtitle', 'alignment')}\n")
            f.write(f"margin_v = {settings.get('subtitle', 'margin_v')}\n\n")
            
            f.write("[theme]\n")
            f.write(f"; Duration range for clips\n")
            f.write(f"min_duration = {settings.get('theme', 'min_duration')}\n")
            f.write(f"max_duration = {settings.get('theme', 'max_duration')}\n")
            f.write(f"; AI theme detection settings\n")
            f.write(f"ai_enabled = {settings.get('theme', 'ai_enabled')}\n")
            f.write(f"ai_model = {settings.get('theme', 'ai_model')}\n")
            f.write(f"ai_provider = {settings.get('theme', 'ai_provider')}\n")
            f.write(f"; Window-based processing for long videos\n")
            f.write(f"window_duration = {settings.get('theme', 'window_duration')}\n")
            f.write(f"window_overlap = {settings.get('theme', 'window_overlap')}\n")
            f.write(f"; Global limit on total themes per video\n")
            f.write(f"theme_cap = {settings.get('theme', 'theme_cap')}\n")
            f.write(f"; Themes to find in each 10-minute window\n")
            f.write(f"target_themes_per_window = {settings.get('theme', 'target_themes_per_window')}\n\n")
            
            f.write("[folder]\n")
            f.write(f"naming_scheme = {settings.get('folder', 'naming_scheme')}\n")
            f.write(f"number_padding = {settings.get('folder', 'number_padding')}\n")
    except Exception as e:
        app_logger.error(f"Error saving settings.ini: {e}")
        return jsonify({'error': str(e)}), 500

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


@app.route('/api/folder/<folder_number>', methods=['GET'])
def get_folder_contents(folder_number: str):
    """Get list of files in a folder before deletion."""
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None

    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    try:
        files = []
        total_size = 0

        def scan_directory(path, relative_path=""):
            nonlocal total_size
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))

            for item in items:
                rel_path = f"{relative_path}/{item.name}" if relative_path else item.name

                if item.is_dir():
                    files.append({
                        'name': item.name,
                        'path': rel_path,
                        'type': 'directory',
                        'size': 0
                    })
                    scan_directory(item, rel_path)
                else:
                    size = item.stat().st_size
                    total_size += size
                    files.append({
                        'name': item.name,
                        'path': rel_path,
                        'type': 'file',
                        'size': size,
                        'size_human': format_size(size)
                    })

        scan_directory(folder)

        return jsonify({
            'folder_name': folder.name,
            'files': files,
            'total_count': len(files),
            'total_size': total_size,
            'total_size_human': format_size(total_size)
        })
    except Exception as e:
        app_logger.error(f"Error listing folder contents: {e}")
        return jsonify({'error': str(e)}), 500


def format_size(bytes_size):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def parse_srt_time(time_str):
    """Convert SRT time (HH:MM:SS,mmm) to seconds."""
    parts = time_str.replace(',', '.').split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def format_srt_time(seconds):
    """Convert seconds to SRT time (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')


@app.route('/api/folder/<folder_number>', methods=['DELETE'])
def delete_folder(folder_number: str):
    """Delete a video folder and all its contents."""
    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None

    # Find the folder by number
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    try:
        # Delete the entire folder and all its contents
        import shutil
        shutil.rmtree(folder)
        app_logger.debug(f"Deleted folder: {folder}")
        return jsonify({'success': True, 'message': f'Folder {folder.name} deleted successfully'})
    except Exception as e:
        app_logger.error(f"Error deleting folder {folder}: {e}")
        return jsonify({'error': f'Failed to delete folder: {str(e)}'}), 500


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
    themes = []
    if themes_file.exists():
        themes = creator.parse_themes_file(themes_file)

    # Check for existing shorts
    shorts_dir = folder / 'shorts'
    existing_shorts = {}
    if shorts_dir.exists():

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

                    title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?:\n\n|\n\*)', content)
                    time_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)', content)

                    # Add adjusted values as separate fields, don't override original
                    if title_match:
                        theme['adjusted_title'] = title_match.group(1).strip()
                    if time_match:
                        theme['adjusted_start'] = time_match.group(1)
                        theme['adjusted_end'] = time_match.group(2)
                    theme['adjusted'] = True
            except Exception as e:
                print(f"Error reading adjust file {adjust_file}: {e}")
        
        # Check for persistent subtitle edits and apply them to theme text
        edits_file = folder / 'shorts' / f"theme_{theme['number']:03d}_edits.json"
        if edits_file.exists():
            try:
                with open(edits_file, 'r', encoding='utf-8') as f:
                    edits = json.load(f)
                
                # If there are edits, we should ideally rebuild the 'text' field
                # For now, let the frontend handle the granular cue edits,
                # but we can mark that edits exist.
                theme['has_subtitle_edits'] = True
            except:
                pass

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


@app.route('/api/add-theme', methods=['POST'])
def add_theme():
    """Add a new custom theme to themes.md."""
    try:
        data = request.json
        folder_number = data.get('folder')
        title = data.get('title', 'Custom Theme')

        if not folder_number:
            return jsonify({'error': 'Folder is required'}), 400

        folder = creator.get_video_folder_by_number(folder_number)
        if not folder:
            return jsonify({'error': 'Folder not found'}), 404

        themes_file = folder / 'themes.md'
        if not themes_file.exists():
            # Create a basic themes.md if it doesn't exist
            with open(themes_file, 'w', encoding='utf-8') as f:
                f.write("# Video Themes\n\n")

        themes = creator.parse_themes_file(themes_file)
        next_num = 1
        if themes:
            next_num = max(int(t.get('number', 0)) for t in themes) + 1

        # Default time range: 0 to 10 seconds
        new_theme_md = f"\n### Theme {next_num}: {title}\n"
        new_theme_md += f"**Time Range:** 00:00:00 - 00:00:10\n"
        new_theme_md += f"**Type:** custom\n"
        new_theme_md += f"**Description:** User added custom theme.\n"

        with open(themes_file, 'a', encoding='utf-8') as f:
            f.write(new_theme_md)

        # Also create initial adjust.md file with all defaults immediately
        shorts_dir = folder / 'shorts'
        shorts_dir.mkdir(exist_ok=True)
        adjust_file = shorts_dir / f'theme_{next_num:03d}_adjust.md'
        
        # Default styling values
        default_settings = {
            'is_custom': True,
            'subtitle_position': 'bottom',
            'subtitle_h_align': 'center',
            'subtitle_v_align': 'bottom',
            'subtitle_bold': False,
            'fontSize': 80,
            'primaryColor': '#ffffff',
            'bgColor': '#000000',
            'bgOpacity': 0.5,
            'fontName': 'Avenir Black',
            'title_font_size': 67,
            'title_bg_type': 'gradient',
            'title_text_color': '#00ff9d',
            'title_font_weight': 800,
            'title_outline_width': 0,
            'title_outline_color': '#000000',
            'title_all_caps': True,
            'show_title': False,
            'title_top': 150
        }
        
        write_theme_adjust_settings(
            adjust_file, 
            next_num, 
            title, 
            "00:00:00 - 00:00:10 (0m 10s)", 
            folder.name, 
            default_settings
        )

        return jsonify({
            'success': True,
            'theme': {
                'number': next_num,
                'title': title,
                'start': '00:00:00',
                'end': '00:00:10',
                'is_custom': True
            }
        })
    except Exception as e:
        import traceback
        app_logger.error(f"Error adding custom theme: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


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
    is_custom = False
    for theme in themes:
        if theme['number'] == theme_number:
            theme['title'] = new_title
            theme['start'] = new_start
            theme['end'] = new_end
            is_custom = theme.get('is_custom', False)
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

    # Preserve existing styling if it exists
    existing_settings = get_theme_adjust_settings(folder, theme_number)
    if is_custom:
        existing_settings['is_custom'] = True

    # Write adjust file with all fields including position and styling
    write_theme_adjust_settings(adjust_file, theme_number, new_title, f"{new_start} - {new_end} ({duration_str})", folder.name, existing_settings)

    # CRITICAL: Immediately refresh the theme-specific SRT and JSON files to match new boundaries
    try:
        # Find master SRT
        srt_files = [f for f in folder.glob('*.srt') if 'theme_' not in f.name and 'adjust' not in f.name]
        if not srt_files: srt_files = list(folder.glob('*.srt'))
        main_srt = srt_files[0] if srt_files else None
        
        if main_srt:
            theme_srt_path = shorts_dir / f"theme_{theme_number:03d}.srt"
            theme_json_path = shorts_dir / f"theme_{theme_number:03d}.json"
            
            # Load edits to inject
            edits = {}
            edits_file = shorts_dir / f"theme_{theme_number:03d}_edits.json"
            if edits_file.exists():
                try:
                    with open(edits_file, 'r', encoding='utf-8') as f:
                        edits = json.load(f)
                except: pass
            
            # Refresh SRT
            creator.create_trimmed_srt(main_srt, start_secs, end_secs, theme_srt_path, edits=edits)
            
            # Refresh JSON (word timestamps)
            word_timestamps_file = next(folder.glob('[!theme]*_word_timestamps.json'), None)
            if word_timestamps_file:
                with open(word_timestamps_file, 'r', encoding='utf-8') as f:
                    all_words = json.load(f).get('words', [])
                
                theme_words = []
                for w in all_words:
                    # Precise overlap check with a tiny 100ms buffer
                    if w['end'] > start_secs - 0.1 and w['start'] < end_secs + 0.1:
                        rw = w.copy()
                        # Timestamps are relative to the start of the theme clip
                        # CLAMP TO RANGE for clean output
                        rw['start'] = max(0, rw['start'] - start_secs)
                        rw['end'] = min(end_secs - start_secs, rw['end'] - start_secs)
                        if rw['end'] > rw['start']:
                            theme_words.append(rw)
                    
                with open(theme_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'theme': str(theme_number),
                        'start_time': start_secs,
                        'end_time': end_secs,
                        'words': theme_words
                    }, f, indent=2)
            
            app_logger.info(f"Theme {theme_number} boundaries updated: Refreshed SRT and JSON.")
    except Exception as e:
        app_logger.warning(f"Failed to auto-refresh theme metadata in update_theme: {e}")

    return jsonify({'success': True, 'message': 'Theme updated successfully'})


@app.route('/api/reset-theme', methods=['POST'])
def reset_theme():
    """Reset theme time adjustment but preserve Position setting."""

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

    # Get original theme data from themes.md
    themes_file = folder / 'themes.md'
    if not themes_file.exists():
        return jsonify({'error': 'themes.md not found'}), 404
    
    themes = creator.parse_themes_file(themes_file)
    theme_data = next((t for t in themes if t['number'] == theme_number), None)
    if not theme_data:
        return jsonify({'error': f'Theme {theme_number} not found in themes.md'}), 404

    theme_start = theme_data['start']
    theme_end = theme_data['end']
    theme_title = theme_data['title']

    # Path to adjust.md file
    adjust_file = folder / 'shorts' / f'theme_{theme_number:03d}_adjust.md'

    # Preserve existing styling if it exists
    existing_settings = get_theme_adjust_settings(folder, theme_number)

    # Rebuild adjust.md with original time range but preserve Position and styling
    start_secs = creator.parse_timestamp_to_seconds(theme_start)
    end_secs = creator.parse_timestamp_to_seconds(theme_end)
    duration_secs = end_secs - start_secs
    minutes = int(duration_secs // 60)
    seconds = int(duration_secs % 60)
    duration_str = f"{minutes}m {seconds}s"

    write_theme_adjust_settings(adjust_file, theme_number, theme_title, f"{theme_start} - {theme_end} ({duration_str})", folder.name, existing_settings)

    # CRITICAL: Immediately refresh the theme-specific SRT and JSON files to match new boundaries
    try:
        shorts_dir = folder / 'shorts'
        # Find master SRT
        srt_files = [f for f in folder.glob('*.srt') if 'theme_' not in f.name and 'adjust' not in f.name]
        if not srt_files: srt_files = list(folder.glob('*.srt'))
        main_srt = srt_files[0] if srt_files else None
        
        if main_srt:
            theme_srt_path = shorts_dir / f"theme_{theme_number:03d}.srt"
            theme_json_path = shorts_dir / f"theme_{theme_number:03d}.json"
            
            # Refresh SRT
            creator.create_trimmed_srt(main_srt, start_secs, end_secs, theme_srt_path)
            
            # Refresh JSON (word timestamps)
            word_timestamps_file = next(folder.glob('[!theme]*_word_timestamps.json'), None)
            if word_timestamps_file:
                with open(word_timestamps_file, 'r', encoding='utf-8') as f:
                    all_words = json.load(f).get('words', [])
                
                theme_words = []
                for w in all_words:
                    if w['end'] > start_secs - 0.1 and w['start'] < end_secs + 0.1:
                        rw = w.copy()
                        rw['start'] = max(0, rw['start'] - start_secs)
                        rw['end'] = min(end_secs - start_secs, rw['end'] - start_secs)
                        if rw['end'] > rw['start']:
                            theme_words.append(rw)
                    
                with open(theme_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'theme': str(theme_number),
                        'start_time': start_secs,
                        'end_time': end_secs,
                        'words': theme_words
                    }, f, indent=2)
    except Exception as e:
        app_logger.error(f"Error refreshing files during reset: {e}")

    return jsonify({'success': True, 'message': 'Theme reset successfully'})


@app.route('/api/theme-subtitles/<folder_number>/<theme_number>', methods=['GET'])
def get_theme_subtitles(folder_number: str, theme_number: str):
    """Get adjusted subtitles for a specific theme, or fall back to original filtered by theme time range."""


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
            time_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)', content)
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

    # Determine source of truth for subtitles
    theme_srt_name = f"theme_{int(theme_number):03d}.srt"
    theme_srt_path = folder / 'shorts' / theme_srt_name
    
    # 1. Ensure theme-specific SRT and JSON files exist in the shorts directory
    # This ensures they exist for editing as soon as adjust.html loads the theme
    try:
        shorts_dir = folder / 'shorts'
        shorts_dir.mkdir(exist_ok=True)
        
        # Determine the fallback main SRT if needed
        srt_files = [f for f in folder.glob('*.srt') if 'theme_' not in f.name and 'adjust' not in f.name]
        if not srt_files: srt_files = list(folder.glob('*.srt'))
        main_srt = srt_files[0] if srt_files else None

        if not theme_srt_path.exists() and main_srt:
            # Load edits to inject
            edits = {}
            edits_file = shorts_dir / f"theme_{int(theme_number):03d}_edits.json"
            if edits_file.exists():
                try:
                    with open(edits_file, 'r', encoding='utf-8') as f:
                        edits = json.load(f)
                except: pass
            
            creator.create_trimmed_srt(main_srt, theme_start_sec, theme_end_sec, theme_srt_path, edits=edits)
            app_logger.info(f"Generated missing theme SRT: {theme_srt_name}")
        
        # Create theme JSON word timestamps (theme_XXX.json) ONLY if it doesn't exist
        theme_json_name = f"theme_{int(theme_number):03d}.json"
        theme_json_path = shorts_dir / theme_json_name
        
        if not theme_json_path.exists():
            # Find original word timestamps
            word_timestamps_file = None
            for file in folder.glob('*_word_timestamps.json'):
                word_timestamps_file = file
                break
                
            if word_timestamps_file and word_timestamps_file.exists():
                with open(word_timestamps_file, 'r', encoding='utf-8') as f:
                    wt_data = json.load(f)
                    all_words = wt_data.get('words', [])
                    
                # Filter words for this theme and make timestamps relative
                theme_words = []
                for w in all_words:
                    # Precise overlap check with a tiny 100ms buffer
                    if w['end'] > theme_start_sec - 0.1 and w['start'] < theme_end_sec + 0.1:
                        rw = w.copy()
                        # Timestamps are relative to the start of the theme clip
                        # CLAMP TO RANGE for clean output
                        rw['start'] = max(0, rw['start'] - theme_start_sec)
                        rw['end'] = min(theme_end_sec - theme_start_sec, rw['end'] - theme_start_sec)
                        if rw['end'] > rw['start']:
                            theme_words.append(rw)
                        
                with open(theme_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'theme': theme_number,
                        'start_time': theme_start_sec,
                        'end_time': theme_end_sec,
                        'words': theme_words
                    }, f, indent=2)
                app_logger.info(f"Generated missing theme metadata JSON: {theme_json_name}")
    except Exception as e:
        app_logger.warning(f"Failed to generate theme metadata: {e}")

    # 2. Select the SRT file to load from
    is_theme_srt = False
    # ALWAYS read from the master SRT for the UI to ensure we have the full context buffer
    srt_files = [f for f in folder.glob('*.srt') if 'theme_' not in f.name and 'adjust' not in f.name]
    if not srt_files: srt_files = list(folder.glob('*.srt'))
    if not srt_files: return "SRT file not found", 404
    srt_file = srt_files[0]

    # Check if adjusted subtitles exist (for info only)
    shorts_dir = folder / 'shorts'
    adjusted_srt = shorts_dir / f'theme_{int(theme_number):03d}_adjust.srt'
    has_adjusted = adjusted_srt.exists()

    # Read and parse SRT file
    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    # Parse SRT into JSON format, filtering by theme time range if it's the main SRT
    cues = []
    filtered_cues = []
    
    # Robust splitting by empty lines
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    
    for block in blocks:
        block_lines = block.strip().split('\n')
        if not block_lines: continue
        
        # Look for the timing line
        for idx, line in enumerate(block_lines):
            time_match = re.search(r'(-?\d{2}:\d{2}:\d{2})[.,](\d{3})\s*-->\s*(-?\d{2}:\d{2}:\d{2})[.,](\d{3})', line)
            if time_match:
                # Sequence is either the line before or a fallback
                seq_str = block_lines[idx-1].strip() if idx > 0 else str(len(filtered_cues) + 1)
                
                # Convert to seconds
                cue_start_sec = creator.parse_timestamp_to_seconds(f"{time_match.group(1)}.{time_match.group(2)}")
                cue_end_sec = creator.parse_timestamp_to_seconds(f"{time_match.group(3)}.{time_match.group(4)}")

                # Filter by theme range
                buffer_sec = 120.0
                if is_theme_srt or (cue_end_sec > (theme_start_sec - buffer_sec) and cue_start_sec < (theme_end_sec + buffer_sec)):
                    # Text is everything after timing line
                    text = '\n'.join(block_lines[idx+1:]).strip()

                    # Convert back to absolute if theme-relative
                    start_str, end_str = "", ""
                    if is_theme_srt:
                        abs_s = cue_start_sec + theme_start_sec
                        abs_e = cue_end_sec + theme_start_sec
                        
                        def format_offset_ts(s):
                            is_neg = s < 0
                            a = abs(s)
                            h, m = int(a // 3600), int((a % 3600) // 60)
                            sec = int(a % 60)
                            ms = int(round((a % 1) * 1000))
                            if ms == 1000: ms = 0; sec += 1
                            if sec == 60: sec = 0; m += 1
                            if m == 60: m = 0; h += 1
                            res = f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"
                            return f"-{res}" if is_neg else res
                        
                        start_str = format_offset_ts(abs_s)
                        end_str = format_offset_ts(abs_e)
                    else:
                        start_str = f"{time_match.group(1)}.{time_match.group(2)}"
                        end_str = f"{time_match.group(3)}.{time_match.group(4)}"

                    filtered_cues.append({
                        'sequence': seq_str, # Preserve as string (could be "1.2")
                        'start': start_str,
                        'end': end_str,
                        'text': text
                    })
                break # Processed this block

    # Load persistent edits if they exist
    edits = {}
    edits_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_edits.json'
    if edits_file.exists():
        try:
            with open(edits_file, 'r', encoding='utf-8') as f:
                raw_edits = json.load(f)
            # Normalize keys to prevent comma/period duplicates
            for k, v in raw_edits.items():
                try:
                    ks, ke = k.split('_')
                    # Use a canonical period-based format for the internal 'edits' dict
                    norm_key = f"{ks.replace(',', '.')}_{ke.replace(',', '.')}"
                    edits[norm_key] = v
                except: edits[k] = v
        except: pass

    # ENFORCE MAX LINES: Split segments for the UI
    try:
        from subtitle_renderer import UniversalSubtitleRenderer
        # Get master word timestamps for splitting absolute cues
        master_wt_file = next(folder.glob('[!theme]*_word_timestamps.json'), None)
        master_words = []
        if master_wt_file:
            with open(master_wt_file, 'r', encoding='utf-8') as f:
                master_words = json.load(f).get('words', [])
        
        # Instantiate renderer just for splitting logic
        # Convert ConfigParser sections to a flat dict for the renderer
        render_settings = {}
        for section in settings.sections():
            for k, v in settings.items(section):
                render_settings[k] = v
                
        try:
            adjust_settings = creator.get_theme_adjust_settings(folder, int(theme_number))
            render_settings.update(adjust_settings)
        except: pass
        
        renderer = UniversalSubtitleRenderer(str(folder), master_words, render_settings)
        
        # Helper to format timestamps consistently for matching
        def format_ts_key(s):
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            ms = int(round((s % 1) * 1000))
            if ms == 1000:
                sec += 1
                ms = 0
                if sec == 60:
                    m += 1
                    sec = 0
            return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

        def format_ts_key_alt(s):
            return format_ts_key(s).replace('.', ',')

        # Keep track of which edits we have used
        used_edit_keys = set()

        # Sort and split FIRST so edits can target the split fragments correctly
        filtered_cues.sort(key=lambda x: creator.parse_timestamp_to_seconds(x['start']))
        filtered_cues = renderer.split_segments_by_max_lines(filtered_cues, master_words)

        # Step 1: Process existing (split) cues and apply edits using robust float matching
        deduped_cues = []
        for cue in filtered_cues:
            cue_start_val = creator.parse_timestamp_to_seconds(cue['start'])
            cue_end_val = creator.parse_timestamp_to_seconds(cue['end'])
            
            is_dup = False
            for edit_key, edited_text in edits.items():
                try:
                    k_start_str, k_end_str = edit_key.split('_')
                    k_start = creator.parse_timestamp_to_seconds(k_start_str)
                    k_end = creator.parse_timestamp_to_seconds(k_end_str)
                    
                    # Robust float matching with 50ms tolerance
                    is_match = abs(cue_start_val - k_start) < 0.05
                    
                    # SMART MATCH: If cue was clamped to theme start, it might match an edit starting earlier
                    if not is_match and abs(cue_start_val - theme_start_sec) < 0.01:
                        if k_start < theme_start_sec:
                            is_match = True # Match found for clamped start
                    
                    if is_match:
                        # Prevent duplicates if master has overlapping segments
                        if edit_key in used_edit_keys:
                            is_dup = True
                            break
                            
                        # Match found: assume this edit REPLACES the existing cue text and timing
                        cue['text'] = edited_text
                        cue['start'] = k_start_str
                        cue['end'] = k_end_str
                        used_edit_keys.add(edit_key)
                        break
                except: continue
            
            if not is_dup:
                deduped_cues.append(cue)
        
        filtered_cues = deduped_cues
        
        # Step 2: Inject "orphaned" edits (gaps filled by user) as new virtual cues
        if edits:
            for key, text in edits.items():
                if key not in used_edit_keys:
                    try:
                        # Parse absolute timestamps from key
                        start_str, end_str = key.split('_')
                        k_start = creator.parse_timestamp_to_seconds(start_str)
                        k_end = creator.parse_timestamp_to_seconds(end_str)
                        
                        # Only inject if it overlaps with the buffered theme range
                        buffer_sec = 120.0
                        if k_end > (theme_start_sec - buffer_sec) and k_start < (theme_end_sec + buffer_sec):
                            filtered_cues.append({
                                'sequence': len(filtered_cues) + 1,
                                'start': start_str,
                                'end': end_str,
                                'text': text.strip(),
                                'is_virtual': True
                            })
                            used_edit_keys.add(key)
                    except: continue

    except Exception as e:
        app_logger.warning(f"Failed to split segments in get_theme_subtitles: {e}")

    return jsonify({
        'cues': filtered_cues,
        'is_adjusted': has_adjusted,
        'edits': edits
    })


@app.route('/api/all-subtitles/<folder_number>', methods=['GET'])
def get_all_subtitles(folder_number: str):
    """Get ALL subtitles from the original SRT file with sequence numbers (no theme filtering)."""

    theme_number = request.args.get('theme')

    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None

    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return "Folder not found", 404

    # Find the SRT file (could have any name, not just folder_number.srt)
    srt_files = [f for f in folder.glob('*.srt') if 'theme_' not in f.name and 'adjust' not in f.name]
    if not srt_files:
        srt_files = list(folder.glob('*.srt'))
        
    if not srt_files:
        return jsonify({'error': 'SRT file not found'}), 404

    srt_file = srt_files[0]  # Use the first (and likely only) SRT file
    print(f"[DEBUG] Using SRT file: {srt_file.name}")

    # If theme is provided, ensure theme-specific files exist
    if theme_number:
        try:
            # Get theme timing
            themes_file = folder / 'themes.md'
            theme_start_sec = None
            theme_end_sec = None
            
            # Check for adjust file first
            adjust_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'
            if adjust_file.exists():
                with open(adjust_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    time_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)', content)
                    if time_match:
                        theme_start_sec = creator.parse_timestamp_to_seconds(time_match.group(1))
                        theme_end_sec = creator.parse_timestamp_to_seconds(time_match.group(2))
            
            if theme_start_sec is None:
                themes = creator.parse_themes_file(themes_file)
                for t in themes:
                    if t['number'] == int(theme_number):
                        theme_start_sec = creator.parse_timestamp_to_seconds(t['start'])
                        theme_end_sec = creator.parse_timestamp_to_seconds(t['end'])
                        break
            
            if theme_start_sec is not None and theme_end_sec is not None:
                shorts_dir = folder / 'shorts'
                shorts_dir.mkdir(exist_ok=True)
                
                # Create theme SRT only if it doesn't exist OR if boundaries changed
                theme_srt_path = shorts_dir / f"theme_{int(theme_number):03d}.srt"
                theme_json_path = shorts_dir / f"theme_{int(theme_number):03d}.json"
                
                # Check if we need a refresh (e.g., boundary change)
                needs_refresh = not theme_srt_path.exists() or not theme_json_path.exists()
                if not needs_refresh and theme_json_path.exists():
                    try:
                        with open(theme_json_path, 'r') as f:
                            old_data = json.load(f)
                            # If boundaries differ by more than 0.1s, refresh
                            if abs(old_data.get('start_time', 0) - theme_start_sec) > 0.1 or \
                               abs(old_data.get('end_time', 0) - theme_end_sec) > 0.1:
                                needs_refresh = True
                                app_logger.info(f"Boundary shift detected for theme {theme_number}, refreshing metadata.")
                    except: needs_refresh = True

                if needs_refresh:
                    # Load edits to inject during refresh
                    edits = {}
                    edits_file = shorts_dir / f"theme_{int(theme_number):03d}_edits.json"
                    if edits_file.exists():
                        try:
                            with open(edits_file, 'r', encoding='utf-8') as f:
                                edits = json.load(f)
                        except: pass
                        
                    creator.create_trimmed_srt(srt_file, theme_start_sec, theme_end_sec, theme_srt_path, edits=edits)
                    app_logger.info(f"Refreshed {theme_srt_path.name} with {len(edits)} injected edits")
                
                # Create/Refresh theme JSON
                if needs_refresh:
                    # Find word timestamps
                    word_timestamps_file = None
                    for file in folder.glob('*_word_timestamps.json'):
                        if not file.name.startswith('theme_'): # Use master for source
                            word_timestamps_file = file
                            break
                        
                    if word_timestamps_file:
                        with open(word_timestamps_file, 'r', encoding='utf-8') as f:
                            wt_data = json.load(f)
                            all_words = wt_data.get('words', [])
                        
                        theme_words = []
                        for w in all_words:
                            # Precise overlap check with a tiny 100ms buffer
                            if w['end'] > theme_start_sec - 0.1 and w['start'] < theme_end_sec + 0.1:
                                rw = w.copy()
                                # Timestamps are relative to the start of the theme clip
                                # CLAMP TO RANGE for clean output
                                rw['start'] = max(0, rw['start'] - theme_start_sec)
                                rw['end'] = min(theme_end_sec - theme_start_sec, rw['end'] - theme_start_sec)
                                if rw['end'] > rw['start']:
                                    theme_words.append(rw)
                            
                        with open(theme_json_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                'theme': theme_number,
                                'start_time': theme_start_sec,
                                'end_time': theme_end_sec,
                                'words': theme_words
                            }, f, indent=2)
                    
                    app_logger.info(f"Auto-generated metadata for theme {theme_number} in get_all_subtitles")
        except Exception as e:
            app_logger.warning(f"Failed to auto-generate theme metadata in get_all_subtitles: {e}")

    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    # Parse SRT into JSON format (no time filtering)
    all_cues = []
    
    # Robust splitting by empty lines
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    
    for block in blocks:
        block_lines = block.strip().split('\n')
        if not block_lines: continue
        
        # Look for the timing line
        for idx, line in enumerate(block_lines):
            time_match = re.search(r'(-?\d{2}:\d{2}:\d{2})[.,](\d{3})\s*-->\s*(-?\d{2}:\d{2}:\d{2})[.,](\d{3})', line)
            if time_match:
                # Sequence is either the line before or a fallback
                seq_str = block_lines[idx-1].strip() if idx > 0 else str(len(all_cues) + 1)
                
                # Re-format timestamps consistently
                def fmt(s_str, ms_str):
                    s = creator.parse_timestamp_to_seconds(f"{s_str}.{ms_str}")
                    h, m, sec = int(s//3600), int((s%3600)//60), s%60
                    return f"{h:02d}:{m:02d}:{sec:06.3f}"
                
                start_ts = fmt(time_match.group(1), time_match.group(2))
                end_ts = fmt(time_match.group(3), time_match.group(4))

                # Text is everything after timing line
                text = '\n'.join(block_lines[idx+1:]).strip()

                all_cues.append({
                    'sequence': seq_str,
                    'start': start_ts,
                    'end': end_ts,
                    'text': text
                })
                break # Processed this block

    # Load persistent edits if they exist
    edits = {}
    if theme_number:
        edits_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_edits.json'
        if edits_file.exists():
            try:
                with open(edits_file, 'r', encoding='utf-8') as f:
                    raw_edits = json.load(f)
                # Normalize keys to prevent comma/period duplicates
                for k, v in raw_edits.items():
                    try:
                        ks, ke = k.split('_')
                        norm_key = f"{ks.replace(',', '.')}_{ke.replace(',', '.')}"
                        edits[norm_key] = v
                    except: edits[k] = v
            except: pass

    # ENFORCE MAX LINES: Split segments for the UI
    try:
        from subtitle_renderer import UniversalSubtitleRenderer
        # Get all words for the video
        word_timestamps_file = next(folder.glob('*_word_timestamps.json'), None)
        all_words = []
        if word_timestamps_file:
            with open(word_timestamps_file, 'r', encoding='utf-8') as f:
                all_words = json.load(f).get('words', [])
        
        # Instantiate renderer just for splitting logic
        # Convert ConfigParser sections to a flat dict for the renderer
        render_settings = {}
        for section in settings.sections():
            for k, v in settings.items(section):
                render_settings[k] = v
                
        try:
            if theme_number:
                adjust_settings = creator.get_theme_adjust_settings(folder, int(theme_number))
                render_settings.update(adjust_settings)
        except: pass
        
        renderer = UniversalSubtitleRenderer(str(folder), all_words, render_settings)
        
        # Sort and split FIRST so edits can target the split fragments correctly
        all_cues.sort(key=lambda x: creator.parse_timestamp_to_seconds(x['start']))
        all_cues = renderer.split_segments_by_max_lines(all_cues, all_words)

        # Apply edits first and track used ones using robust float matching
        used_edit_keys = set()
        deduped_cues = []
        for cue in all_cues:
            cue_start_val = creator.parse_timestamp_to_seconds(cue['start'])
            cue_end_val = creator.parse_timestamp_to_seconds(cue['end'])
            
            is_dup = False
            for edit_key, edited_text in edits.items():
                try:
                    k_start_str, k_end_str = edit_key.split('_')
                    k_start = creator.parse_timestamp_to_seconds(k_start_str)
                    k_end = creator.parse_timestamp_to_seconds(k_end_str)
                    
                    # Robust matching by start time
                    if abs(cue_start_val - k_start) < 0.05:
                        if edit_key in used_edit_keys:
                            is_dup = True
                            break
                            
                        # Match found: use the edit's text and NEW timing
                        cue['text'] = edited_text
                        cue['start'] = k_start_str
                        cue['end'] = k_end_str
                        used_edit_keys.add(edit_key)
                        break
                except: continue
            
            if not is_dup:
                deduped_cues.append(cue)
        
        all_cues = deduped_cues
        
        # Inject orphaned edits as virtual blocks
        if edits:
            for key, text in edits.items():
                if key not in used_edit_keys:
                    try:
                        start_str, end_str = key.split('_')
                        all_cues.append({
                            'sequence': len(all_cues) + 1,
                            'start': start_str,
                            'end': end_str,
                            'text': text.strip(),
                            'is_virtual': True
                        })
                        used_edit_keys.add(key)
                    except: continue

    except Exception as e:
        app_logger.warning(f"Failed to split segments in get_all_subtitles: {e}")

    return jsonify({
        'cues': all_cues,
        'total': len(all_cues),
        'edits': edits
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


@app.route('/api/ensure-adjust-file/<folder_number>/<theme_number>', methods=['POST'])
def ensure_adjust_file(folder_number: str, theme_number: str):
    """Ensure the theme adjust.md file exists, creating it if necessary."""

    base_dir = Path(settings.get('video', 'output_dir'))
    folder = None
    for f in base_dir.iterdir():
        if f.is_dir() and f.name.startswith(f"{folder_number}_"):
            folder = f
            break

    if not folder:
        return jsonify({'error': 'Folder not found'}), 404

    shorts_dir = folder / 'shorts'
    shorts_dir.mkdir(exist_ok=True)

    adjust_file = shorts_dir / f'theme_{int(theme_number):03d}_adjust.md'

    # If adjust file doesn't exist or is very small/incomplete, create it with default values
    is_incomplete = False
    if adjust_file.exists():
        with open(adjust_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if len(content) < 100 or "**Folder:**" not in content or "**Time Range:**" not in content:
                is_incomplete = True

    if not adjust_file.exists() or is_incomplete:
        # Get theme info from themes.md
        themes_file = folder / 'themes.md'
        title = f"Theme {theme_number}"
        time_range = ""
        is_custom_theme = False

        if themes_file.exists():
            with open(themes_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Find the theme block first
                theme_block_pattern = rf'###\s*Theme\s+{re.escape(theme_number)}:.*?(?=### Theme|\*\*Why this works:\*\*|$)'
                block_match = re.search(theme_block_pattern, content, re.DOTALL)
                if block_match:
                    block = block_match.group(0)
                    
                    # Extract title
                    title_match = re.search(rf'###\s*Theme\s+{re.escape(theme_number)}:\s*(.+?)(?:\n|$)', block)
                    if title_match:
                        title = title_match.group(1).strip()

                    # Extract time range
                    time_match = re.search(r'\*\*Time Range:\*\*\s*(.+?)(?:\n|$)', block)
                    if time_match:
                        time_range = time_match.group(1).strip()
                    
                    # Extract custom type
                    if '**Type:** custom' in block:
                        is_custom_theme = True
                    else:
                        is_custom_theme = False

        # Write default adjust file with all default styling values
        default_settings = {
            'is_custom': is_custom_theme,
            'subtitle_position': 'bottom',
            'subtitle_h_align': 'center',
            'subtitle_v_align': 'bottom',
            'subtitle_bold': False,
            'fontSize': 80,
            'primaryColor': '#ffffff',
            'bgColor': '#000000',
            'bgOpacity': 0.5,
            'fontName': 'Avenir Black',
            'title_font_size': 67,
            'title_bg_type': 'gradient',
            'title_text_color': '#00ff9d',
            'title_font_weight': 800,
            'title_outline_width': 0,
            'title_outline_color': '#000000',
            'title_all_caps': True,
            'show_title': False,
            'title_top': 150
        }

        write_theme_adjust_settings(
            adjust_file,
            theme_number,
            title,
            time_range,
            folder.name,
            default_settings
        )

    return jsonify({'success': True, 'adjust_file': str(adjust_file)})


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

    formatting = {}
    if formatting_file.exists():
        with open(formatting_file, 'r', encoding='utf-8') as f:
            formatting = json.load(f)

    # Also load theme adjust settings which contain subtitle positioning and styling
    adjust_settings = get_theme_adjust_settings(folder, int(theme_number))

    # Merge adjust settings into formatting (adjust settings take precedence)
    result = {**formatting, **adjust_settings}

    return jsonify({'formatting': result})


@app.route('/api/save-cue-text', methods=['POST'])
def save_cue_text():
    """Save individual subtitle edit to SRT file and persistent JSON."""
    app_logger.debug("[DEBUG] save_cue_text endpoint called")
    try:
        try:
            data = request.json
        except Exception as e:
            app_logger.error(f"[ERROR] Failed to parse JSON: {e}")
            return jsonify({'error': 'Invalid JSON'}), 400
        
        folder_number = data.get('folder')
        theme_number = data.get('theme')
        theme_start = data.get('theme_start')
        cue_start = data.get('cue_start')
        cue_end = data.get('cue_end')
        text = data.get('text')

        if not folder_number or not theme_number or cue_start is None or cue_end is None or text is None:
            return jsonify({'error': 'Missing required fields'}), 400

        base_dir = Path(settings.get('video', 'output_dir'))
        folder = None
        for f in base_dir.iterdir():
            if f.is_dir() and f.name.startswith(f"{folder_number}_"):
                folder = f
                break

        if not folder:
            return jsonify({'error': 'Folder not found'}), 404

        # 1. ALWAYS save to persistent edits JSON FIRST using ABSOLUTE keys
        # This is the single source of truth for all manual edits.
        edits_data = {}
        try:
            shorts_dir = folder / 'shorts'
            shorts_dir.mkdir(exist_ok=True)
            edits_file = shorts_dir / f'theme_{int(theme_number):03d}_edits.json'
            if edits_file.exists():
                with open(edits_file, 'r', encoding='utf-8') as f:
                    edits_data = json.load(f)
            
            # Key MUST be absolute HH:MM:SS.mmm
            edit_key = f"{cue_start.replace(',', '.')}_{cue_end.replace(',', '.')}"
            
            # Robust duplicate prevention: Remove any existing keys that match this time (float comparison)
            target_start = creator.parse_timestamp_to_seconds(cue_start)
            target_end = creator.parse_timestamp_to_seconds(cue_end)
            keys_to_remove = [k for k in edits_data.keys() if k != edit_key] # Don't remove the one we're about to set
            
            for k in keys_to_remove:
                try:
                    ks, ke = k.split('_')
                    if abs(creator.parse_timestamp_to_seconds(ks) - target_start) < 0.05 and \
                       abs(creator.parse_timestamp_to_seconds(ke) - target_end) < 0.05:
                        del edits_data[k]
                        print(f"[DEBUG] Removed duplicate edit key: {k}")
                except: pass

            edits_data[edit_key] = text.strip()
            with open(edits_file, 'w', encoding='utf-8') as f:
                json.dump(edits_data, f, indent=2)
            print(f"[DEBUG] Saved persistent UI edit to {edits_file} with key {edit_key}")
        except Exception as ee:
            print(f"[ERROR] JSON save failed: {ee}")

        # 2. Get theme metadata for SRT matching
        theme_start_sec = 0
        if theme_start is not None and str(theme_start).strip() != '':
            try: theme_start_sec = float(theme_start)
            except: pass

        adjust_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'
        theme_end_sec = theme_start_sec + 60
        if adjust_file.exists():
            with open(adjust_file, 'r', encoding='utf-8') as f:
                c = f.read()
                tm = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)', c)
                if tm:
                    if theme_start_sec == 0:
                        theme_start_sec = creator.parse_timestamp_to_seconds(tm.group(1))
                    theme_end_sec = creator.parse_timestamp_to_seconds(tm.group(2))

        # 3. Update theme SRT (physically)
        theme_srt = folder / 'shorts' / f'theme_{int(theme_number):03d}.srt'
        
        # ALWAYS regenerate theme SRT for simplicity and robustness with "Virtual Blocks"
        # This ensures the injection logic in shorts_creator.py is always used.
        srt_files = [f for f in folder.glob('*.srt') if 'theme_' not in f.name and 'adjust' not in f.name]
        main_srt = srt_files[0] if srt_files else None
        
        if main_srt:
            creator.create_trimmed_srt(main_srt, theme_start_sec, theme_end_sec, theme_srt, edits=edits_data)
            print(f"[DEBUG] Theme SRT {theme_srt.name} regenerated with latest edits.")

        return jsonify({'success': True, 'message': 'Saved successfully'})
    except Exception as e:
        import traceback
        app_logger.error(f"[ERROR] save_cue_text: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-cue-timing', methods=['POST'])
def save_cue_timing():
    """Update start/end timing for an individual cue."""
    try:
        data = request.json
        folder_number = data.get('folder')
        theme_number = data.get('theme')
        theme_start = data.get('theme_start')
        
        sequence = data.get('sequence') # To identify which block to change
        old_start = data.get('old_start')
        old_end = data.get('old_end')
        new_start = data.get('new_start')
        new_end = data.get('new_end')

        if not all([folder_number, theme_number, sequence, old_start, old_end, new_start, new_end]):
            return jsonify({'error': 'Missing required fields'}), 400

        base_dir = Path(settings.get('video', 'output_dir'))
        folder = next((f for f in base_dir.iterdir() if f.is_dir() and f.name.startswith(f"{folder_number}_")), None)
        if not folder: return jsonify({'error': 'Folder not found'}), 404

        # 1. Update ONLY the theme SRT file (Master is read-only)
        theme_srt = folder / 'shorts' / f'theme_{int(theme_number):03d}.srt'
        
        theme_start_sec = 0
        try: theme_start_sec = float(theme_start or 0)
        except: pass

        if theme_srt.exists():
            target_file = theme_srt
            is_trimmed = True
            
            with open(target_file, 'r', encoding='utf-8') as f: content = f.read()
            blocks = content.split('\n\n')
            new_blocks = []
            
            target_old_start = parse_srt_time(old_start.replace(',', '.'))
            if is_trimmed: target_old_start -= theme_start_sec

            for block in blocks:
                if not block.strip(): continue
                lines = block.strip().split('\n')
                if len(lines) < 2: 
                    new_blocks.append(block); continue
                
                # Check sequence number or timestamp to match
                ts_match = re.match(r'(\d+:\d+:[\d,]+)\s*-->\s*(\d+:\d+:[\d,]+)', lines[1])
                if not ts_match:
                    new_blocks.append(block); continue
                
                block_start = parse_srt_time(ts_match.group(1).replace(',', '.'))
                # Match by sequence OR timestamp (more robust)
                if str(lines[0]) == str(sequence) or abs(block_start - target_old_start) < 0.1:
                    # Update this block
                    ns_sec = parse_srt_time(new_start.replace(',', '.'))
                    ne_sec = parse_srt_time(new_end.replace(',', '.'))
                    if is_trimmed:
                        ns_sec -= theme_start_sec
                        ne_sec -= theme_start_sec
                    
                    ns_str = format_srt_time(ns_sec)
                    ne_str = format_srt_time(ne_sec)
                    lines[1] = f"{ns_str} --> {ne_str}"
                    new_blocks.append('\n'.join(lines))
                else:
                    new_blocks.append(block)
            
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(new_blocks) + '\n\n')

        # 2. Update persistent edits JSON (migrate key)
        edits_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_edits.json'
        if edits_file.exists():
            with open(edits_file, 'r', encoding='utf-8') as f: edits_data = json.load(f)
            old_key = f"{old_start}_{old_end}"
            new_key = f"{new_start}_{new_end}"
            if old_key in edits_data:
                edits_data[new_key] = edits_data.pop(old_key)
                with open(edits_file, 'w', encoding='utf-8') as f: json.dump(edits_data, f, indent=2)

        return jsonify({'success': True})
    except Exception as e:
        app_logger.error(f"save_cue_timing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-subtitle-restructure', methods=['POST'])
def save_subtitle_restructure():
    """Overwrite the entire theme SRT with a new set of cues (Split/Join)."""
    try:
        data = request.json
        folder_number = data.get('folder')
        theme_number = data.get('theme')
        theme_start = float(data.get('theme_start', 0))
        theme_end = float(data.get('theme_end', 0))
        cues = data.get('cues', [])

        if not folder_number or not theme_number or not cues:
            return jsonify({'error': 'Missing required fields'}), 400

        base_dir = Path(settings.get('video', 'output_dir'))
        folder = next((f for f in base_dir.iterdir() if f.is_dir() and f.name.startswith(f"{folder_number}_")), None)
        if not folder: return jsonify({'error': 'Folder not found'}), 404

        # Update Theme SRT (this one can be overwritten as it's theme-specific)
        theme_srt = folder / 'shorts' / f'theme_{int(theme_number):03d}.srt'
        with open(theme_srt, 'w', encoding='utf-8') as f:
            for i, cue in enumerate(cues, 1):
                try:
                    abs_start = parse_srt_time(cue['start'].replace(',', '.'))
                    abs_end = parse_srt_time(cue['end'].replace(',', '.'))
                    
                    start_rel = max(0, abs_start - theme_start)
                    end_rel = max(0, abs_end - theme_start)
                    
                    f.write(f"{i}\n")
                    f.write(f"{format_srt_time(start_rel)} --> {format_srt_time(end_rel)}\n")
                    f.write(f"{cue['text']}\n\n")
                except: continue

        # Preserve edits JSON but mark the SRT update
        app_logger.info(f"Theme SRT restructured for theme {theme_number}, preserving edits JSON.")

        return jsonify({'success': True})
    except Exception as e:
        app_logger.error(f"save_subtitle_restructure error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-global-position', methods=['POST'])
def save_global_position():
    """Save global subtitle position for a theme to the adjust.md file."""
    try:
        data = request.json
        folder_number = data.get('folder')
        theme_number = data.get('theme')
        position = data.get('position', 'bottom')
        # Custom position data (optional)
        custom_left = data.get('left')
        custom_top = data.get('top')
        h_align = data.get('h_align', 'center')
        v_align = data.get('v_align', 'middle')
        
        # Additional styling fields
        font_size = data.get('fontSize')
        subtitle_bold = data.get('subtitleBold')
        primary_color = data.get('primaryColor')
        bg_color = data.get('bgColor')
        bg_opacity = data.get('bgOpacity')
        font_name = data.get('fontName')
        
        # Title styling fields
        title_font_size = data.get('titleFontSize')
        title_bg_type = data.get('titleBgType')
        title_text_color = data.get('titleTextColor')
        title_font_weight = data.get('titleFontWeight')
        title_outline_width = data.get('titleOutlineWidth')
        title_outline_color = data.get('titleOutlineColor')
        title_all_caps = data.get('titleAllCaps')
        title_top = data.get('titleTop')
        show_title = data.get('showTitle')

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
        
        # Preserve existing styling if not provided
        existing_settings = get_theme_adjust_settings(folder, theme_number)
        
        # Update settings with new values
        if position: existing_settings['subtitle_position'] = position
        if custom_left is not None: existing_settings['subtitle_left'] = custom_left
        if custom_top is not None: existing_settings['subtitle_top'] = custom_top
        if h_align: existing_settings['subtitle_h_align'] = h_align
        if v_align: existing_settings['subtitle_v_align'] = v_align
        
        if font_size is not None: existing_settings['fontSize'] = font_size
        if subtitle_bold is not None: existing_settings['subtitle_bold'] = subtitle_bold
        if primary_color is not None: existing_settings['primaryColor'] = primary_color
        if bg_color is not None: existing_settings['bgColor'] = bg_color
        if bg_opacity is not None: existing_settings['bgOpacity'] = bg_opacity
        if font_name is not None: existing_settings['fontName'] = font_name
        
        # Update title styling fields
        if title_font_size is not None: existing_settings['title_font_size'] = title_font_size
        if title_bg_type is not None: existing_settings['title_bg_type'] = title_bg_type
        if title_text_color is not None: existing_settings['title_text_color'] = title_text_color
        if title_font_weight is not None: existing_settings['title_font_weight'] = title_font_weight
        if title_outline_width is not None: existing_settings['title_outline_width'] = title_outline_width
        if title_outline_color is not None: existing_settings['title_outline_color'] = title_outline_color
        if title_all_caps is not None: existing_settings['title_all_caps'] = title_all_caps
        if title_top is not None: existing_settings['title_top'] = title_top
        if show_title is not None: existing_settings['show_title'] = show_title

        # Use existing title/time range if not in existing_settings
        title = existing_settings.get('title')
        time_range = existing_settings.get('time_range')
        
        if not title or not time_range:
            # No existing file or missing fields, fetch theme data from themes.md
            themes_file = folder / 'themes.md'
            if themes_file.exists():
                themes = creator.parse_themes_file(themes_file)
                for theme in themes:
                    if theme['number'] == int(theme_number):
                        if not title: title = theme.get('title', 'Theme Title')
                        if not time_range:
                            # Calculate duration
                            start_secs = creator.parse_timestamp_to_seconds(theme['start'])
                            end_secs = creator.parse_timestamp_to_seconds(theme['end'])
                            duration_secs = end_secs - start_secs
                            minutes = int(duration_secs // 60)
                            seconds = int(duration_secs % 60)
                            duration_str = f"{minutes}m {seconds}s"
                            time_range = f"{theme['start']} - {theme['end']} ({duration_str})"
                        break
            
            # Set defaults if still not found
            if not title: title = "Theme Title"
            if not time_range: time_range = "--:--:-- - --:--:--"

        # Rebuild file with updated position and styling
        write_theme_adjust_settings(adjust_file, theme_number, title, time_range, folder.name, existing_settings)

        print(f"[DEBUG] Saved global position: folder={folder_number}, theme={theme_number}, position={position}, custom={custom_left}")

        return jsonify({
            'success': True,
            'message': f'Saved global position for theme {theme_number}',
            'position': position
        })
    except Exception as e:
        import traceback
        app_logger.error(f"Error in save_global_position: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-global-position', methods=['GET'])
def get_global_position():
    """Get global subtitle position for a theme from the adjust.md file."""

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

        # Always try to read styling fields if they exist
        font_size_match = re.search(r'\*\*subtitle_font_size:\*\*\s*(\d+)', content)
        primary_color_match = re.search(r'\*\*subtitle_primary_color:\*\*\s*(#[0-9a-fA-F]+)', content)
        bg_color_match = re.search(r'\*\*subtitle_bg_color:\*\*\s*(#[0-9a-fA-F]+)', content)
        bg_opacity_match = re.search(r'\*\*subtitle_bg_opacity:\*\*\s*([0-9.]+)', content)
        font_name_match = re.search(r'\*\*subtitle_font_name:\*\*\s*(.+)', content)
        subtitle_bold_match = re.search(r'\*\*subtitle_bold:\*\*\s*(true|false)', content)
        
        # Title styling fields
        tfs_match = re.search(r'\*\*title_font_size:\*\*\s*(\d+)', content)
        tbt_match = re.search(r'\*\*title_bg_type:\*\*\s*(\w+)', content)
        ttc_match = re.search(r'\*\*title_text_color:\*\*\s*(#[0-9a-fA-F]+)', content)
        tfw_match = re.search(r'\*\*title_font_weight:\*\*\s*(\d+)', content)
        tow_match = re.search(r'\*\*title_outline_width:\*\*\s*([0-9.]+)', content)
        toc_match = re.search(r'\*\*title_outline_color:\*\*\s*(#[0-9a-fA-F]+)', content)
        tac_match = re.search(r'\*\*title_all_caps:\*\*\s*(true|false)', content)
        st_match = re.search(r'\*\*show_title:\*\*\s*(true|false)', content)

        styling_data = {
            'font_size': int(font_size_match.group(1)) if font_size_match else None,
            'subtitle_bold': subtitle_bold_match.group(1) == 'true' if subtitle_bold_match else None,
            'primary_color': primary_color_match.group(1) if primary_color_match else None,
            'bg_color': bg_color_match.group(1) if bg_color_match else None,
            'bg_opacity': float(bg_opacity_match.group(1)) if bg_opacity_match else None,
            'font_name': font_name_match.group(1).strip() if font_name_match else None,
            
            # Title styling
            'title_font_size': int(tfs_match.group(1)) if tfs_match else None,
            'title_bg_type': tbt_match.group(1) if tbt_match else None,
            'title_text_color': ttc_match.group(1) if ttc_match else None,
            'title_font_weight': int(tfw_match.group(1)) if tfw_match else None,
            'title_outline_width': float(tow_match.group(1)) if tow_match else None,
            'title_outline_color': toc_match.group(1) if toc_match else None,
            'title_all_caps': tac_match.group(1) == 'true' if tac_match else None,
            'show_title': st_match.group(1) == 'true' if st_match else None,
            'available_fonts': get_available_fonts()
        }

        # Check if it's a custom position with coordinates
        position_match = re.search(r'\*\*subtitle_position:\*\*\s*custom', content)
        if position_match:
            # Read custom coordinates
            left_match = re.search(r'\*\*subtitle_left:\*\*\s*(\d+)', content)
            top_match = re.search(r'\*\*subtitle_top:\*\*\s*(\d+)', content)
            h_align_match = re.search(r'\*\*subtitle_h_align:\*\*\s*(\w+)', content)
            v_align_match = re.search(r'\*\*subtitle_v_align:\*\*\s*(\w+)', content)
            
            result = {
                'position': 'custom',
                'left': int(left_match.group(1)) if left_match else None,
                'top': int(top_match.group(1)) if top_match else None,
                'h_align': h_align_match.group(1) if h_align_match else 'center',
                'v_align': v_align_match.group(1) if v_align_match else 'middle'
            }
            result.update(styling_data)
            print(f"[DEBUG] Found global custom position: folder={folder_number}, theme={theme_number}, result={result}")
            return jsonify(result)

        # Check for preset position
        position_match = re.search(r'\*\*subtitle_position:\*\*\s*(top|middle|bottom)', content)
        if position_match:
            position = position_match.group(1)
            result = {'position': position}
            result.update(styling_data)
            print(f"[DEBUG] Found global position: folder={folder_number}, theme={theme_number}, position={position}")
            return jsonify(result)

    # No position found, return default
    print(f"[DEBUG] No global position found: folder={folder_number}, theme={theme_number}, using default='bottom'")
    return jsonify({
        'position': 'bottom',
        'available_fonts': get_available_fonts()
    })


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


@app.route('/api/youtube-search', methods=['GET'])
def youtube_search():
    """Proxy YouTube search using yt-dlp to avoid CORS."""
    query = request.args.get('q', '').strip()
    page = request.args.get('page', 1, type=int)

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    try:
        # Use yt-dlp to search YouTube
        import yt_dlp

        # Get 20 results per page (2 pages of 10 for pagination)
        results_per_page = 20

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'playlistend': results_per_page,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_url = f'ytsearch{results_per_page}:{query}'
            result = ydl.extract_info(search_url, download=False)

            if not result or 'entries' not in result:
                return jsonify({'error': 'No results found'}), 404

            # Format results to match expected structure
            videos = []
            for entry in result['entries']:
                if entry:
                    video_id = entry.get('id', '')

                    # Build thumbnails array with different sizes
                    thumbnails = []
                    for quality in ['maxresdefault', 'sddefault', 'hqdefault', 'mqdefault']:
                        thumbnails.append({
                            'url': f'https://img.youtube.com/vi/{video_id}/{quality}.jpg',
                            'quality': quality
                        })

                    videos.append({
                        'videoId': video_id,
                        'title': entry.get('title', 'Unknown'),
                        'author': entry.get('uploader', 'Unknown'),
                        'lengthSeconds': entry.get('duration', 0),
                        'videoThumbnails': thumbnails,
                        'authorThumbnails': []
                    })

            return jsonify(videos)

    except Exception as e:
        app_logger.error(f"YouTube search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_video():
    """Process a new video (URL or local file)."""
    global task_counter
    ensure_manager()

    data = request.json
    url = data.get('url', '').strip()
    local_file = data.get('local_file', '').strip()
    model = data.get('model', settings.get('whisper', 'model'))
    language = data.get('language', settings.get('whisper', 'language'))
    resolution = data.get('resolution', 'best')

    if not url and not local_file:
        return jsonify({'error': 'Either URL or local file must be provided'}), 400

    with task_lock:
        task_counter += 1
        task_id = f"task_{task_counter}"
        tasks[task_id] = task_manager.dict({
            'type': 'process',
            'status': 'pending',
            'url': url if url else local_file,
            'log': '',
            'cancelled': False
        })

    # Start background task - pass task_id directly (daemon=True to allow Ctrl+C)
    thread = threading.Thread(
        target=run_task_with_callback,
        args=(task_id, _process_video, url, local_file, model, language, resolution),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


def _process_video(url: str, local_file: str, model: str, language: str, resolution: str = 'best', progress_callback=None, cancel_check=None):
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

    creator.generate_subtitles(video_info, model_size=model, language=language, progress_callback=progress_callback)

    if cancel_check and cancel_check():
        raise Exception('Cancelled by user')

    creator.generate_themes(video_info, ai_generator=ai_generator, model_size=model, language=language, progress_callback=progress_callback)

    return {
        'folder': video_info['folder'],
        'folder_number': video_info['folder_number'],
        'title': video_info['title']
    }


@app.route('/api/regenerate-themes', methods=['POST'])
def regenerate_themes():
    """Regenerate themes for an existing video."""
    global task_counter
    ensure_manager()

    data = request.json
    folder_number = data.get('folder_number', '').strip()
    model = data.get('model', settings.get('whisper', 'model'))

    if not folder_number:
        return jsonify({'error': 'folder_number is required'}), 400

    with task_lock:
        task_counter += 1
        task_id = f"task_{task_counter}"
        tasks[task_id] = task_manager.dict({
            'type': 'regenerate',
            'status': 'pending',
            'folder_number': folder_number,
            'log': ''
        })

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


@app.route('/api/test-log')
def test_log():
    app_logger.info("TEST LOG ENDPOINT CALLED")
    with open('server.log', 'a') as f:
        f.write(f"{datetime.now()} - [DIRECT_TEST] Manual write to log\n")
    return jsonify({'status': 'ok'})


@app.route('/api/create-shorts', methods=['POST'])
def create_shorts():
    """Create shorts for selected themes."""
    app_logger.info("CREATE_SHORTS ENDPOINT HIT")
    global task_counter
    ensure_manager()

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
        tasks[task_id] = task_manager.dict({
            'type': 'create_shorts',
            'status': 'pending',
            'folder_number': folder_number,
            'themes': themes
        })

    # Switch to simple thread instead of run_task_with_callback (multiprocessing)
    # to ensure logging works and to avoid child process startup issues
    def task_wrapper():
        try:
            with task_lock:
                tasks[task_id]['status'] = 'processing'
            
            # Simple progress callback for the thread
            def thread_progress_cb(msg):
                with task_lock:
                    current = dict(tasks[task_id])
                    current['log'] = (current.get('log', '') + '\n' + msg).strip()
                    # Extract percentage if present

                    match = re.search(r'Progress:\s*(\d+)%', msg)
                    if match:
                        current['progress'] = int(match.group(1))
                    tasks[task_id] = current

            # Direct call
            result = _create_shorts(folder_number, themes, progress_callback=thread_progress_cb)
            
            with task_lock:
                current = dict(tasks[task_id])
                current['status'] = 'complete'
                current['progress'] = 100
                current['result'] = result
                tasks[task_id] = current
        except Exception as e:
            import traceback
            app_logger.error(f"Task error: {traceback.format_exc()}")
            with task_lock:
                current = dict(tasks[task_id])
                current['status'] = 'error'
                current['error'] = str(e)
                tasks[task_id] = current

    thread = threading.Thread(target=task_wrapper, daemon=True)
    thread.start()

    return jsonify({'task_id': task_id})


def _create_shorts(folder_number: str, themes: List, progress_callback=None, cancel_check=None):
    """Create shorts in background."""
    # Direct file writing as a fail-safe for background process logging
    with open('server.log', 'a') as f_log:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        f_log.write(f"{timestamp} - INFO - [BG_TASK] _create_shorts started for folder {folder_number}\n")
    
    # Re-initialize logging for the child process
    import logging
    from logging.handlers import RotatingFileHandler
    
    # Simple setup to ensure child process logs to the same file
    log_h = RotatingFileHandler('server.log', maxBytes=100*1024, backupCount=1)
    log_h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Get the loggers used in shorts_creator
    l1 = logging.getLogger('server.py')
    l1.setLevel(logging.INFO)
    l1.addHandler(log_h)
    
    l2 = logging.getLogger('subtitle_renderer')
    l2.setLevel(logging.INFO)
    l2.addHandler(log_h)
    
    l1.info(f"[DEBUG] _create_shorts: folder={folder_number}, themes={themes}")
    
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
    ensure_manager()
    with task_lock:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
        return jsonify(dict(tasks[task_id]))


@app.route('/api/task/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id: str):
    """Cancel a background task by terminating subprocesses and cleaning up files."""
    ensure_manager()
    import subprocess
    import signal
    import shutil
    import time

    print(f"[CANCEL] Cancel requested for task: {task_id}")
    url = ''
    with task_lock:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404

        # Allow cancelling tasks in pending, processing or running status
        if tasks[task_id]['status'] in ['pending', 'processing', 'running']:
            print(f"[CANCEL] Task {task_id} found with status {tasks[task_id]['status']}. Setting to cancelled.")
            # Set cancelled flag and get URL
            task_data = dict(tasks[task_id])
            task_data['cancelled'] = True
            task_data['status'] = 'cancelled'
            url = task_data.get('url', '')
            
            # Write back to shared memory
            tasks[task_id] = task_data
            print(f"[CANCEL] Task {task_id} status now: {tasks[task_id]['status']}")
            
            # Forcefully terminate the process if it exists
            if task_id in task_processes:
                try:
                    task_processes[task_id].terminate()
                    print(f"[CANCEL] Terminated process for task: {task_id}")
                except Exception as e:
                    print(f"[CANCEL] Error terminating process: {e}")

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
                    themes_file = folder / 'themes.md'

                    # If has video but no subtitles/themes, it's a partial download
                    if len(video_files) > 0 and len(srt_files) == 0 and not themes_file.exists():
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


@app.route('/api/re-transcribe-settings/<folder_number>', methods=['GET'])
def get_retranscribe_settings(folder_number: str):
    """Get the language and model used for transcribing this video (from theme.md)."""
    try:
        # Find the folder
        base_dir = Path(settings.get('video', 'output_dir'))
        folder = None
        for f in base_dir.iterdir():
            if f.is_dir() and f.name.startswith(f"{folder_number}_"):
                folder = f
                break

        if not folder:
            return jsonify({'error': 'Folder not found'}), 404

        # Read theme.md to get current settings
        themes_file = folder / 'themes.md'
        if not themes_file.exists():
            return jsonify({'error': 'themes.md not found'}), 404

        with open(themes_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse model and language from theme.md

        # Handle markdown formatting with **
        model_match = re.search(r'\*\*Whisper Model:\*\*\s*(\w+)', content)
        language_match = re.search(r'\*\*Language:\*\*\s*(\w+)', content)

        current_model = model_match.group(1) if model_match else settings.get('whisper', 'model')
        current_language = language_match.group(1) if language_match else settings.get('whisper', 'language')

        # Get video file info
        video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
        video_info = None
        if video_files:
            video_file = video_files[0]
            import cv2
            cap = cv2.VideoCapture(str(video_file), cv2.CAP_FFMPEG)
            if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
            
            if not cap.isOpened():
                cap = cv2.VideoCapture(str(video_file))
                if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)

            duration_sec = 0
            fps = 0
            width = 0
            height = 0
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps > 0:
                    duration_sec = frame_count / fps
                cap.release()

            video_size = video_file.stat().st_size

            # Format duration
            minutes = int(duration_sec // 60)
            seconds = int(duration_sec % 60)
            duration_str = f"{minutes}:{seconds:02d}" if minutes > 0 else f"{seconds}s"

            video_info = {
                'filename': video_file.name,
                'duration_seconds': duration_sec,
                'duration_formatted': duration_str,
                'size_bytes': video_size,
                'size_formatted': format_size(video_size),
                'resolution': f"{width}x{height}" if width > 0 and height > 0 else 'unknown',
                'fps': round(fps, 2) if fps > 0 else 0
            }

        return jsonify({
            'current_model': current_model,
            'current_language': current_language,
            'video_info': video_info
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

def _run_retranscribe_task(folder_number, folder, video_file, model, language, base_dir, progress_callback=None, cancel_check=None):
    """Module-level function for re-transcription task."""
    app_logger.info(f"DEBUG: _run_retranscribe_task started for folder {folder_number}")
    sys.__stdout__.write(f"DEBUG: _run_retranscribe_task started for folder {folder_number}\n")
    sys.__stdout__.flush()
    
    # Set initial progress via callback
    if progress_callback:
        progress_callback("Progress: 1% Initializing re-transcription process...")

    try:
        # Phase 1: Transcription (0-80%)
        # Phase 2: Theme Identification (80-90%)
        # Phase 3: Title/Reason Generation (90-100%)
        current_phase = [1] 

        # Create video_info dict
        video_info = {
            'folder': str(folder),
            'folder_number': folder_number,
            'video_path': str(video_file),
            'title': video_file.stem
        }

        # Internal callback that scales percentages for the UI
        def scaled_callback(msg):
            msg_str = str(msg)
            # Use raw stdout to bypass any redirection issues
            app_logger.info(f"[Re-transcribe Progress] {msg_str}")
            sys.__stdout__.write(f"[Re-transcribe Progress] {msg_str}\n")
            sys.__stdout__.flush()
            
            display_percent = None

            match = re.search(r'Progress:\s*(\d+)%', msg_str)
            if match:
                raw_percent = int(match.group(1))
                
                if current_phase[0] == 1:
                    display_percent = int(raw_percent * 0.80)
                elif current_phase[0] == 2:
                    display_percent = 80 + int(raw_percent * 0.10)
                else:
                    display_percent = 90 + int(raw_percent * 0.10)
                
                display_percent = max(1, min(100, display_percent))
                # Rewrite the log message to show the display percentage
                msg_str = re.sub(r'Progress:\s*\d+%', f'Progress: {display_percent}%', msg_str)
            
            # Use the provided progress_callback to update the task state in shared memory
            if progress_callback:
                progress_callback(msg_str)

        # Get or create YouTubeShortsCreator instance
        scaled_callback("DEBUG: Initializing YouTubeShortsCreator...")
        from shorts_creator import YouTubeShortsCreator
        creator = YouTubeShortsCreator(base_dir)

        # Step 1: Transcription (Phase 1)
        scaled_callback("DEBUG: Starting Phase 1: Subtitle Generation")
        srt_path = creator.generate_subtitles(
            video_info,
            model_size=model,
            language=language,
            progress_callback=scaled_callback,
            cancel_check=cancel_check
        )

        # Transition to Phase 2
        current_phase[0] = 2
        
        # Initialize AI generator
        ai_generator = None
        try:
            from ai_theme_generator import AIThemeGenerator
            ai_generator = AIThemeGenerator()
            if not ai_generator.is_available():
                ai_generator = None
        except ImportError:
            pass

        # Step 2 & 3: Theme Generation
        # Detect phase change within generate_themes
        orig_scaled_callback = scaled_callback
        def phase_aware_callback(msg):
            if 'Progress:' in msg and current_phase[0] == 2:
                if 'total themes' in msg or 'Theme 1:' in msg:
                    current_phase[0] = 3
            orig_scaled_callback(msg)

        scaled_callback('Generating new themes...')
        creator.generate_themes(
            video_info,
            ai_generator=ai_generator,
            model_size=model,
            language=language,
            progress_callback=phase_aware_callback,
            cancel_check=cancel_check
        )

        # Mark as complete via callback
        if progress_callback:
            progress_callback("Progress: 100% Transcription and theme generation complete!")

    except Exception as e:
        if "Cancelled by user" in str(e):
            # The run_task wrapper handles 'cancelled' status if it sees the flag,
            # but we can make it explicit here just in case.
            app_logger.info(f"Retranscription for {folder_number} was successfully cancelled.")
            raise e
        
        import traceback
        error_msg = str(e)
        print(f"[Re-transcribe Error] {error_msg}\n{traceback.format_exc()}")
        raise e


@app.route('/api/re-transcribe', methods=['POST'])
def re_transcribe():
    """Re-transcribe an existing video in the library with progress tracking."""
    ensure_manager()
    sys.__stdout__.write("API: /api/re-transcribe called\n")
    sys.__stdout__.flush()
    try:
        data = request.json
        folder_number = data.get('folder')
        model = data.get('model', settings.get('whisper', 'model'))
        language = data.get('language', settings.get('whisper', 'language'))

        if not folder_number:
            return jsonify({'error': 'Folder number is required'}), 400

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
        video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
        if not video_files:
            return jsonify({'error': 'No video file found'}), 404
        video_file = video_files[0]

        # Create a task ID for progress tracking
        task_id = f"retranscribe_{folder_number}_{int(time.time())}"

        # Initialize task tracking
        with task_lock:
            tasks[task_id] = task_manager.dict({
                'type': 'retranscribe',
                'status': 'pending',
                'folder': folder_number,
                'model': model,
                'language': language,
                'log': 'Initializing...',
                'progress': 0,
                'error': None,
                'cancelled': False
            })
            app_logger.debug(f"[DEBUG] Task created: {task_id}, tasks dict keys: {list(tasks.keys())}")

        # Start background task using the unified callback wrapper
        app_logger.debug(f"[DEBUG] Starting task {task_id} with run_task_with_callback")
        thread = threading.Thread(
            target=run_task_with_callback,
            args=(task_id, _run_retranscribe_task, folder_number, folder, video_file, model, language, base_dir),
            daemon=True
        )
        thread.start()
        app_logger.debug(f"[DEBUG] Thread started for task {task_id}, is_alive: {thread.is_alive()}")

        return jsonify({'task_id': task_id})

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/re-transcribe-status/<task_id>')
def retranscribe_status(task_id: str):
    """Get the status of a re-transcribe task."""
    ensure_manager()
    app_logger.debug(f"[DEBUG] Status requested for task {task_id}, tasks keys: {list(tasks.keys())}")
    with task_lock:
        if task_id not in tasks:
            app_logger.debug(f"[DEBUG] Task {task_id} NOT FOUND in tasks dict")
            return jsonify({'error': 'Task not found'}), 404
        app_logger.debug(f"[DEBUG] Task {task_id} found: {tasks[task_id]}")

        task = dict(tasks[task_id])
        response = jsonify({
            'status': task['status'],
            'log': task.get('log', ''),
            'error': task.get('error'),
            'progress': task.get('progress', 0)
        })
        app_logger.debug(f"[DEBUG] Returning response for {task_id}")
        return response

# Add a simple test route
@app.route('/api/test')
def test_route():
    return jsonify({'status': 'ok', 'message': 'Test route works'})


# Track edit processes for cancellation
edit_processes: Dict[str, Dict] = {}
edit_counter = 0


def _extract_audio_levels(video_path: Path, start_time: float, end_time: float) -> List[float]:
    """Extract audio volume levels from a video segment using FFmpeg."""
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_tmp = Path(tmp_dir) / "audio.wav"
        
        # Extract audio segment to WAV
        cmd = [
            'ffmpeg', '-y', '-ss', str(start_time), '-i', str(video_path),
            '-t', str(end_time - start_time), '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', str(audio_tmp)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        if not audio_tmp.exists():
            return []
            
        # Read WAV data
        import numpy as np
        import wave
        
        with wave.open(str(audio_tmp), 'rb') as wav:
            n_frames = wav.getnframes()
            data = wav.readframes(n_frames)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            
        # Calculate RMS levels in small windows (e.g., 100ms)
        sample_rate = 16000
        window_size = int(sample_rate * 0.1) # 100ms
        levels = []
        
        for i in range(0, len(audio), window_size):
            window = audio[i:i+window_size]
            if len(window) == 0: continue
            rms = np.sqrt(np.mean(window**2))
            levels.append(float(rms))
            
        # Normalize levels to 0.0 - 1.0 range
        if not levels: return []
        max_level = max(levels)
        if max_level > 0:
            levels = [l / max_level for l in levels]
            
        return levels

@app.route('/api/audio-library')
def get_audio_library():
    """Returns a list of audio files available in media/audio."""
    audio_dir = Path('media/audio')
    if not audio_dir.exists():
        return jsonify([])
    
    # Get all audio files
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.aac', '*.m4a', '*.ogg']:
        for f in audio_dir.glob(ext):
            audio_files.append({
                'name': f.name,
                'path': f'audio/{f.name}'
            })
            
    # Sort alphabetically
    audio_files.sort(key=lambda x: x['name'].lower())
    return jsonify(audio_files)

@app.route('/api/image-library')
def get_image_library():
    """Returns a list of image files available in media/images or specified subfolder."""
    subfolder = request.args.get('subfolder', 'images')
    image_dir = Path('media') / subfolder
    if not image_dir.exists():
        return jsonify([])

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.eps', '*.svg']:
        for f in image_dir.glob(ext):
            image_files.append({
                'name': f.name,
                'path': f'{subfolder}/{f.name}'
            })

    image_files.sort(key=lambda x: x['name'].lower())
    return jsonify(image_files)
@app.route('/api/outro-library')
def get_outro_library():
    """Returns a list of video and image files available in media/outro."""
    outro_dir = Path('media/outro')
    if not outro_dir.exists():
        outro_dir.mkdir(parents=True, exist_ok=True)
        return jsonify([])

    files = []
    # Video extensions
    video_exts = ['*.mp4', '*.mkv', '*.webm', '*.mov', '*.avi']
    # Image extensions
    image_exts = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.svg']
    
    for ext in video_exts:
        for f in outro_dir.glob(ext):
            duration = 5.0
            try:
                import subprocess
                cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', str(f)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
            except: pass
            
            files.append({
                'name': f.name,
                'path': f'outro/{f.name}',
                'type': 'video',
                'duration': duration
            })

    for ext in image_exts:
        for f in outro_dir.glob(ext):
            files.append({
                'name': f.name,
                'path': f'outro/{f.name}',
                'type': 'image'
            })

    files.sort(key=lambda x: x['name'].lower())
    return jsonify(files)

@app.route('/api/broll-library')
def get_broll_library():
    """Returns a list of video files available in media/video."""
    video_dir = Path('media/video')
    if not video_dir.exists():
        return jsonify([])
    
    video_files = []
    for ext in ['*.mp4', '*.mkv', '*.webm', '*.mov', '*.avi']:
        for f in video_dir.glob(ext):
            # Try to get duration if possible
            duration = 5.0
            try:
                import subprocess
                cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', str(f)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
            except: pass

            video_files.append({
                'name': f.name,
                'path': f'video/{f.name}',
                'duration': duration
            })
            
    video_files.sort(key=lambda x: x['name'].lower())
    return jsonify(video_files)

@app.route('/media/<path:filename>')
def serve_media(filename):
    """Serve media files (audio, images, etc.) from the media directory."""
    return send_from_directory('media', filename)

@app.route('/api/upload-media', methods=['POST'])
def upload_media():
    """Handle media upload (images, audio, b-roll) for video editing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            # Ensure media directory exists
            media_dir = Path('media')
            media_dir.mkdir(exist_ok=True)

            # Handle EPS conversion if necessary
            is_eps = filename.lower().endswith('.eps')
            if is_eps:
                # Change extension to .png for the saved file
                filename = Path(filename).with_suffix('.png').name

            target_path = media_dir / filename
            
            if is_eps:
                # Convert EPS to PNG using direct Ghostscript call (more robust than Pillow for EPS)
                try:
                    # We need to save the uploaded file temporarily first
                    temp_eps = media_dir / f"temp_{filename}.eps"
                    file.save(str(temp_eps))
                    
                    # Run Ghostscript to convert EPS to PNG
                    # -dEPSCrop: crop to EPS bounding box
                    # -sDEVICE=pngalpha: PNG with transparency
                    # -r300: 300 DPI for good quality
                    import subprocess
                    gs_cmd = [
                        "gs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dEPSCrop",
                        "-sDEVICE=pngalpha", "-r300",
                        f"-sOutputFile={target_path}", str(temp_eps)
                    ]
                    app_logger.info(f"Running GS: {' '.join(gs_cmd)}")
                    gs_res = subprocess.run(gs_cmd, capture_output=True, text=True)
                    
                    # Remove temporary EPS
                    if temp_eps.exists():
                        temp_eps.unlink()
                        
                    if gs_res.returncode != 0:
                        raise Exception(f"Ghostscript error: {gs_res.stderr}")
                        
                    app_logger.info(f"Converted EPS logo to PNG via Ghostscript: {filename}")
                except Exception as e:
                    import traceback
                    app_logger.error(f"Failed to convert EPS to PNG via GS: {e}\n{traceback.format_exc()}")
                    return jsonify({'error': f'Failed to process EPS file: {str(e)}'}), 500
            else:
                # Save the file normally
                file.save(str(target_path))

            # Try to get duration if it's a video
            duration = None
            if filename.lower().endswith(('.mp4', '.mkv', '.webm', '.mov', '.avi')):
                try:
                    import subprocess
                    cmd = [
                        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', str(target_path)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        duration = float(result.stdout.strip())
                except Exception as e:
                    app_logger.warning(f"Could not get duration for {filename}: {e}")

            return jsonify({
                'filename': filename,
                'url': f'/media/{filename}',
                'duration': duration
            })
    except Exception as e:
        import traceback
        app_logger.error(f"CRITICAL UPLOAD ERROR: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-edit', methods=['POST'])
def process_video_edit():
    """Process video with effects including face tracking."""
    global edit_counter
    ensure_manager()

    data = request.json
    app_logger.info(f"RAW INCOMING DATA: {data}")
    video_path = data.get('video_path')
    edit_settings = data.get('settings', {})
    quality_preset = data.get('quality', 'standard')

    # CRF Mapping for quality
    crf_map = {
        'fast': 28,
        'standard': 22,
        'high': 18,
        'ultra': 12
    }
    crf = crf_map.get(quality_preset, 22)
    edit_settings['export_crf'] = crf
    
    # Use settings.ini default if available, otherwise use fast presets
    default_preset = settings.get('video', 'export_preset', fallback='ultrafast')
    if quality_preset == 'fast':
        edit_settings['export_preset'] = 'ultrafast'
    else:
        edit_settings['export_preset'] = default_preset

    if not video_path:
        return jsonify({'error': 'video_path is required'}), 400

    # Construct full video path
    base_dir = Path(settings.get('video', 'output_dir'))
    input_video = base_dir / video_path
    
    # CRITICAL: Check if we are starting from a theme clip BEFORE potentially switching to master
    is_editing_theme_clip = 'theme_' in input_video.name

    if not input_video.exists():
        return jsonify({'error': f'Video not found: {video_path}'}), 404
    
    # Try to fetch theme title and subtitle settings if folder/theme provided
    f_num = edit_settings.get('folder_number')
    t_num = edit_settings.get('theme_number')
    
    # ALWAYS put edited_shorts inside the shorts folder
    folder_root = input_video.parent
    if folder_root.name != 'shorts':
        # If we are looking at master root, we want to go into shorts
        folder_root = folder_root / 'shorts'
        
    output_dir = folder_root / 'edited_shorts'
    output_dir.mkdir(parents=True, exist_ok=True)

    if f_num and t_num:
        try:
            # Find the actual folder path
            folder = None
            for f in base_dir.iterdir():
                if f.is_dir() and f.name.startswith(f"{f_num}_"):
                    folder = f
                    break
            
            if folder:
                # We ALWAYS enable reburn for theme edits now to ensure 
                # all overlays (B-roll, Images, Subtitles) are correctly layered.
                reburn_enabled = True
                edit_settings['subtitles']['reburn'] = True
                
                # Get adjustments for this theme to get metadata timing
                adjust_settings = get_theme_adjust_settings(folder, t_num)
                
                theme_meta_start = 0
                theme_meta_end = 0
                time_range = adjust_settings.get('time_range')
                app_logger.info(f"DEBUG: Processing Folder {f_num}, Theme {t_num}")
                app_logger.info(f"DEBUG: Metadata Time Range from adjust.md: {time_range}")
                
                if time_range:
                    # Format: "00:01:40 - 00:02:33 (0m 53s)" or similar
                    t_match = re.search(r'(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)', time_range)
                    if t_match:
                        theme_meta_start = creator.parse_timestamp_to_seconds(t_match.group(1).replace(',', '.'))
                        theme_meta_end = creator.parse_timestamp_to_seconds(t_match.group(2).replace(',', '.'))
                        app_logger.info(f"DEBUG: Parsed Theme Metadata: Start={theme_meta_start}, End={theme_meta_end}")

                # Identify if we are using an already-subtitled theme clip
                # themes are stored in folder/shorts/
                is_theme_subtitled = 'theme_' in input_video.name and 'shorts' in str(input_video.parent)
                is_manual_clean = 'clean_' in input_video.name
                
                if reburn_enabled:
                    # Switch to master if we are reburning, to ensure we use absolute metadata timestamps correctly
                    master_videos = [f for f in folder.glob('*.mp4') if 'theme_' not in f.name and 'edited_' not in f.name and 'clean_' not in f.name]
                    if master_videos:
                        input_video = master_videos[0]
                        app_logger.info(f"Switching to clean master video for re-burn: {input_video}")

                if 'subtitles' not in edit_settings:
                    edit_settings['subtitles'] = {}
                
                # Merge into edit_settings (but let incoming ones override)
                for k, v in adjust_settings.items():
                    if k not in edit_settings['subtitles']:
                        edit_settings['subtitles'][k] = v
                
                # CRITICAL: Also specifically map karaoke settings from highlight_style.json
                highlight_file = folder / 'highlight_style.json'
                if highlight_file.exists():
                    try:
                        with open(highlight_file, 'r', encoding='utf-8') as f:
                            h_style = json.load(f)
                            subs_conf = edit_settings['subtitles']
                            if 'karaoke_mode' in h_style: subs_conf['mode'] = h_style['karaoke_mode']
                            if 'effectType' in h_style: subs_conf['effect_type'] = h_style['effectType']
                            if 'textColor' in h_style: subs_conf['textColor'] = h_style['textColor']
                            if 'glowColor' in h_style: subs_conf['glowColor'] = h_style['glowColor']
                            if 'glowBlur' in h_style: subs_conf['glowBlur'] = h_style['glowBlur']
                            if 'past_color' in h_style: subs_conf['pastColor'] = h_style['past_color']
                    except Exception as e:
                        app_logger.warning(f"Failed to load highlight settings for Edit: {e}")
                
                # Handle trim translation
                user_trim = edit_settings.get('trim', {})
                user_start_str = user_trim.get('start', '00:00:00')
                user_end_str = user_trim.get('end', '')
                user_start_sec = creator.parse_timestamp_to_seconds(user_start_str)
                user_end_sec = creator.parse_timestamp_to_seconds(user_end_str) if user_end_str else 0

                app_logger.info(f"DEBUG: User trim: start={user_start_str} ({user_start_sec}s), end={user_end_str} ({user_end_sec}s)")
                app_logger.info(f"DEBUG: theme_meta: start={theme_meta_start}s, end={theme_meta_end}s, reburn={reburn_enabled}")

                # ALWAYS use theme metadata as the base if available AND we are reburning (using master)
                if theme_meta_start > 0 and reburn_enabled:
                    if 'trim' not in edit_settings: edit_settings['trim'] = {}
                    
                    # If user_start_sec is close to theme_meta_start or very large, 
                    # it is likely a confused absolute time from the UI. 
                    # We reset to metadata start.
                    if user_start_sec == 0 or abs(user_start_sec - theme_meta_start) < 60:
                        new_start = theme_meta_start
                        app_logger.info(f"Using metadata start {new_start} (ignoring UI {user_start_sec})")
                    else:
                        # Significant difference, treat as relative offset
                        new_start = theme_meta_start + user_start_sec
                        app_logger.info(f"Applying relative user offset {user_start_sec} to theme start {theme_meta_start}")

                    # Calculate end time
                    meta_duration = theme_meta_end - theme_meta_start
                    
                    if user_end_sec == 0 or abs(user_end_sec - theme_meta_end) < 60:
                        new_end = theme_meta_end
                    elif abs(user_end_sec - meta_duration) < 10:
                        # If user_end_sec is close to meta duration, it is likely the full clip duration from UI
                        new_end = theme_meta_end
                        app_logger.info(f"User end {user_end_sec} is close to metadata duration {meta_duration}, using metadata end {new_end}")
                    else:
                        new_end = theme_meta_start + user_end_sec
                        app_logger.info(f"Using relative user end offset {user_end_sec} -> absolute {new_end}")
                        
                    edit_settings['trim']['start'] = format_srt_time(new_start).replace(',', '.')
                    edit_settings['trim']['end'] = format_srt_time(new_end).replace(',', '.')
                    app_logger.info(f"Final Enforced Theme Trim: {edit_settings['trim']['start']} - {edit_settings['trim']['end']}")
                else:
                    # If reburn is False, we are editing the theme clip directly.
                    # UI might send absolute master-video times; we must translate them to 0-based relative.
                    if theme_meta_start > 0 and user_start_sec >= theme_meta_start - 60:
                        rel_start = max(0, user_start_sec - theme_meta_start)
                        # Ensure we don't end up with invalid negative or zero durations
                        if user_end_sec > user_start_sec:
                            rel_end = user_end_sec - theme_meta_start
                        else:
                            rel_end = theme_meta_end - theme_meta_start
                        
                        edit_settings['trim']['start'] = format_srt_time(rel_start).replace(',', '.')
                        edit_settings['trim']['end'] = format_srt_time(rel_end).replace(',', '.')
                        app_logger.info(f"Translated absolute UI trim to relative for theme clip: {edit_settings['trim']['start']} - {edit_settings['trim']['end']}")
                    else:
                        app_logger.info(f"Using provided trim as-is (reburn={reburn_enabled}): {user_start_str} - {user_end_str}")
                        if 'trim' not in edit_settings: edit_settings['trim'] = {}
                        edit_settings['trim']['start'] = user_start_str
                        edit_settings['trim']['end'] = user_end_str

                # Also try to get title from themes.md or adjust.md
                theme_title = ""
                adjust_file = folder / 'shorts' / f'theme_{int(t_num):03d}_adjust.md'
                if adjust_file.exists():
                    with open(adjust_file, 'r', encoding='utf-8') as f:
                        adj_content = f.read()
                    t_match = re.search(r'\*\*Title:\*\*\s*(.+)', adj_content)
                    if t_match: theme_title = t_match.group(1).strip()
                
                if not theme_title:
                    themes_file = folder / 'themes.md'
                    if themes_file.exists():
                        with open(themes_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        theme_section_pattern = rf'### Theme {t_num}:(.*?)(?=### Theme|\Z)'
                        section_match = re.search(theme_section_pattern, content, re.DOTALL)
                        if section_match:
                            title_match = re.search(rf'### Theme {t_num}:\s*(.*?)\n', section_match.group(0))
                            if title_match: theme_title = title_match.group(1).strip()
                
                if theme_title:
                    edit_settings['title'] = theme_title
                    edit_settings['subtitles']['title'] = theme_title
                    # Respect the show_title setting from adjust.md (default to False if not set)
                    show_title_from_adjust = adjust_settings.get('show_title', False)
                    edit_settings['subtitles']['show_title'] = show_title_from_adjust
                
                # IMPORTANT: Find the correct SRT file for this specific theme
                # This ensures we use the theme boundaries and any manual edits
                theme_srt = folder / 'shorts' / f'theme_{int(t_num):03d}.srt'
                if not theme_srt.exists():
                    # Fallback to theme title based filename
                    sanitized = creator.sanitize_title(theme_title)
                    pattern = f"theme_{int(t_num):03d}_{sanitized}.srt"
                    matches = list((folder / 'shorts').glob(pattern))
                    if matches:
                        theme_srt = matches[0]
                
                if theme_srt.exists():
                    edit_settings['subtitles']['srt_path'] = str(theme_srt)
                    app_logger.info(f"Using theme-specific SRT for edit: {theme_srt}")
                    
        except Exception as e:
            app_logger.warning(f"Failed to fetch theme metadata for edit: {e}")

    # Generate output path
    # Sanitize filename to avoid issues with special characters like | or pipes
    clean_stem = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in input_video.stem])
    theme_suffix = f"_theme_{int(t_num):03d}" if t_num else ""
    output_filename = f"edited_{clean_stem.strip()}{theme_suffix}_{quality_preset}.mp4"
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
        args=(edit_id, str(input_video), str(output_video), edit_settings, is_editing_theme_clip, quality_preset),
        daemon=True
    )
    thread.start()

    return jsonify({'edit_id': edit_id, 'status': 'started'})


def run_edit_task(edit_id: str, input_video: str, output_video: str, edit_settings: dict, is_editing_theme_clip: bool = False, quality_preset: str = 'standard'):
    """Run video edit in background thread."""
    app_logger.info(f"THREAD ENTERED: run_edit_task for {edit_id}")
    from video_processor import VideoProcessor
    from subtitle_renderer import render_canvas_karaoke_video
    import os
    import time
    import cv2
    import shutil
    import tempfile
    import subprocess
    from pathlib import Path

    # Helper to log directly to status and file
    def log_message(msg):
        if not msg.startswith("PROGRESS:"):
            app_logger.info(f"[EDIT_TASK {edit_id}] {msg}")
        with task_lock:
            if edit_id in edit_processes:
                if msg.startswith("PROGRESS:"):
                    try:
                        parts = msg.split("|")
                        percent_str = parts[0].split(":")[1].replace("%", "")
                        edit_processes[edit_id]['progress'] = int(percent_str)
                        if len(parts) > 1:
                            edit_processes[edit_id]['log'] = f"STAGE:{parts[1]}\n"
                        return 
                    except: pass
                current_log = edit_processes[edit_id].get('log', '')
                edit_processes[edit_id]['log'] = current_log + msg + '\n'

    def to_sec(ts):
        if not ts: return 0.0
        if isinstance(ts, (int, float)): return float(ts)
        try:
            parts = list(map(float, str(ts).replace(',', '.').split(':')))
            return parts[-1] + (parts[-2] * 60 if len(parts) > 1 else 0) + (parts[-3] * 3600 if len(parts) > 2 else 0)
        except: return 0.0

    try:
        with task_lock:
            edit_processes[edit_id]['status'] = 'running'
            edit_processes[edit_id]['progress'] = 0
            edit_processes[edit_id]['log'] = ''

        # Determine folder and theme context
        f_num = edit_settings.get('folder_number')
        base_dir = Path(settings.get('video', 'output_dir'))
        folder_path = None
        for f in base_dir.iterdir():
            if f.is_dir() and f.name.startswith(f"{f_num}_"):
                folder_path = f
                break
        if not folder_path: folder_path = Path(input_video).parent.parent
        t_num = edit_settings.get('theme_number', 1)

        # Timestamps for extraction
        start_str = edit_settings.get('trim', {}).get('start', '0')
        end_str = edit_settings.get('trim', {}).get('end')
        log_message(f"DEBUG: run_edit_task start_str={start_str}, end_str={end_str}")
        actual_start = to_sec(start_str)
        actual_end = to_sec(end_str)
        log_message(f"DEBUG: Calculated actual_start={actual_start}, actual_end={actual_end}")
        
        reburn_enabled = edit_settings.get('subtitles', {}).get('reburn', True)
        final_burn_success = False

        # --- STEP 1: SYNCED SEGMENT EXTRACTION ---
        if reburn_enabled:
            log_message("STEP 1: Extracting clean segment from master for sync...")
            temp_clean = os.path.join(str(GLOBAL_TEMP_DIR), f"clean_{edit_id}.mp4")
            
            # Find the clean master
            master_video = None
            master_videos = [f for f in folder_path.glob('*.mp4') if 'theme_' not in f.name and 'edited_' not in f.name and 'clean_' not in f.name]
            if master_videos: master_video = master_videos[0]
            else: master_video = Path(input_video)
            
            # Stage 1: Fast Stream Copy Buffer.
            # We copy a segment starting 30s before the target. This is instant and safe.
            seek_buffer = 30.0
            fast_ss = max(0, actual_start - seek_buffer)
            buffer_dur = (actual_end - actual_start) + seek_buffer + 10.0
            
            extract_cmd = [
                'ffmpeg', '-nostdin', '-y',
                '-i', str(master_video.absolute()),
                '-ss', str(fast_ss),
                '-t', str(buffer_dur),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                temp_clean
            ]
            app_logger.info(f"Step 1 Buffer Copy Command: {' '.join(extract_cmd)}")
            subprocess.run(extract_cmd, capture_output=True, check=True)
            
            input_video = temp_clean
            # Step 2 logic: Now we perform the FRAME-ACCURATE trim relative to the buffer start.
            # Since the buffer starts at fast_ss, the target start is at (actual_start - fast_ss).
            # VideoProcessor re-encodes, so this trim will be perfect.
            relative_start = actual_start - fast_ss
            relative_end = actual_end - fast_ss
            edit_settings['trim']['start'] = format_srt_time(relative_start).replace(',', '.')
            edit_settings['trim']['end'] = format_srt_time(relative_end).replace(',', '.')
            app_logger.info(f"Step 2 Relative Trim: {edit_settings['trim']['start']} - {edit_settings['trim']['end']}")

            input_video = temp_clean
            # Step 2 logic will now be 0-based relative to this clean clip
            # --- FIXED: We DO NOT reset to 0:00 because temp_clean has a 30s buffer. ---
            # We keep the relative_start/end calculated above.

        # --- STEP 2: APPLY EFFECTS & OVERLAYS ---
        log_message("STEP 2: Applying Logo, Audio, and B-roll overlays...")
        
        # DEBUG: Log images and B-roll data
        images_to_process = edit_settings.get('images', [])
        broll_to_process = edit_settings.get('broll_markers', [])
        log_message(f"DEBUG: Images to process: {len(images_to_process)}")
        for i, img in enumerate(images_to_process):
            log_message(f"  - Image {i}: {img.get('name')} ({len(img.get('markers', []))} markers)")
        
        log_message(f"DEBUG: B-roll markers to process: {len(broll_to_process)}")
        for i, br in enumerate(broll_to_process):
            log_message(f"  - B-roll {i}: {br.get('name')} at {br.get('start_time')}s to {br.get('end_time')}s")

        # Calculate display_duration for marker time mapping (main + outro)
        # We trust the client-provided display_duration as it accounts for extended outro clips.
        if 'display_duration' in edit_settings:
            display_duration = float(edit_settings['display_duration'])
        else:
            main_dur = actual_end - actual_start
            outro_conf = edit_settings.get('outro', {})
            # If the end time was already extended by the client, this might be redundant but safe
            outro_video_data = outro_conf.get('outroVideo')
            outro_video_dur = float(outro_video_data.get('duration', 0)) if (outro_video_data and outro_video_data.get('path')) else 0
            display_duration = main_dur + outro_video_dur
        
        edit_settings['display_duration'] = display_duration
        log_message(f"DEBUG: Final display_duration for marker mapping: {display_duration}")
        
        processor = VideoProcessor(input_video)
        intermediate_video = processor.apply_effects(output_video, edit_settings, cancel_flag=lambda: edit_processes.get(edit_id, {}).get('cancelled', False), log_callback=log_message)
        
        # --- STEP 3: BURN SUBTITLES ---
        if reburn_enabled:
            log_message("STEP 3: Burning Subtitles & Title...")
            
            # Ensure title from metadata if missing
            if not edit_settings.get('subtitles', {}).get('title'):
                adjust_settings = get_theme_adjust_settings(folder_path, t_num)
                if adjust_settings.get('title'):
                    edit_settings.setdefault('subtitles', {})['title'] = adjust_settings['title']

            # Use relative theme SRT because Step 1 created a 0-based clip
            # PRIORITIZE srt_path passed from process_video_edit (which already found the best match)
            srt_path = edit_settings.get('subtitles', {}).get('srt_path')
            
            if not srt_path or not os.path.exists(srt_path):
                srt_path = str(folder_path / 'shorts' / f'theme_{int(t_num):03d}.srt')
                if not os.path.exists(srt_path):
                    matches = list((folder_path / 'shorts').glob(f"theme_{int(t_num):03d}_*.srt"))
                    if matches: srt_path = str(matches[0])
            
            word_timestamps_file = next(folder_path.glob('*_word_timestamps.json'), None)
            
            if os.path.exists(srt_path) and word_timestamps_file:
                # Setup render context
                edit_render_settings = edit_settings.get('subtitles', {}).copy()
                edit_render_settings['folder_number'] = f_num
                edit_render_settings['theme_number'] = t_num

                # Metadata start for word lookup
                adjust_settings = get_theme_adjust_settings(folder_path, t_num)
                t_match = re.search(r'(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)', adjust_settings.get('time_range', ''))
                theme_meta_start = to_sec(t_match.group(1).replace(',', '.')) if t_match else 0
                log_message(f"DEBUG: theme_meta_start from adjust.md={theme_meta_start}")

                # The renderer uses theme_meta_start as the offset for word lookup *relative to the SRT*.
                # This must match the original theme start time because the SRT is relative to that.
                edit_render_settings['theme_meta_start'] = theme_meta_start
                log_message(f"DEBUG: Final theme_meta_start for renderer={edit_render_settings['theme_meta_start']}")

                # Pass animations and effect markers for frame-by-frame rendering
                edit_render_settings['animations'] = edit_settings.get('animations', {})
                edit_render_settings['effect_markers'] = edit_settings.get('effect_markers', [])
                edit_render_settings['effects'] = edit_settings.get('effects', {})

                # Load per-cue formatting from JSON if exists
                formatting_file = folder_path / 'shorts' / f'theme_{int(t_num):03d}_formatting.json'
                if formatting_file.exists():
                    try:
                        with open(formatting_file, 'r', encoding='utf-8') as f:
                            edit_render_settings['formatting'] = json.load(f)
                            log_message(f"DEBUG: Loaded per-cue formatting for {len(edit_render_settings['formatting'])} cues")
                    except Exception as e:
                        log_message(f"Warning: Failed to load formatting file: {e}")
                else:
                    # Fallback to formatting passed in request if not found on disk
                    edit_render_settings['formatting'] = edit_settings.get('settings', {}).get('formatting', {})

                # CRITICAL: Merge adjust_settings into edit_render_settings so saved font, colors, etc. are applied
                # These fields always come from adjust.md (saved settings):
                always_from_adjust = ['fontName', 'fontSize', 'primaryColor', 'bgColor', 'bgOpacity',
                                      'subtitle_bold', 'subtitle_position', 'subtitle_left', 'subtitle_top',
                                      'subtitle_h_align', 'subtitle_v_align',
                                      'title_font_size', 'title_bg_type', 'title_text_color', 'title_font_weight',
                                      'title_outline_width', 'title_outline_color', 'title_all_caps', 'title_top',
                                      'title']
                for key in always_from_adjust:
                    if key in adjust_settings and adjust_settings[key] is not None:
                        edit_render_settings[key] = adjust_settings[key]

                # Explicitly apply show_title from edit UI (priority)
                # If NOT in UI settings, fall back to adjust.md
                if 'show_title' in edit_settings.get('subtitles', {}):
                    edit_render_settings['show_title'] = edit_settings['subtitles']['show_title']
                elif 'show_title' in adjust_settings:
                    edit_render_settings['show_title'] = adjust_settings['show_title']

                if 'h_align' in edit_render_settings: edit_render_settings['subtitle_h_align'] = edit_render_settings['h_align']
                if 'v_align' in edit_render_settings: edit_render_settings['subtitle_v_align'] = edit_render_settings['v_align']

                # Progress weighting
                pass_count = 2
                p_scale = 1.0 / pass_count
                p_offset = 50.0

                def render_progress_cb(percent, stage, msg):
                    scaled_percent = min(99, int(p_offset + (percent * p_scale)))
                    log_message(f"PROGRESS:{scaled_percent}%|Final Subtitle Burn ({msg})")

                log_message(f"DEBUG: Calling render_canvas_karaoke_video with actual_start={actual_start}, actual_end={actual_end}, srt={srt_path}")
                # CRITICAL: render_canvas_karaoke_video expects absolute timestamps for data lookup,
                # but is_already_trimmed=True handles the 0-based video sync.
                final_burn_success = render_canvas_karaoke_video(
                    intermediate_video, str(word_timestamps_file), srt_path, output_video,
                    actual_start, actual_end, edit_render_settings,
                    progress_callback=render_progress_cb, is_already_trimmed=True, srt_mode='relative'
                )
            else:
                log_message(f"Warning: SRT or Timestamps missing. Skipping Step 3.")

        # FINALIZATION
        if not final_burn_success:
            if os.path.exists(intermediate_video) and str(intermediate_video) != str(output_video):
                if os.path.exists(output_video): os.remove(output_video)
                shutil.move(intermediate_video, output_video)

        # VALIDATION: Ensure output file is valid and not empty
        if not os.path.exists(output_video) or os.path.getsize(output_video) < 1000:
            raise Exception("Output video is missing or empty. FFmpeg may have failed due to invalid seek range.")

        # --- STEP 4: APPEND OUTRO VIDEO ---
        outro_conf = edit_settings.get('outro', {})
        outro_video_data = outro_conf.get('outroVideo')
        if outro_video_data and outro_video_data.get('path'):
            log_message("STEP 4: Appending Outro Video...")
            outro_path = folder_path / 'media' / outro_video_data['path']
            if not outro_path.exists():
                outro_path = Path('media') / outro_video_data['path']
            
            if not outro_path.exists():
                # Recursive search fallback
                try:
                    matches = list(Path('media').rglob(outro_video_data['path']))
                    if matches: outro_path = matches[0]
                except: pass
            
            if not outro_path.exists():
                outro_path = Path(outro_video_data['path'])
            
            if outro_path.exists():
                try:
                    # 1. Transcode outro to match the main video format (1080x1920, libx264, aac)
                    # This ensures the concat demuxer works reliably.
                    temp_outro = os.path.join(str(GLOBAL_TEMP_DIR), f"outro_ready_{edit_id}.mp4")
                    
                    # Get actual duration first so it's available for filters
                    dur_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(outro_path)]
                    outro_dur_raw = subprocess.run(dur_cmd, capture_output=True).stdout.strip()
                    try:
                        # If image, ffprobe duration is empty, fall back to setting or default
                        actual_dur = float(outro_dur_raw) if outro_dur_raw else float(outro_video_data.get('duration', 3.0))
                    except (ValueError, TypeError):
                        actual_dur = 3.0

                    # Robust check for audio in the outro clip
                    check_outro_audio_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', str(outro_path)]
                    outro_has_audio = subprocess.run(check_outro_audio_cmd, capture_output=True).stdout.strip() != b""
                    
                    # Build transcode command
                    trans_cmd = ['ffmpeg', '-nostdin', '-y']
                    
                    if not outro_has_audio:
                        # Add silent audio source if outro has none
                        trans_cmd.extend(['-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo'])
                    
                    # Check if it's an image to use -loop 1
                    is_image = outro_video_data.get('type') == 'image' or str(outro_path).lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.svg'))
                    if is_image:
                        trans_cmd.extend(['-loop', '1'])
                    
                    trans_cmd.extend(['-i', str(outro_path.absolute())])
                    
                    # Unified filter complex to handle:
                    # - Transparency (PNG) by overlaying on a black background
                    # - "Fit" (contain) scaling strategy
                    # - Smooth video fade-in transition
                    v_in_idx = 1 if not outro_has_audio else 0
                    fc = f"color=c=black:s=1080x1920:d={actual_dur}[bg];" \
                         f"[{v_in_idx}:v]format=rgba,scale=1080:1920:force_original_aspect_ratio=decrease[scaled];" \
                         f"[bg][scaled]overlay=(W-w)/2:(H-h)/2"
                    
                    if outro_conf.get('fadeVideoSmooth'):
                        # Fade in over 1 second (or 1/3 of duration if very short)
                        fade_dur = min(1.0, actual_dur / 3.0)
                        fc += f",fade=t=in:st=0:d={fade_dur}"
                    
                    trans_cmd.extend(['-filter_complex', f"{fc}[v]", '-map', '[v]', '-map', '0:a'])
                    trans_cmd.extend(['-t', str(actual_dur)])

                    # Add audio fade if requested
                    if outro_conf.get('fadeAudio'):
                        # Start fade 1 second before actual end
                        fade_st = max(0, actual_dur - 1)
                        trans_cmd.extend(['-af', f'afade=t=out:st={fade_st}:d=1'])
                    
                    trans_cmd.extend([
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', str(edit_settings.get('export_crf', 22)),
                        '-r', '30', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', 
                        temp_outro
                    ])
                    
                    log_message(f"Transcoding outro clip (has_audio={outro_has_audio}): {' '.join(trans_cmd)}")
                    subprocess.run(trans_cmd, capture_output=True, check=True)
                    
                    # 2. Concatenate the main video and the prepared outro
                    final_combined = os.path.join(str(GLOBAL_TEMP_DIR), f"final_combined_{edit_id}.mp4")
                    
                    # We use a robust re-encoding join.
                    # First, we check if the main video has audio.
                    check_audio_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', str(output_video)]
                    has_audio = subprocess.run(check_audio_cmd, capture_output=True).stdout.strip() != b""
                    
                    log_message(f"Joining main video (has_audio={has_audio}) and outro...")
                    
                    # If both have audio, we can use simple concat
                    if has_audio:
                        cmd_str = f"ffmpeg -nostdin -y -i '{output_video}' -i '{temp_outro}' -filter_complex '[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]' -map '[v]' -map '[a]' -c:v libx264 -preset ultrafast -crf {edit_settings.get('export_crf', 22)} -c:a aac -b:a 192k -ar 44100 '{final_combined}'"
                    else:
                        # Add silence to main video if it doesn't have it
                        cmd_str = f"ffmpeg -nostdin -y -i '{output_video}' -i '{temp_outro}' -f lavfi -i anullsrc=r=44100:cl=stereo -filter_complex '[0:v][2:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]' -map '[v]' -map '[a]' -c:v libx264 -preset ultrafast -crf {edit_settings.get('export_crf', 22)} -c:a aac -b:a 192k -ar 44100 '{final_combined}'"
                    
                    log_message(f"Running Join Command: {cmd_str}")
                    subprocess.run(cmd_str, shell=True, check=True, capture_output=True)
                    
                    # 3. Swap the combined video for the output_video
                    if os.path.exists(final_combined):
                        if os.path.exists(output_video): os.remove(output_video)
                        shutil.move(final_combined, output_video)
                        log_message("Outro successfully appended.")
                    
                    # Cleanup
                    if os.path.exists(temp_outro): os.remove(temp_outro)
                    
                except Exception as e:
                    log_message(f"Warning: Failed to append outro: {e}")
            else:
                log_message(f"Warning: Outro video file not found: {outro_path}")

        with task_lock:
            edit_processes[edit_id]['status'] = 'completed'
            edit_processes[edit_id]['success'] = True
            try: rel_path = Path(output_video).relative_to(Path.cwd())
            except: rel_path = output_video
            edit_processes[edit_id]['output_path'] = str(rel_path)

    except Exception as e:
        import traceback
        log_message(f"Processing error: {traceback.format_exc()}")
        with task_lock:
            edit_processes[edit_id]['status'] = 'cancelled' if edit_processes.get(edit_id, {}).get('cancelled') else 'failed'
            edit_processes[edit_id]['error'] = str(e)
    finally: pass

@app.route('/api/process-edit/<edit_id>/cancel', methods=['POST'])
def cancel_edit(edit_id: str):
    """Cancel a running video edit."""
    with task_lock:
        if edit_id not in edit_processes:
            return jsonify({'error': 'Edit not found'}), 404

        edit_data = edit_processes[edit_id]
        edit_data['cancelled'] = True
        edit_data['status'] = 'cancelled'
        edit_processes[edit_id] = edit_data

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
    ensure_manager()
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
        tasks[task_id] = task_manager.dict({
            'type': 'create_word_timestamps',
            'status': 'pending',
            'folder_number': folder_number,
            'video_path': str(video_file)
        })

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

    data = request.json
    folder_number = data.get('folder')
    style_data = {
        'preset': data.get('preset', 'yellow-glow'),
        'textColor': data.get('textColor', '#ffff00'),
        'glowColor': data.get('glowColor', '#ffff00'),
        'glowBlur': data.get('glowBlur', '10'),
        'fontWeight': data.get('fontWeight', 'bold'),
        # Karaoke mode and effect settings
        'karaoke_mode': data.get('karaokeMode', 'normal'),
        'effectType': data.get('effectType', 'none'),
        'autoEmoji': data.get('autoEmoji', False),
        'keywordScaling': data.get('keywordScaling', False),
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


def get_theme_adjust_settings(folder_path, theme_number):
    """Read theme adjustment settings from theme_XXX_adjust.md."""
    adjust_file = folder_path / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'
    
    # Load defaults from settings.ini
    default_font = settings.get('subtitle', 'font_name', fallback='Arial')
    default_size = int(settings.get('subtitle', 'font_size', fallback='80'))
    default_primary = settings.get('subtitle', 'primary_colour', fallback='#FFFFFF')
    default_align = settings.get('subtitle', 'alignment', fallback='bottom')
    
    settings_dict = {
        'fontName': default_font,
        'fontSize': default_size,
        'primaryColor': default_primary,
        'subtitle_position': default_align,
        'subtitle_left': None,
        'subtitle_top': None,
        'subtitle_h_align': 'center',
        'subtitle_v_align': 'bottom',
        'subtitle_bold': False,
        'title': None,
        'time_range': None,
        'folder': None,
        'title_font_size': None,
        'title_bg_type': 'gradient',
        'title_text_color': '#00ff9d',
        'title_font_weight': 800,
        'title_outline_width': 0,
        'title_outline_color': '#000000',
        'title_all_caps': True,
        'show_title': False
    }
    
    if adjust_file.exists():
        with open(adjust_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract title and time range
        title_match = re.search(r'\*\*Title:\*\*\s*(.+)', content)
        if title_match: settings_dict['title'] = title_match.group(1).strip()
        
        time_match = re.search(r'\*\*Time Range:\*\*\s*(.+)', content)
        if time_match: settings_dict['time_range'] = time_match.group(1).strip()
        
        folder_match = re.search(r'\*\*Folder:\*\*\s*(.+)', content)
        if folder_match: settings_dict['folder'] = folder_match.group(1).strip()

        # Extract subtitle position
        pos_match = re.search(r'\*\*subtitle_position:\*\*\s*(\w+)', content)
        if pos_match:
            settings_dict['subtitle_position'] = pos_match.group(1)
            
        # Extract custom coordinates if they exist
        left_match = re.search(r'\*\*subtitle_left:\*\*\s*(\d+)', content)
        if left_match:
            settings_dict['subtitle_left'] = int(left_match.group(1))
            
        top_match = re.search(r'\*\*subtitle_top:\*\*\s*(\d+)', content)
        if top_match:
            settings_dict['subtitle_top'] = int(top_match.group(1))
            
        h_align_match = re.search(r'\*\*subtitle_h_align:\*\*\s*(\w+)', content)
        if h_align_match:
            settings_dict['subtitle_h_align'] = h_align_match.group(1)
            
        v_align_match = re.search(r'\*\*subtitle_v_align:\*\*\s*(\w+)', content)
        if v_align_match:
            settings_dict['subtitle_v_align'] = v_align_match.group(1)
            
        # Additional styling fields
        font_size_match = re.search(r'\*\*subtitle_font_size:\*\*\s*(\d+)', content)
        if font_size_match:
            settings_dict['fontSize'] = int(font_size_match.group(1))
            
        primary_color_match = re.search(r'\*\*subtitle_primary_color:\*\*\s*(#[0-9a-fA-F]+)', content)
        if primary_color_match:
            settings_dict['primaryColor'] = primary_color_match.group(1)
            
        bg_color_match = re.search(r'\*\*subtitle_bg_color:\*\*\s*(#[0-9a-fA-F]+)', content)
        if bg_color_match:
            settings_dict['bgColor'] = bg_color_match.group(1)
            
        bg_opacity_match = re.search(r'\*\*subtitle_bg_opacity:\*\*\s*([0-9.]+)', content)
        if bg_opacity_match:
            settings_dict['bgOpacity'] = float(bg_opacity_match.group(1)) / 100.0 if float(bg_opacity_match.group(1)) > 1.0 else float(bg_opacity_match.group(1))
            
        font_name_match = re.search(r'\*\*subtitle_font_name:\*\*\s*(.+)', content)
        if font_name_match:
            settings_dict['fontName'] = font_name_match.group(1).strip()
            
        subtitle_bold_match = re.search(r'\*\*subtitle_bold:\*\*\s*(true|false)', content)
        if subtitle_bold_match:
            settings_dict['subtitle_bold'] = subtitle_bold_match.group(1) == 'true'

        karaoke_match = re.search(r'\*\*karaoke_highlighting:\*\*\s*(true|false)', content)
        if karaoke_match:
            settings_dict['karaoke_enabled'] = karaoke_match.group(1) == 'true'
            
        # Title styling fields
        ttop_match = re.search(r'\*\*title_top:\*\*\s*(\d+)', content)
        if ttop_match: settings_dict['title_top'] = int(ttop_match.group(1))

        tfs_match = re.search(r'\*\*title_font_size:\*\*\s*(\d+)', content)
        if tfs_match: settings_dict['title_font_size'] = int(tfs_match.group(1))
        
        tbt_match = re.search(r'\*\*title_bg_type:\*\*\s*(\w+)', content)
        if tbt_match: settings_dict['title_bg_type'] = tbt_match.group(1)
        
        ttc_match = re.search(r'\*\*title_text_color:\*\*\s*(#[0-9a-fA-F]+)', content)
        if ttc_match: settings_dict['title_text_color'] = ttc_match.group(1)
        
        tfw_match = re.search(r'\*\*title_font_weight:\*\*\s*(\d+)', content)
        if tfw_match: settings_dict['title_font_weight'] = int(tfw_match.group(1))
        
        tow_match = re.search(r'\*\*title_outline_width:\*\*\s*([0-9.]+)', content)
        if tow_match: settings_dict['title_outline_width'] = float(tow_match.group(1))
        
        toc_match = re.search(r'\*\*title_outline_color:\*\*\s*(#[0-9a-fA-F]+)', content)
        if toc_match: settings_dict['title_outline_color'] = toc_match.group(1)
        
        tac_match = re.search(r'\*\*title_all_caps:\*\*\s*(true|false)', content)
        if tac_match: settings_dict['title_all_caps'] = tac_match.group(1) == 'true'

        type_match = re.search(r'\*\*Type:\*\*\s*custom', content)
        if type_match:
            settings_dict['is_custom'] = True

        st_match = re.search(r'\*\*show_title:\*\*\s*(true|false)', content)
        if st_match: settings_dict['show_title'] = st_match.group(1) == 'true'
            
    # Load global highlight style if it exists
    highlight_file = folder_path / 'highlight_style.json'
    if highlight_file.exists():
        try:
            with open(highlight_file, 'r', encoding='utf-8') as f:
                h_style = json.load(f)
                # Map highlight style fields to the settings keys expected by the renderer
                if 'textColor' in h_style: settings_dict['textColor'] = h_style['textColor']
                if 'glowColor' in h_style: settings_dict['glowColor'] = h_style['glowColor']
                if 'glowBlur' in h_style: settings_dict['glowBlur'] = h_style['glowBlur']
                if 'fontWeight' in h_style: settings_dict['font_weight'] = h_style['fontWeight']
                if 'effectType' in h_style: settings_dict['effect_type'] = h_style['effectType']
                if 'karaoke_mode' in h_style: settings_dict['mode'] = h_style['karaoke_mode']
                if 'autoEmoji' in h_style: settings_dict['auto_emoji'] = h_style['autoEmoji']
                if 'keywordScaling' in h_style: settings_dict['keyword_scaling'] = h_style['keywordScaling']
                if 'past_color' in h_style: settings_dict['pastColor'] = h_style['past_color']
        except Exception as e:
            app_logger.warning(f"Failed to load highlight_style.json: {e}")

    return settings_dict


def write_theme_adjust_settings(adjust_file, theme_number, title, time_range, folder_name, settings_dict):
    """Write theme adjustment settings to theme_XXX_adjust.md, preserving all fields."""
    with open(adjust_file, 'w', encoding='utf-8') as f:
        f.write(f"# Theme {int(theme_number)}\n\n")
        if title:
            f.write(f"**Title:** {title}\n\n")
        if time_range:
            f.write(f"**Time Range:** {time_range}\n")

        if settings_dict.get('is_custom'):
            f.write(f"**Type:** custom\n")

        # Write position - either preset or custom with coordinates
        position = settings_dict.get('subtitle_position', 'bottom')
        f.write(f"**subtitle_position:** {position}\n")
        
        if settings_dict.get('subtitle_left') is not None:
            f.write(f"**subtitle_left:** {settings_dict['subtitle_left']}\n")
        if settings_dict.get('subtitle_top') is not None:
            f.write(f"**subtitle_top:** {settings_dict['subtitle_top']}\n")
            
        f.write(f"**subtitle_h_align:** {settings_dict.get('subtitle_h_align', 'center')}\n")
        f.write(f"**subtitle_v_align:** {settings_dict.get('subtitle_v_align', 'bottom')}\n")
        
        # Write additional styling
        if 'fontSize' in settings_dict:
            f.write(f"**subtitle_font_size:** {settings_dict['fontSize']}\n")
        
        if 'subtitle_bold' in settings_dict:
            f.write(f"**subtitle_bold:** {'true' if settings_dict['subtitle_bold'] else 'false'}\n")
            
        if 'primaryColor' in settings_dict:
            f.write(f"**subtitle_primary_color:** {settings_dict['primaryColor']}\n")
            
        if 'bgColor' in settings_dict:
            f.write(f"**subtitle_bg_color:** {settings_dict['bgColor']}\n")
            
        if 'bgOpacity' in settings_dict:
            # We save the value as-is (could be 0.0-1.0 or 0-100)
            f.write(f"**subtitle_bg_opacity:** {settings_dict['bgOpacity']}\n")
            
        if 'fontName' in settings_dict:
            f.write(f"**subtitle_font_name:** {settings_dict['fontName']}\n")

        # Preserve karaoke highlighting if present
        if 'karaoke_enabled' in settings_dict:
            f.write(f"**karaoke_highlighting:** {'true' if settings_dict['karaoke_enabled'] else 'false'}\n")

        # Write title styling
        if 'title_font_size' in settings_dict:
            f.write(f"**title_font_size:** {settings_dict['title_font_size']}\n")
        if 'title_bg_type' in settings_dict:
            f.write(f"**title_bg_type:** {settings_dict['title_bg_type']}\n")
        if 'title_text_color' in settings_dict:
            f.write(f"**title_text_color:** {settings_dict['title_text_color']}\n")
        if 'title_font_weight' in settings_dict:
            f.write(f"**title_font_weight:** {settings_dict['title_font_weight']}\n")
        if 'title_outline_width' in settings_dict:
            f.write(f"**title_outline_width:** {settings_dict['title_outline_width']}\n")
        if 'title_outline_color' in settings_dict:
            f.write(f"**title_outline_color:** {settings_dict['title_outline_color']}\n")
        if 'title_all_caps' in settings_dict:
            f.write(f"**title_all_caps:** {'true' if settings_dict['title_all_caps'] else 'false'}\n")
        if 'show_title' in settings_dict:
            f.write(f"**show_title:** {'true' if settings_dict['show_title'] else 'false'}\n")
        if settings_dict.get('title_top') is not None:
            f.write(f"**title_top:** {settings_dict['title_top']}\n")

        if folder_name:
            f.write(f"\n**Folder:** {folder_name}\n")
        f.write(f"**Last Modified:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


@app.route('/api/encode-canvas-karaoke', methods=['POST'])
def encode_canvas_karaoke():
    """Generate and encode canvas karaoke video server-side (fast)."""
    app_logger.info("ENCODE_CANVAS_KARAOKE ENDPOINT HIT")
    ensure_manager()
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
        video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
        if not video_files:
            return jsonify({'error': 'No video file found'}), 404
        video_file = video_files[0]

        # Get theme timing - prefer client-provided times, fall back to adjust.md, then themes.md
        theme_start = data.get('themeStart')
        theme_end = data.get('themeEnd')
        
        start_time = 0
        end_time = 0

        if theme_start is not None and theme_end is not None:
            # Use times provided by client (current theme length in browser)
            start_time = float(theme_start)
            end_time = float(theme_end)
        else:
            # Try to read from adjust.md
            adjust_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'
            if adjust_file.exists():
                with open(adjust_file, 'r', encoding='utf-8') as f:
                    adj_content = f.read()

                tr_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)', adj_content)
                if tr_match:
                    start_time = parse_timestamp(tr_match.group(1))
                    end_time = parse_timestamp(tr_match.group(2))
            
            # Fall back to reading from themes.md if still 0
                
            if start_time == 0 and end_time == 0:
                themes_file = folder / 'themes.md'
                if not themes_file.exists():
                    return jsonify({'error': 'themes.md not found and no client times provided'}), 400

                with open(themes_file, 'r', encoding='utf-8') as f:
                    themes_content = f.read()

                # Parse time range

                time_pattern = r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)'
                time_match = re.search(time_pattern, themes_content)

                if not time_match:
                    return jsonify({'error': 'Could not parse theme time range from themes.md'}), 400

                def parse_time_str(time_str):
                    time_str = time_str.replace(',', '.')
                    h, m, s = time_str.split(':')
                    return int(h) * 3600 + int(m) * 60 + float(s)

                start_time = parse_time_str(time_match.group(1))
                end_time = parse_time_str(time_match.group(2))

        # Get word timestamps
        word_timestamps_file = None
        for file in folder.glob('*_word_timestamps.json'):
            word_timestamps_file = file
            break

        if not word_timestamps_file or not word_timestamps_file.exists():
            return jsonify({'error': 'Word timestamps not found'}), 404

        # Prefer the theme-specific SRT file if it exists, as it contains user edits and restructuring.
        # Fall back to the main (absolute) SRT only if theme-specific is missing.
        srt_file = folder / 'shorts' / f'theme_{int(theme_number):03d}.srt'
        if not srt_file.exists():
            srt_file = folder / 'adjust.srt'
        
        if not srt_file.exists():
            srt_files = [f for f in folder.glob('*.srt') if 'theme_' not in f.name and 'adjust' not in f.name]
            if srt_files:
                srt_file = srt_files[0]
        
        if not srt_file or not srt_file.exists():
            # Last resort: any SRT
            for srt in folder.glob('*.srt'):
                if not any(x in srt.name.lower() for x in ['transcribe']):
                    srt_file = srt
                    break

        if not srt_file or not srt_file.exists():
            return jsonify({'error': 'Subtitle file not found'}), 404

        # Output path
        output_path = folder / 'shorts' / f'theme_{theme_number}_canvas_karaoke.mp4'
        output_path.parent.mkdir(exist_ok=True)

        # Get theme adjust settings for subtitle positioning
        adjust_settings = get_theme_adjust_settings(folder, theme_number)
        
        # Merge into karaoke_settings (but let incoming data override if present)
        final_settings = karaoke_settings.copy()
        
        # Ensure base fields are present if missing
        if 'fontSize' not in final_settings: final_settings['fontSize'] = 100
        if 'fontName' not in final_settings: final_settings['fontName'] = 'Avenir Black'
        if 'bgColor' not in final_settings: final_settings['bgColor'] = '#000000'
        if 'bgOpacity' not in final_settings: final_settings['bgOpacity'] = 0.63
        
        final_settings.update({
            'show_title': karaoke_settings.get('show_title', False),
            'title': adjust_settings.get('title', 'Theme Title')
        })
        
        # Merge in saved adjustments (overrides incoming if present)
        final_settings.update(adjust_settings)
        
        # Ensure title fields are robustly set from either source
        mapping = {
            'title_font_size': 'titleFontSize',
            'title_bg_type': 'titleBgType',
            'title_text_color': 'titleTextColor',
            'title_font_weight': 'titleFontWeight',
            'title_outline_width': 'titleOutlineWidth',
            'title_outline_color': 'titleOutlineColor',
            'title_all_caps': 'titleAllCaps',
            'title_top': 'titleTop'
        }
        for snake, camel in mapping.items():
            if snake not in final_settings and camel in karaoke_settings:
                final_settings[snake] = karaoke_settings[camel]

        # Extract audio levels for reactive effects if needed
        if final_settings.get('effect_type') == 'volume_shake':
            try:
                app_logger.info(f"Extracting audio levels for theme {theme_number}...")
                audio_levels = _extract_audio_levels(video_file, start_time, end_time)
                final_settings['audio_levels'] = audio_levels
            except Exception as ae:
                app_logger.warning(f"Failed to extract audio levels: {ae}")

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
                        'complete': (stage == 'completed'),
                        'error': (message if stage == 'error' else None)
                    })

        # Background rendering function
        def render_in_background():
            try:
                app_logger.info(f"Starting server-side canvas karaoke export for theme {theme_number}")

                # Import the renderer
                from subtitle_renderer import render_canvas_karaoke_video

                # Determine SRT mode explicitly based on the chosen file
                srt_mode = 'relative' if 'theme_' in srt_file.name or 'adjust' in srt_file.name else 'absolute'

                # DEBUG: Log the settings being passed to the renderer
                app_logger.info(f"DEBUG: Passing to renderer - fontName={final_settings.get('fontName')}, fontSize={final_settings.get('fontSize')}")

                # Render video with progress callback
                success = render_canvas_karaoke_video(
                    str(video_file),
                    str(word_timestamps_file),
                    str(srt_file),
                    str(output_path),
                    start_time,
                    end_time,
                    final_settings,
                    progress_callback,
                    srt_mode=srt_mode
                )

                if not success:
                    progress_callback(-1, 'error', 'Rendering failed')
                else:
                    # Save a copy of the SRT and JSON metadata alongside the video
                    try:
                        import shutil
                        video_name = Path(output_path).stem
                        
                        # Save trimmed SRT
                        dest_srt = Path(output_path).parent / f"{video_name}.srt"
                        
                        # Load edits for this theme if it's a theme export
                        edits = {}
                        if theme_number:
                            edits_file = folder / 'shorts' / f"theme_{int(theme_number):03d}_edits.json"
                            if edits_file.exists():
                                try:
                                    with open(edits_file, 'r', encoding='utf-8') as f:
                                        edits = json.load(f)
                                except: pass
                        
                        # Generate a fresh trimmed SRT from the updated main SRT source
                        # This ensures edits are included and the timing is relative to the clip
                        creator.create_trimmed_srt(srt_file, start_time, end_time, dest_srt, edits=edits)
                        
                        # Save a JSON with theme-specific word timestamps
                        dest_json = Path(output_path).parent / f"{video_name}.json"
                        
                        # Load word timestamps if not already in memory
                        with open(word_timestamps_file, 'r', encoding='utf-8') as f:
                            wt_data = json.load(f)
                            all_words = wt_data.get('words', [])
                        
                        # Filter word timestamps for this theme
                        theme_words = []
                        for w in all_words:
                            if w['end'] > start_time - 0.1 and w['start'] < end_time + 0.1:
                                # Create relative timestamps for the JSON
                                # CLAMP TO RANGE for clean output
                                rw = w.copy()
                                rw['start'] = max(0, w['start'] - start_time)
                                rw['end'] = min(end_time - start_time, w['end'] - start_time)
                                if rw['end'] > rw['start']:
                                    theme_words.append(rw)
                        
                        with open(dest_json, 'w', encoding='utf-8') as f:
                            json.dump({
                                'theme': theme_number,
                                'start_time': start_time,
                                'end_time': end_time,
                                'words': theme_words
                            }, f, indent=2)
                            
                        app_logger.info(f"Saved metadata to {dest_srt} and {dest_json}")
                    except Exception as me:
                        app_logger.warning(f"Failed to save metadata: {me}")

                    progress_callback(100, 'completed', 'Video saved successfully')

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
    ensure_manager()
    with canvas_karaoke_lock:
        if job_id not in canvas_karaoke_progress:
            return jsonify({'error': 'Job not found'}), 404

        progress_info = canvas_karaoke_progress[job_id].copy()

        # Clean up old completed jobs (older than 5 minutes)
        if progress_info.get('complete') or progress_info.get('error'):
            # Keep it for now so client can get final status
            pass

        return jsonify(progress_info)


@app.route('/api/download-canvas-karaoke/<folder_number>/<theme>')
def download_canvas_karaoke(folder_number, theme):
    """Download the canvas karaoke video."""
    try:
        base_dir = Path(settings.get('video', 'output_dir'))
        folder_path = None

        for f in base_dir.iterdir():
            if f.is_dir() and f.name.startswith(f"{folder_number}_"):
                folder_path = f
                break

        if not folder_path:
            return jsonify({'error': 'Folder not found'}), 404

        # Use glob to find the file since it now includes the title
        pattern = f'theme_{int(theme):03d}_*_canvas_karaoke.mp4'
        matches = list((folder_path / 'shorts').glob(pattern))
        
        if not matches:
            # Fallback to old format
            video_path = folder_path / 'shorts' / f'theme_{theme}_canvas_karaoke.mp4'
        else:
            video_path = matches[0]

        if not video_path.exists():
            return jsonify({'error': 'Video not found'}), 404

        return send_file(str(video_path), as_attachment=True, download_name=video_path.name)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


from subtitle_renderer import render_canvas_karaoke_video

@app.route('/api/export-canvas-karaoke', methods=['POST'])
def export_canvas_karaoke():
    """Export canvas karaoke video using FFmpeg on server (fast processing)."""
    app_logger.info("EXPORT_CANVAS_KARAOKE ENDPOINT HIT")
    ensure_manager()
    try:
        data = request.get_json()
        folder_number = data.get('folder')
        theme_number = data.get('theme')
        theme_start_provided = data.get('themeStart')
        theme_end_provided = data.get('themeEnd')
        karaoke_settings = data.get('settings', {})
        quality = data.get('quality', 'standard')

        if not folder_number or not theme_number:
            return jsonify({'error': 'Missing folder or theme number'}), 400

        # Create job ID
        job_id = f"{folder_number}_{theme_number}_{int(datetime.now().timestamp())}"

        # Initialize progress tracking
        with canvas_karaoke_lock:
            canvas_karaoke_progress[job_id] = {
                'status': 'starting',
                'progress': 0,
                'message': 'Initializing export...',
                'complete': False
            }

        def run_export_thread(jid, f_num, t_num, settings_dict, start_override=None, end_override=None, quality_preset='standard'):
            app_logger.info(f"THREAD START: run_export_thread jid={jid} folder={f_num} theme={t_num} quality={quality_preset}")
            try:
                # CRF Mapping for quality
                crf_map = {
                    'fast': 28,
                    'standard': 22,
                    'high': 18,
                    'ultra': 12
                }
                crf = crf_map.get(quality_preset, 22)
                settings_dict['export_crf'] = crf
                
                # Use settings.ini default if available, otherwise use fast presets
                default_preset = settings.get('video', 'export_preset', fallback='ultrafast')
                if quality_preset == 'fast':
                    settings_dict['export_preset'] = 'ultrafast'
                else:
                    settings_dict['export_preset'] = default_preset

                # Get folder path
                base_dir = Path(settings.get('video', 'output_dir'))
                folder = None
                for f in base_dir.iterdir():
                    if f.is_dir() and f.name.startswith(f"{f_num}_"):
                        folder = f
                        break
                
                if not folder:
                    with canvas_karaoke_lock:
                        canvas_karaoke_progress[jid] = {'status': 'error', 'error': 'Folder not found', 'complete': False}
                    return

                # Get video file
                video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
                if not video_files:
                    with canvas_karaoke_lock:
                        canvas_karaoke_progress[jid] = {'status': 'error', 'error': 'No video file found', 'complete': False}
                    return
                video_file = video_files[0]

                # Get word timestamps (Always use the master absolute one for Pass 2 lookups)
                word_timestamps_file = next(folder.glob('*_word_timestamps.json'), None)
                if not word_timestamps_file:
                    with canvas_karaoke_lock:
                        canvas_karaoke_progress[jid] = {'status': 'error', 'error': 'Word timestamps not found', 'complete': False}
                    return

                # CRITICAL: If we found a theme-specific JSON (starts with 'theme_'), try to find the master one
                if word_timestamps_file.name.startswith('theme_'):
                    master_json = next(folder.glob('[!theme]*_word_timestamps.json'), None)
                    if master_json:
                        word_timestamps_file = master_json
                        app_logger.info(f"Using master word timestamps for export: {word_timestamps_file.name}")

                # Prefer the theme-specific SRT file if it exists, as it contains user edits and restructuring.
                # Fall back to the main (absolute) SRT only if theme-specific is missing.
                srt_file = folder / 'shorts' / f'theme_{int(t_num):03d}.srt'
                if not srt_file.exists():
                    srt_file = folder / 'adjust.srt'
                
                if not srt_file.exists():
                    srt_files = [f for f in folder.glob('*.srt') if 'theme_' not in f.name and 'adjust' not in f.name]
                    if srt_files:
                        srt_file = srt_files[0]
                
                if not srt_file or not srt_file.exists():
                    # Last resort: any SRT
                    for srt in folder.glob('*.srt'):
                        if not any(x in srt.name.lower() for x in ['transcribe']):
                            srt_file = srt
                            break
                
                if not srt_file or not srt_file.exists():
                    with canvas_karaoke_lock:
                        canvas_karaoke_progress[jid] = {'status': 'error', 'error': 'Subtitle file not found', 'complete': False}
                    return

                # Determine start and end times
                start_time = 0
                end_time = 0
                theme_title = ""

                if start_override is not None and end_override is not None:
                    start_time = parse_timestamp(start_override)
                    end_time = parse_timestamp(end_override)
                    app_logger.info(f"Using client-provided times: {start_time} - {end_time}")
                
                # Check adjust.md if times not provided or to get title
                adjust_file = folder / 'shorts' / f'theme_{int(t_num):03d}_adjust.md'
                if adjust_file.exists():
                    with open(adjust_file, 'r', encoding='utf-8') as f:
                        adj_content = f.read()
                    
                    t_match = re.search(r'\*\*Title:\*\*\s*(.+)', adj_content)
                    if t_match: theme_title = t_match.group(1).strip()
                    
                    if start_time == 0 and end_time == 0:
                        # Allow both HH:MM:SS and HH:MM:SS,mmm or HH:MM:SS.mmm
                        tr_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)', adj_content)
                        if tr_match:
                            start_time = parse_timestamp(tr_match.group(1))
                            end_time = parse_timestamp(tr_match.group(2))
                            app_logger.info(f"Using adjust.md times: {start_time} - {end_time}")

                # Fallback to themes.md
                if start_time == 0 and end_time == 0:
                    themes_file = folder / 'themes.md'
                    if themes_file.exists():
                        with open(themes_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        try:
                            # 1. Try to find the section for this theme
                            theme_section_pattern = rf'### Theme {t_num}:(.*?)(?=### Theme|\Z)'
                            section_match = re.search(theme_section_pattern, content, re.DOTALL)
                            if section_match:
                                section = section_match.group(0)
                                # Extract Title
                                if not theme_title:
                                    title_match = re.search(rf'### Theme {t_num}:\s*(.*?)\n', section)
                                    if title_match: theme_title = title_match.group(1).strip()
                                
                                # Extract Time Range
                                time_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)', section)
                                if time_match:
                                    def parse_t(s):
                                        s = s.replace(',', '.')
                                        h, m, sec = s.split(':')
                                        return int(h) * 3600 + int(m) * 60 + float(sec)
                                    start_time = parse_t(time_match.group(1))
                                    end_time = parse_t(time_match.group(2))
                                    app_logger.info(f"Using themes.md times: {start_time} - {end_time}")
                        except Exception as e:
                            app_logger.error(f"Error parsing themes.md: {e}")

                # Direct log for debugging
                with open('server.log', 'a') as f_log:
                    f_log.write(f"{datetime.now()} - [DEBUG_EXPORT] Theme {t_num}: title='{theme_title}', start={start_time}, end={end_time}\n")

                # EXTRA: Also find the ORIGINAL theme start time from themes.md to help with SRT sync
                theme_meta_start = 0
                themes_file = folder / 'themes.md'
                if themes_file.exists():
                    try:
                        with open(themes_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        theme_section_pattern = rf'### Theme {t_num}:(.*?)(?=### Theme|\Z)'
                        section_match = re.search(theme_section_pattern, content, re.DOTALL)
                        if section_match:
                            time_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)', section_match.group(0))
                            if time_match:
                                theme_meta_start = creator.parse_timestamp_to_seconds(time_match.group(1))
                    except: pass

                # Output path
                sanitized_title = creator.sanitize_title(theme_title) if theme_title else "untitled"
                
                # Determine suffix based on mode
                suffix = "standard" if settings_dict.get('effect_type') == 'none' else "canvas_karaoke"
                output_filename = f'theme_{int(t_num):03d}_{sanitized_title}_{suffix}_{quality_preset}.mp4'
                
                output_path = folder / 'shorts' / output_filename
                output_path.parent.mkdir(exist_ok=True)

                def progress_cb(percent, stage, msg):
                    with canvas_karaoke_lock:
                        # Update the shared task dict
                        current = dict(canvas_karaoke_progress[jid])
                        current.update({
                            'progress': int(percent),
                            'message': f"{stage}: {msg}",
                            'status': 'processing'
                        })
                        canvas_karaoke_progress[jid] = current

                # Call the actual renderer using multi-pass strategy (same as edit)
                from video_processor import VideoProcessor
                import time as py_time
                
                export_start_perf = py_time.time()
                processor = VideoProcessor(str(video_file))
                
                # Setup settings for Pass 1 (Trim & Vertical Crop)
                # We reuse the logic from process_video_edit
                pass1_settings = {
                    'trim': {
                        'start': str(start_time),
                        'end': str(end_time)
                    },
                    'subtitles': {
                        'reburn': True,
                        'srt_path': str(srt_file),
                        'native_burn': settings_dict.get('effect_type') == 'none', # FAST LANE DETECTION
                        'fontSize': settings_dict.get('fontSize', 80),
                        'primaryColor': settings_dict.get('primaryColor', '#ffffff'),
                        'show_title': settings_dict.get('show_title', False),
                        'title': theme_title,
                        'title_top': settings_dict.get('titleTop', 150),
                        'title_font_size': settings_dict.get('titleFontSize', 80),
                        'title_text_color': settings_dict.get('titleTextColor', '#00ff9d'),
                    },
                    'folder_number': f_num,
                    'theme_number': t_num,
                    # Include all overlay and effect settings from the request
                    'images': settings_dict.get('images', []),
                    'broll_markers': settings_dict.get('broll_markers', []),
                    'audio': settings_dict.get('audio', {}),
                    'outro': settings_dict.get('outro', {}),
                    'effects': settings_dict.get('effects', {}),
                    'lighting': settings_dict.get('lighting', {}),
                    'animations': settings_dict.get('animations', {}),
                    'effect_markers': settings_dict.get('effect_markers', []),
                    'display_duration': end_time - start_time
                }
                
                # Check if we can use the "Fast Lane" (Native FFmpeg for everything)
                use_fast_lane = pass1_settings['subtitles']['native_burn']
                if use_fast_lane:
                    app_logger.info("⚡ FAST LANE DETECTED: Using native FFmpeg for subtitle burning.")
                
                # Pass 1: Trim, Vertical Crop, Audio mixing (FFmpeg passes)
                # Calculate progress (Pass 1 in VideoProcessor now handles its own internal offsets)
                def pass1_progress_cb(msg):
                    if "PROGRESS:" in msg:
                        try:
                            # Extract percentage from VideoProcessor (0-100 range)
                            percent = int(msg.split(":")[1].split("%")[0])
                            # If we ARE doing the slow Python pass, FFmpeg is only about 40% of the work.
                            # If we ARE in Fast Lane (Native FFmpeg), then FFmpeg is 100% of the work.
                            if use_fast_lane:
                                progress_cb(percent, "FFmpeg Export", "Generating final video")
                            else:
                                progress_cb(percent * 0.4, "Optimizing Video", "Processing effects & overlays")
                        except: pass

                app_logger.info(f"Starting Pass 1 (FFmpeg): theme {t_num}")
                pass1_start = py_time.time()
                intermediate_video = processor.apply_effects(str(output_path), pass1_settings, log_callback=pass1_progress_cb)
                pass1_duration = py_time.time() - pass1_start
                app_logger.info(f"PASS 1 COMPLETE: {pass1_duration:.2f}s")
                
                if use_fast_lane:
                    # In Fast Lane, the output is already complete!
                    # Move from intermediate temp path to final output path
                    import shutil
                    if os.path.exists(intermediate_video) and str(Path(intermediate_video).absolute()) != str(output_path.absolute()):
                        if output_path.exists():
                            os.remove(output_path)
                        shutil.move(intermediate_video, str(output_path))
                        
                    total_duration = py_time.time() - export_start_perf
                    app_logger.info(f"FAST LANE EXPORT SUMMARY (Theme {t_num}):")
                    app_logger.info(f"  - Total Duration: {total_duration:.2f}s")
                    # We skip the Python loop entirely
                    success = True
                else:
                    # SLOW LANE: Proceed to Python rendering loop
                    # Get adjustments for this theme to ensure latest styling is used
                    adjust_settings = get_theme_adjust_settings(folder, t_num)
                    
                    # Setup final settings for the renderer, merging incoming with adjust.md
                    final_render_settings = settings_dict.copy()
                    final_render_settings['folder_number'] = f_num
                    final_render_settings['theme_number'] = t_num
                    final_render_settings['themeNumber'] = t_num
                    
                    # ... (keep existing merging/mapping logic)
                    # Ensure base subtitle fields are present if missing
                    if 'fontSize' not in final_render_settings: final_render_settings['fontSize'] = 100
                    if 'fontName' not in final_render_settings: final_render_settings['fontName'] = 'Avenir Black'
                    if 'bgColor' not in final_render_settings: final_render_settings['bgColor'] = '#000000'
                    if 'bgOpacity' not in final_render_settings: final_render_settings['bgOpacity'] = 0.63
                    
                    # Merge in saved adjustments first
                    final_render_settings.update(adjust_settings)

                    # Apply user's explicit show_title choice (overrides saved setting)
                    if 'show_title' in settings_dict:
                        final_render_settings['show_title'] = settings_dict['show_title']

                    # Also ensure title is set
                    final_render_settings['title'] = theme_title
                    
                    # Ensure title fields are robustly set from either source (handle camelCase from UI)
                    mapping = {
                        'title_font_size': 'titleFontSize',
                        'title_bg_type': 'titleBgType',
                        'title_text_color': 'titleTextColor',
                        'title_font_weight': 'titleFontWeight',
                        'title_outline_width': 'titleOutlineWidth',
                        'title_outline_color': 'titleOutlineColor',
                        'title_all_caps': 'titleAllCaps',
                        'title_top': 'titleTop',
                        'subtitle_h_align': 'h_align',
                        'subtitle_v_align': 'v_align',
                        'reburn': 'reburn'
                    }
                    for snake, camel in mapping.items():
                        if snake not in final_render_settings and camel in settings_dict:
                            final_render_settings[snake] = settings_dict[camel]
                    
                    # Store original theme start for SRT sync
                    final_render_settings['theme_meta_start'] = theme_meta_start
                    
                    # Extract audio levels for reactive effects if needed
                    if final_render_settings.get('effect_type') == 'volume_shake':
                        try:
                            app_logger.info(f"Extracting audio levels for export of theme {t_num}...")
                            audio_levels = _extract_audio_levels(video_file, start_time, end_time)
                            final_render_settings['audio_levels'] = audio_levels
                            final_render_settings['base_time'] = float(start_time)
                        except Exception as ae:
                            app_logger.warning(f"Failed to extract audio levels for export: {ae}")
                    
                    # Pass 2: Canvas Subtitles (The slower Python loop, but now on a trimmed clip)
                    def render_progress_cb(percent, stage, msg):
                        # VideoProcessor (Pass 1) took up 40% of the work
                        progress_cb(40 + (percent * 0.6), "Rendering Subtitles", msg)

                    app_logger.info(f"Starting Pass 2: Subtitle Rendering for theme {t_num}")
                    # Determine SRT mode explicitly
                    srt_mode = 'relative' if 'theme_' in srt_file.name else 'absolute'

                    # DEBUG: Log the settings being passed to the renderer
                    app_logger.info(f"DEBUG: Passing to renderer (slow lane) - fontName={final_render_settings.get('fontName')}, fontSize={final_render_settings.get('fontSize')}")

                    pass2_start = py_time.time()
                    success = render_canvas_karaoke_video(
                        str(intermediate_video),
                        str(word_timestamps_file),
                        str(srt_file),
                        str(output_path),
                        float(start_time),
                        float(end_time),
                        final_render_settings,
                        progress_callback=render_progress_cb,
                        is_already_trimmed=True,
                        srt_mode=srt_mode
                    )
                    pass2_duration = py_time.time() - pass2_start
                    total_duration = py_time.time() - export_start_perf
                    
                    app_logger.info(f"EXPORT PERFORMANCE SUMMARY (Theme {t_num}):")
                    app_logger.info(f"  - Pass 1 (FFmpeg): {pass1_duration:.2f}s")
                    app_logger.info(f"  - Pass 2 (Python): {pass2_duration:.2f}s")
                    app_logger.info(f"  - TOTAL DURATION: {total_duration:.2f}s")

                with canvas_karaoke_lock:
                    if success:
                        canvas_karaoke_progress[jid] = {
                            'status': 'completed',
                            'progress': 100,
                            'complete': True,
                            'output_path': str(output_path),
                            'download_url': f'/api/download-canvas-karaoke/{f_num}/{t_num}'
                        }
                    else:
                        canvas_karaoke_progress[jid] = {'status': 'error', 'error': 'Rendering failed', 'complete': False}

            except Exception as e:
                import traceback
                app_logger.error(f"Thread export error: {traceback.format_exc()}")
                with canvas_karaoke_lock:
                    canvas_karaoke_progress[jid] = {'status': 'error', 'error': str(e), 'complete': False}

        # Start the thread
        threading.Thread(target=run_export_thread, args=(job_id, folder_number, theme_number, karaoke_settings, theme_start_provided, theme_end_provided, quality)).start()

        return jsonify({'success': True, 'job_id': job_id})

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
        folder = creator.get_video_folder_by_number(folder)
        if not folder:
            return jsonify({'error': 'Folder not found'}), 404

        if type == 'canvas_karaoke':
            # Use glob to find the file since it now includes the title
            # Look for either standard or canvas_karaoke versions
            pattern = f'theme_{int(theme):03d}_*_{{standard,canvas_karaoke}}.mp4'
            # Note: glob doesn't support curly braces by default in all versions, 
            # so let's check for both manually
            matches = list((folder / 'shorts').glob(f'theme_{int(theme):03d}_*_standard.mp4'))
            matches += list((folder / 'shorts').glob(f'theme_{int(theme):03d}_*_canvas_karaoke.mp4'))
            
            if not matches:
                # Fallback to old format just in case
                video_path = folder / 'shorts' / f'theme_{theme}_canvas_karaoke.mp4'
            else:
                # Get the newest match
                matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                video_path = matches[0]
        else:
            return jsonify({'error': 'Invalid type'}), 400

        if not video_path.exists():
            return jsonify({'error': 'Video not found'}), 404

        return send_file(str(video_path), as_attachment=True, download_name=video_path.name)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import signal
    
    # Initialize shared state
    ensure_manager()

    now = datetime.now()
    start_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    print("YouTube Shorts Creator - Web Server")
    print(f"Started at: {start_time_str}")
    print("=" * 60)
    print(f"Server running at: http://localhost:5000")
    print(f"Video directory: {settings.get('video', 'output_dir')}")
    print("=" * 60)

    # Log server startup
    app_logger.info("=" * 60)
    app_logger.info(f"YouTube Shorts Creator - Web Server STARTED at {start_time_str}")
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
                if task.get('status') in ['pending', 'processing', 'running']:
                    task['cancelled'] = True
        print("\033[92mServer shut down cleanly.\033[0m")
        app_logger.debug("Server shut down cleanly")
        import sys
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)

    # Note: use_reloader=True enables auto-reload on code changes.
    # This may start the process twice initially, but enables development workflow.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)
