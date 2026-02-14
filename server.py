#!/usr/bin/env python3
"""
Flask server for the YouTube Shorts Creator web GUI.
"""

from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import subprocess
import threading
import time
import queue
import os
import sys
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
# Max size set to roughly 1000 lines (avg 100 bytes per line = 100KB)
log_handler = RotatingFileHandler('server.log', maxBytes=100*1024, backupCount=1)
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

# Ensure canvas_karaoke_exporter logger also logs to file
exporter_logger = logging.getLogger('canvas_karaoke_exporter')
exporter_logger.setLevel(logging.INFO)
exporter_logger.addHandler(log_handler)

# Keep console output too
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)
flask_logger.addHandler(console_handler)
app_logger.addHandler(console_handler)
exporter_logger.addHandler(console_handler)

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
task_counter = 0

import multiprocessing

# Global process manager for shared state
# We initialize it here as None and will set up the actual objects 
# inside the if __name__ == '__main__': block to avoid recursion errors.
task_manager = None
tasks = {}
task_processes = {}
task_lock = threading.RLock() 

# Canvas karaoke progress tracking
canvas_karaoke_progress = {}
canvas_karaoke_lock = threading.RLock()

def run_task_with_callback(task_id: str, func, *args, **kwargs):
    """Run a function in a background process with shared state support."""
    global tasks, task_manager, task_processes
    if task_manager is None:
        # Fallback for unexpected environments, though it should be initialized in main
        task_manager = multiprocessing.Manager()
        tasks = task_manager.dict()

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
    import re
    import traceback
    
    # Define a callback that updates the SHARED dictionary
    def process_callback(msg):
        msg_str = str(msg)
        # Log to the real system stdout so we can see it in terminal
        sys.__stdout__.write(f"[{task_id}] {msg_str}\n")
        sys.__stdout__.flush()
        
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
    srt_files = [f for f in folder.glob('*.srt') if 'theme_' not in f.name and 'adjust' not in f.name]
    if not srt_files:
        srt_files = list(folder.glob('*.srt'))
        
    if not srt_files:
        return "SRT file not found", 404

    srt_file = srt_files[0]

    # Create theme-specific SRT and JSON files in the shorts directory
    # This ensures they exist for editing as soon as adjust.html loads the theme
    try:
        shorts_dir = folder / 'shorts'
        shorts_dir.mkdir(exist_ok=True)
        
        # 1. Create theme SRT (theme_XXX.srt)
        theme_srt_name = f"theme_{int(theme_number):03d}.srt"
        theme_srt_path = shorts_dir / theme_srt_name
        
        # Use creator to create the trimmed SRT
        creator.create_trimmed_srt(srt_file, theme_start_sec, theme_end_sec, theme_srt_path)
        
        # 2. Create theme JSON word timestamps (theme_XXX.json)
        theme_json_name = f"theme_{int(theme_number):03d}.json"
        theme_json_path = shorts_dir / theme_json_name
        
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
                if w['start'] >= theme_start_sec - 1.0 and w['end'] <= theme_end_sec + 1.0:
                    rw = w.copy()
                    rw['start'] = max(0, w['start'] - theme_start_sec)
                    rw['end'] = max(0, w['end'] - theme_start_sec)
                    theme_words.append(rw)
                    
            with open(theme_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'theme': theme_number,
                    'start_time': theme_start_sec,
                    'end_time': theme_end_sec,
                    'words': theme_words
                }, f, indent=2)
                
        app_logger.info(f"Generated theme metadata: {theme_srt_name} and {theme_json_name}")
    except Exception as e:
        app_logger.warning(f"Failed to generate theme metadata: {e}")

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

    # Load persistent edits if they exist
    edits = {}
    edits_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_edits.json'
    if edits_file.exists():
        try:
            with open(edits_file, 'r', encoding='utf-8') as f:
                edits = json.load(f)
        except:
            pass

    return jsonify({
        'cues': filtered_cues,
        'is_adjusted': has_adjusted,
        'edits': edits
    })


@app.route('/api/all-subtitles/<folder_number>', methods=['GET'])
def get_all_subtitles(folder_number: str):
    """Get ALL subtitles from the original SRT file with sequence numbers (no theme filtering)."""
    import re
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
                    time_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})', content)
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
                
                # Create theme SRT
                theme_srt_path = shorts_dir / f"theme_{int(theme_number):03d}.srt"
                creator.create_trimmed_srt(srt_file, theme_start_sec, theme_end_sec, theme_srt_path)
                
                # Create theme JSON
                theme_json_path = shorts_dir / f"theme_{int(theme_number):03d}.json"
                
                # Find word timestamps
                word_timestamps_file = None
                for file in folder.glob('*_word_timestamps.json'):
                    word_timestamps_file = file
                    break
                    
                if word_timestamps_file:
                    with open(word_timestamps_file, 'r', encoding='utf-8') as f:
                        wt_data = json.load(f)
                        all_words = wt_data.get('words', [])
                    
                    theme_words = [w.copy() for w in all_words 
                                   if w['start'] >= theme_start_sec - 1.0 and w['end'] <= theme_end_sec + 1.0]
                    for w in theme_words:
                        w['start'] = max(0, w['start'] - theme_start_sec)
                        w['end'] = max(0, w['end'] - theme_start_sec)
                        
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
                    # Use period for fractional seconds (standard timestamp format)
                    start_ts = f"{match.group(1)}.{match.group(2)}"
                    end_ts = f"{match.group(3)}.{match.group(4)}"

                    # Get subtitle text (may be multiple lines)
                    i += 1
                    text_lines = []
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                        text_lines.append(lines[i].strip())
                        i += 1

                    all_cues.append({
                        'sequence': seq_num,
                        'start': start_ts,
                        'end': end_ts,
                        'text': '\n'.join(text_lines)
                    })
                    continue
            else:
                i += 1
        else:
            i += 1

    # Load persistent edits if they exist
    edits = {}
    if theme_number:
        edits_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_edits.json'
        if edits_file.exists():
            try:
                with open(edits_file, 'r', encoding='utf-8') as f:
                    edits = json.load(f)
            except:
                pass

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

        if not folder_number or not theme_number or cue_start is None or cue_end is None or not text:
            return jsonify({'error': 'Missing required fields'}), 400

        base_dir = Path(settings.get('video', 'output_dir'))
        folder = None
        for f in base_dir.iterdir():
            if f.is_dir() and f.name.startswith(f"{folder_number}_"):
                folder = f
                break

        if not folder:
            return jsonify({'error': 'Folder not found'}), 404

        import re

        try:
            # Incoming timestamps from UI are ABSOLUTE HH:MM:SS.mmm
            target_start_seconds = parse_srt_time(cue_start.replace(',', '.'))
            target_end_seconds = parse_srt_time(cue_end.replace(',', '.'))
        except (ValueError, TypeError) as e:
            app_logger.error(f"[ERROR] Invalid cue times: {e}")
            return jsonify({'error': f'Invalid time values: {str(e)}'}), 400

        # Get theme start time for normalization
        theme_start_sec = None
        if theme_start is not None and str(theme_start).strip() != '':
            try: theme_start_sec = float(theme_start)
            except: pass

        if theme_start_sec is None:
            adjust_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_adjust.md'
            if adjust_file.exists():
                with open(adjust_file, 'r', encoding='utf-8') as f:
                    c = f.read()
                    tm = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})', c)
                    if tm: theme_start_sec = creator.parse_timestamp_to_seconds(tm.group(1))
            
            if theme_start_sec is None:
                themes_file = folder / 'themes.md'
                if themes_file.exists():
                    themes = creator.parse_themes_file(themes_file)
                    for t in themes:
                        if t['number'] == int(theme_number):
                            theme_start_sec = creator.parse_timestamp_to_seconds(t['start'])
                            break

        # Paths
        theme_srt = folder / 'shorts' / f'theme_{int(theme_number):03d}.srt'
        srt_files = [f for f in folder.glob('*.srt') if 'theme_' not in f.name and 'adjust' not in f.name]
        if not srt_files: srt_files = list(folder.glob('*.srt'))
        main_srt = srt_files[0] if srt_files else None

        files_to_update = []
        if main_srt: files_to_update.append(main_srt)
        if theme_srt.exists(): files_to_update.append(theme_srt)

        matched_absolute_start = None
        matched_absolute_end = None

        for target_file in files_to_update:
            is_trimmed = 'theme_' in target_file.name
            try:
                with open(target_file, 'r', encoding='utf-8-sig') as f: content = f.read()
            except:
                try:
                    with open(target_file, 'r', encoding='utf-8') as f: content = f.read()
                except: continue

            blocks = []
            slines = content.split('\n')
            si = 0
            while si < len(slines):
                line = slines[si].strip()
                if not line: si += 1; continue
                block = {'number': line, 'timestamp': None, 'text_lines': [], 'start': None, 'end': None}
                si += 1
                if si < len(slines):
                    ts_line = slines[si].strip(); block['timestamp'] = ts_line
                    ts_m = re.match(r'(\d+:\d+:[\d,]+)\s*-->\s*(\d+:\d+:[\d,]+)', ts_line)
                    if ts_m:
                        block['start'] = parse_srt_time(ts_m.group(1).replace(',', '.'))
                        block['end'] = parse_srt_time(ts_m.group(2).replace(',', '.'))
                    si += 1
                    while si < len(slines):
                        t_line = slines[si].strip()
                        if not t_line or t_line.isdigit(): break
                        block['text_lines'].append(t_line); si += 1
                    blocks.append(block)

            res_lines = []
            match_found = False
            for block in blocks:
                res_lines.append(block['number'])
                res_lines.append(block['timestamp'])
                
                # Match logic:
                # If target_file is main_srt, we use target_start_seconds directly (ABSOLUTE)
                # If target_file is theme_srt, it has RELATIVE times. We compare against (target_start_seconds - theme_start_sec)
                compare_start = target_start_seconds
                if is_trimmed and theme_start_sec is not None:
                    compare_start = target_start_seconds - theme_start_sec
                
                is_match = False
                if block['start'] is not None:
                    # Tolerance for minor format differences
                    if abs(block['start'] - compare_start) < 0.15: 
                        is_match = True
                    # Special case: clamped start at 0.0 in trimmed file
                    elif is_trimmed and block['start'] == 0 and (target_start_seconds - (theme_start_sec or 0)) < 0.1:
                        is_match = True

                if is_match and not match_found:
                    res_lines.append(text.strip()); match_found = True
                    if not is_trimmed:
                        matched_absolute_start = format_srt_time(block['start']).replace(',', '.')
                        matched_absolute_end = format_srt_time(block['end']).replace(',', '.')
                else: res_lines.append('\n'.join(block['text_lines']))
                res_lines.append('')

            if match_found:
                with open(target_file, 'w', encoding='utf-8') as f: f.write('\n'.join(res_lines))

        # Save to persistent edits JSON using ABSOLUTE keys
        # This is what ensures edits persist across page reloads and theme boundary shifts
        try:
            edits_file = folder / 'shorts' / f'theme_{int(theme_number):03d}_edits.json'
            edits_data = {}
            if edits_file.exists():
                with open(edits_file, 'r', encoding='utf-8') as f: edits_data = json.load(f)
            
            # Key MUST be absolute to match all-subtitles
            # Use matched_absolute_* if available, otherwise use incoming cue_start/end
            final_start = matched_absolute_start or cue_start
            final_end = matched_absolute_end or cue_end
            
            edit_key = f"{final_start}_{final_end}"
            edits_data[edit_key] = text.strip()
            with open(edits_file, 'w', encoding='utf-8') as f: json.dump(edits_data, f, indent=2)
            print(f"[DEBUG] Saved persistent UI edit to {edits_file} with key {edit_key}")
        except Exception as ee: print(f"[ERROR] JSON save failed: {ee}")

        return jsonify({'success': True, 'message': 'Saved successfully'})
    except Exception as e:
        import traceback
        app_logger.error(f"[ERROR] save_cue_text: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

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
        tasks[task_id] = task_manager.dict({
            'type': 'create_shorts',
            'status': 'pending',
            'folder_number': folder_number,
            'themes': themes
        })

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
        return jsonify(dict(tasks[task_id]))


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
        import re
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
            sys.__stdout__.write(f"[Re-transcribe Progress] {msg_str}\n")
            sys.__stdout__.flush()
            
            display_percent = None
            import re
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
    settings = {
        'subtitle_position': 'bottom',
        'subtitle_left': None,
        'subtitle_top': None,
        'subtitle_h_align': 'center',
        'subtitle_v_align': 'bottom'
    }
    
    if adjust_file.exists():
        with open(adjust_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract subtitle position
        import re
        pos_match = re.search(r'\*\*subtitle_position:\*\*\s*(\w+)', content)
        if pos_match:
            settings['subtitle_position'] = pos_match.group(1)
            
        # Extract custom coordinates if they exist
        left_match = re.search(r'\*\*subtitle_left:\*\*\s*(\d+)', content)
        if left_match:
            settings['subtitle_left'] = int(left_match.group(1))
            
        top_match = re.search(r'\*\*subtitle_top:\*\*\s*(\d+)', content)
        if top_match:
            settings['subtitle_top'] = int(top_match.group(1))
            
        h_align_match = re.search(r'\*\*subtitle_h_align:\*\*\s*(\w+)', content)
        if h_align_match:
            settings['subtitle_h_align'] = h_align_match.group(1)
            
        v_align_match = re.search(r'\*\*subtitle_v_align:\*\*\s*(\w+)', content)
        if v_align_match:
            settings['subtitle_v_align'] = v_align_match.group(1)
            
    return settings


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

        # Get theme timing - prefer client-provided times, fall back to themes.md
        theme_start = data.get('themeStart')
        theme_end = data.get('themeEnd')

        if theme_start is not None and theme_end is not None:
            # Use times provided by client (current theme length in browser)
            start_time = float(theme_start)
            end_time = float(theme_end)
        else:
            # Fall back to reading from themes.md
            themes_file = folder / 'themes.md'
            if not themes_file.exists():
                return jsonify({'error': 'themes.md not found and no client times provided'}), 400

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

        # Fall back to main video SRT if adjust.srt doesn't exist
        if not srt_file.exists():
            for srt in folder.glob('*.srt'):
                if 'transcribe' not in srt.name and 'theme' not in srt.name:
                    srt_file = srt
                    break

        if not srt_file.exists():
            return jsonify({'error': 'Subtitle file not found'}), 404

        # Output path
        output_path = folder / 'shorts' / f'theme_{theme_number}_canvas_karaoke.mp4'
        output_path.parent.mkdir(exist_ok=True)

        # Get theme adjust settings for subtitle positioning
        adjust_settings = get_theme_adjust_settings(folder, theme_number)
        
        # Merge into karaoke_settings (but let incoming data override if present)
        final_settings = {
            'fontSize': karaoke_settings.get('fontSize', 48) * 2,  # Double for 1080x1920
            'fontName': karaoke_settings.get('fontName', 'Arial'),
            'textColor': karaoke_settings.get('textColor', '#ffff00'),
            'pastColor': karaoke_settings.get('pastColor', '#808080'),
            'mode': karaoke_settings.get('mode', 'normal')
        }
        final_settings.update(adjust_settings)

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
                    final_settings,
                    progress_callback
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
                        
                        # Generate a fresh trimmed SRT from the updated main SRT source
                        # This ensures edits are included and the timing is relative to the clip
                        creator.create_trimmed_srt(srt_file, start_time, end_time, dest_srt)
                        
                        # Save a JSON with theme-specific word timestamps
                        dest_json = Path(output_path).parent / f"{video_name}.json"
                        
                        # Load word timestamps if not already in memory
                        with open(word_timestamps_file, 'r', encoding='utf-8') as f:
                            wt_data = json.load(f)
                            all_words = wt_data.get('words', [])
                        
                        # Filter word timestamps for this theme
                        theme_words = []
                        for w in all_words:
                            if w['start'] >= start_time - 1.0 and w['end'] <= end_time + 1.0:
                                # Create relative timestamps for the JSON
                                rw = w.copy()
                                rw['start'] = max(0, w['start'] - start_time)
                                rw['end'] = max(0, w['end'] - start_time)
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

                    progress_callback(100, 'complete', 'Video saved successfully')

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

        video_path = folder_path / 'shorts' / f'theme_{theme}_canvas_karaoke.mp4'

        if not video_path.exists():
            return jsonify({'error': 'Video not found'}), 404

        return send_file(str(video_path), as_attachment=True, download_name=video_path.name)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


from subtitle_renderer import render_canvas_karaoke_video

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

        # Create a unique job ID
        import uuid
        job_id = str(uuid.uuid4())
        
        # Initialize task state
        with canvas_karaoke_lock:
            canvas_karaoke_progress[job_id] = {
                'status': 'processing',
                'progress': 0,
                'message': 'Starting export...',
                'folder': folder_number,
                'theme': theme_number,
                'complete': False,
                'error': None
            }

        def run_export_thread(jid, f_num, t_num, settings_dict):
            try:
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
                video_files = list(folder.glob('*.mp4'))
                if not video_files:
                    with canvas_karaoke_lock:
                        canvas_karaoke_progress[jid] = {'status': 'error', 'error': 'No video file found', 'complete': False}
                    return
                video_file = video_files[0]

                # Get word timestamps
                word_timestamps_file = next(folder.glob('*_word_timestamps.json'), None)
                if not word_timestamps_file:
                    with canvas_karaoke_lock:
                        canvas_karaoke_progress[jid] = {'status': 'error', 'error': 'Word timestamps not found', 'complete': False}
                    return

                # Get theme subtitles
                srt_file = folder / 'shorts' / f'theme_{int(t_num):03d}.srt'
                if not srt_file.exists():
                    srt_file = folder / 'adjust.srt'
                
                if not srt_file.exists():
                    with canvas_karaoke_lock:
                        canvas_karaoke_progress[jid] = {'status': 'error', 'error': 'Subtitle file not found', 'complete': False}
                    return

                # Get theme timing from themes.md
                themes_file = folder / 'themes.md'
                start_time = 0
                end_time = 0
                if themes_file.exists():
                    with open(themes_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    import re
                    pattern = rf'### Theme {t_num}:.*?\*\*Time Range:\*\*\s*(\d{{2}}:\d{{2}}:\d{{2}})\s*-\s*(\d{{2}}:\d{{2}}:\d{{2}})'
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        def parse_t(s):
                            h, m, sec = map(int, s.split(':'))
                            return h * 3600 + m * 60 + sec
                        start_time = parse_t(match.group(1))
                        end_time = parse_t(match.group(2))

                # Output path
                output_path = folder / 'shorts' / f'theme_{t_num}_canvas_karaoke.mp4'
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

                # Call the actual renderer
                success = render_canvas_karaoke_video(
                    str(video_file),
                    str(word_timestamps_file),
                    str(srt_file),
                    str(output_path),
                    float(start_time),
                    float(end_time),
                    settings_dict,
                    progress_callback=progress_cb
                )

                with canvas_karaoke_lock:
                    if success:
                        canvas_karaoke_progress[jid] = {
                            'status': 'complete',
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
        threading.Thread(target=run_export_thread, args=(job_id, folder_number, theme_number, karaoke_settings)).start()

        return jsonify({'success': True, 'job_id': job_id})

    except Exception as e:
        import traceback
        app_logger.error(f"Export error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


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
    
    # Initialize shared state
    task_manager = multiprocessing.Manager()
    tasks = task_manager.dict()
    canvas_karaoke_progress = task_manager.dict()

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
                if task.get('status') in ['pending', 'processing', 'running']:
                    task['cancelled'] = True
        print("\033[92mServer shut down cleanly.\033[0m")
        app_logger.debug("Server shut down cleanly")
        import sys
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)

    app.run(host='0.0.0.0', port=5000, debug=True)
