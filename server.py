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
from pathlib import Path
from typing import Dict, List
import configparser

from shorts_creator import YouTubeShortsCreator, load_settings

app = Flask(__name__, static_folder='static')
CORS(app)

# Global state
creator = YouTubeShortsCreator()
settings = load_settings()

# Background task management
tasks: Dict[str, Dict] = {}
task_counter = 0
task_lock = threading.Lock()


def run_task(task_id: str, func, *args, **kwargs):
    """Run a function in a background thread and update task status."""
    try:
        with task_lock:
            tasks[task_id]['status'] = 'running'

        result = func(*args, **kwargs)

        with task_lock:
            tasks[task_id]['status'] = 'completed'
            tasks[task_id]['result'] = result
    except Exception as e:
        with task_lock:
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['error'] = str(e)


@app.route('/')
def index():
    """Serve the HTML frontend."""
    return send_from_directory('.', 'index.html')


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
    return send_file(str(video_path), mimetype='video/mp4')


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
    if video_info_file.exists():
        with open(video_info_file, 'r') as f:
            for line in f:
                if line.startswith('Title:'):
                    video_title = line.split(':', 1)[1].strip()
                    break

    return jsonify({
        'folder': folder.name,
        'title': video_title,
        'themes': themes
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

    if not url and not local_file:
        return jsonify({'error': 'Either URL or local file must be provided'}), 400

    with task_lock:
        task_counter += 1
        task_id = f"task_{task_counter}"
        tasks[task_id] = {
            'type': 'process',
            'status': 'pending',
            'url': url if url else local_file
        }

    # Start background task
    thread = threading.Thread(
        target=run_task,
        args=(task_id, _process_video, url, local_file, model)
    )
    thread.start()

    return jsonify({'task_id': task_id})


def _process_video(url: str, local_file: str, model: str):
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

    if url:
        video_info = creator.download_video(url)
    else:
        video_info = creator.process_local_video(local_file)

    creator.create_video_info(video_info)
    creator.generate_subtitles(video_info, model_size=model)
    creator.generate_themes(video_info, ai_generator=ai_generator, model_size=model)

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
            'folder_number': folder_number
        }

    thread = threading.Thread(
        target=run_task,
        args=(task_id, _regenerate_themes, folder_number, model)
    )
    thread.start()

    return jsonify({'task_id': task_id})


def _regenerate_themes(folder_number: str, model: str):
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

    creator.generate_themes(video_info, ai_generator=ai_generator, model_size=model)

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
        target=run_task,
        args=(task_id, _create_shorts, folder_number, themes)
    )
    thread.start()

    return jsonify({'task_id': task_id})


def _create_shorts(folder_number: str, themes: List):
    """Create shorts in background."""
    theme_str = 'all' if themes == 'all' else ','.join(map(str, themes))
    creator.create_shorts(folder_number, theme_str)

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
    """Cancel a background task."""
    with task_lock:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404

        if tasks[task_id]['status'] == 'running':
            # Note: We can't actually cancel running threads easily
            # This just marks it as cancelled
            tasks[task_id]['status'] = 'cancelled'

        return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 60)
    print("YouTube Shorts Creator - Web Server")
    print("=" * 60)
    print(f"Server running at: http://localhost:5000")
    print(f"Video directory: {settings.get('video', 'output_dir')}")
    print("=" * 60)

    app.run(host='127.0.0.1', port=5000, debug=True)
