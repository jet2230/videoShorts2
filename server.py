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
from datetime import datetime

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
            with task_lock:
                if task_id in tasks and 'log' in tasks[task_id]:
                    current_log = tasks[task_id]['log']
                    if not isinstance(current_log, str):
                        current_log = str(current_log) if current_log else ''
                    tasks[task_id]['log'] = current_log + msg_str + '\n'
        except Exception as e:
            print(f"Callback error: {e}")

    # Create cancel flag checker
    def check_cancelled():
        with task_lock:
            return tasks.get(task_id, {}).get('cancelled', False)

    # Add callback and cancel checker to kwargs
    kwargs['progress_callback'] = progress_callback
    kwargs['cancel_check'] = check_cancelled

    # Run using the original run_task function
    run_task(task_id, func, *args, **kwargs)


def run_task(task_id: str, func, *args, **kwargs):
    """Run a function in a background thread and update task status."""
    # Set up output capture to also send to callback
    class OutputCaptureWithCallback:
        def __init__(self, callback=None):
            self.callback = callback

        def write(self, text):
            try:
                if text is not None and text.strip():
                    text_str = str(text) if not isinstance(text, str) else text
                    # Send to callback if available
                    if self.callback:
                        self.callback(text_str.rstrip())
            except:
                pass  # Ignore write errors

        def flush(self):
            pass

    # Get progress_callback from kwargs if available, but don't pop it
    progress_callback = kwargs.get('progress_callback', None)

    # Set up output capture with callback
    capture = OutputCaptureWithCallback(progress_callback)

    # Redirect stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
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

    # Start background task - pass task_id directly
    thread = threading.Thread(
        target=run_task_with_callback,
        args=(task_id, _process_video, url, local_file, model, resolution)
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
        args=(task_id, _regenerate_themes, folder_number, model)
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
    """Cancel a background task by terminating subprocesses and cleaning up files."""
    import subprocess
    import signal
    import shutil
    import time

    url = ''
    with task_lock:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404

        if tasks[task_id]['status'] == 'running':
            # Set cancelled flag and get URL
            tasks[task_id]['cancelled'] = True
            tasks[task_id]['status'] = 'cancelled'
            url = tasks[task_id].get('url', '')

    # Try to kill yt-dlp and whisper processes
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

    # Start background task
    thread = threading.Thread(
        target=run_edit_task,
        args=(edit_id, str(input_video), str(output_video), edit_settings)
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
        processor.apply_effects(output_video, edit_settings, cancel_flag=lambda: edit_processes.get(edit_id, {}).get('cancelled', False), progress_callback=log_message)

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



if __name__ == '__main__':
    print("=" * 60)
    print("YouTube Shorts Creator - Web Server")
    print("=" * 60)
    print(f"Server running at: http://localhost:5000")
    print(f"Video directory: {settings.get('video', 'output_dir')}")
    print("=" * 60)

    app.run(host='127.0.0.1', port=5000, debug=True)
