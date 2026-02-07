#!/usr/bin/env python3
"""
Automatic YouTube Shorts Creation App
Downloads videos, generates subtitles, and identifies themes for shorts.
"""

import os
import re
import json
import argparse
import subprocess
import gc
import pty
import configparser
from pathlib import Path
from typing import Dict, List, Optional
import yt_dlp

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def load_settings(settings_file: str = 'settings.ini') -> configparser.ConfigParser:
    """Load settings from settings.ini file."""
    config = configparser.ConfigParser()

    if Path(settings_file).exists():
        config.read(settings_file)
        return config
    else:
        # Create default settings if file doesn't exist
        config['whisper'] = {
            'model': 'small',
            'language': 'ar',
            'task': 'transcribe'
        }
        config['video'] = {
            'output_dir': 'videos',
            'aspect_ratio': '9:16',
            'resolution_width': '1080',
            'resolution_height': '1920',
            'codec': 'libx264',
            'preset': 'medium',
            'crf': '23'
        }
        config['subtitle'] = {
            'font_name': 'Arial',
            'font_size': '24',
            'primary_colour': '&HFFFFFF',
            'back_colour': '&H80000000',
            'outline_colour': '&H00000000',
            'alignment': '2',
            'margin_v': '35'
        }
        config['theme'] = {
            'min_duration': '30',
            'max_duration': '240',
            'ai_enabled': 'true',
            'ai_model': 'llama3',
            'ai_provider': 'ollama',
            'window_duration': '600',
            'window_overlap': '120'
        }
        config['folder'] = {
            'naming_scheme': 'numbered',
            'number_padding': '3'
        }

        # Save default settings
        with open(settings_file, 'w') as f:
            config.write(f)

        return config


# Load settings at module level
settings = load_settings()


def write_srt(segments, file):
    """Write segments to SRT format."""
    for i, segment in enumerate(segments, start=1):
        # Format timestamps: 00:00:00,000
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip()

        file.write(f"{i}\n")
        file.write(f"{start} --> {end}\n")
        file.write(f"{text}\n\n")


def format_timestamp(seconds: float) -> str:
    """Format seconds to timestamp string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


class YouTubeShortsCreator:
    """Main class for creating YouTube shorts from long-form videos."""

    def __init__(self, base_dir: str = "videos"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def sanitize_title(self, title: str) -> str:
        """Sanitize video title for use as folder name."""
        # Remove invalid characters
        title = re.sub(r'[<>:"/\\|?*]', '', title)
        # Replace multiple spaces with single space
        title = re.sub(r'\s+', '_', title)
        # Limit length
        return title[:100]

    def get_next_folder_number(self) -> int:
        """Get the next folder number."""
        existing = [d for d in self.base_dir.iterdir() if d.is_dir()]
        if not existing:
            return 1

        numbers = []
        for folder in existing:
            match = re.match(r'(\d+)_', folder.name)
            if match:
                # Only count folders that actually have video files
                folder_path = self.base_dir / folder.name
                has_video = any(folder_path.glob("*.mp4")) or any(folder_path.glob("*.mkv")) or any(folder_path.glob("*.webm"))
                if has_video:
                    numbers.append(int(match.group(1)))

        return max(numbers) + 1 if numbers else 1

    def download_video(self, url: str, resolution: str = 'best', progress_callback=None) -> Dict[str, str]:
        """Download YouTube video in specified quality."""
        def _log_msg(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        _log_msg(f"Downloading video from: {url} (resolution: {resolution})")

        # Get video info first
        ydl_opts_info = {
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'unknown_title')
            sanitized_title = self.sanitize_title(title)

        # Check if folder with this title already exists
        existing_folder = None
        for folder in self.base_dir.iterdir():
            if folder.is_dir() and folder.name.endswith(sanitized_title):
                # Found existing folder with same title
                existing_folder = folder
                break

        if existing_folder:
            # Reuse existing folder
            output_folder = existing_folder
            # Extract folder number from existing folder name
            folder_num = int(existing_folder.name.split('_')[0])
            _log_msg(f"Reusing existing folder: {output_folder}")
        else:
            # Create new folder
            folder_num = self.get_next_folder_number()
            folder_name = f"{folder_num:03d}_{sanitized_title}"
            output_folder = self.base_dir / folder_name
            output_folder.mkdir(exist_ok=True)
            _log_msg(f"Created folder: {output_folder}")

        # Map resolution to yt-dlp format string
        format_map = {
            'best': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '1080': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            '720': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]',
            '480': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[height<=480]',
            '360': 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=360]+bestaudio/best[height<=360]',
        }

        # Download video
        def ydl_progress_hook(d):
            """Progress hook for yt-dlp to log download progress."""
            if d['status'] == 'downloading':
                downloaded = d.get('downloaded_bytes', 0)
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                if total > 0:
                    percent = (downloaded / total) * 100
                    _log_msg(f"[download] {percent:.1f}%")
            elif d['status'] == 'finished':
                _log_msg("[download] 100%")

        ydl_opts_download = {
            'format': format_map.get(resolution, format_map['best']),
            'outtmpl': str(output_folder / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [ydl_progress_hook]
        }

        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
            ydl.download([url])

        # Find downloaded video file
        video_files = list(output_folder.glob("*.mp4"))
        video_path = video_files[0] if video_files else None

        # Optimize video for seeking (faststart)
        if video_path:
            _log_msg("Optimizing video for streaming (faststart)...")
            try:
                optimized_path = output_folder / f"{video_path.stem}_temp.mp4"
                result = subprocess.run([
                    'ffmpeg', '-i', str(video_path),
                    '-c', 'copy',  # Copy streams without re-encoding (fast)
                    '-movflags', 'faststart',
                    str(optimized_path)
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    # Replace original with optimized
                    video_path.unlink()
                    optimized_path.rename(video_path)
                    _log_msg("Video optimization complete")
                else:
                    _log_msg(f"Warning: Optimization failed - {result.stderr}")
                    # Clean up temp file if it exists
                    if optimized_path.exists():
                        optimized_path.unlink()
            except FileNotFoundError:
                _log_msg("Warning: ffmpeg not found, skipping optimization")
            except Exception as e:
                _log_msg(f"Warning: Optimization failed - {e}")

        return {
            'folder': str(output_folder),
            'title': title,
            'sanitized_title': sanitized_title,
            'video_path': str(video_path) if video_path else None,
            'url': url,
            'folder_number': folder_num
        }

    def process_local_video(self, video_path: str, progress_callback=None) -> Dict[str, str]:
        """Process a local video file by copying it to the project structure."""
        def _log_msg(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        import shutil

        video_file = Path(video_path).resolve()
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Check if video is already in the correct folder structure (videos/XXX_name/)
        parent_folder = video_file.parent
        folder_name = parent_folder.name

        # Check if parent is the base_dir and folder matches the pattern XXX_name
        if parent_folder.parent.resolve() == self.base_dir.resolve():
            # Folder name should start with 3 digits and underscore
            import re
            if re.match(r'^\d{3}_', folder_name):
                # Already in correct structure, use existing folder
                _log_msg(f"Video already in correct folder structure: {parent_folder}")
                folder_num = int(folder_name.split('_')[0])
                filename = video_file.stem
                return {
                    'folder': str(parent_folder),
                    'title': filename,
                    'sanitized_title': self.sanitize_title(filename),
                    'video_path': str(video_file),
                    'url': 'Local video source (not downloaded from YouTube)',
                    'folder_number': folder_num,
                    'is_local': True
                }

        # Not in correct structure, create new folder
        filename = video_file.stem  # filename without extension
        sanitized_title = self.sanitize_title(filename)

        # Check if folder with this title already exists
        existing_folder = None
        for folder in self.base_dir.iterdir():
            if folder.is_dir() and folder.name.endswith(sanitized_title):
                # Found existing folder with same title
                existing_folder = folder
                break

        if existing_folder:
            # Reuse existing folder
            output_folder = existing_folder
            # Extract folder number from existing folder name
            folder_num = int(existing_folder.name.split('_')[0])
            _log_msg(f"Reusing existing folder: {output_folder}")
        else:
            # Create new folder
            folder_num = self.get_next_folder_number()
            folder_name = f"{folder_num:03d}_{sanitized_title}"
            output_folder = self.base_dir / folder_name
            output_folder.mkdir(exist_ok=True)
            _log_msg(f"Created folder: {output_folder}")

        # Copy video to project folder
        output_video_path = output_folder / video_file.name
        shutil.copy2(video_file, output_video_path)
        _log_msg(f"Copied video to: {output_video_path}")

        # Optimize video for seeking (faststart)
        _log_msg("Optimizing video for streaming (faststart)...")
        try:
            optimized_path = output_folder / f"{output_video_path.stem}_temp.mp4"
            result = subprocess.run([
                'ffmpeg', '-i', str(output_video_path),
                '-c', 'copy',  # Copy streams without re-encoding (fast)
                '-movflags', 'faststart',
                str(optimized_path)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                # Replace original with optimized
                output_video_path.unlink()
                optimized_path.rename(output_video_path)
                _log_msg("Video optimization complete")
            else:
                _log_msg(f"Warning: Optimization failed - {result.stderr}")
                # Clean up temp file if it exists
                if optimized_path.exists():
                    optimized_path.unlink()
        except FileNotFoundError:
            _log_msg("Warning: ffmpeg not found, skipping optimization")
        except Exception as e:
            _log_msg(f"Warning: Optimization failed - {e}")

        return {
            'folder': str(output_folder),
            'title': filename,
            'sanitized_title': sanitized_title,
            'video_path': str(output_video_path),
            'url': 'Local video source (not downloaded from YouTube)',
            'folder_number': folder_num,
            'is_local': True
        }

    def create_video_info(self, video_info: Dict[str, str], progress_callback=None) -> None:
        """Create video info.txt file."""
        def _log_msg(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        info_path = Path(video_info['folder']) / 'video info.txt'

        content = f"""Video Information
==================
Title: {video_info['title']}
Source: {video_info['url']}
Folder: {video_info['folder']}
Folder Number: {video_info['folder_number']}
Video Path: {video_info['video_path']}
"""

        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(content)

        _log_msg(f"Created video info file: {info_path}")

    def generate_subtitles(self, video_info: Dict[str, str], model_size: str = None, progress_callback=None) -> str:
        """Generate subtitles using Whisper CLI."""
        def _log_msg(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        # Use settings if model_size not provided
        if model_size is None:
            model_size = settings.get('whisper', 'model')

        # Get language and task from settings
        language = settings.get('whisper', 'language')
        task = settings.get('whisper', 'task')

        # Stop AI model to free GPU memory
        try:
            subprocess.run(['ollama', 'stop', 'llama3'], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            print("\033[91mStopped Ollama llama3 model\033[0m")
        except Exception:
            pass

        video_path = video_info['video_path']
        folder = video_info['folder']

        _log_msg(f"Transcribing with Whisper ({model_size})...")

        # Use Python API with simulated progress
        import whisper as whisper_module
        from whisper.utils import get_writer
        import time
        import threading

        # Load model
        _log_msg(f"Loading Whisper model ({model_size})...")
        model = whisper_module.load_model(model_size)
        _log_msg(f"Model loaded. Processing audio...")

        # Start a background thread to simulate progress
        stop_progress = threading.Event()

        def progress_simulator():
            """Simulate progress updates during transcription."""
            for i in range(0, 101, 10):
                if stop_progress.is_set():
                    break
                _log_msg(f"Progress: {i}%")
                time.sleep(1)

        progress_thread = threading.Thread(target=progress_simulator)
        progress_thread.start()

        try:
            # Transcribe
            result = model.transcribe(
                str(video_path),
                language=language,
                task=task,
                verbose=False
            )
        finally:
            stop_progress.set()
            progress_thread.join()
            _log_msg("Progress: 100%")

        # Save SRT file
        base_name = Path(video_path).stem
        srt_path = Path(folder) / f"{base_name}.srt"

        # Use get_writer to write SRT
        writer = get_writer('srt', str(folder))
        writer(result, srt_path)

        if srt_path.exists():
            _log_msg(f"Created subtitles: {srt_path}")
            _log_msg("Subtitles: Complete")
        else:
            raise FileNotFoundError(f"Whisper CLI did not create expected subtitle file: {srt_path}")

        return str(srt_path)

    def generate_themes(self, video_info: Dict[str, str], ai_generator=None, model_size='base', progress_callback=None) -> None:
        """Generate themes for YouTube Shorts from subtitles."""
        def _log_msg(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        # Read from SRT file for both transcript and timing
        srt_file = Path(video_info['folder']) / f"{Path(video_info['video_path']).stem}.srt"

        if not srt_file.exists():
            _log_msg("No subtitle file found, skipping theme generation.")
            return

        _log_msg("Generating themes for shorts...")
        _log_msg("Progress: 0%")

        # Parse SRT to get segments with timing
        segments = self._parse_srt_segments(srt_file)

        # Extract transcript text from segments
        transcript = ' '.join(seg['text'] for seg in segments)

        # Identify themes - use AI for boundary detection if available, otherwise pattern-based
        ai_used = False
        ai_model = None
        themes = []

        if ai_generator and ai_generator.is_available():
            _log_msg(f"  Using AI ({ai_generator.model}) for theme identification...")
            ai_used = True
            ai_model = ai_generator.model

            # Use AI to identify theme boundaries
            ai_themes = ai_generator.identify_theme_boundaries(segments)

            if ai_themes:
                _log_msg(f"    AI identified {len(ai_themes)} themes")
                # Convert AI themes to standard format
                for ai_theme in ai_themes:
                    # Find segments within this time range
                    theme_segments = [
                        s for s in segments
                        if s['start'] >= ai_theme['start_time'] and s['end'] <= ai_theme['end_time']
                    ]

                    if theme_segments:
                        combined_text = ' '.join(s['text'] for s in theme_segments)
                        themes.append({
                            'start': self._format_timestamp(ai_theme['start_time']),
                            'end': self._format_timestamp(ai_theme['end_time']),
                            'duration': self._format_duration(ai_theme['duration']),
                            'text': combined_text[:800],
                            'type': 'ai'
                        })

                # Use AI to generate titles for AI-identified themes
                total_themes = len(themes)
                for idx, theme in enumerate(themes):
                    progress = int((idx + 1) / total_themes * 100)
                    _log_msg(f"Progress: {progress}%")
                    ai_title = ai_generator.generate_title(theme['text'], theme['duration'])
                    if ai_title:
                        theme['title'] = ai_title
                        _log_msg(f"    Theme {idx + 1}: {ai_title[:50]}...")
                    else:
                        theme['title'] = self._generate_theme_title(theme['text'])

                    # Generate reason
                    ai_reason = ai_generator.generate_reason(theme['text'], theme['title'])
                    if ai_reason:
                        theme['reason'] = ai_reason
                    else:
                        theme['reason'] = self._get_theme_reason(theme['text'])

        # Fallback to pattern-based if AI didn't return themes
        if not themes:
            if ai_used:
                _log_msg("    AI boundary detection failed, using pattern-based approach")

            themes = self._identify_themes(segments, transcript)

            # Use AI to generate better titles if available
            if ai_generator and ai_generator.is_available():
                total_themes = len(themes)
                for idx, theme in enumerate(themes):
                    progress = int((idx + 1) / total_themes * 100)
                    _log_msg(f"Progress: {progress}%")
                    ai_title = ai_generator.generate_title(theme['text'], theme['duration'])
                    if ai_title:
                        theme['title'] = ai_title
                        _log_msg(f"    Theme {idx + 1}: {ai_title[:50]}...")

                    # Also try to generate a better reason
                    ai_reason = ai_generator.generate_reason(theme['text'], theme['title'])
                    if ai_reason:
                        theme['reason'] = ai_reason
                    else:
                        theme['reason'] = self._get_theme_reason(theme['text'])
            else:
                # Add pattern-based titles and reasons
                for theme in themes:
                    theme['title'] = self._generate_theme_title(theme['text'])
                    theme['reason'] = self._get_theme_reason(theme['text'])

        # Save themes to markdown file
        themes_path = Path(video_info['folder']) / 'themes.md'
        with open(themes_path, 'w', encoding='utf-8') as f:
            f.write(f"# Themes for YouTube Shorts\n\n")
            f.write(f"**Video:** {video_info['title']}\n\n")
            total_duration = self._format_duration(segments[-1]['end']) if segments else 'N/A'
            f.write(f"**Total Duration:** {total_duration}\n\n")
            f.write(f"**Number of Themes:** {len(themes)}\n\n")
            f.write(f"**Whisper Model:** {model_size}\n\n")

            # Indicate whether AI was used for title generation
            if ai_used:
                f.write(f"**Title Generation:** AI-powered ({ai_model})\n\n")
            else:
                f.write(f"**Title Generation:** Pattern-based\n\n")

            f.write(f"---\n\n")

            if themes:
                # Summary table at the top
                f.write(f"## Theme Summary\n\n")
                f.write(f"| # | Theme | Duration | Time Range |\n")
                f.write(f"|---|-------|----------|------------|\n")
                for i, theme in enumerate(themes, 1):
                    f.write(f"| {i} | {theme['title']} | {theme['duration']} | {theme['start']} - {theme['end']} |\n")
                f.write(f"\n---\n\n")

                # Detailed descriptions
                f.write(f"## Detailed Theme Descriptions\n\n")
                for i, theme in enumerate(themes, 1):
                    f.write(f"### Theme {i}: {theme['title']}\n\n")
                    f.write(f"**Time Range:** {theme['start']} - {theme['end']} ({theme['duration']})\n\n")
                    f.write(f"**Why this works:** {theme['reason']}\n\n")
                    f.write(f"**Transcript Preview:**\n```\n{theme['text']}\n```\n\n")
                    f.write(f"---\n\n")
            else:
                f.write("No clear themes identified. The video may need manual review.\n")

        _log_msg("Progress: 100%")
        print(f"Created themes file: {themes_path}")

    def _parse_srt_segments(self, srt_path: Path) -> List[Dict]:
        """Parse SRT file to get segments with timing."""
        segments = []

        with open(srt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for timestamp lines (format: 00:00:00,000 --> 00:00:00,000)
            if '-->' in line:
                times = line.split('-->')
                start_time = self._parse_timestamp(times[0].strip())
                end_time = self._parse_timestamp(times[1].strip().split()[0])

                # Get text for this segment
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip() and not '-->' in lines[i]:
                    text_lines.append(lines[i].strip())
                    i += 1

                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': ' '.join(text_lines)
                })
            else:
                i += 1

        return segments

    def _parse_timestamp(self, timestamp: str) -> float:
        """Parse SRT timestamp to seconds."""
        # Format: 00:00:00,000
        time_part = timestamp.replace(',', '.')
        h, m, s = time_part.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def _identify_themes(self, segments: List[Dict], transcript: str) -> List[Dict]:
        """Identify potential themes for shorts from segments."""
        themes = []
        min_duration = 20  # 20 seconds
        max_duration = 240  # 4 minutes

        # Strategy 1: Find numbered list items (very common in lectures)
        number_patterns = [
            r'(?:number|no\.|#)\s*(\d+)',
            r'the\s+(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)',
            r'(\d+)\s*(?:st|nd|rd|th)\s*(?:point|thing|reason|way|one)',
        ]

        # Strategy 2: Find question patterns
        question_indicators = [
            r'how often',
            r'do you (?:see|know|think|feel)',
            r'what (?:is|was|does|did)',
            r'why (?:is|was|does|did)',
            r'ask yourself'
        ]

        # Strategy 3: Find story/narrative patterns
        story_indicators = [
            r'(?:he|she|they) said',
            r'the story of',
            r'for example',
            r'let me (?:tell|share)',
        ]

        # Strategy 4: Find "reasons why" patterns
        reason_patterns = [
            r'(?:reason|cause).*because',
            r'why.*because',
            r'the reason is',
        ]

        import re

        # First pass: identify all potential theme boundaries
        theme_boundaries = []
        for i, seg in enumerate(segments):
            text_lower = seg['text'].lower()
            text = seg['text']

            # Check for numbered list patterns
            for pattern in number_patterns:
                if re.search(pattern, text_lower):
                    theme_boundaries.append((i, 'number', seg))
                    break

            # Check for question patterns
            for pattern in question_indicators:
                if re.search(pattern, text_lower):
                    theme_boundaries.append((i, 'question', seg))
                    break

            # Check for story patterns
            for pattern in story_indicators:
                if re.search(pattern, text_lower):
                    theme_boundaries.append((i, 'story', seg))
                    break

            # Check for reason patterns
            for pattern in reason_patterns:
                if re.search(pattern, text_lower):
                    theme_boundaries.append((i, 'reason', seg))
                    break

            # Check for questions or exclamations
            if '?' in text or '!' in text:
                theme_boundaries.append((i, 'emotion', seg))

        # If we found themes at boundaries, create them
        if len(theme_boundaries) >= 3:
            # Sort boundaries by segment index
            theme_boundaries.sort(key=lambda x: x[0])

            # Create themes from consecutive boundary points
            last_end_idx = 0
            for j in range(len(theme_boundaries) - 1):
                start_idx = theme_boundaries[j][0]
                end_idx = theme_boundaries[j + 1][0]

                # Ensure no overlap: start from last_end_idx if available
                if start_idx < last_end_idx:
                    start_idx = last_end_idx

                # Find natural end point to avoid cutting off mid-sentence
                end_idx = self.find_natural_end_point(segments, end_idx)

                # Combine segments from start_idx to end_idx
                theme_segments = segments[start_idx:end_idx + 1]
                if not theme_segments:
                    continue

                start_time = theme_segments[0]['start']
                end_time = theme_segments[-1]['end']
                duration = end_time - start_time

                if min_duration <= duration <= max_duration:
                    combined_text = ' '.join(s['text'] for s in theme_segments)

                    themes.append({
                        'start': self._format_timestamp(start_time),
                        'end': self._format_timestamp(end_time),
                        'duration': self._format_duration(duration),
                        'text': combined_text[:800],  # Limit text length
                        'type': theme_boundaries[j][1]
                    })

                    # Update last_end_idx to prevent overlap
                    last_end_idx = end_idx + 1

            # Try to get up to 15 themes
            themes = themes[:15]

        # Fallback strategy: if not enough themes found, use sliding window
        if len(themes) < 3:
            themes = []
            window_size = int(60 / (segments[1]['start'] - segments[0]['start']) if len(segments) > 1 else 10)  # ~60 seconds

            for i in range(0, len(segments), window_size):
                end_idx = min(i + window_size, len(segments))

                # Find natural end point
                end_idx = self.find_natural_end_point(segments, end_idx)

                theme_segments = segments[i:end_idx + 1]

                if not theme_segments:
                    continue

                start_time = theme_segments[0]['start']
                end_time = theme_segments[-1]['end']
                duration = end_time - start_time

                if min_duration <= duration <= max_duration:
                    combined_text = ' '.join(s['text'] for s in theme_segments)

                    themes.append({
                        'start': self._format_timestamp(start_time),
                        'end': self._format_timestamp(end_time),
                        'duration': self._format_duration(duration),
                        'text': combined_text[:800],
                        'type': 'window'
                    })

            themes = themes[:12]

        # Add titles and reasons
        for theme in themes:
            theme['title'] = self._generate_theme_title(theme['text'])
            theme['reason'] = self._get_theme_reason(theme['text'])

        return themes[:15]

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"

    def _generate_theme_title(self, text: str) -> str:
        """Generate a meaningful title for a theme based on its content."""
        text_lower = text.lower()

        # Define topics with their keywords - returns short, catchy titles
        # Order matters: more specific patterns should come first
        topic_patterns = [
            ('The Mizan (Scales of Justice)', ['mizan', 'mawazin', 'wazan', 'scales', 'scale of justice', 'weighed', 'weighing', 'khafifatani', 'thakilatani']),
            ('Why Your Dua Isn\'t Answered', ['dua', 'answered', 'respond', 'accept', 'prayers']),
            ('Self-Reflection: Know Yourself', ['faults', 'critic', 'yourself', 'shortcomings', 'mistakes']),
            ('Grave and Death Reminder', ['grave', 'death', 'died', 'janasah', 'symmetry', 'bury']),
            ('Fearing Hellfire', ['hellfire', 'jahannam', 'fear', 'running', 'fleeing']),
            ('Paradise Requires Effort', ['paradise', 'jannah', 'striving', 'working', 'enter']),
            ('Bridge Between Knowledge and Action', ['knowledge', 'action', 'gap', 'bridge', 'practicing']),
            ('Gratitude for Blessings', ['gratitude', 'blessing', 'thankful', 'enjoy', 'appreciate']),
            ('No Silence in Prayer', ['salah', 'silence', 'prayer', 'tasbih', 'vicar', 'sujood']),
            ('Grave Questioning Explained', ['grave', 'questioning', 'questioned', 'actions', 'accompany']),
            ('Two Angels in the Grave', ['angels', 'munkar', 'nakir', 'malakan', 'come to that person']),
            ('Barzakh is Real and Physical', ['barzakh', 'physical', 'real', 'dimension', 'parallel']),
            ('Imams Warned Against Blind Following', ['imam', 'takleed', 'encouraged', 'evidences', 'shafi', 'malik']),
            ('Mental Person is Forgiven', ['majnu', 'forgiven', 'mental capacity', 'no sin']),
            ('Grave vs Dunya Rulings', ['barzakh', 'rulings', 'dunya', 'apply', 'parallel']),
            ('Hearing Footsteps After Death', ['footsteps', 'relatives', 'walk away', 'dies', 'hears']),
            ('World of the Unseen', ['unseen', 'ghaybi', 'alamul', 'cannot understand']),
            ('No Accountability Without Sanity', ['hisab', 'pen of responsibility', 'lifted', 'wajib']),
            ('Practical Sunnah to Practice', ['sunnah', 'practice', 'act upon', 'introduce', 'neglecting']),
            ('Asking Allah for Paradise', ['jannah', 'naar', 'fire', 'seek refuge', 'allahumma inni']),
            ('Siraat Bridge Over Hell', ['siraat', 'bridge', 'assirat', 'pass over', 'lightning']),
            ('What to Say in Sujood', ['sujood', 'allahumma', 'jannah', 'naar', 'remain silent']),
            ('Do Not Remain Silent in Prayer', ['silent', 'silence', 'taslim', 'remain silent']),
            ('Making Dua for Parents', ['parents', 'dua', 'making', 'father', 'mother']),
            ('Love for the Prophet', ['love the prophet', 'prophet muhammad', 'beloved prophet', 'loving prophet']),
        ]

        # Score each topic based on keyword matches
        best_topic = None
        best_score = 0

        for topic, keywords in topic_patterns:
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_topic = topic

        if best_score >= 1:  # Even 1 match is good enough
            return best_topic

        # Look for question patterns to create engaging titles
        import re
        if '?' in text:
            question_parts = text.split('?')
            for q in question_parts[:2]:
                q = q.strip()
                if len(q.split()) >= 4:
                    words = q.split()
                    # Get the key words from the question
                    meaningful_words = []
                    skip_words = {'do', 'you', 'what', 'how', 'why', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'to', 'for'}
                    for word in words:
                        w = word.lower().strip('?,.!;:')
                        if w and w not in skip_words and len(w) > 3:
                            meaningful_words.append(word)
                    if meaningful_words:
                        title = ' '.join(meaningful_words[:4]).capitalize()
                        return title + '?'

        # Try to extract from explanatory sentences
        explanatory_matches = [
            (r'don\'t ([^.]+)', 'Don\'t {}'),
            (r'never ([^.]+)', 'Never {}'),
            (r'always ([^.]+)', 'Always {}'),
            (r'this shows (?:that )?([^,]+)', 'This Shows: {}'),
        ]

        for pattern, prefix in explanatory_matches:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phrase = match.group(1).strip()
                # Clean and shorten
                words = phrase.split()
                # Remove starting filler words
                while words and words[0].lower() in ['that', 'the', 'a', 'an', 'it', 'they', 'there', 'is', 'are', 'to', 'for']:
                    words.pop(0)
                if words:
                    short_phrase = ' '.join(words[:5])
                    if prefix:
                        return prefix.format(short_phrase.capitalize())

        # Look for key concept combinations
        if 'allah' in text_lower and 'prophet' in text_lower:
            return "Allah and the Prophet's Guidance"
        if 'hadith' in text_lower or 'evidence' in text_lower:
            return "Based on Authentic Hadith"
        if 'real' in text_lower and 'physical' in text_lower:
            return "The Physical Reality"

        return "Key Islamic Teaching"

    def _clean_sentence_for_title(self, sentence: str) -> str:
        """Clean a sentence to make it suitable as a title."""
        # Remove filler phrases at the beginning
        filler_prefixes = [
            'so', 'now', 'and then', 'then', 'basically',
            'what this means is', 'the thing is', 'for example',
            'so like', 'and', 'or', 'but'
        ]

        sentence = sentence.strip()
        sentence_lower = sentence.lower()

        for prefix in filler_prefixes:
            if sentence_lower.startswith(prefix + ' '):
                sentence = sentence[len(prefix):].strip()
                sentence_lower = sentence.lower()

        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        return sentence

    def _extract_title_from_sentence(self, sentence: str, topic: str) -> str:
        """Extract a concise title from a sentence about a specific topic."""
        # Remove filler phrases at the start
        sentence = sentence.strip()
        prefixes_to_remove = [
            'so,', 'now,', 'and then', 'then,', 'basically,',
            'what this means is', 'the thing is', 'for example'
        ]

        sentence_lower = sentence.lower()
        for prefix in prefixes_to_remove:
            if sentence_lower.startswith(prefix):
                sentence = sentence[len(prefix):].strip()

        # Limit length
        if len(sentence) > 60:
            sentence = sentence[:57] + '...'

        return sentence.capitalize()

    def _get_theme_reason(self, text: str) -> str:
        """Generate a reason for why this theme works as a short."""
        text_lower = text.lower()

        # Check for different engagement factors
        if '?' in text:
            return "Contains a thought-provoking question"
        elif any(word in text_lower for word in ['story', 'imagine', 'believe']):
            return "Narrative that engages viewers emotionally"
        elif any(word in text_lower for word in ['important', 'remember', 'key', 'secret', 'lesson']):
            return "Shares valuable knowledge viewers want to save"
        elif any(word in text_lower for word in ['proof', 'evidence', 'hadith', 'verse']):
            return "Provides scriptural evidence that adds credibility"
        elif any(word in text_lower for word in ['mistake', 'wrong', 'not permitted', 'should not']):
            return "Addresses common misconceptions - high value content"
        elif '!' in text:
            return "Emotional delivery that creates impact"
        elif any(word in text_lower for word in ['so', 'therefore', 'thus', 'this shows']):
            return "Clear logical conclusion that's satisfying"
        elif any(word in text_lower for word in ['allah', 'prophet', 'muhammad', 'god']):
            return "Spiritual content that resonates with the audience"
        else:
            return "Self-contained segment with a clear message"

    def process_video(self, url: str, whisper_model: str = 'base') -> Dict[str, str]:
        """Complete pipeline: download, subtitle, and theme generation."""
        print("=" * 60)
        print("YouTube Shorts Creator - Processing Pipeline")
        print("=" * 60)

        # Step 1: Download
        video_info = self.download_video(url)

        # Step 2: Create info file
        self.create_video_info(video_info)

        # Step 3: Generate subtitles
        self.generate_subtitles(video_info, model_size=whisper_model)

        # Step 4: Generate themes
        self.generate_themes(video_info, model_size=whisper_model)

        print("=" * 60)
        print(f"Processing complete! Folder: {video_info['folder']}")
        print("=" * 60)

        return video_info

    def get_video_folder_by_number(self, folder_number: str) -> Optional[Path]:
        """Find a video folder by its number (e.g., '001')."""
        for folder in self.base_dir.iterdir():
            if folder.is_dir() and folder.name.startswith(f"{folder_number}_"):
                # Check if it has a video file
                if any(folder.glob("*.mp4")) or any(folder.glob("*.mkv")) or any(folder.glob("*.webm")):
                    return folder
        return None

    def get_video_file(self, folder: Path) -> Optional[Path]:
        """Get the video file from a folder."""
        for ext in ['*.mp4', '*.mkv', '*.webm']:
            videos = list(folder.glob(ext))
            if videos:
                return videos[0]
        return None

    def parse_themes_file(self, themes_file: Path) -> List[Dict]:
        """Parse themes.md file to extract theme information."""
        if not themes_file.exists():
            raise FileNotFoundError(f"Themes file not found: {themes_file}")

        themes = []
        with open(themes_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for theme markers: "### Theme N: Title"
            if line.startswith('### Theme'):
                # Extract theme number and title
                match = re.match(r'### Theme (\d+):\s*(.+)', line)
                if match:
                    theme_num = int(match.group(1))
                    title = match.group(2)

                    # Look for time range in next few lines
                    i += 1
                    time_range = None
                    while i < len(lines) and not lines[i].startswith('**Why this works:**'):
                        if '**Time Range:**' in lines[i]:
                            time_match = re.search(r'(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})', lines[i])
                            if time_match:
                                time_range = {
                                    'start': time_match.group(1),
                                    'end': time_match.group(2)
                                }
                                break
                        i += 1

                    if time_range:
                        themes.append({
                            'number': theme_num,
                            'title': title,
                            'start': time_range['start'],
                            'end': time_range['end']
                        })
            i += 1

        return themes

    def parse_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp (HH:MM:SS) to seconds."""
        parts = timestamp.split(':')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s

    def seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def find_natural_end_point(self, segments: List[Dict], end_idx: int, max_lookahead: int = 30) -> int:
        """Find a natural stopping point (sentence ending) near the end index.

        Looks for segments ending with periods, complete sentences, or natural pauses.
        """
        # Look ahead from end_idx for a natural stopping point
        for i in range(end_idx, min(end_idx + max_lookahead, len(segments))):
            text = segments[i]['text'].strip()

            # Check for sentence ending patterns
            if text.endswith('.') or text.endswith('!') or text.endswith('?'):
                return i

            # Check for Arabic period (U+06D4)
            if text.endswith('۔') or text.endswith('।'):
                return i

            # Check for common ending phrases in Islamic content
            lower_text = text.lower()
            ending_phrases = [
                'na\'udhu billah',
                'subhanallah',
                'alhamdulillah',
                'wallahu \'alam',
                'wasallam',
                'rahimullah',
                'this is with regards to',
                'so again',
                'and then',
                'allah subhanahu',
                'sallallahu alayhi'
            ]
            for phrase in ending_phrases:
                if phrase in lower_text:
                    return i

        # If no natural ending found, return original end_idx
        return end_idx

    def _apply_bidi_to_ass(self, text: str) -> str:
        """For Arabic text in ASS - NO REVERSAL NEEDED.

        ASS/libass handles RTL (right-to-left) text rendering automatically.
        We just need to pass the text through without modification.

        The original implementation that reversed words was incorrect - it assumed
        manual reversal was needed, but ASS format already handles Arabic RTL.
        """
        # Just return text as-is - ASS will handle RTL rendering
        return text

    def _rgb_to_ass_color(self, rgb_string: str) -> str:
        """Convert RGB string like 'rgb(255, 0, 0)' to ASS color format '&H00BBGGRR'."""
        match = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', rgb_string)
        if not match:
            return '&H00FFFFFF'  # Default white

        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        # ASS color format: &H00BBGGRR (BGR order with alpha prefix)
        return f'&H00{b:02X}{g:02X}{r:02X}'

    def _parse_html_formatting(self, html_text: str) -> str:
        """Parse HTML formatting and convert to ASS tags.

        When styling is applied to PART of an Arabic phrase, we reverse the Arabic
        word order BEFORE applying styling, so it displays correctly in RTL.
        """
        from html import unescape

        # Check if we have Arabic with partial styling
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in html_text)
        has_styling = '<span' in html_text or '<strong>' in html_text or '<b>' in html_text

        if has_arabic and has_styling:
            # Get plain text (no HTML)
            plain_text = unescape(re.sub(r'<[^>]+>', '', html_text))
            words = plain_text.split()

            # Find Arabic words
            arabic_indices = [i for i, w in enumerate(words) if any('\u0600' <= c <= '\u06FF' for c in w)]

            # Reverse Arabic words
            arabic_words_rev = [words[i] for i in arabic_indices][::-1]

            # Rebuild with reversed Arabic
            new_words = words[:]
            for idx, rev_word in zip(arabic_indices, arabic_words_rev):
                new_words[idx] = rev_word

            # The HTML tags in original now apply to the reversed positions
            # We need to map: original styled content -> new position
            # For now, simpler: work with reversed plain text and skip original HTML
            # We'll apply the styles extracted from original to the reversed text

            # Extract styles from original HTML
            styles = []
            color_match = re.search(r'<span[^>]*style="[^"]*color:\s*rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[^"]*"', html_text)
            if color_match:
                r, g, b = int(color_match.group(1)), int(color_match.group(2)), int(color_match.group(3))
                styles.append(('color', f'&H00{b:02X}{g:02X}{r:02X}'))

            size_match = re.search(r'font-size:\s*([\d.]+)em', html_text)
            if size_match:
                styles.append(('size', int(48 * float(size_match.group(1)))))

            # Check for bold/italic
            if '<strong>' in html_text or '<b>' in html_text:
                styles.append(('bold', True))

            # Apply styles to first Arabic word only (the one that was styled in original)
            if styles:
                result_parts = []
                styled_word_idx = arabic_indices[0] if arabic_indices else -1

                for i, word in enumerate(new_words):
                    word_has_arabic = any('\u0600' <= c <= '\u06FF' for c in word)

                    if i == styled_word_idx and word_has_arabic:
                        # This is the styled word - add tags + word + resets together
                        tags = []
                        for style_type, style_val in styles:
                            if style_type == 'color':
                                tags.append(f'{{\\c{style_val}}}')
                            elif style_type == 'size':
                                tags.append(f'{{\\fs{style_val}}}')
                            elif style_type == 'bold':
                                tags.append('{\\b1}')
                        # tags + word + resets all together without space
                        result_parts.append(''.join(tags) + word + '{\\r}{\\r}')
                    else:
                        result_parts.append(word)

                # Join with spaces between elements
                return ' '.join(result_parts)

        # Default: simple HTML to ASS conversion
        decoded = unescape(html_text)
        result = decoded

        # Convert <span style="color: rgb(...)"> to ASS color tags
        color_pattern = r'<span\s+style="[^"]*color:\s*rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[^"]*">(.*?)</span>'
        result = re.sub(color_pattern, lambda m: f'{{\\c&H00{int(m.group(3)):02X}{int(m.group(2)):02X}{int(m.group(1)):02X}}}{m.group(4)}{{\\r}}', result, flags=re.DOTALL)

        # Convert <strong> and <b> to ASS bold tags
        result = re.sub(r'<strong>(.*?)</strong>', r'{\\b1}\1{\\b0}', result, flags=re.DOTALL)
        result = re.sub(r'<b>(.*?)</b>', r'{\\b1}\1{\\b0}', result, flags=re.DOTALL)

        # Convert <em> and <i> to ASS italic tags
        result = re.sub(r'<em>(.*?)</em>', r'{\\i1}\1{\\i0}', result, flags=re.DOTALL)
        result = re.sub(r'<i>(.*?)</i>', r'{\\i1}\1{\\i0}', result, flags=re.DOTALL)

        # Convert font-size (e.g., "font-size: 1.2em")
        size_pattern = r'<span\s+style="[^"]*font-size:\s*([\d.]+)em[^"]*">(.*?)</span>'
        def convert_size(m):
            size_em = float(m.group(1))
            size_px = int(48 * size_em)
            return f'{{\\fs{size_px}}}{m.group(2)}{{\\r}}'
        result = re.sub(size_pattern, convert_size, result, flags=re.DOTALL)

        # Remove any remaining HTML tags
        result = re.sub(r'<[^>]+>', '', result)

        return result

    def _create_ass_file(self, trimmed_srt_path: Path, formatting_json_path: Path, output_ass_path: Path) -> bool:
        """Create ASS subtitle file from SRT and formatting JSON."""
        import json

        # Load formatting data
        with open(formatting_json_path, 'r', encoding='utf-8') as f:
            formatting_data = json.load(f)

        # Parse SRT segments
        segments = self._parse_srt_segments(trimmed_srt_path)

        # Build ASS header
        ass_header = """[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1080
PlayResY: 1920
Timer: 100.0000
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,0,1,2,1,2,10,10,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        with open(output_ass_path, 'w', encoding='utf-8') as f:
            f.write(ass_header)

            for segment in segments:
                start_time = segment['start']
                end_time = segment['end']

                # Format timestamps for ASS (H:MM:SS.CC)
                start_h = int(start_time // 3600)
                start_m = int((start_time % 3600) // 60)
                start_s = start_time % 60
                start_ass = f"{start_h}:{start_m:02d}:{start_s:05.2f}"

                end_h = int(end_time // 3600)
                end_m = int((end_time % 3600) // 60)
                end_s = end_time % 60
                end_ass = f"{end_h}:{end_m:02d}:{end_s:05.2f}"

                # Find formatting for this timestamp
                # JSON uses "HH:MM:SS.mmm" format (with dot)
                timestamp_json = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d}.{int((start_time % 1) * 1000):03d}"
                formatting = formatting_data.get(timestamp_json, {})

                # Get text - use html if available, otherwise plain text
                text = formatting.get('html', segment['text'])

                # Parse HTML formatting to ASS tags
                text = self._parse_html_formatting(text)

                # Apply Arabic RTL word reversal
                text = self._apply_bidi_to_ass(text)

                # Escape special ASS characters (but not backslashes in tags)
                # First protect ASS tags
                ass_tags = re.findall(r'{\\[^}]+}', text)
                temp_text = text
                for i, tag in enumerate(ass_tags):
                    temp_text = temp_text.replace(tag, f"__ASS_TAG_{i}__")

                # Escape special characters in the remaining text
                temp_text = temp_text.replace('{', '\\{').replace('}', '\\}')

                # Restore ASS tags
                for i, tag in enumerate(ass_tags):
                    temp_text = temp_text.replace(f"__ASS_TAG_{i}__", tag)

                text = temp_text

                # Write dialogue line
                f.write(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}\n")

        return True

    def create_trimmed_srt(self, original_srt: Path, start_seconds: float, end_seconds: float, output_path: Path) -> Path:
        """Create a trimmed SRT file with timestamps offset to start from 0."""
        segments = self._parse_srt_segments(original_srt)

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, start=1):
                # Adjust timestamps by subtracting start time
                seg_start = max(0, segment['start'] - start_seconds)
                seg_end = max(0, segment['end'] - start_seconds)

                # Only include subtitles that appear within our clip range
                if seg_end > 0 and seg_start < (end_seconds - start_seconds):
                    start_ts = self.seconds_to_timestamp(seg_start)
                    end_ts = self.seconds_to_timestamp(seg_end)
                    text = segment['text'].strip()

                    f.write(f"{i}\n")
                    f.write(f"{start_ts} --> {end_ts}\n")
                    f.write(f"{text}\n\n")

        return output_path

    def create_short(self, video_path: Path, theme: Dict, output_dir: Path, srt_path: Path, progress_callback=None) -> str:
        """Create a short video clip using ffmpeg with 9:16 aspect ratio and burnt-in subtitles."""
        output_file = output_dir / f"theme_{theme['number']:03d}_{theme['title'][:30].replace(' ', '_')}.mp4"

        # Convert timestamps to seconds
        start_seconds = self.parse_timestamp_to_seconds(theme['start'])
        end_seconds = self.parse_timestamp_to_seconds(theme['end'])
        duration = end_seconds - start_seconds

        print(f"  Creating: {output_file.name}")
        print(f"    Time range: {theme['start']} - {theme['end']} ({duration:.1f}s)")
        print(f"    Aspect ratio: 9:16 (1080x1920)")

        # Create trimmed SRT file with adjusted timestamps for the clip
        trimmed_srt_name = f"theme_{theme['number']:03d}.srt"
        trimmed_srt_path = output_dir / trimmed_srt_name
        self.create_trimmed_srt(srt_path, start_seconds, end_seconds, trimmed_srt_path)
        print(f"    Created trimmed subtitles: {trimmed_srt_name}")

        # Check if formatting JSON exists for this theme
        formatting_json_path = output_dir / f"theme_{theme['number']:03d}_formatting.json"
        subtitle_file_for_ffmpeg = str(trimmed_srt_path).replace(':', '\\:').replace('\\', '\\\\').replace("'", "\\'")
        use_ass = False

        if formatting_json_path.exists():
            print(f"    Found subtitle formatting - creating ASS file...")
            # Create ASS file from trimmed SRT and formatting JSON
            ass_output_path = output_dir / f"theme_{theme['number']:03d}.ass"
            try:
                self._create_ass_file(trimmed_srt_path, formatting_json_path, ass_output_path)
                print(f"    Created ASS subtitles: {ass_output_path.name}")
                subtitle_file_for_ffmpeg = str(ass_output_path).replace(':', '\\:').replace('\\', '\\\\').replace("'", "\\'")
                use_ass = True
            except Exception as e:
                print(f"    Warning: Failed to create ASS file, using SRT: {e}")

        # YouTube Shorts format: 9:16 aspect ratio
        # Scale and crop (not pad) to achieve proper 9:16 ratio

        print(f"    Burning subtitles...")

        # Get settings
        width = settings.get('video', 'resolution_width')
        height = settings.get('video', 'resolution_height')
        font_name = settings.get('subtitle', 'font_name')
        font_size = settings.get('subtitle', 'font_size')
        margin_v = settings.get('subtitle', 'margin_v')
        alignment = settings.get('subtitle', 'alignment')
        codec = settings.get('video', 'codec')
        preset = settings.get('video', 'preset')
        crf = settings.get('video', 'crf')

        # Filter chain:
        # Scale and crop to target resolution, then burn in subtitles
        # For ASS files, don't apply force_style (formatting is embedded)
        # For SRT files, apply force_style settings
        if use_ass:
            vf_filter = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}:(iw-ow)/2:(ih-oh)/2,subtitles={subtitle_file_for_ffmpeg}"
        else:
            vf_filter = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}:(iw-ow)/2:(ih-oh)/2,subtitles={subtitle_file_for_ffmpeg}:force_style='FontSize={font_size},MarginV={margin_v},Alignment={alignment},FontName={font_name}'"

        cmd = [
            'ffmpeg',
            '-ss', str(start_seconds),  # Input seeking
            '-i', str(video_path),
            '-t', str(duration),
            '-vf', vf_filter,
            '-c:v', codec,
            '-preset', preset,
            '-crf', crf,
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-y',
            str(output_file)
        ]

        try:
            # Run ffmpeg and capture progress in real-time from stderr
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )

            last_progress = 0
            import time

            while True:
                # Check if process is done
                if process.poll() is not None:
                    break

                # Read stderr line with timeout
                line = process.stderr.readline()
                if not line:
                    time.sleep(0.05)
                    continue

                line = line.strip()
                # Parse ffmpeg progress from stderr: frame=  123 fps= 45 ... time=00:00:05.12
                if 'time=' in line and 'frame=' in line:
                    try:
                        # Extract time= value
                        time_match = line.split('time=')[1].split()[0]
                        # Parse time format HH:MM:SS.milliseconds
                        parts = time_match.split(':')
                        if len(parts) == 3:
                            hours = int(parts[0])
                            minutes = int(parts[1])
                            seconds_parts = parts[2].split('.')
                            seconds = int(seconds_parts[0])
                            current_time = hours * 3600 + minutes * 60 + seconds

                            # Calculate percentage
                            if duration > 0:
                                percent = min(100, int((current_time / duration) * 100))
                                # Send progress on every 5% change
                                if percent >= last_progress + 5 or percent == 100:
                                    last_progress = percent
                                    # Call the callback directly if available, otherwise print
                                    if progress_callback:
                                        progress_callback(f"Progress: {percent}%")
                                    else:
                                        print(f"Progress: {percent}%")
                    except (ValueError, IndexError):
                        pass

            # Wait for process to complete
            return_code = process.wait()
            if return_code == 0:
                print(f"    ✓ Created successfully")
                return str(output_file)
            else:
                stderr = process.stderr.read()
                print(f"    ✗ Failed with code {return_code}")
                return None

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_shorts(self, folder_number: str, theme_numbers: str = None, progress_callback=None, cancel_check=None) -> None:
        """Create short video clips based on themes."""
        def _log_msg(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(f"[NO_CALLBACK] {msg}")

        _log_msg("=" * 60)
        _log_msg("Creating shorts...")
        _log_msg("Progress: 0%")

        # Find the video folder
        folder = self.get_video_folder_by_number(folder_number)
        if not folder:
            _log_msg(f"Error: Video folder '{folder_number}_' not found")
            return

        _log_msg(f"Video folder: {folder.name}")

        # Get the video file
        video_path = self.get_video_file(folder)
        if not video_path:
            _log_msg(f"Error: No video file found in {folder}")
            return

        _log_msg(f"Video file: {video_path.name}")

        # Parse themes file
        themes_file = folder / 'themes.md'
        if not themes_file.exists():
            _log_msg(f"Error: themes.md not found in {folder}")
            _log_msg("Please run the script on this video first to generate themes.")
            return

        themes = self.parse_themes_file(themes_file)
        _log_msg(f"Found {len(themes)} themes")

        # Check for adjusted theme files and update themes with adjusted values
        for theme in themes:
            adjust_file = folder / 'shorts' / f"theme_{theme['number']:03d}_adjust.md"
            if adjust_file.exists():
                try:
                    with open(adjust_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Parse adjusted values
                    import re
                    title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?:\n\n|\n\*)', content)
                    time_match = re.search(r'\*\*Time Range:\*\*\s*(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})', content)
                    if title_match:
                        theme['title'] = title_match.group(1).strip()
                    if time_match:
                        theme['start'] = time_match.group(1)
                        theme['end'] = time_match.group(2)
                    theme['adjusted'] = True
                    _log_msg(f"  Theme {theme['number']}: Using adjusted values from adjust file")
                except Exception as e:
                    _log_msg(f"  Warning: Could not read adjust file for theme {theme['number']}: {e}")

        # Determine which themes to create
        if theme_numbers == 'all':
            selected_themes = themes
        else:
            # Parse comma-separated theme numbers
            try:
                requested_nums = [int(n.strip()) for n in theme_numbers.split(',')]
                selected_themes = [t for t in themes if t['number'] in requested_nums]

                if not selected_themes:
                    _log_msg(f"Error: No matching themes found for numbers: {theme_numbers}")
                    return

                # Check for requested themes that don't exist
                found_nums = {t['number'] for t in selected_themes}
                missing_nums = set(requested_nums) - found_nums
                if missing_nums:
                    _log_msg(f"Warning: Theme(s) {sorted(missing_nums)} not found")
            except ValueError:
                _log_msg(f"Error: Invalid theme numbers format: {theme_numbers}")
                _log_msg("Use comma-separated numbers (e.g., '1,2,5') or 'all'")
                return

        total_themes = len(selected_themes)
        _log_msg(f"Creating {total_themes} short(s)...")

        # Create output directory for shorts
        shorts_dir = folder / 'shorts'
        shorts_dir.mkdir(exist_ok=True)

        # Find the original SRT subtitle file (fallback if no adjusted subs exist)
        original_srt_path = None
        for ext in ['*.srt']:
            srt_files = list(folder.glob(ext))
            if srt_files:
                original_srt_path = srt_files[0]
                break

        if original_srt_path:
            _log_msg(f"Original subtitles: {original_srt_path.name}")
        else:
            _log_msg("Warning: No SRT subtitle file found in folder")

        # Check if there are adjusted subtitle files for any of the selected themes
        has_adjusted_subs = any(
            (folder / 'shorts' / f"theme_{t['number']:03d}_adjust.srt").exists()
            for t in selected_themes
        )

        if has_adjusted_subs:
            _log_msg("Note: Some themes have adjusted subtitle files (theme_XXX_adjust.srt)")
        shorts_dir = folder / 'shorts'
        shorts_dir.mkdir(exist_ok=True)

        # Create each short
        successful = 0
        for idx, theme in enumerate(selected_themes):
            # Check for cancellation
            if cancel_check and cancel_check():
                _log_msg("Task cancelled by user")
                return

            theme_num = theme['number']
            theme_title = theme['title'][:50]

            _log_msg(f"Short {idx + 1}/{total_themes}: Theme {theme_num} - {theme_title}...")

            # Check for adjusted subtitle file for this specific theme
            theme_adjust_srt = shorts_dir / f"theme_{theme_num:03d}_adjust.srt"
            if theme_adjust_srt.exists():
                _log_msg(f"  Using adjusted subtitles: {theme_adjust_srt.name}")
                srt_path_for_theme = theme_adjust_srt
            else:
                # Use the default SRT file (already set above)
                srt_path_for_theme = original_srt_path
                if not srt_path_for_theme:
                    _log_msg(f"  Warning: No SRT file found for theme {theme_num}")

            # Create wrapper callback that adds overall progress
            def make_progress_callback(current_idx, total):
                def callback(msg):
                    if progress_callback:
                        # Check if message contains progress percentage
                        import re
                        match = re.search(r'Progress:\s*(\d+)%', msg)
                        if match:
                            short_progress = int(match.group(1))
                            # Calculate overall progress:
                            # (completed shorts * 100) + (current short progress)
                            overall_progress = int(((current_idx * 100) + short_progress) / total)
                            # Clamp to 100
                            overall_progress = min(100, overall_progress)
                            # Show both which short and overall progress
                            result = f"Progress: {overall_progress}% ({current_idx + 1}/{total} done)"
                            # Don't print here - it gets captured and causes issues!
                            progress_callback(result)
                        else:
                            # Forward other messages as-is
                            progress_callback(msg)
                return callback

            result = self.create_short(
                video_path, theme, shorts_dir, srt_path_for_theme,
                progress_callback=make_progress_callback(idx, total_themes)
            )
            if result:
                successful += 1
                _log_msg(f"  ✓ Short created successfully")
            else:
                _log_msg(f"  ✗ Failed to create short")

        _log_msg("Progress: 100% (complete)")
        _log_msg(f"Created {successful}/{total_themes} shorts in: {shorts_dir}")


def main():
    """Main entry point."""
    # Reload settings to get latest values
    global settings
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description='Automatic YouTube Shorts Creation App'
    )
    parser.add_argument('input', help='YouTube video URL, path to local video file, or video folder number (e.g., 001)')
    parser.add_argument(
        '--model',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default=settings.get('whisper', 'model'),
        help=f'Whisper model size (default: {settings.get("whisper", "model")})'
    )
    parser.add_argument(
        '--output-dir',
        default=settings.get('video', 'output_dir'),
        help=f'Base directory for downloaded videos (default: {settings.get("video", "output_dir")})'
    )
    parser.add_argument(
        '--theme',
        type=str,
        help='Create shorts for specific theme(s). Use comma-separated numbers (e.g., "1,2,5") or "all"'
    )
    parser.add_argument(
        '--regenerate-themes-only',
        action='store_true',
        help='Only regenerate themes.md from existing subtitles (skip transcription)'
    )

    args = parser.parse_args()

    creator = YouTubeShortsCreator(base_dir=args.output_dir)

    # Mode 1: Create shorts from existing video
    if args.theme:
        # Check if input is a folder number (e.g., "001", "002", etc.)
        if re.match(r'^\d{1,3}$', args.input):
            creator.create_shorts(args.input, args.theme)
        else:
            print("Error: When using --theme, the input must be a video folder number (e.g., '001')")
            print("Example: python shorts_creator.py 001 --theme=2")
        return

    # Mode 2: Process a new video (URL or local file) or regenerate themes
    # Check if regenerating themes only
    if args.regenerate_themes_only:
        # Input should be an existing folder number
        if re.match(r'^\d{1,3}$', args.input):
            matching_folders = list(Path(args.output_dir).glob(f"{args.input.zfill(3)}_*"))

            if not matching_folders:
                print(f"Error: Folder {args.input} not found")
                return

            folder = matching_folders[0]
            # Find the video file in the folder
            video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.mkv')) + list(folder.glob('*.webm'))
            if not video_files:
                print(f"Error: No video file found in {folder}")
                return

            video_path = video_files[0]

            # Check if subtitle file exists
            srt_file = folder / f"{video_path.stem}.srt"
            if not srt_file.exists():
                print(f"⚠ No subtitle file found: {srt_file.name}")
                response = input("Do you want to generate subtitles now? (y/n): ").strip().lower()
                if response == 'y':
                    video_info = {
                        'title': folder.name.split('_', 1)[1] if '_' in folder.name else folder.name,
                        'url': 'Existing video',
                        'folder': str(folder),
                        'folder_number': args.input,
                        'video_path': str(video_path),
                        'is_local': True
                    }
                    print(f"Generating subtitles for: {video_path.name}")
                    creator.generate_subtitles(video_info, model_size=args.model)
                else:
                    print("Skipping. Cannot generate themes without subtitles.")
                    return

            video_info = {
                'title': folder.name.split('_', 1)[1] if '_' in folder.name else folder.name,
                'url': 'Existing video',
                'folder': str(folder),
                'folder_number': args.input,
                'video_path': str(video_path),
                'is_local': True
            }

            print(f"Regenerating themes for: {folder.name}")

            # Initialize AI generator
            ai_generator = None
            try:
                from ai_theme_generator import AIThemeGenerator
                ai_generator = AIThemeGenerator()
                if ai_generator.is_available():
                    print(f"✓ AI enabled: Using {ai_generator.model} for theme identification")
                else:
                    ai_generator = None
            except ImportError:
                pass

            creator.generate_themes(video_info, ai_generator=ai_generator, model_size=args.model)
            print("=" * 60)
            print(f"Themes regenerated! Folder: {folder}")
            print("=" * 60)
            return
        else:
            print("Error: --regenerate-themes-only requires a folder number (e.g., '001')")
            return

    # Initialize AI generator (default)
    ai_generator = None
    try:
        from ai_theme_generator import AIThemeGenerator
        ai_generator = AIThemeGenerator()
        if ai_generator.is_available():
            print(f"✓ AI enabled: Using {ai_generator.model} for theme identification")
        else:
            print("⚠ AI not available. Using pattern-based theme identification instead.")
            ai_generator = None
    except ImportError:
        print("⚠ AI module not found. Using pattern-based theme identification instead.")
        ai_generator = None

    # Auto-detect if input is a URL or local file
    input_lower = args.input.lower()
    is_url = (
        args.input.startswith(('http://', 'https://')) or
        args.input.startswith('www.') or
        'youtube.com/' in input_lower or
        'youtu.be/' in input_lower
    )

    if is_url:
        # Download from YouTube
        # Add https:// if missing for URLs starting with www or youtube.com
        if args.input.startswith('www.'):
            args.input = 'https://' + args.input
        elif not args.input.startswith(('http://', 'https://')):
            args.input = 'https://' + args.input
        print(f"Detected URL: {args.input}")
        video_info = creator.download_video(args.input)
    else:
        # Process local video file
        print(f"Detected local file: {args.input}")
        video_info = creator.process_local_video(args.input)

    # Create info file
    creator.create_video_info(video_info)

    # Generate subtitles
    creator.generate_subtitles(video_info, model_size=args.model)

    # Generate themes (with AI if enabled)
    creator.generate_themes(video_info, ai_generator=ai_generator, model_size=args.model)

    print("=" * 60)
    print(f"Processing complete! Folder: {video_info['folder']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
