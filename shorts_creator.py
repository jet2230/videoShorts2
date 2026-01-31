#!/usr/bin/env python3
"""
Automatic YouTube Shorts Creation App
Downloads videos, generates subtitles, and identifies themes for shorts.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import yt_dlp
import whisper


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


def write_vtt(segments, file):
    """Write segments to VTT format."""
    file.write("WEBVTT\n\n")

    for i, segment in enumerate(segments, start=1):
        # Format timestamps: 00:00:00.000
        start = format_timestamp(segment['start'], vtt=True)
        end = format_timestamp(segment['end'], vtt=True)
        text = segment['text'].strip()

        file.write(f"{i}\n")
        file.write(f"{start} --> {end}\n")
        file.write(f"{text}\n\n")


def format_timestamp(seconds: float, vtt: bool = False) -> str:
    """Format seconds to timestamp string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)

    separator = '.' if vtt else ','
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


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
        """Get the next available folder number."""
        existing = [d for d in self.base_dir.iterdir() if d.is_dir()]
        if not existing:
            return 1

        numbers = []
        for folder in existing:
            match = re.match(r'(\d+)_', folder.name)
            if match:
                numbers.append(int(match.group(1)))

        return max(numbers) + 1 if numbers else 1

    def download_video(self, url: str) -> Dict[str, str]:
        """Download YouTube video in highest quality."""
        print(f"Downloading video from: {url}")

        # Get video info first
        ydl_opts_info = {
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'unknown_title')
            sanitized_title = self.sanitize_title(title)

        # Create folder
        folder_num = self.get_next_folder_number()
        folder_name = f"{folder_num:03d}_{sanitized_title}"
        output_folder = self.base_dir / folder_name
        output_folder.mkdir(exist_ok=True)

        print(f"Created folder: {output_folder}")

        # Download video
        ydl_opts_download = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(output_folder / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
            ydl.download([url])

        # Find downloaded video file
        video_files = list(output_folder.glob("*.mp4"))
        video_path = video_files[0] if video_files else None

        return {
            'folder': str(output_folder),
            'title': title,
            'sanitized_title': sanitized_title,
            'video_path': str(video_path) if video_path else None,
            'url': url,
            'folder_number': folder_num
        }

    def create_video_info(self, video_info: Dict[str, str]) -> None:
        """Create video info.txt file."""
        info_path = Path(video_info['folder']) / 'video info.txt'

        content = f"""Video Information
==================
Title: {video_info['title']}
Download URL: {video_info['url']}
Folder: {video_info['folder']}
Folder Number: {video_info['folder_number']}
Video Path: {video_info['video_path']}
"""

        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Created video info file: {info_path}")

    def generate_subtitles(self, video_info: Dict[str, str], model_size: str = 'base') -> str:
        """Generate subtitles using Whisper."""
        print(f"Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)

        video_path = video_info['video_path']
        print(f"Transcribing: {video_path}")

        # Transcribe with transliteration support
        result = model.transcribe(
            video_path,
            task='transcribe',
            language=None,  # Auto-detect
            verbose=False
        )

        # Save subtitles in multiple formats
        base_name = Path(video_path).stem
        srt_path = Path(video_info['folder']) / f"{base_name}.srt"
        vtt_path = Path(video_info['folder']) / f"{base_name}.vtt"
        txt_path = Path(video_info['folder']) / f"{base_name}_subtitles.txt"

        # Save SRT
        with open(srt_path, 'w', encoding='utf-8') as f:
            write_srt(result['segments'], f)

        # Save VTT
        with open(vtt_path, 'w', encoding='utf-8') as f:
            write_vtt(result['segments'], f)

        # Save plain text
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])

        print(f"Created subtitles: {srt_path}, {vtt_path}, {txt_path}")

        return str(txt_path)

    def generate_themes(self, video_info: Dict[str, str]) -> None:
        """Generate themes for YouTube Shorts from subtitles."""
        subtitle_file = Path(video_info['folder']) / f"{Path(video_info['video_path']).stem}_subtitles.txt"

        if not subtitle_file.exists():
            print("No subtitle file found, skipping theme generation.")
            return

        print("Generating themes for shorts...")

        with open(subtitle_file, 'r', encoding='utf-8') as f:
            transcript = f.read()

        # Parse subtitle segments from SRT file for timing
        srt_file = subtitle_file.with_suffix('.srt')
        segments = self._parse_srt_segments(srt_file)

        # Identify potential themes
        themes = self._identify_themes(segments, transcript)

        # Save themes to markdown file
        themes_path = Path(video_info['folder']) / 'themes.md'
        with open(themes_path, 'w', encoding='utf-8') as f:
            f.write(f"# Themes for YouTube Shorts\n\n")
            f.write(f"**Video:** {video_info['title']}\n\n")
            f.write(f"**Total Duration:** {segments[-1]['end'] if segments else 'N/A'}\n\n")
            f.write(f"---\n\n")
            f.write(f"## Identified Themes\n\n")

            if themes:
                for i, theme in enumerate(themes, 1):
                    f.write(f"### Theme {i}: {theme['title']}\n\n")
                    f.write(f"**Time Range:** {theme['start']} - {theme['end']} ({theme['duration']})\n\n")
                    f.write(f"**Why this works:** {theme['reason']}\n\n")
                    f.write(f"**Transcript Preview:**\n```\n{theme['text']}\n```\n\n")
                    f.write(f"---\n\n")
            else:
                f.write("No clear themes identified. The video may need manual review.\n")

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
        min_duration = 30  # 30 seconds
        max_duration = 240  # 4 minutes

        # Strategy: Find segments with interesting content
        # Look for: questions, emotional statements, key insights, stories

        # Combine consecutive segments to create themes
        current_theme = None
        current_segments = []

        for i, seg in enumerate(segments):
            text = seg['text'].lower()
            duration = seg['end'] - seg['start']

            # Check if this segment could start a good theme
            starters = [
                'here', 'so', 'basically', 'the thing is',
                'let me tell', 'story', 'imagine',
                'important', 'remember', 'basically',
                'actually', 'in fact', 'believe'
            ]

            # Check for interesting content
            has_question = '?' in seg['text']
            has_exclamation = '!' in seg['text']
            has_starter = any(starter in text for starter in starters)

            if has_question or has_exclamation or has_starter or not current_theme:
                if not current_theme:
                    current_theme = {
                        'start': self._format_timestamp(seg['start']),
                        'end_raw': seg['start'],
                        'text': seg['text']
                    }
                    current_segments = [seg]
                else:
                    current_segments.append(seg)
                    current_theme['text'] += ' ' + seg['text']
                    current_theme['end_raw'] = seg['end']
            else:
                # End current theme if it's long enough
                if current_theme:
                    duration = current_theme['end_raw'] - self._parse_timestamp(current_theme['start'])
                    current_theme['end'] = self._format_timestamp(current_theme['end_raw'])
                    current_theme['duration'] = self._format_duration(duration)

                    if min_duration <= duration <= max_duration:
                        current_theme['title'] = self._generate_theme_title(current_theme['text'])
                        current_theme['reason'] = self._get_theme_reason(current_theme['text'])
                        themes.append(current_theme)

                    current_theme = None
                    current_segments = []

        # Add any remaining theme
        if current_theme:
            duration = current_theme['end_raw'] - self._parse_timestamp(current_theme['start'])
            current_theme['end'] = self._format_timestamp(current_theme['end_raw'])
            current_theme['duration'] = self._format_duration(duration)

            if min_duration <= duration <= max_duration:
                current_theme['title'] = self._generate_theme_title(current_theme['text'])
                current_theme['reason'] = self._get_theme_reason(current_theme['text'])
                themes.append(current_theme)

        # Limit to top themes
        return themes[:10]

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
        """Generate a title for a theme based on its content."""
        # Take first meaningful sentence or phrase
        words = text.split()
        if len(words) <= 8:
            return text[:50] + '...' if len(text) > 50 else text
        else:
            return ' '.join(words[:8]) + '...'

    def _get_theme_reason(self, text: str) -> str:
        """Generate a reason for why this theme works as a short."""
        text_lower = text.lower()

        if '?' in text:
            return "Contains a question that can hook viewers"
        elif any(word in text_lower for word in ['story', 'imagine', 'believe']):
            return "Narrative content that engages viewers"
        elif any(word in text_lower for word in ['important', 'remember', 'key', 'secret']):
            return "Promises valuable information"
        elif '!' in text:
            return "Has emotional impact or excitement"
        else:
            return "Self-contained segment with clear message"

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
        self.generate_themes(video_info)

        print("=" * 60)
        print(f"Processing complete! Folder: {video_info['folder']}")
        print("=" * 60)

        return video_info


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Automatic YouTube Shorts Creation App'
    )
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument(
        '--model',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='base',
        help='Whisper model size (default: base)'
    )
    parser.add_argument(
        '--output-dir',
        default='videos',
        help='Base directory for downloaded videos (default: videos)'
    )

    args = parser.parse_args()

    creator = YouTubeShortsCreator(base_dir=args.output_dir)
    creator.process_video(args.url, whisper_model=args.model)


if __name__ == '__main__':
    main()
