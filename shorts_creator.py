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

    def process_local_video(self, video_path: str) -> Dict[str, str]:
        """Process a local video file by copying it to the project structure."""
        import shutil

        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Use filename (without extension) as the folder name
        filename = video_file.stem  # filename without extension
        sanitized_title = self.sanitize_title(filename)

        # Create folder
        folder_num = self.get_next_folder_number()
        folder_name = f"{folder_num:03d}_{sanitized_title}"
        output_folder = self.base_dir / folder_name
        output_folder.mkdir(exist_ok=True)

        print(f"Created folder: {output_folder}")

        # Copy video to project folder
        output_video_path = output_folder / video_file.name
        shutil.copy2(video_file, output_video_path)
        print(f"Copied video to: {output_video_path}")

        return {
            'folder': str(output_folder),
            'title': filename,
            'sanitized_title': sanitized_title,
            'video_path': str(output_video_path),
            'url': 'Local video source (not downloaded from YouTube)',
            'folder_number': folder_num,
            'is_local': True
        }

    def create_video_info(self, video_info: Dict[str, str]) -> None:
        """Create video info.txt file."""
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

        print(f"Created video info file: {info_path}")

    def generate_subtitles(self, video_info: Dict[str, str], model_size: str = 'base') -> str:
        """Generate subtitles using Whisper."""
        print(f"Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)

        video_path = video_info['video_path']
        print(f"Transcribing: {video_path}")

        # Transcribe with Arabic as default language for proper Arabic script
        # Use beam search and lower temperature for better accuracy
        result = model.transcribe(
            video_path,
            task='transcribe',
            language='ar',  # Default to Arabic for proper Arabic script output
            verbose=False,
            # Improved accuracy settings
            temperature=0.0,  # Lower = more deterministic, less hallucination
            beam_size=5,  # Beam search for better accuracy
            best_of=5,  # Try multiple times and pick best
            patience=1.0,  # Beam search patience
        )

        # Save subtitles
        base_name = Path(video_path).stem
        srt_path = Path(video_info['folder']) / f"{base_name}.srt"
        txt_path = Path(video_info['folder']) / f"{base_name}_subtitles.txt"

        # Save SRT
        with open(srt_path, 'w', encoding='utf-8') as f:
            write_srt(result['segments'], f)

        # Save plain text
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])

        print(f"Created subtitles: {srt_path}, {txt_path}")

        return str(txt_path)

    def generate_themes(self, video_info: Dict[str, str], ai_generator=None) -> None:
        """Generate themes for YouTube Shorts from subtitles."""
        subtitle_file = Path(video_info['folder']) / f"{Path(video_info['video_path']).stem}_subtitles.txt"

        if not subtitle_file.exists():
            print("No subtitle file found, skipping theme generation.")
            return

        print("Generating themes for shorts...")

        with open(subtitle_file, 'r', encoding='utf-8') as f:
            transcript = f.read()

        # Parse subtitle segments from SRT file for timing
        # The SRT file is named {base_name}.srt while the text file is {base_name}_subtitles.txt
        srt_file = Path(video_info['folder']) / f"{Path(video_info['video_path']).stem}.srt"
        if not srt_file.exists():
            print(f"SRT file not found: {srt_file}, skipping theme generation.")
            return
        segments = self._parse_srt_segments(srt_file)

        # Identify themes - use AI for boundary detection if available, otherwise pattern-based
        ai_used = False
        ai_model = None
        themes = []

        if ai_generator and ai_generator.is_available():
            print(f"  Using AI ({ai_generator.model}) for theme identification...")
            ai_used = True
            ai_model = ai_generator.model

            # Use AI to identify theme boundaries
            ai_themes = ai_generator.identify_theme_boundaries(segments)

            if ai_themes:
                print(f"    AI identified {len(ai_themes)} themes")
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
                for idx, theme in enumerate(themes):
                    ai_title = ai_generator.generate_title(theme['text'], theme['duration'])
                    if ai_title:
                        theme['title'] = ai_title
                        print(f"    Theme {idx + 1}: {ai_title[:50]}...")
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
                print("    AI boundary detection failed, using pattern-based approach")

            themes = self._identify_themes(segments, transcript)

            # Use AI to generate better titles if available
            if ai_generator and ai_generator.is_available():
                for idx, theme in enumerate(themes):
                    ai_title = ai_generator.generate_title(theme['text'], theme['duration'])
                    if ai_title:
                        theme['title'] = ai_title
                        print(f"    Theme {idx + 1}: {ai_title[:50]}...")

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
        self.generate_themes(video_info)

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

    def create_trimmed_srt(self, original_srt: Path, start_seconds: float, end_seconds: float, output_path: Path) -> Path:
        """Create a trimmed SRT file with timestamps adjusted for the clip."""
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

    def create_short(self, video_path: Path, theme: Dict, output_dir: Path, srt_path: Path) -> str:
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

        # YouTube Shorts format: 1080x1920 (9:16 aspect ratio)
        # Scale and crop (not pad) to achieve proper 9:16 ratio
        # For ffmpeg subtitles, we need to escape colons and backslashes in the path
        trimmed_srt_for_ffmpeg = str(trimmed_srt_path).replace(':', '\\:').replace('\\', '\\\\').replace("'", "\\'")

        print(f"    Burning subtitles...")

        # Filter chain:
        # 1. Scale to ensure minimum dimensions, then crop to exactly 1080x1920
        # 2. Burn in subtitles
        vf_filter = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(iw-ow)/2:(ih-oh)/2,subtitles={trimmed_srt_for_ffmpeg}:force_style='FontSize=16,MarginV=35,Alignment=2,FontName=Arial'"

        cmd = [
            'ffmpeg',
            '-ss', str(start_seconds),
            '-i', str(video_path),
            '-t', str(duration),
            '-vf', vf_filter,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-y',
            str(output_file)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            print(f"    ✓ Created successfully")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Failed: {e.stderr.decode()}")
            return None

    def create_shorts(self, folder_number: str, theme_numbers: str = None) -> None:
        """Create short video clips based on themes."""
        print("=" * 60)
        print("YouTube Shorts Creator - Creating Shorts")
        print("=" * 60)

        # Find the video folder
        folder = self.get_video_folder_by_number(folder_number)
        if not folder:
            print(f"Error: Video folder '{folder_number}_' not found")
            return

        print(f"Video folder: {folder.name}")

        # Get the video file
        video_path = self.get_video_file(folder)
        if not video_path:
            print(f"Error: No video file found in {folder}")
            return

        print(f"Video file: {video_path.name}")

        # Parse themes file
        themes_file = folder / 'themes.md'
        if not themes_file.exists():
            print(f"Error: themes.md not found in {folder}")
            print("Please run the script on this video first to generate themes.")
            return

        themes = self.parse_themes_file(themes_file)
        print(f"Found {len(themes)} themes")

        # Determine which themes to create
        if theme_numbers == 'all':
            selected_themes = themes
        else:
            # Parse comma-separated theme numbers
            try:
                requested_nums = [int(n.strip()) for n in theme_numbers.split(',')]
                selected_themes = [t for t in themes if t['number'] in requested_nums]

                if not selected_themes:
                    print(f"Error: No matching themes found for numbers: {theme_numbers}")
                    return

                # Check for requested themes that don't exist
                found_nums = {t['number'] for t in selected_themes}
                missing_nums = set(requested_nums) - found_nums
                if missing_nums:
                    print(f"Warning: Theme(s) {sorted(missing_nums)} not found")
            except ValueError:
                print(f"Error: Invalid theme numbers format: {theme_numbers}")
                print("Use comma-separated numbers (e.g., '1,2,5') or 'all'")
                return

        print(f"Creating {len(selected_themes)} short(s)...")

        # Find the SRT subtitle file
        srt_path = None
        for ext in ['*.srt']:
            srt_files = list(folder.glob(ext))
            if srt_files:
                srt_path = srt_files[0]
                break

        if not srt_path:
            print("Error: No SRT subtitle file found. Cannot create shorts with burnt-in subtitles.")
            return

        print(f"Using subtitles: {srt_path.name}")

        # Create output directory for shorts
        shorts_dir = folder / 'shorts'
        shorts_dir.mkdir(exist_ok=True)

        # Create each short
        successful = 0
        for theme in selected_themes:
            print(f"\nTheme {theme['number']}: {theme['title']}")
            result = self.create_short(video_path, theme, shorts_dir, srt_path)
            if result:
                successful += 1

        print("\n" + "=" * 60)
        print(f"Created {successful}/{len(selected_themes)} shorts in: {shorts_dir}")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Automatic YouTube Shorts Creation App'
    )
    parser.add_argument('input', help='YouTube video URL, path to local video file, or video folder number (e.g., 001)')
    parser.add_argument(
        '--model',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='medium',
        help='Whisper model size (default: medium)'
    )
    parser.add_argument(
        '--output-dir',
        default='videos',
        help='Base directory for downloaded videos (default: videos)'
    )
    parser.add_argument(
        '--ai',
        action='store_true',
        help='Use local AI (Llama 3 via Ollama) for theme title generation'
    )
    parser.add_argument(
        '--theme',
        type=str,
        help='Create shorts for specific theme(s). Use comma-separated numbers (e.g., "1,2,5") or "all"'
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

    # Mode 2: Process a new video (URL or local file)
    # Initialize AI generator if requested
    ai_generator = None
    if args.ai:
        try:
            from ai_theme_generator import AIThemeGenerator
            ai_generator = AIThemeGenerator()
            if ai_generator.is_available():
                print(f"✓ AI enabled: Using {ai_generator.model}")
            else:
                print("⚠ AI requested but not available. Using pattern-based titles instead.")
                ai_generator = None
        except ImportError:
            print("⚠ AI module not found. Using pattern-based titles instead.")
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
    creator.generate_themes(video_info, ai_generator=ai_generator)

    print("=" * 60)
    print(f"Processing complete! Folder: {video_info['folder']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
