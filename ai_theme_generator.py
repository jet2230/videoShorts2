#!/usr/bin/env python3
"""
AI-powered theme generation using local LLM models.
Supports Ollama (Llama 3) and llama-cpp.
"""

import json
import requests
from typing import Optional, List, Dict


class AIThemeGenerator:
    """Generate theme titles using local AI models."""

    def __init__(self, model: str = 'llama3', api_base: str = 'http://localhost:11434'):
        """
        Initialize AI theme generator.

        Args:
            model: Model name (default: llama3)
            api_base: API base URL (default: Ollama)
        """
        self.model = model
        self.api_base = api_base

    def is_available(self) -> bool:
        """Check if the AI service is available."""
        try:
            response = requests.get(f'{self.api_base}/api/tags', timeout=2)
            return response.status_code == 200
        except:
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = requests.get(f'{self.api_base}/api/tags', timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
        except:
            pass
        return []

    def generate_title(self, transcript: str, duration: str = "") -> str:
        """
        Generate a theme title using AI.

        Args:
            transcript: The transcript text
            duration: Duration string (for context)

        Returns:
            Generated title
        """
        # Use more transcript for better context
        transcript_text = transcript[:2000]

        prompt = f"""You are a YouTube Shorts content expert. Generate a SHORT, CATCHY title for a short video clip.

Transcript ({duration}):
{transcript_text}

CRITICAL RULES:
- 2-6 words maximum
- ONLY use words/terms that ACTUALLY appear in the transcript above
- NO made-up terms, books, or concepts not in the text
- Focus on the MAIN topic discussed (look for repeated terms)
- No filler words like "The", "A", "An" at the start
- Make viewers want to watch

Return ONLY the title, nothing else."""

        try:
            response = requests.post(
                f'{self.api_base}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.6,  # Lower to reduce hallucinations
                        'num_predict': 35,
                        'top_p': 0.85,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                title = response.json().get('response', '').strip()
                # Clean up
                title = title.strip('"\'').strip()
                # Remove any common prefixes
                for prefix in ['Title:', 'The title is', 'The video is about']:
                    if title.lower().startswith(prefix.lower()):
                        title = title[len(prefix):].strip()

                # Fix common AI typos
                title = self._fix_common_typos(title)

                # Capitalize first letter
                if title:
                    title = title[0].upper() + title[1:]

                return title if len(title) > 3 else "Islamic Teaching"
            else:
                return None

        except Exception as e:
            print(f"  AI generation error: {e}")
            return None

    def _fix_common_typos(self, text: str) -> str:
        """Fix common typos in AI-generated text."""
        # Common corrections: (typo -> correct, add possessive s variant)
        corrections = [
            ('duaat', 'Dua', True),
            ('duah', 'Dua', True),
            ('duaa', 'Dua', True),
            ('salla', 'Salla', False),
            ('allah', 'Allah', False),
            ('quran', 'Quran', False),
            ('hadith', 'Hadith', False),
            ('jannah', 'Jannah', False),
            ('jahannam', 'Jahannam', False),
            ('sunnah', 'Sunnah', False),
            ('sujood', 'Sujood', False),
        ]

        for typo, correct, possessive in corrections:
            import re
            # Case-insensitive replacement
            pattern = re.compile(r'\b' + typo + r's\b', re.IGNORECASE)
            if possessive:
                text = pattern.sub(correct + "'s", text)
            else:
                text = pattern.sub(correct + "s", text)

            # Fix standalone version
            pattern = re.compile(r'\b' + typo + r'\b', re.IGNORECASE)
            text = pattern.sub(correct, text)

        # Capitalize first letter if needed
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text

    def generate_reason(self, transcript: str, title: str) -> str:
        """
        Generate a reason why this theme works as a short.

        Args:
            transcript: The transcript text
            title: The theme title

        Returns:
            Generated reason
        """
        prompt = f"""Explain in 10 words or less why this clip makes a good YouTube Short:

Title: {title}
Transcript: {transcript[:300]}

Return the reason only, nothing else."""

        try:
            response = requests.post(
                f'{self.api_base}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'num_predict': 30,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                reason = response.json().get('response', '').strip()
                reason = reason.strip('"\'').strip()
                return reason if len(reason) > 5 else "Engaging content for viewers"
            else:
                return None

        except Exception:
            return None

    def identify_theme_boundaries(self, transcript_segments: List[Dict], progress_callback=None, cancel_check=None) -> List[Dict]:
        """
        Use AI to identify natural theme boundaries from transcript segments.

        Uses overlapping 10-minute windows to handle arbitrarily long videos:
        - Window 1: 0-10 minutes
        - Window 2: 8-18 minutes (2 min overlap)
        - Window 3: 16-26 minutes (2 min overlap)
        - etc.

        This prevents themes from being cut at window boundaries.

        Args:
            transcript_segments: List of segments with 'start', 'end', 'text' keys
            progress_callback: Optional callback for progress updates
            cancel_check: Optional function that returns True if task should be cancelled

        Returns:
            List of theme dicts with 'start_time', 'end_time', and 'description'
        """
        if not transcript_segments:
            return []

        def _log_msg(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        # Check for initial cancellation
        if cancel_check and cancel_check():
            raise Exception("Cancelled by user")

        # Get total duration in seconds
        total_duration = int(transcript_segments[-1]['end'])

        # Create overlapping windows (10 minutes each, 2 minute overlap)
        windows = []
        current_start = 0
        while current_start < total_duration:
            current_end = min(current_start + 600, total_duration)  # 10 minutes = 600 seconds
            windows.append((current_start, current_end))
            if current_end >= total_duration:
                break
            current_start = current_end - 120  # Backtrack 2 minutes (120 seconds) for overlap

        _log_msg(f"  AI Identifying themes across {len(windows)} windows...")

        # Process each window and collect themes
        all_themes = []
        for window_idx, (window_start, window_end) in enumerate(windows):
            # Check for cancellation before each window
            if cancel_check and cancel_check():
                raise Exception("Cancelled by user")

            # Calculate and report progress (0-100% within this phase)
            progress = int((window_idx) / len(windows) * 100)
            _log_msg(f"  Progress: {progress}% (Window {window_idx + 1}/{len(windows)})")

            # Filter segments for this window
            window_segments = [
                seg for seg in transcript_segments
                if seg['start'] >= window_start and seg['start'] < window_end
            ]

            if not window_segments:
                continue

            # Create transcript for this window
            transcript_parts = []
            for i in range(0, len(window_segments), 3):
                group = window_segments[i:i+3]
                if group:
                    start_sec = int(group[0]['start'])
                    text = " ".join(seg['text'][:100].strip() for seg in group)
                    transcript_parts.append(f"[{start_sec}s] {text}")

            window_transcript = "\n".join(transcript_parts)

            # Generate themes for this window
            window_themes = self._process_window(
                window_transcript,
                window_start,
                window_end,
                window_idx + 1,
                len(windows)
            )

            if window_themes:
                all_themes.extend(window_themes)

        # Merge overlapping themes and deduplicate
        merged_themes = self._merge_overlapping_themes(all_themes)

        _log_msg(f"  AI identified {len(merged_themes)} total themes. Finalizing... Progress: 100%")
        return merged_themes

    def _process_window(self, transcript: str, window_start: int, window_end: int,
                        window_num: int, total_windows: int) -> List[Dict]:
        """
        Process a single window and identify themes.

        Args:
            transcript: Transcript text for this window
            window_start: Window start time in seconds
            window_end: Window end time in seconds
            window_num: Current window number (for progress)
            total_windows: Total number of windows

        Returns:
            List of themes for this window
        """
        # Adjust target number of themes based on window size
        # For a 10-minute window, aim for 3-5 themes (45-90 seconds each)
        target_themes = 4 if total_windows > 1 else 15

        prompt = f"""Identify YouTube Shorts clips from this Islamic lecture. Find {target_themes} clips.

TRANSCRIPT COVERS: {window_start // 60}m{window_start % 60:02d}s to {window_end // 60}m{window_end % 60:02d}s

TIMESTAMPS are in SECONDS (just use the numbers like [120s]).

CRITICAL RULES:
- MINIMUM 45 seconds per clip
- MAXIMUM 2 minutes per clip
- Most clips should be 45-90 seconds
- Start at topic transitions
- End at complete thoughts
- No overlaps
- ALL timestamps must be between {window_start} and {window_end}

Your JSON output:
[
  {{"start": {window_start + 30}, "end": {window_start + 90}, "reason": "explains concept"}},
  {{"start": {window_start + 100}, "end": {window_start + 160}, "reason": "discusses topic"}}
]

Aim for shorter clips (45-90 seconds) rather than long ones.

Transcript (timestamps in seconds):
{transcript}

Return ONLY valid JSON. Nothing else."""

        try:
            response = requests.post(
                f'{self.api_base}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.5,
                        'num_predict': 800,
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                themes_data = self._parse_ai_response(result)

                # Validate and convert to theme format
                themes = []
                for theme in themes_data:
                    if 'start' in theme and 'end' in theme:
                        start_time = float(theme['start'])
                        end_time = float(theme['end'])
                        duration = end_time - start_time

                        # Enforce duration limits and window bounds
                        if (30 <= duration <= 240 and
                            window_start <= start_time < window_end and
                            window_start < end_time <= window_end):
                            themes.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': duration,
                                'reason': theme.get('reason', 'Engaging content')
                            })

                return themes
            else:
                return []

        except Exception as e:
            print(f" [error: {e}]")
            return []

    def _parse_ai_response(self, result: str) -> List[Dict]:
        """
        Parse AI response to extract theme data.

        Args:
            result: Raw AI response text

        Returns:
            List of theme dicts with 'start', 'end', 'reason'
        """
        import re
        themes_data = []

        # Try to parse as JSON first
        start_idx = result.rfind('[')
        if start_idx != -1:
            bracket_count = 0
            for i in range(start_idx, len(result)):
                if result[i] == '[':
                    bracket_count += 1
                elif result[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        try:
                            json_str = result[start_idx:i+1]
                            themes_data = json.loads(json_str)
                            break
                        except:
                            pass

        # Try format with backticks: 1. `{"start": 0, "end": 30}`
        if not themes_data:
            tick_pattern = r'(\d+)\.\s*`{"start":\s*(\d+),\s*"end":\s*(\d+)}`'
            matches = re.findall(tick_pattern, result)
            for match in matches:
                start_time = float(match[2])
                end_time = float(match[3])
                themes_data.append({'start': start_time, 'end': end_time, 'reason': 'AI-identified theme'})

        # If JSON parsing failed, try to extract from numbered list format
        if not themes_data:
            list_pattern = r'(\d+)\.\s*"([^"]+)"\s*\((\d+):(\d+)\s*-\s*(\d+):(\d+)\)'
            matches = re.findall(list_pattern, result)
            for match in matches:
                title = match[1]
                start_min = int(match[2])
                start_sec = int(match[3])
                end_min = int(match[4])
                end_sec = int(match[5])
                start_time = start_min * 60 + start_sec
                end_time = end_min * 60 + end_sec
                themes_data.append({'start': start_time, 'end': end_time, 'reason': title})

        # Try format without titles: 1. "TIMESTAMPS: MM:SS - MM:SS"
        if not themes_data:
            list_pattern = r'(\d+)\.\s*"TIMESTAMPS:\s*(\d+):(\d+)\s*-\s*(\d+):(\d+)'
            matches = re.findall(list_pattern, result)
            for match in matches:
                start_min = int(match[1])
                start_sec = int(match[2])
                end_min = int(match[3])
                end_sec = int(match[4])
                start_time = start_min * 60 + start_sec
                end_time = end_min * 60 + end_sec
                themes_data.append({'start': start_time, 'end': end_time, 'reason': 'AI-identified theme'})

        return themes_data

    def _merge_overlapping_themes(self, themes: List[Dict]) -> List[Dict]:
        """
        Merge overlapping or duplicate themes from different windows.

        Args:
            themes: List of all themes from all windows

        Returns:
            Deduplicated and merged list of themes
        """
        if not themes:
            return []

        # Sort by start time
        sorted_themes = sorted(themes, key=lambda x: x['start_time'])

        # Merge overlapping themes
        merged = []
        for theme in sorted_themes:
            if not merged:
                merged.append(theme)
                continue

            last = merged[-1]
            # Check if themes significantly overlap (>50% overlap)
            overlap_start = max(last['start_time'], theme['start_time'])
            overlap_end = min(last['end_time'], theme['end_time'])
            overlap_duration = overlap_end - overlap_start if overlap_end > overlap_start else 0

            shorter_duration = min(last['duration'], theme['duration'])

            if overlap_duration > shorter_duration * 0.5:
                # Significant overlap, keep the longer one
                if theme['duration'] > last['duration']:
                    merged[-1] = theme
            elif theme['start_time'] < last['end_time'] + 10:
                # Slight overlap or very close, skip the shorter one
                if theme['duration'] > last['duration']:
                    merged[-1] = theme
            else:
                # No overlap, add as new theme
                merged.append(theme)

        return merged


def test_ai_generator():
    """Test the AI theme generator."""
    print("Testing AI Theme Generator...")

    ai = AIThemeGenerator()

    if not ai.is_available():
        print("✗ AI service not available")
        return False

    models = ai.get_available_models()
    print(f"✓ Available models: {models}")

    # Test transcript
    test_transcript = """
    which is sometimes you ask yourself but it seems like my dua is not answering
    it do you get that feeling sometimes you say like my dua but it seems like my dua
    is not being answered this is something that also happened to so many people
    """

    title = ai.generate_title(test_transcript, "0m 30s")
    print(f"✓ Generated title: \"{title}\"")

    reason = ai.generate_reason(test_transcript, title)
    print(f"✓ Generated reason: \"{reason}\"")

    return True


if __name__ == '__main__':
    test_ai_generator()
