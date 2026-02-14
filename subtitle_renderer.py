#!/usr/bin/env python3
"""
Server-side canvas karaoke video renderer.
Renders karaoke subtitles on video frames using OpenCV and PIL.
"""

import os

# Disable hardware acceleration for FFmpeg backend to fix AV1 decoding issues
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;none"

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UniversalSubtitleRenderer:
    """Universal renderer for all subtitle styles (standard and karaoke)."""

    def __init__(self, video_path: str, word_timestamps: List[Dict], settings: Dict, formatting: Dict = None):
        """
        Initialize the renderer.

        Args:
            video_path: Path to video file
            word_timestamps: List of word timestamps with 'word', 'start', 'end'
            settings: Rendering settings
            formatting: Optional formatting for specific subtitles
        """
        self.video_path = Path(video_path)
        self.word_timestamps = word_timestamps or []
        self.settings = settings
        self.formatting = formatting or {}

        # Video properties
        self.cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
        
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(str(video_path))

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Rendering settings (always 1080x1920 vertical)
        self.output_width = 1080
        self.output_height = 1920
        self.font_size = int(settings.get('fontSize', 80))
        self.font_name = settings.get('fontName', 'Arial')
        
        # Convert hex colors to RGB tuples for PIL
        self.text_color = self._hex_to_rgb(settings.get('textColor', '#ffff00'))
        self.primary_color = self._hex_to_rgb(settings.get('primaryColor', '#ffffff'))
        self.past_color = self._hex_to_rgb(settings.get('pastColor', '#808080'))
        self.outline_color = self._hex_to_rgb(settings.get('outlineColor', '#000000'))
        
        # Glow settings
        self.glow_color = self._hex_to_rgb(settings.get('glowColor', '#ffff00'))
        self.glow_blur = int(settings.get('glowBlur', 0))
        
        # Background settings
        self.bg_color = self._hex_to_rgb(settings.get('bgColor', '#000000'))
        self.bg_opacity = float(settings.get('bgOpacity', 0.63))
        
        self.mode = settings.get('mode', 'standard')  # 'standard', 'normal', 'cumulative'
        self.font_weight = settings.get('font_weight', 'normal')

        # Subtitle positioning
        self.subtitle_position = settings.get('subtitle_position', 'bottom')
        self.subtitle_left = settings.get('subtitle_left') # Custom X
        self.subtitle_top = settings.get('subtitle_top')   # Custom Y
        self.subtitle_h_align = settings.get('subtitle_h_align', 'center')
        self.subtitle_v_align = settings.get('subtitle_v_align', 'bottom')

        # Pre-calculate word timings
        self.words_by_time = sorted(self.word_timestamps, key=lambda w: w['start'])

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert #RRGGBB to (R, G, B)."""
        if not hex_color: return (255, 255, 255)
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        if len(hex_color) != 6: return (255, 255, 255)
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def parse_subtitle_html(self, html: str) -> List[Dict]:
        """Parse HTML-style subtitle text into styled word objects."""
        if not html: return []
        
        import re
        from html.parser import HTMLParser

        class SubtitleHTMLParser(HTMLParser):
            def __init__(self, base_style):
                super().__init__()
                self.result = []
                self.style_stack = [base_style]

            def handle_starttag(self, tag, attrs):
                current = self.style_stack[-1].copy()
                if tag in ('b', 'strong'): current['bold'] = True
                elif tag in ('i', 'em'): current['italic'] = True
                elif tag == 'font':
                    for name, value in attrs:
                        if name == 'color': current['color'] = value
                        elif name == 'size':
                            size_map = {'1': 0.5, '2': 0.7, '3': 0.9, '4': 1.0, '5': 1.2, '6': 1.4, '7': 1.6}
                            current['sizeMultiplier'] = size_map.get(value, 1.0)
                elif tag == 'span':
                    for name, value in attrs:
                        if name == 'style':
                            if 'color' in value:
                                match = re.search(r'color:\s*([^;]+)', value)
                                if match: current['color'] = match.group(1).strip()
                            if 'font-weight' in value:
                                if 'bold' in value or re.search(r'font-weight:\s*([789]00)', value):
                                    current['bold'] = True
                            if 'font-style' in value and 'italic' in value:
                                current['italic'] = True
                            if 'font-size' in value:
                                em_match = re.search(r'font-size:\s*([\d.]+)em', value)
                                if em_match: current['sizeMultiplier'] = float(em_match.group(1))
                                else:
                                    px_match = re.search(r'font-size:\s*([\d.]+)px', value)
                                    if px_match: current['sizeMultiplier'] = float(px_match.group(1)) / 16.0
                self.style_stack.append(current)

            def handle_endtag(self, tag):
                if len(self.style_stack) > 1:
                    self.style_stack.pop()

            def handle_data(self, data):
                words = data.split()
                style = self.style_stack[-1]
                for word in words:
                    self.result.append({
                        'text': word,
                        'bold': style.get('bold', False),
                        'italic': style.get('italic', False),
                        'color': style.get('color'),
                        'sizeMultiplier': style.get('sizeMultiplier', 1.0)
                    })

        parser = SubtitleHTMLParser({'bold': False, 'italic': False, 'sizeMultiplier': 1.0})
        parser.feed(html)
        return parser.result

    def get_words_at_time(self, current_time: float, subtitle_text: str) -> Tuple[List[Dict], int]:
        """
        Get word list and highlighted word index for current time.
        
        Args:
            current_time: Current time in seconds
            subtitle_text: Subtitle text
        """
        words = subtitle_text.split()
        
        # Determine highlighted index if in karaoke mode
        highlighted_index = -1
        current_word_ts = None
        if self.mode != 'standard':
            for i, word_ts in enumerate(self.words_by_time):
                if word_ts['start'] <= current_time < word_ts['end']:
                    current_word_ts = word_ts
                    break
        
        if current_word_ts:
            ts_word_lower = current_word_ts['word'].lower().strip('.,!?;:"\'')
            for j, word in enumerate(words):
                if word.lower().strip('.,!?;:"\'') == ts_word_lower:
                    highlighted_index = j
                    break

        # For coloring, we need to find which words in the subtitle
        # correspond to timestamps before/after current_time
        # Approach: Find words in subtitle whose timestamps are < current_time
        colored_words = []

        # Count how many words in global timeline have ended before current_time
        words_before = 0
        for word_ts in self.words_by_time:
            if word_ts['end'] <= current_time:
                words_before += 1
            else:
                break

        # Now assign colors based on word position in subtitle
        # Words before current position get "past" color, current gets highlight
        for j, word in enumerate(words):
            word_color = self.primary_color  # Default primary color

            # Simple heuristic: first N words where N = words_before % subtitle_length
            # This is approximate but works for sequential word timing
            word_position_in_global = words_before - (len(words) - 1 - j)
            if word_position_in_global >= 0:
                # This word should be colored based on current_time vs its timestamp
                # Find the timestamp for this position
                if word_position_in_global < len(self.words_by_time):
                    word_ts = self.words_by_time[word_position_in_global]
                    if current_time >= word_ts['end']:
                        # Word already spoken
                        if self.mode == 'cumulative':
                            word_color = self.past_color
                        else:
                            word_color = self.primary_color
                    elif current_time >= word_ts['start']:
                        # Currently being spoken
                        word_color = self.text_color

            colored_words.append({
                'text': word,
                'color': word_color
            })

        return colored_words, highlighted_index

    def get_words_for_subtitle(self, subtitle_start: float, subtitle_end: float, subtitle_text: str) -> List[Dict]:
        """
        Get word timestamps that fall within a subtitle's time range.

        Args:
            subtitle_start: Subtitle start time in seconds
            subtitle_end: Subtitle end time in seconds
            subtitle_text: Subtitle text

        Returns:
            List of word timestamps that fall within this subtitle's time range
        """
        words = subtitle_text.split()
        subtitle_words = []

        # Find all word timestamps within the subtitle time range
        for word_ts in self.words_by_time:
            # Check if this word falls within the subtitle time range
            # Use a small buffer for edge cases
            if word_ts['start'] >= subtitle_start - 0.1 and word_ts['end'] <= subtitle_end + 0.1:
                subtitle_words.append(word_ts)

        # If we don't have exact matches, try to match by word count
        # Find words that start around subtitle_start
        if len(subtitle_words) < len(words):
            subtitle_words = []
            # Find the word timestamp closest to subtitle_start
            for i, word_ts in enumerate(self.words_by_time):
                if word_ts['start'] >= subtitle_start - 0.5:
                    # Take this word and the next N words
                    for j in range(len(words)):
                        if i + j < len(self.words_by_time):
                            subtitle_words.append(self.words_by_time[i + j])
                    break

        return subtitle_words[:len(words)]  # Limit to word count

    def get_words_at_time_for_subtitle(self, current_time: float, subtitle_data: str, subtitle_start: float, subtitle_end: float) -> Tuple[List[Dict], int]:
        """
        Get word list and highlighted word index for current time within a subtitle.

        Args:
            current_time: Current time in seconds
            subtitle_data: Subtitle text or HTML content
            subtitle_start: Subtitle start time
            subtitle_end: Subtitle end time

        Returns:
            Tuple of (words_with_colors, highlighted_index)
        """
        # Parse data into styled words
        if '<' in subtitle_data or '&' in subtitle_data:
            words_info = self.parse_subtitle_html(subtitle_data)
        else:
            words_info = [{'text': w, 'bold': False, 'italic': False, 'color': None, 'sizeMultiplier': 1.0} 
                         for w in subtitle_data.split()]

        # Get word timestamps for this subtitle (based on plain text count)
        plain_text = ' '.join([w['text'] for w in words_info])
        subtitle_word_timestamps = self.get_words_for_subtitle(subtitle_start, subtitle_end, plain_text)

        # Find highlighted word index
        highlighted_index = -1
        for i, word_ts in enumerate(subtitle_word_timestamps):
            if word_ts['start'] <= current_time < word_ts['end']:
                highlighted_index = i
                break

        # Color words based on timing and individual word styles
        colored_words = []
        highlight_color_rgb = self.text_color # settings.textColor

        for j, word_obj in enumerate(words_info):
            # Base color: use word's custom color if set, else primary global color
            base_color = self._hex_to_rgb(word_obj['color']) if word_obj.get('color') else self.primary_color
            word_color = base_color

            if self.mode != 'standard':
                # Get the timestamp for this word (by position)
                if j < len(subtitle_word_timestamps):
                    word_ts = subtitle_word_timestamps[j]
                    if current_time >= word_ts['end']:
                        # Word already spoken
                        if self.mode == 'cumulative':
                            word_color = self.past_color
                        else:
                            word_color = base_color
                    elif current_time >= word_ts['start']:
                        # Currently being spoken
                        word_color = highlight_color_rgb
                elif highlighted_index == -1 and j < len(words_info) and current_time >= subtitle_end:
                    if self.mode == 'cumulative':
                        word_color = self.past_color

            colored_words.append({
                'text': word_obj['text'],
                'color': word_color,
                'bold': word_obj.get('bold', False),
                'italic': word_obj.get('italic', False),
                'sizeMultiplier': word_obj.get('sizeMultiplier', 1.0)
            })

        return colored_words, highlighted_index

    def render_frame(self, frame: np.ndarray, current_time: float, subtitle_text: str, subtitle_start: float = None, subtitle_end: float = None, subtitle_seq: int = None) -> np.ndarray:
        """
        Render karaoke subtitles on a video frame.

        Args:
            frame: Input video frame (numpy array)
            current_time: Current time in seconds
            subtitle_text: Subtitle text to display
            subtitle_start: Subtitle start time (optional)
            subtitle_end: Subtitle end time (optional)
            subtitle_seq: Subtitle sequence number for formatting lookup (optional)

        Returns:
            Frame with rendered subtitles
        """
        # Look up formatting for this subtitle
        subtitle_data = subtitle_text
        if subtitle_seq is not None and str(subtitle_seq) in self.formatting:
            formatting = self.formatting[str(subtitle_seq)]
            if formatting.get('html'):
                subtitle_data = formatting['html']

        # Use cover/crop to fill entire output (no black bars)
        input_height, input_width = frame.shape[:2]

        # Calculate scale to cover output dimensions
        scale_w = self.output_width / input_width
        scale_h = self.output_height / input_height
        scale = max(scale_w, scale_h)

        # Calculate new dimensions
        new_width = int(input_width * scale)
        new_height = int(input_height * scale)

        # Resize frame
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        if int(current_time * 10) % 50 == 0:
            logger.info(f"Frame scaled to {new_width}x{new_height}, scale={scale:.2f}")

        # Calculate crop offsets (center crop)
        x_offset = (new_width - self.output_width) // 2
        y_offset = (new_height - self.output_height) // 2

        # Crop to exact output dimensions
        frame_cropped = frame_resized[y_offset:y_offset + self.output_height, x_offset:x_offset + self.output_width]

        # Convert to PIL for text rendering
        pil_image = Image.fromarray(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))
        
        # Debug watermark if enabled
        if self.settings.get('debug', False):
            try:
                draw = ImageDraw.Draw(pil_image)
                debug_font = self._get_font(80)
                draw.text((50, 50), f"DEBUG: {current_time:.2f}s", fill=(255, 0, 0), font=debug_font)
                
                if not subtitle_text:
                    if int(current_time * 10) % 50 == 0:
                        with open('server.log', 'a') as f:
                            f.write(f"[{datetime.now()}] RENDER DEBUG: NO TEXT at {current_time:.2f}s\n")
                            f.flush()
                    draw.text((50, 150), "NO SUBTITLE TEXT FOUND", fill=(255, 0, 0), font=debug_font)
            except Exception as e:
                with open('server.log', 'a') as f:
                    f.write(f"[{datetime.now()}] RENDER DEBUG ERROR: {str(e)}\n")
                    f.flush()

        # Skip if no subtitle text
        if not subtitle_text or not subtitle_text.strip():
            if int(current_time * 10) % 50 == 0:
                logger.info(f"No subtitle text at {current_time:.2f}s")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
        if int(current_time * 10) % 50 == 0: # Log roughly every 5 seconds
            logger.info(f"Rendering subtitle at {current_time:.2f}s: {subtitle_text[:30]}...")

        # Get colored words - use subtitle time range if available
        if subtitle_start is not None and subtitle_end is not None:
            colored_words, _ = self.get_words_at_time_for_subtitle(current_time, subtitle_data, subtitle_start, subtitle_end)
        else:
            colored_words, _ = self.get_words_at_time(current_time, subtitle_text)

        # Word wrapping logic needs to use individual word fonts
        lines = []
        current_line = []
        current_line_width = 0
        max_width = self.output_width - 80  # 40px padding
        
        for word_info in colored_words:
            word_font = self._get_word_font(word_info)
            word_width = self._get_word_width(word_info['text'], word_font)
            space_width = self._get_space_width(word_font)
            
            if not current_line:
                current_line = [word_info]
                current_line_width = word_width
            elif current_line_width + space_width + word_width <= max_width:
                current_line.append(word_info)
                current_line_width += space_width + word_width
            else:
                lines.append(current_line)
                current_line = [word_info]
                current_line_width = word_width
        
        if current_line:
            lines.append(current_line)

        # Calculate total height of all lines
        line_height = int(self.font_size * 1.2)
        total_height = len(lines) * line_height

        # Calculate starting Y position based on settings
        if self.subtitle_position == 'custom' and self.subtitle_left is not None and self.subtitle_top is not None:
            # For custom positions (dragged), subtitle_top is ALWAYS the center point
            # because that's how dragging is initialized and updated in the UI
            startY = self.subtitle_top - (total_height / 2)
        else:
            # Preset positions
            if self.subtitle_position == 'top':
                margin_top = 100
                startY = margin_top
            elif self.subtitle_position == 'middle':
                startY = (self.output_height - total_height) / 2
            else: # bottom
                margin_bottom = 100
                startY = self.output_height - margin_bottom - total_height

        # Render each line
        if int(current_time * 10) % 50 == 0:
            logger.info(f"Rendering {len(lines)} lines at {current_time:.2f}s (y={startY}, text: {subtitle_text[:20]}...)")

        # Convert to RGBA for glow/transparency support
        pil_image = pil_image.convert("RGBA")
        
        # Create a separate layer for background and text
        # This allows proper alpha blending of background boxes
        overlay_layer = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay_layer)
        
        # Create a separate layer for glow
        glow_layer = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)

        # First pass: Draw background boxes on overlay layer
        y_copy = startY
        bg_alpha = int(self.bg_opacity * 255)
        bg_fill = self.bg_color + (bg_alpha,)
        
        for line in lines:
            line_width = 0
            for i, word_info in enumerate(line):
                wf = self._get_word_font(word_info)
                line_width += self._get_word_width(word_info['text'], wf)
                if i < len(line) - 1:
                    line_width += self._get_space_width(wf)

            if (self.subtitle_position == 'custom' and self.subtitle_left is not None):
                # For custom positions (dragged), subtitle_left is ALWAYS the center point
                xCenter = self.subtitle_left
            else:
                xCenter = self.output_width / 2

            # Draw background box for the line on overlay layer
            box_padding = 10
            overlay_draw.rectangle(
                [xCenter - (line_width / 2) - box_padding, y_copy, xCenter + (line_width / 2) + box_padding, y_copy + line_height],
                fill=bg_fill
            )
            y_copy += line_height

        # Second pass: Draw text (glow, then outline, then main text) on overlay layer
        y_text = startY
        for line in lines:
            line_width = 0
            for i, word_info in enumerate(line):
                wf = self._get_word_font(word_info)
                line_width += self._get_word_width(word_info['text'], wf)
                if i < len(line) - 1:
                    line_width += self._get_space_width(wf)

            if self.subtitle_position == 'custom' and self.subtitle_left is not None:
                if self.subtitle_h_align == 'left':
                    x = self.subtitle_left
                elif self.subtitle_h_align == 'right':
                    x = self.subtitle_left - line_width
                else: # center
                    x = self.subtitle_left - (line_width / 2)
            else:
                x = (self.output_width - line_width) / 2

            text_y = y_text + (line_height / 2)
            
            for word_info in line:
                word_text = word_info['text']
                word_color = word_info['color']
                word_font = self._get_word_font(word_info)
                
                # Check if this word should glow (if it's the highlight color)
                highlight_color_rgb = self.text_color
                
                if word_color == highlight_color_rgb and self.glow_blur > 0:
                    glow_draw.text((x, text_y), word_text, fill=self.glow_color + (255,), font=word_font, anchor="lm")
                
                # Draw outline/shadow on overlay layer
                offsets = [(-2,-2), (2,-2), (-2,2), (2,2)]
                for dx, dy in offsets:
                    overlay_draw.text((x+dx, text_y+dy), word_text, fill=self.outline_color + (255,), font=word_font, anchor="lm")
                
                # Draw main text on overlay layer
                overlay_draw.text((x, text_y), word_text, fill=word_color + (255,), font=word_font, anchor="lm")
                
                x += self._get_word_width(word_text, word_font) + self._get_space_width(word_font)

            y_text += line_height

        # Apply blur to glow layer
        if self.glow_blur > 0:
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=self.glow_blur))
            # Composite glow onto overlay first
            overlay_layer = Image.alpha_composite(overlay_layer, glow_layer)

        # Finally composite everything onto the main image
        pil_image = Image.alpha_composite(pil_image, overlay_layer)

        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    def _get_word_font(self, word_info: Dict) -> ImageFont.FreeTypeFont:
        """Get the font for a specific word based on its style."""
        size = int(self.font_size * word_info.get('sizeMultiplier', 1.0))
        
        # Determine weight and style
        weight = "Bold" if word_info.get('bold', False) or self.font_weight == 'bold' else "Regular"
        style = "Italic" if word_info.get('italic', False) else ""
        
        # Combine into something fc-match might understand
        requested_font = f"{self.font_name}:{weight}{style}"
        
        return self._get_font(size, requested_font)

    def _get_font(self, size: int, font_name: str = None) -> ImageFont.FreeTypeFont:
        """Get font with fallbacks."""
        if font_name is None:
            font_name = self.font_name
            
        # Try to load the specified font with various extensions and paths
        font_paths = [
            f"/usr/share/fonts/truetype/{font_name}.ttf",
            f"/usr/share/fonts/truetype/{font_name}.ttc",
            f"/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
            f"/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            f"/usr/share/fonts/google-noto-vf/NotoSansArabic[wght].ttf",
            f"/usr/share/fonts/open-sans/OpenSans-Bold.ttf",
        ]
        
        for path in font_paths:
            try:
                if os.path.exists(path):
                    font = ImageFont.truetype(path, size)
                    return font
            except:
                pass

        # Try using fc-match to find the best match
        try:
            import subprocess
            result = subprocess.run(['fc-match', '-f', '%{file}', font_name], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return ImageFont.truetype(result.stdout.strip(), size)
        except:
            pass

        # Use default as absolute last resort (will be small)
        return ImageFont.load_default()

    def _wrap_words(self, colored_words: List[Dict], font, max_width: int) -> List[List[Dict]]:
        """Wrap words into lines that fit within max_width."""
        lines = []
        current_line = []
        current_width = 0

        for word_info in colored_words:
            word_width = self._get_word_width(word_info['text'], font)

            if current_line and current_width + word_width <= max_width:
                current_line.append(word_info)
                current_width += word_width + self._get_space_width(font)
            else:
                # Start new line
                if current_line:
                    lines.append(current_line)
                current_line = [word_info]
                current_width = word_width

        if current_line:
            lines.append(current_line)

        return lines

    def _get_word_width(self, text: str, font) -> int:
        """Get the width of a word in pixels."""
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]

    def _get_space_width(self, font) -> int:
        """Get the width of a space character."""
        return self._get_word_width(' ', font)

    def _get_text_width(self, words: List[Dict], font) -> int:
        """Get total width of a line of words."""
        width = 0
        for i, word_info in enumerate(words):
            width += self._get_word_width(word_info['text'], font)
            if i < len(words) - 1:
                width += self._get_space_width(font)
        return width

    def get_frame_at_time(self, time_sec: float) -> Optional[np.ndarray]:
        """Get video frame at specific time using random access."""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, int(time_sec * 1000))
        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def seek_to_time(self, time_sec: float) -> bool:
        """Seek video to specific time position for sequential reading."""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, int(time_sec * 1000))
        return True

    def read_next_frame(self) -> Optional[np.ndarray]:
        """Read next frame sequentially (much faster than random access)."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()


def render_canvas_karaoke_video(
    video_path: str,
    word_timestamps_path: str,
    subtitle_srt_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    settings: Dict,
    progress_callback = None
) -> bool:
    """
    Render a canvas karaoke video.

    Args:
        video_path: Path to source video
        word_timestamps_path: Path to word timestamps JSON
        subtitle_srt_path: Path to subtitle SRT file
        output_path: Path for output video
        start_time: Start time in seconds
        end_time: End time in seconds
        settings: Rendering settings
        progress_callback: Optional callback function(progress_percent, stage, message)

    Returns:
        True if successful
    """
    # Load word timestamps
    with open(word_timestamps_path, 'r') as f:
        data = json.load(f)
        word_timestamps = data.get('words', [])

    if not word_timestamps:
        logger.error("No word timestamps found")
        return False

    # Detect if SRT file has absolute times (main video SRT) or relative times (theme SRT)
    srt_path_obj = Path(subtitle_srt_path)
    srt_filename = srt_path_obj.name.lower()
    is_theme_srt = 'theme_' in srt_filename
    
    logger.info(f"Loading subtitles from: {subtitle_srt_path}")

    subtitles = _parse_srt(subtitle_srt_path)
    
    # Load formatting if available
    formatting = {}
    srt_path = Path(subtitle_srt_path)
    # Check for formatting JSON: theme_001_formatting.json
    formatting_file = srt_path.parent / srt_path.name.replace('.srt', '_formatting.json')
    if formatting_file.exists():
        try:
            with open(formatting_file, 'r', encoding='utf-8') as f:
                formatting = json.load(f)
            logger.info(f"Loaded formatting for {len(formatting)} subtitles from {formatting_file}")
        except Exception as e:
            logger.warning(f"Failed to load formatting file {formatting_file}: {e}")

    with open('server.log', 'a') as f:
        msg = f"[DEBUG_SRT] Path: {subtitle_srt_path}, Count: {len(subtitles)}\n"
        if subtitles:
            msg += f"[DEBUG_SRT] First sub: {subtitles[0]['start']}s - {subtitles[0]['end']}s: {subtitles[0]['text'][:20]}\n"
        f.write(msg)
        f.flush()

    if subtitles:
        logger.info(f"Parsed {len(subtitles)} subtitles. First sub: {subtitles[0]['start']}s - {subtitles[0]['end']}s: {subtitles[0]['text'][:30]}")
        # Calculate how many subtitles overlap with the theme's time range if treated as absolute
        overlapping_count = 0
        for sub in subtitles:
            if sub['start'] >= start_time - 5.0 and sub['start'] <= end_time + 5.0:
                overlapping_count += 1
        
        # If many subtitles overlap when treated as absolute, it's definitely absolute
        if overlapping_count > 0:
            is_theme_srt = False
            timing_msg = f"Detected ABSOLUTE times: {overlapping_count} subs overlap theme [{start_time}-{end_time}]\n"
        elif not is_theme_srt:
            # If it's not named 'theme_' and no overlap, check the first sub
            first_sub_start = subtitles[0]['start']
            if first_sub_start > end_time:
                is_theme_srt = False
                timing_msg = f"Detected ABSOLUTE times: First sub {first_sub_start}s > theme end {end_time}s\n"
            elif first_sub_start < 10.0 and start_time > 2.0:
                is_theme_srt = True
                timing_msg = f"Detected RELATIVE times: First sub {first_sub_start}s while theme start {start_time}s\n"
            else:
                # If it starts near 0 and theme starts near 0, could be either, but usually relative for edited files
                is_theme_srt = True 
                timing_msg = f"SRT type unclear, defaulting to RELATIVE (most likely for edited/theme files)\n"
        else:
            timing_msg = f"Defaulting to RELATIVE based on filename\n"
            
        with open('server.log', 'a') as f:
            f.write(f"[DEBUG_TIMING] {timing_msg}")
            f.flush()
    else:
        logger.warning("No subtitles parsed from SRT file")

    # Create renderer
    renderer = UniversalSubtitleRenderer(video_path, word_timestamps, settings, formatting)
    
    with open('server.log', 'a') as f:
        f.write(f"[{datetime.now()}] RENDERER START: {video_path} from {start_time} to {end_time}, subs={len(subtitles)}\n")
        f.flush()

    # Setup FFmpeg command
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp(prefix='short_renderer_'))

    try:
        # Generate frames at original video FPS for proper sync
        fps = renderer.fps
        if fps <= 0 or fps > 120:
            logger.warning(f"Invalid FPS {fps}, using 30")
            fps = 30
        total_frames = int((end_time - start_time) * fps)
        frame_pattern = temp_dir / 'frame_%06d.jpg'  # Use JPEG for faster I/O

        logger.info(f"Generating {total_frames} frames from {start_time}s to {end_time}s at {fps}fps (original video FPS)")
        
        # Log to server.log directly if needed
        with open('server.log', 'a') as f:
            f.write(f"RENDERER DEBUG: fps={fps}, total_frames={total_frames}, start={start_time}, end={end_time}\n")
            f.flush()

        if progress_callback:
            progress_callback(0, "rendering", "Seeking to start position...")

        # OPTIMIZATION 1: Sequential frame reading instead of random access
        # Seek to start position once, then read frames sequentially
        # This is much faster because video codecs are optimized for sequential playback
        start_frame_number = int(start_time * fps)
        
        # Try seeking by frame first
        success_seek = False
        
        def try_seek_and_read(pos_frame):
            renderer.cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)
            ret, frame = renderer.cap.read()
            return ret, frame

        # Try 1: Standard frame-based seek
        ret, first_frame = try_seek_and_read(start_frame_number)
        if ret:
            success_seek = True
        else:
            logger.info(f"Initial frame-based seek failed for frame {start_frame_number}. Trying millisecond-based seek...")
            # Try 2: Millisecond-based seek
            renderer.cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)
            ret, first_frame = renderer.cap.read()
            if ret:
                success_seek = True
                logger.info("Millisecond-based seek succeeded.")
            else:
                logger.warning("Millisecond-based seek failed. Re-initializing VideoCapture without explicit backend...")
                # Try 3: Re-initialize without explicit FFmpeg backend and try again
                renderer.release()
                renderer.cap = cv2.VideoCapture(str(video_path))
                if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                    renderer.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
                ret, first_frame = try_seek_and_read(start_frame_number)
                if ret:
                    success_seek = True
                    logger.info("Seek after re-initialization succeeded.")
                else:
                    # Try 4: Attempt slow sequential read from start
                    logger.warning("Seek still failing. Attempting slow sequential read from start...")
                    renderer.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # Check if we can read even the first frame
                    ret, first_frame = renderer.cap.read()
                    if ret:
                        # If we can read the first frame, try to skip to the target
                        # But skip the loop if it's too far (it's already been 2 seconds of failure)
                        for _ in range(start_frame_number - 1):
                            if not renderer.cap.grab(): # grab() is faster than read()
                                break
                        ret, first_frame = renderer.cap.read()
                        if ret:
                            success_seek = True
                            logger.info("Sequential read from start succeeded.")

        # FINAL FALLBACK: Transcode segment if everything else failed
        if not success_seek:
            logger.warning("OpenCV cannot read this video format directly. Attempting automatic transcode fallback...")
            
            # Create a temporary transcoded segment
            transcoded_src = temp_dir / "transcoded_segment.mp4"
            
            # Use FFmpeg to transcode just the segment we need to H.264
            # This is fast and extremely reliable
            transcode_cmd = [
                'ffmpeg', '-y',
                '-ss', str(max(0, start_time - 2.0)), # Start 2s early for safety
                '-i', str(video_path),
                '-t', str(end_time - start_time + 4.0), # Duration + buffer
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '18',
                '-c:a', 'copy',
                str(transcoded_src)
            ]
            
            logger.info(f"Running fallback transcode: {' '.join(transcode_cmd)}")
            ts_result = subprocess.run(transcode_cmd, capture_output=True, text=True)
            
            if ts_result.returncode == 0 and transcoded_src.exists():
                logger.info("Transcode successful. Re-initializing renderer with transcoded source.")
                renderer.release()
                renderer.cap = cv2.VideoCapture(str(transcoded_src))
                # The transcoded source starts at start_time - 2.0
                # We need to find the new start frame relative to the transcode start
                actual_transcode_start = max(0, start_time - 2.0)
                offset_time = start_time - actual_transcode_start
                offset_frames = int(offset_time * fps)
                
                renderer.cap.set(cv2.CAP_PROP_POS_FRAMES, offset_frames)
                ret, first_frame = renderer.cap.read()
                if ret:
                    success_seek = True
                    logger.info(f"Successfully read first frame from transcoded source (offset {offset_frames} frames).")
                else:
                    # Try reading from 0 if seek failed in temp file
                    renderer.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, first_frame = renderer.cap.read()
                    if ret:
                        success_seek = True
                        logger.info("Successfully read first frame from transcoded source (at 0).")

        if not success_seek:
            logger.error(f"Could not seek to frame {start_frame_number} or time {start_time}s after all retries, including transcode fallback.")
            return False

        # If we successfully read a frame, we need to seek back to start_frame_number
        # because the loop will read it again. 
        # Actually, since we already read the first frame, we should just adjust the loop.
        # But for simplicity, let's seek back. If it fails again, we have a problem.
        # To be safe, let's just use the first_frame we already have for the first iteration.
        
        if progress_callback:
            progress_callback(1, "rendering", "Rendering frames...")

        for frame_idx in range(total_frames):
            current_time = start_time + (frame_idx / fps)

            if frame_idx == 0:
                frame = first_frame
            else:
                # OPTIMIZATION 1: Read next frame sequentially (no random seek)
                ret, frame = renderer.cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {frame_idx} at time {current_time}s")
                    break

            # Find subtitle for this time
            # Handle both absolute (main video SRT) and relative (theme SRT) times
            subtitle_text = ""
            subtitle_start = None
            subtitle_end = None
            subtitle_seq = None

            if is_theme_srt:
                # Theme SRT: times are relative to theme start (0 = theme start)
                relative_time = current_time - start_time
                for sub in subtitles:
                    if sub['start'] <= relative_time <= sub['end']:
                        subtitle_text = sub['text']
                        subtitle_seq = sub.get('sequence')
                        # Convert to absolute time for word timestamp lookup
                        subtitle_start = sub['start'] + start_time
                        subtitle_end = sub['end'] + start_time
                        break
            else:
                # Main video SRT: times are absolute (from video start)
                for sub in subtitles:
                    if sub['start'] <= current_time <= sub['end']:
                        subtitle_text = sub['text']
                        subtitle_seq = sub.get('sequence')
                        # Already absolute, use directly
                        subtitle_start = sub['start']
                        subtitle_end = sub['end']
                        break

            # Render karaoke subtitles
            rendered_frame = renderer.render_frame(frame, current_time, subtitle_text, subtitle_start, subtitle_end, subtitle_seq)

            # Save frame
            # OPTIMIZATION 2: Use JPEG instead of PNG for faster compression and smaller files
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = temp_dir / frame_filename
            # JPEG quality 95 provides good quality with much faster compression
            cv2.imwrite(str(frame_path), rendered_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Progress callback (every 10 frames for responsiveness)
            if frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 70  # Rendering is 70% of total work
                if progress_callback:
                    progress_callback(progress, "rendering", f"Generating frame {frame_idx}/{total_frames}")
                elif frame_idx % 100 == 0:  # Less frequent logging if no callback
                    logger.info(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")

        # Encode with FFmpeg
        if progress_callback:
            progress_callback(70, "encoding", "Encoding video with FFmpeg (using NVIDIA GPU)...")

        # Update frame pattern to use JPEG
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-start_number', '0',
            '-i', str(temp_dir / 'frame_%06d.jpg'),
            '-ss', str(start_time),
            '-i', str(video_path),
            '-t', str(end_time - start_time),
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'h264_nvenc', # Use NVIDIA Hardware Encoder
            '-preset', 'p4',       # NVENC preset (p4 is medium/balanced)
            '-cq', '23',           # Constant quality for NVENC
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            str(output_path)
        ]

        logger.info(f"Encoding video with NVIDIA GPU: {' '.join(ffmpeg_cmd)}")

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(f"NVIDIA encoding failed, falling back to CPU: {result.stderr}")
            if progress_callback:
                progress_callback(75, "encoding", "NVIDIA encoding failed, falling back to CPU...")
            
            # Fallback to CPU encoding (libx264)
            ffmpeg_cmd_cpu = [
                'ffmpeg',
                '-y',
                '-framerate', str(fps),
                '-start_number', '0',
                '-i', str(temp_dir / 'frame_%06d.jpg'),
                '-ss', str(start_time),
                '-i', str(video_path),
                '-t', str(end_time - start_time),
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                str(output_path)
            ]
            result = subprocess.run(ffmpeg_cmd_cpu, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            if progress_callback:
                progress_callback(-1, "error", "FFmpeg encoding failed")
            return False

        if progress_callback:
            progress_callback(100, "complete", "Video saved successfully")

        logger.info(f"Video saved to {output_path}")
        return True

    finally:
        renderer.release()
        shutil.rmtree(temp_dir, ignore_errors=True)


def _parse_srt(srt_path: str) -> List[Dict]:
    """Parse SRT subtitle file with robust regex and fallback."""
    import re
    from datetime import datetime

    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(srt_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
    # Standardize line endings
    content = content.replace('\r\n', '\n')
    
    # Try robust regex first
    pattern = r'(\d+)\s*\n(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*\n(.*?)(?=\n\s*\n|\n\s*\d+\s*\n|\Z)'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    # Fallback to simpler split if regex fails
    if not matches:
        # Split by double newline or single newline followed by a number
        blocks = re.split(r'\n\s*\n', content.strip())
        subtitles = []
        for block in blocks:
            lines = [l.strip() for l in block.split('\n') if l.strip()]
            if len(lines) >= 3 and '-->' in lines[1]:
                try:
                    ts_line = lines[1]
                    times = ts_line.split('-->')
                    start_str = times[0].strip().replace(',', '.')
                    end_str = times[1].strip().split()[0].replace(',', '.')
                    text = ' '.join(lines[2:])
                    
                    def parse_time(time_str):
                        h, m, s = map(float, time_str.split(':'))
                        return h * 3600 + m * 60 + s
                    
                    subtitles.append({
                        'sequence': int(lines[0]),
                        'start': parse_time(start_str),
                        'end': parse_time(end_str),
                        'text': text
                    })
                except: continue
        return subtitles

    subtitles = []
    for match in matches:
        try:
            sequence = int(match.group(1))
            start_str = match.group(2).replace(',', '.')
            end_str = match.group(3).replace(',', '.')
            text = match.group(4).strip()

            def parse_time(time_str):
                h, m, s = map(float, time_str.split(':'))
                return h * 3600 + m * 60 + s

            subtitles.append({
                'sequence': sequence,
                'start': parse_time(start_str),
                'end': parse_time(end_str),
                'text': text
            })
        except: continue

    return subtitles


if __name__ == '__main__':
    # Test usage
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 6:
        print("Usage: python canvas_karaoke_exporter.py <video_path> <word_timestamps.json> <subtitle.srt> <output.mp4> <start_time> <end_time>")
        sys.exit(1)

    video_path = sys.argv[1]
    word_timestamps_path = sys.argv[2]
    subtitle_srt_path = sys.argv[3]
    output_path = sys.argv[4]
    start_time = float(sys.argv[5])
    end_time = float(sys.argv[6])

    settings = {
        'fontSize': 48,
        'fontName': 'Arial',
        'textColor': '#ffff00',
        'pastColor': '#808080',
        'mode': 'normal'
    }

    success = render_canvas_karaoke_video(
        video_path, word_timestamps_path, subtitle_srt_path, output_path,
        start_time, end_time, settings
    )

    if success:
        print("✅ Export complete!")
    else:
        print("❌ Export failed")
        sys.exit(1)
