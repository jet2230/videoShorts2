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
from subtitle_effects import SubtitleEffects

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
        
        # Initialize effects handler
        self.effects = SubtitleEffects(settings)

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
        
        # Font path cache to avoid repeated subprocess calls
        self._font_path_cache = {}
        
        self.mode = settings.get('mode', 'standard')  # 'standard', 'normal', 'cumulative'
        
        # Priority: explicit font_weight, then subtitle_bold flag
        self.font_weight = settings.get('font_weight')
        if not self.font_weight:
            self.font_weight = 'bold' if settings.get('subtitle_bold') else 'normal'

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
        colored_words = []

        # Count how many words in global timeline have ended before current_time
        words_before = 0
        for word_ts in self.words_by_time:
            if word_ts['end'] <= current_time:
                words_before += 1
            else:
                break

        # Now assign colors based on word position in subtitle
        for j, word in enumerate(words):
            word_color = self.primary_color  # Default primary color

            word_position_in_global = words_before - (len(words) - 1 - j)
            if word_position_in_global >= 0:
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
        """Get word timestamps that fall within a subtitle's time range."""
        words = subtitle_text.split()
        subtitle_words = []

        # Find all word timestamps within the subtitle time range
        for word_ts in self.words_by_time:
            if word_ts['start'] >= subtitle_start - 0.1 and word_ts['end'] <= subtitle_end + 0.1:
                subtitle_words.append(word_ts)

        # Fallback match by word count
        if len(subtitle_words) < len(words):
            subtitle_words = []
            for i, word_ts in enumerate(self.words_by_time):
                if word_ts['start'] >= subtitle_start - 0.5:
                    for j in range(len(words)):
                        if i + j < len(self.words_by_time):
                            subtitle_words.append(self.words_by_time[i + j])
                    break

        return subtitle_words[:len(words)]

    def get_words_at_time_for_subtitle(self, current_time: float, subtitle_data: str, subtitle_start: float, subtitle_end: float) -> Tuple[List[Dict], int]:
        """Get word list and highlighted word index for current time within a subtitle."""
        if '<' in subtitle_data or '&' in subtitle_data:
            words_info = self.parse_subtitle_html(subtitle_data)
        else:
            words_info = [{'text': w, 'bold': False, 'italic': False, 'color': None, 'sizeMultiplier': 1.0} 
                         for w in subtitle_data.split()]

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
        highlight_color_rgb = self.text_color

        for j, word_obj in enumerate(words_info):
            base_color = self._hex_to_rgb(word_obj['color']) if word_obj.get('color') else self.primary_color
            word_color = base_color

            if self.mode != 'standard':
                if j < len(subtitle_word_timestamps):
                    word_ts = subtitle_word_timestamps[j]
                    if current_time >= word_ts['end']:
                        if self.mode == 'cumulative':
                            word_color = self.past_color
                        else:
                            word_color = base_color
                    elif current_time >= word_ts['start']:
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
        """Render subtitles on a video frame."""
        # Look up formatting
        subtitle_data = subtitle_text
        if subtitle_seq is not None and str(subtitle_seq) in self.formatting:
            formatting = self.formatting[str(subtitle_seq)]
            if formatting.get('html'):
                subtitle_data = formatting['html']

        # Crop to vertical
        input_height, input_width = frame.shape[:2]
        scale = max(self.output_width / input_width, self.output_height / input_height)
        new_width, new_height = int(input_width * scale), int(input_height * scale)
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        x_offset = (new_width - self.output_width) // 2
        y_offset = (new_height - self.output_height) // 2
        frame_cropped = frame_resized[y_offset:y_offset + self.output_height, x_offset:x_offset + self.output_width]

        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))
        
        if not subtitle_text or not subtitle_text.strip():
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Get words and timestamps
        subtitle_word_timestamps = []
        if subtitle_start is not None and subtitle_end is not None:
            colored_words, highlighted_index = self.get_words_at_time_for_subtitle(current_time, subtitle_data, subtitle_start, subtitle_end)
            plain_text = ' '.join([w['text'] for w in colored_words])
            subtitle_word_timestamps = self.get_words_for_subtitle(subtitle_start, subtitle_end, plain_text)
        else:
            colored_words, highlighted_index = self.get_words_at_time(current_time, subtitle_text)

        # Word wrapping
        lines = []
        current_line = []
        current_line_width = 0
        max_width = self.output_width - 80
        
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
        
        if current_line: lines.append(current_line)

        # Layout
        line_height = int(self.font_size * 1.2)
        total_height = len(lines) * line_height

        if self.settings.get('effect_type') == 'flash':
            startY = (self.output_height - line_height) / 2
        elif self.subtitle_position == 'custom' and self.subtitle_left is not None and self.subtitle_top is not None:
            startY = self.subtitle_top - (total_height / 2)
        else:
            if self.subtitle_position == 'top': startY = 100
            elif self.subtitle_position == 'middle': startY = (self.output_height - total_height) / 2
            else: startY = self.output_height - 100 - total_height

        # Rendering Layers
        pil_image = pil_image.convert("RGBA")
        overlay_layer = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay_layer)
        glow_layer = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)

        # Pass 1: Background boxes
        y_copy = startY
        bg_alpha = int(self.bg_opacity * 255)
        bg_fill = self.bg_color + (bg_alpha,)
        
        for line in lines:
            line_width = 0
            for i, word_info in enumerate(line):
                wf = self._get_word_font(word_info)
                line_width += self._get_word_width(word_info['text'], wf)
                if i < len(line) - 1: line_width += self._get_space_width(wf)

            if self.settings.get('effect_type') == 'flash' or self.settings.get('effect_type') == 'dynamic_box':
                xCenter = self.output_width / 2
            elif (self.subtitle_position == 'custom' and self.subtitle_left is not None):
                xCenter = self.subtitle_left
            else:
                xCenter = self.output_width / 2

            if self.settings.get('effect_type') != 'dynamic_box':
                overlay_draw.rectangle(
                    [xCenter - (line_width / 2) - 10, y_copy, xCenter + (line_width / 2) + 10, y_copy + line_height],
                    fill=bg_fill
                )
            y_copy += line_height

        # Pass 2: Text
        y_text = startY
        words_before_line = 0
        for line in lines:
            line_width = 0
            for i, word_info in enumerate(line):
                wf = self._get_word_font(word_info)
                line_width += self._get_word_width(word_info['text'], wf)
                if i < len(line) - 1: line_width += self._get_space_width(wf)

            if self.settings.get('effect_type') == 'flash': x = self.output_width / 2
            elif self.subtitle_position == 'custom' and self.subtitle_left is not None:
                if self.subtitle_h_align == 'left': x = self.subtitle_left
                elif self.subtitle_h_align == 'right': x = self.subtitle_left - line_width
                else: x = self.subtitle_left - (line_width / 2)
            else:
                x = (self.output_width - line_width) / 2

            text_y = y_text + (line_height / 2)
            for j, word_info in enumerate(line):
                word_idx = words_before_line + j
                is_highlighted = (word_idx == highlighted_index)
                
                # Timing
                if word_idx < len(subtitle_word_timestamps):
                    ts = subtitle_word_timestamps[word_idx]
                    w_start, w_end = ts['start'], ts['end']
                else:
                    w_start = subtitle_start if subtitle_start is not None else current_time
                    w_end = subtitle_end if subtitle_end is not None else current_time + 1.0
                
                # Apply Effects
                effect_mods = self.effects.apply_word_effect(word_info, current_time, w_start, w_end, is_highlighted=is_highlighted)
                word_text = effect_mods['text']
                if not word_text:
                    x += self._get_word_width(word_info['text'], self._get_word_font(word_info)) + self._get_space_width(self._get_word_font(word_info))
                    continue

                # Get and normalize color
                word_color = effect_mods['color']
                if isinstance(word_color, str):
                    word_color = self._hex_to_rgb(word_color)
                
                word_font = self._get_word_font({**word_info, 'sizeMultiplier': word_info.get('sizeMultiplier', 1.0) * effect_mods['scale']})
                x_off, y_off = effect_mods['offset_x'], effect_mods['offset_y']
                
                # Glow
                if word_color == self.text_color and effect_mods['glow_blur'] > 0:
                    glow_draw.text((x + x_off, text_y + y_off), word_text, fill=self.glow_color + (255,), font=word_font, anchor="lm")
                
                # Outline
                if effect_mods.get('custom_render') and self.settings.get('effect_type') == 'shadow_3d':
                    self.effects.apply_3d_shadow(overlay_draw, x + x_off, text_y + y_off, word_text, word_font, word_color + (255,))
                else:
                    offsets = [(-2,-2), (2,-2), (-2,2), (2,2)]
                    for dx, dy in offsets:
                        overlay_draw.text((x + x_off + dx, text_y + y_off + dy), word_text, fill=self.outline_color + (255,), font=word_font, anchor="lm")
                
                # Dynamic Box
                if self.settings.get('effect_type') == 'dynamic_box' and is_highlighted:
                    bbox = overlay_draw.textbbox((x + x_off, text_y + y_off), word_text, font=word_font, anchor="lm")
                    pad = 10
                    overlay_draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=self.text_color+(255,))
                    # Re-normalize word_color if it changed (e.g. to bgColor)
                    word_color = self._hex_to_rgb(self.settings.get('bgColor', '#000000'))

                # Progressive Fill
                if self.settings.get('effect_type') == 'progressive_fill' and is_highlighted:
                    overlay_draw.text((x + x_off, text_y + y_off), word_text, fill=self.primary_color + (255,), font=word_font, anchor="lm")
                    prog = max(0, min(1, (current_time - w_start) / (w_end - w_start)))
                    bbox = overlay_draw.textbbox((x + x_off, text_y + y_off), word_text, font=word_font, anchor="lm")
                    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                    if tw > 0 and th > 0:
                        word_layer = Image.new("RGBA", (int(tw)+20, int(th)+20), (0,0,0,0))
                        word_draw = ImageDraw.Draw(word_layer)
                        word_draw.text((10, 10 + th/2), word_text, fill=word_color+(255,), font=word_font, anchor="lm")
                        mask = self.effects.get_progressive_fill_mask(word_layer.width, word_layer.height, prog)
                        overlay_layer.paste(word_layer, (int(bbox[0])-10, int(bbox[1])-10), mask)
                else:
                    # Final draw with normalized color
                    overlay_draw.text((x + x_off, text_y + y_off), word_text, fill=word_color + (255,), font=word_font, anchor="lm")
                
                # Emoji
                if 'emoji' in effect_mods:
                    emoji_font = self._get_font(int(self.font_size * 1.2), "NotoColorEmoji")
                    overlay_draw.text((x + x_off, text_y + y_off - line_height), effect_mods['emoji'], fill=(255,255,255,255), font=emoji_font, anchor="mm")

                x += self._get_word_width(word_text, word_font) + self._get_space_width(word_font)

            y_text += line_height
            words_before_line += len(line)

        # Compositing
        if self.glow_blur > 0:
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=self.glow_blur))
            overlay_layer = Image.alpha_composite(overlay_layer, glow_layer)
        pil_image = Image.alpha_composite(pil_image, overlay_layer)
        return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    def _get_word_font(self, word_info: Dict) -> ImageFont.FreeTypeFont:
        """Get the font for a specific word based on its style."""
        size = int(self.font_size * word_info.get('sizeMultiplier', 1.0))
        
        # Word is bold if specifically tagged OR if global font_weight is bold
        is_bold = word_info.get('bold', False) or (self.font_weight == 'bold')
        weight = "Bold" if is_bold else "Regular"
        
        style = "Italic" if word_info.get('italic', False) else ""
        return self._get_font(size, f"{self.font_name}:{weight}{style}")

    def _get_font(self, size: int, font_name: str = None) -> ImageFont.FreeTypeFont:
        """Get font with fallbacks."""
        if font_name is None: font_name = self.font_name
        
        # Check cache first
        if font_name in self._font_path_cache:
            return ImageFont.truetype(self._font_path_cache[font_name], size)

        # 1. Try direct paths (essential for local font files)
        direct_paths = [
            f"/usr/share/fonts/truetype/{font_name}.ttf",
            f"/usr/share/fonts/truetype/{font_name}.ttc",
        ]
        for path in direct_paths:
            try:
                if os.path.exists(path):
                    self._font_path_cache[font_name] = path
                    return ImageFont.truetype(path, size)
            except: pass
            
        # 2. Try fc-match (the most robust way on Linux to resolve "Family:Style")
        try:
            # -f '%{file}' returns the absolute path to the best matching font file
            result = subprocess.run(['fc-match', '-f', '%{file}', font_name], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                match_path = result.stdout.strip()
                if os.path.exists(match_path):
                    self._font_path_cache[font_name] = match_path
                    return ImageFont.truetype(match_path, size)
        except: pass
        
        # 3. Last resort fallbacks
        # We separate regular and bold fallbacks to avoid "Always Bold" issue
        fallbacks = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
        ]
        
        # If bold was explicitly requested, prioritize bold fallbacks
        if ":Bold" in font_name:
            fallbacks = [
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
            ] + fallbacks
            
        for path in fallbacks:
            try:
                if os.path.exists(path):
                    self._font_path_cache[font_name] = path
                    return ImageFont.truetype(path, size)
            except: pass
            
        # Default fallback (no caching for default as it's built-in)
        return ImageFont.load_default()

    def _get_word_width(self, text: str, font) -> int:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]

    def _get_space_width(self, font) -> int:
        return self._get_word_width(' ', font)

    def release(self):
        if self.cap: self.cap.release()


def render_canvas_karaoke_video(video_path, word_timestamps_path, subtitle_srt_path, output_path, start_time, end_time, settings, progress_callback=None):
    with open(word_timestamps_path, 'r') as f:
        word_timestamps = json.load(f).get('words', [])
    if not word_timestamps: return False

    subtitles = _parse_srt(subtitle_srt_path)
    formatting = {}
    srt_path = Path(subtitle_srt_path)
    formatting_file = srt_path.parent / srt_path.name.replace('.srt', '_formatting.json')
    if formatting_file.exists():
        with open(formatting_file, 'r', encoding='utf-8') as f: formatting = json.load(f)

    renderer = UniversalSubtitleRenderer(video_path, word_timestamps, settings, formatting)
    fps = renderer.fps if 0 < renderer.fps <= 120 else 30
    total_frames = int((end_time - start_time) * fps)
    
    import tempfile, shutil
    temp_dir = Path(tempfile.mkdtemp(prefix='short_renderer_'))
    
    # Pre-extract segment to a temporary file for better OpenCV compatibility (especially AV1)
    # and much faster seeking.
    temp_segment = temp_dir / "segment.mp4"
    logger.info(f"Extracting segment {start_time}-{end_time} to {temp_segment}...")
    
    extract_cmd = [
        'ffmpeg', '-y', '-ss', str(start_time), '-i', str(video_path),
        '-t', str(end_time - start_time), '-c:v', 'libx264', '-preset', 'ultrafast',
        '-crf', '18', '-c:a', 'copy', str(temp_segment)
    ]
    subprocess.run(extract_cmd, capture_output=True, check=True)
    
    # Re-initialize renderer with the small segment
    segment_renderer = UniversalSubtitleRenderer(str(temp_segment), word_timestamps, settings, formatting)
    
    try:
        # segment_renderer starts at 0, so we use frame indices relative to start of segment
        for frame_idx in range(total_frames):
            current_time = start_time + (frame_idx / fps)
            # Use segment time which is relative to its start
            current_segment_time = frame_idx / fps
            
            ret, frame = segment_renderer.cap.read()
            if not ret: break

            subtitle_text, subtitle_start, subtitle_end, subtitle_seq = "", None, None, None
            is_theme_srt = 'theme_' in Path(subtitle_srt_path).name.lower()
            
            for sub in subtitles:
                s_start = sub['start'] + (start_time if is_theme_srt else 0)
                s_end = sub['end'] + (start_time if is_theme_srt else 0)
                if s_start <= current_time <= s_end:
                    subtitle_text, subtitle_seq = sub['text'], sub.get('sequence')
                    subtitle_start, subtitle_end = s_start, s_end
                    break

            rendered_frame = segment_renderer.render_frame(frame, current_time, subtitle_text, subtitle_start, subtitle_end, subtitle_seq)
            cv2.imwrite(str(temp_dir / f"frame_{frame_idx:06d}.jpg"), rendered_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if progress_callback and frame_idx % 10 == 0:
                progress_callback((frame_idx / total_frames) * 70, "rendering", f"Frame {frame_idx}/{total_frames}")

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-framerate', str(fps), '-start_number', '0', '-i', str(temp_dir / 'frame_%06d.jpg'),
            '-ss', str(start_time), '-i', str(video_path), '-t', str(end_time - start_time),
            '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(output_path)
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True)
        return True
    finally:
        segment_renderer.release()
        renderer.release()
        shutil.rmtree(temp_dir, ignore_errors=True)

def _parse_srt(srt_path: str) -> List[Dict]:
    import re
    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
    pattern = r'(\d+)\s*\n(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*\n(.*?)(?=\n\s*\n|\n\s*\d+\s*\n|\Z)'
    subtitles = []
    for match in re.finditer(pattern, content, re.DOTALL):
        def parse_time(ts):
            h, m, s = map(float, ts.replace(',', '.').split(':'))
            return h * 3600 + m * 60 + s
        subtitles.append({
            'sequence': int(match.group(1)),
            'start': parse_time(match.group(2)),
            'end': parse_time(match.group(3)),
            'text': match.group(4).strip()
        })
    return subtitles
