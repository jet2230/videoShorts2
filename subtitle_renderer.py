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
        self.font_size = int(settings.get('fontSize') or 80)
        self.font_name = settings.get('fontName') or 'Arial'
        
        # Convert hex colors to RGB tuples for PIL
        self.text_color = self._hex_to_rgb(settings.get('textColor') or '#ffff00')
        self.primary_color = self._hex_to_rgb(settings.get('primaryColor') or '#ffffff')
        self.past_color = self._hex_to_rgb(settings.get('pastColor') or '#808080')
        self.outline_color = self._hex_to_rgb(settings.get('outlineColor') or '#000000')
        
        # Glow settings
        self.glow_color = self._hex_to_rgb(settings.get('glowColor') or '#ffff00')
        self.glow_blur = int(settings.get('glowBlur') or 0)

        # Background settings
        self.bg_color = self._hex_to_rgb(settings.get('bgColor') or '#000000')
        self.bg_opacity = float(settings.get('bgOpacity') or 0.63)
        self._title_pre_rendered = False  # Track if title has been pre-rendered
        self._font_path_cache = {}
        self._font_obj_cache = {}
        
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
        self._subtitle_word_cache = {} # Cache for word-to-subtitle mappings
        
        # Performance buffers
        self._overlay_layer = None
        self._glow_layer = None
        self._last_sub_index = 0
        self._title_cache = None
        self._title_cache_size = None
        self._last_render_state = {}
        self._last_rendered_frame = None
        self._glow_cache = None
        self._last_glow_key = None

        # Title styling
        title_font_size = settings.get('title_font_size') or (self.font_size * 0.6)
        self.title_font_size_pref = int(title_font_size)
        self.title_bg_type = settings.get('title_bg_type') or 'gradient'
        self.title_text_color = self._hex_to_rgb(settings.get('title_text_color') or '#00ff9d')
        self.title_font_weight = str(settings.get('title_font_weight') or '800')
        self.title_outline_width = float(settings.get('title_outline_width') or 0)
        self.title_outline_color = self._hex_to_rgb(settings.get('title_outline_color') or '#000000')
        self.title_all_caps = settings.get('title_all_caps') is not False
        self.show_title = settings.get('show_title') is True
        self.title_text = settings.get('title') or 'Theme Title'
        self.title_top = settings.get('title_top') # Custom Title Y

        # Pre-calculate word timings
        self.words_by_time = sorted(self.word_timestamps, key=lambda w: w['start'])
        
        # Detect if text is Arabic (RTL)
        self.is_rtl = self._is_arabic(self.word_timestamps)

    def _is_arabic(self, word_timestamps: List[Dict]) -> bool:
        """Detect if the content contains Arabic characters (RTL)."""
        if not word_timestamps:
            return False
            
        # Sample first few words
        sample_size = min(20, len(word_timestamps))
        arabic_chars = 0
        total_chars = 0
        
        for i in range(sample_size):
            word = word_timestamps[i].get('word', '')
            total_chars += len(word)
            # Check for Arabic Unicode range
            for char in word:
                if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' or '\u08A0' <= char <= '\u08FF':
                    arabic_chars += 1
                    
        return arabic_chars > (total_chars * 0.3) if total_chars > 0 else False
        
        # Pre-calculate title properties to save CPU every frame
        self._title_pre_rendered = False  # Track if title has been pre-rendered
        # Note: We don't call _precalculate_title_properties() here to avoid overhead
        # It will be called lazily on first frame that needs it

    def _precalculate_title_properties(self):
        """Pre-calculate title font size and pre-render title layer to avoid doing it every frame."""
        if not self.show_title or not self.title_text:
            return

        title = self.title_text.upper() if self.title_all_caps else self.title_text
        
        # Base title font size from settings
        max_title_width = self.output_width * 0.85 # Safe area
        
        # Font weight from settings
        weight_suffix = "Bold"
        fw = str(self.title_font_weight)
        if fw == "400": weight_suffix = "Regular"
        elif fw == "600": weight_suffix = "SemiBold"
        
        curr_size = self.title_font_size_pref
        title_font = self._get_font(curr_size, f"{self.font_name}:{weight_suffix}")
        
        # Dummy draw for bbox calculation
        dummy_img = Image.new("RGBA", (self.output_width, 400), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy_img)
        
        # Auto-scale font size if title is too long
        while curr_size > 10:
            bbox = draw.textbbox((0, 0), title, font=title_font)
            if (bbox[2] - bbox[0]) <= max_title_width:
                break
            curr_size -= 2
            title_font = self._get_font(curr_size, f"{self.font_name}:{weight_suffix}")

        self.resolved_title_text = title
        self.resolved_title_font = title_font
        self.resolved_font_size = curr_size  # Store the actual font size used
        
        # Determine Title Y position (default 150 center)
        ui_title_center_y = self.title_top if self.title_top is not None else 150
        
        # Get bounding box for the background relative to center
        # 'mm' anchor ensures the center of the text is at ui_title_center_y
        bbox = draw.textbbox((self.output_width / 2, ui_title_center_y), title, font=title_font, anchor="mm")
        padding = 20
        bg_bbox = [bbox[0] - padding, bbox[1] - padding/2, bbox[2] + padding, bbox[3] + padding/2]
        
        # Create a dedicated title surface
        self.title_surface = Image.new("RGBA", (self.output_width, self.output_height), (0, 0, 0, 0))
        s_draw = ImageDraw.Draw(self.title_surface)
        
        # Draw background on surface
        box_outline_color = self.title_text_color + (100,)
        if self.title_bg_type == 'gradient':
            s_draw.rounded_rectangle(bg_bbox, radius=8, fill=(0, 0, 0, 180), outline=box_outline_color, width=2)
        elif self.title_bg_type == 'solid':
            s_draw.rounded_rectangle(bg_bbox, radius=8, fill=(0, 0, 0, 230), outline=(50, 50, 50, 255), width=2)
            
        # Draw outline manually
        if self.title_outline_width > 0:
            outline_color = self.title_outline_color + (255,)
            font_scale_ratio = self.resolved_font_size / self.title_font_size_pref
            scaled_outline_width = self.title_outline_width * font_scale_ratio
            stroke_pixels = int(round(scaled_outline_width))

            for dx, dy in [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]:
                s_draw.text(
                    (self.output_width / 2 + dx * stroke_pixels, ui_title_center_y + dy * stroke_pixels),
                    title,
                    fill=outline_color,
                    font=title_font,
                    anchor="mm"
                )

        # Draw main text on top
        s_draw.text(
            (self.output_width / 2, ui_title_center_y),
            title,
            fill=self.title_text_color + (255,),
            font=title_font,
            anchor="mm"
        )
        
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

        # First try to match by TEXT content (more robust than time-based matching)
        # This handles cases where SRT timestamps don't perfectly align with word timestamps
        if len(words) > 0:
            # Normalize words for comparison (remove punctuation, lowercase)
            def normalize_word(w):
                return w.lower().strip('.,!?;:"\'-')

            normalized_subtitle_words = [normalize_word(w) for w in words]

            # Search through word timestamps to find a matching sequence
            for i in range(len(self.words_by_time) - len(words) + 1):
                match = True
                potential_match = []
                for j in range(len(words)):
                    if i + j >= len(self.words_by_time):
                        match = False
                        break
                    ts_word = normalize_word(self.words_by_time[i + j]['word'])
                    if ts_word != normalized_subtitle_words[j]:
                        match = False
                        break
                    potential_match.append(self.words_by_time[i + j])

                if match and len(potential_match) == len(words):
                    # Found a text match! Verify it's reasonably close in time
                    first_word_time = potential_match[0]['start']
                    # Allow up to 5 seconds difference (handles small theme SRT timestamp offsets)
                    if abs(first_word_time - subtitle_start) <= 5:
                        return potential_match

        # Fallback: Try time-based matching
        # Find all word timestamps that overlap with the subtitle time range
        for word_ts in self.words_by_time:
            # Overlap check: word starts before subtitle ends AND word ends after subtitle starts
            if word_ts['start'] < subtitle_end - 0.05 and word_ts['end'] > subtitle_start + 0.05:
                subtitle_words.append(word_ts)

        # Fallback match by word count
        if len(subtitle_words) < len(words):
            subtitle_words = []
            for i, word_ts in enumerate(self.words_by_time):
                if abs(word_ts['start'] - subtitle_start) < 2.0:
                    for j in range(len(words)):
                        if i + j < len(self.words_by_time):
                            ts = self.words_by_time[i + j]
                            # Only include if it's not too far past the subtitle end
                            if ts['start'] < subtitle_end + 2.0:
                                subtitle_words.append(ts)
                    break

        return subtitle_words[:len(words)]

    def get_words_at_time_for_subtitle(self, current_time: float, subtitle_data: str, subtitle_start: float, subtitle_end: float) -> Tuple[List[Dict], int, List[Dict]]:
        """Get word list, highlighted word index, and timestamps for current time within a subtitle."""
        if '<' in subtitle_data or '&' in subtitle_data:
            words_info = self.parse_subtitle_html(subtitle_data)
        else:
            words_info = [{'text': w, 'bold': False, 'italic': False, 'color': None, 'sizeMultiplier': 1.0} 
                         for w in subtitle_data.split()]

        plain_text = ' '.join([w['text'] for w in words_info])
        
        # Use cache if available
        cache_key = f"{subtitle_start}_{subtitle_end}_{plain_text}"
        if cache_key in self._subtitle_word_cache:
            subtitle_word_timestamps = self._subtitle_word_cache[cache_key]
        else:
            subtitle_word_timestamps = self.get_words_for_subtitle(subtitle_start, subtitle_end, plain_text)
            self._subtitle_word_cache[cache_key] = subtitle_word_timestamps

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

        return colored_words, highlighted_index, subtitle_word_timestamps

    def render_frame(self, frame: np.ndarray, current_time: float, subtitle_text: str, subtitle_start: float = None, subtitle_end: float = None, subtitle_seq: int = None) -> np.ndarray:
        """Render subtitles on a video frame."""
        # Look up formatting
        subtitle_data = subtitle_text
        if subtitle_seq is not None and str(subtitle_seq) in self.formatting:
            formatting = self.formatting[str(subtitle_seq)]
            if formatting.get('html'):
                subtitle_data = formatting['html']

        # Input is now guaranteed to be output resolution thanks to FFmpeg extraction
        frame_cropped = frame

        # FAST PATH: Check if there is anything to draw before expensive PIL conversion
        has_title = self.show_title and self.title_text
        has_subs = subtitle_text and subtitle_text.strip()

        if not (has_title or has_subs):
            self._last_rendered_frame = None
            self._last_render_state = {}
            # Convert BGR frame to RGB bytes for FFmpeg
            return cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB).tobytes()

        # SMART REUSE: Check if we can reuse the previous render
        # Only works for 'standard' mode where there are no word-by-word highlights
        if self.mode == 'standard' and self.effects.effect_type in ['standard', 'none']:
            current_state = {
                'text': subtitle_data,
                'has_title': has_title,
                'title_text': self.title_text
            }
            if current_state == self._last_render_state and self._last_rendered_frame is not None:
                # Same text, no effects, no title change - just reuse the overlay logic
                # However, since the background video frame has changed, we still need to composite
                # So we only skip the PIL DRAWING part, not the composition.
                pass 

        # 1. Pre-render title once (not every frame)
        if has_title and not self._title_pre_rendered:
            self._precalculate_title_properties()
            self._title_pre_rendered = True

        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.convert("RGBA")
        
        # Reuse or create overlay layer
        if self._overlay_layer is None or self._overlay_layer.size != pil_image.size:
            self._overlay_layer = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        else:
            # Much faster than creating a new image: clear existing buffer
            self._overlay_layer.paste((0, 0, 0, 0), [0, 0, pil_image.size[0], pil_image.size[1]])
            
        overlay_layer = self._overlay_layer
        overlay_draw = ImageDraw.Draw(overlay_layer)
        
        # Coordinate scaling: Map UI pixels (1080x1920) to actual frame pixels
        scale_x = pil_image.width / 1080
        scale_y = pil_image.height / 1920
        # Use average scale for font to preserve aspect ratio
        font_scale = (scale_x + scale_y) / 2

        # Glow layer only if blur > 0
        glow_layer = None
        if self.glow_blur > 0:
            if self._glow_layer is None or self._glow_layer.size != pil_image.size:
                self._glow_layer = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
            else:
                self._glow_layer.paste((0, 0, 0, 0), [0, 0, pil_image.size[0], pil_image.size[1]])
            glow_layer = self._glow_layer
            glow_draw = ImageDraw.Draw(glow_layer)

        # 1. Pre-render title once (not every frame)
        if has_title and not self._title_pre_rendered:
            self._precalculate_title_properties()
            self._title_pre_rendered = True

        # 2. Render pre-rendered title surface
        if has_title and hasattr(self, 'title_surface'):
            # Scale title surface to match current frame scaling
            if pil_image.size != self.title_surface.size:
                scaled_title = self.title_surface.resize(pil_image.size, Image.LANCZOS)
                overlay_layer.paste(scaled_title, (0, 0), scaled_title)
            else:
                overlay_layer.paste(self.title_surface, (0, 0), self.title_surface)

        # 2. Render subtitles if they exist
        if has_subs:
            # Get words and timestamps
            subtitle_word_timestamps = []
            if subtitle_start is not None and subtitle_end is not None:
                colored_words, highlighted_index, subtitle_word_timestamps = self.get_words_at_time_for_subtitle(current_time, subtitle_data, subtitle_start, subtitle_end)
            else:
                colored_words, highlighted_index = self.get_words_at_time(current_time, subtitle_text)

            # Word wrapping
            lines = []
            current_line = []
            current_line_width = 0
            max_width = self.output_width - 80
            
            # Cache fonts to avoid repeated _get_word_font calls
            font_cache = {}
            
            for word_info in colored_words:
                cache_key = (word_info.get('bold'), word_info.get('italic'), word_info.get('sizeMultiplier', 1.0))
                if cache_key not in font_cache:
                    font_cache[cache_key] = self._get_word_font(word_info)
                
                word_font = font_cache[cache_key]
                word_width = self._get_word_width(word_info['text'], word_font)
                space_width = self._get_space_width(word_font)
                
                if not current_line:
                    current_line = [{**word_info, 'font': word_font, 'width': word_width, 'space_width': space_width}]
                    current_line_width = word_width
                elif current_line_width + space_width + word_width <= max_width:
                    current_line.append({**word_info, 'font': word_font, 'width': word_width, 'space_width': space_width})
                    current_line_width += space_width + word_width
                else:
                    lines.append(current_line)
                    current_line = [{**word_info, 'font': word_font, 'width': word_width, 'space_width': space_width}]
                    current_line_width = word_width
            
            if current_line: lines.append(current_line)

            # Layout
            line_height = int(self.font_size * 1.2 * font_scale)
            total_height = len(lines) * line_height

            if self.settings.get('effect_type') == 'flash':
                startY = (pil_image.height - line_height) / 2
            elif self.subtitle_position == 'custom' and self.subtitle_left is not None and self.subtitle_top is not None:
                startY = (self.subtitle_top * scale_y) - (total_height / 2)
            else:
                if self.subtitle_position == 'top': startY = 100 * scale_y
                elif self.subtitle_position == 'middle': startY = (pil_image.height - total_height) / 2
                else: startY = pil_image.height - (100 * scale_y) - total_height

            # Pass 1: Background boxes
            y_copy = startY
            bg_alpha = int(self.bg_opacity * 255)
            if bg_alpha > 0:
                bg_fill = self.bg_color + (bg_alpha,)
                for line in lines:
                    line_width = 0
                    for i, word_info in enumerate(line):
                        line_width += word_info['width']
                        if i < len(line) - 1: line_width += word_info['space_width']

                    if self.settings.get('effect_type') == 'flash' or self.settings.get('effect_type') == 'dynamic_box':
                        xCenter = pil_image.width / 2
                    elif (self.subtitle_position == 'custom' and self.subtitle_left is not None):
                        xCenter = self.subtitle_left * scale_x
                    else:
                        xCenter = pil_image.width / 2

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
                    line_width += word_info['width']
                    if i < len(line) - 1: line_width += word_info['space_width']

                # Calculate line starting X position
                if self.settings.get('effect_type') == 'flash':
                    x_center = pil_image.width / 2
                elif (self.subtitle_position == 'custom' and self.subtitle_left is not None):
                    x_center = self.subtitle_left * scale_x
                else:
                    x_center = pil_image.width / 2

                # In RTL mode, the first word of the line (index 0) is the rightmost
                # We start drawing from the right edge of the line
                if self.is_rtl:
                    x = x_center + (line_width / 2)
                else:
                    x = x_center - (line_width / 2)

                text_y = y_text + (line_height / 2)

                for j, word_info in enumerate(line):
                    word_text = word_info['text']
                    word_color = word_info['color']
                    word_font = word_info['font']
                    
                    is_highlighted = (words_before_line + j) == highlighted_index
                    
                    # Effect modifications
                    w_start = None
                    w_end = None
                    word_index = words_before_line + j
                    if word_index < len(subtitle_word_timestamps):
                        ts = subtitle_word_timestamps[word_index]
                        w_start, w_end = ts['start'], ts['end']
                    
                    effect_mods = self.effects.apply_word_effect(word_info, current_time, w_start, w_end, is_highlighted)
                    
                    word_text = effect_mods['text']
                    word_color = effect_mods['color']
                    if isinstance(word_color, str):
                        word_color = self._hex_to_rgb(word_color)
                    
                    # Update font if scale changed
                    if effect_mods['scale'] != 1.0:
                        word_font = self._get_word_font({**word_info, 'sizeMultiplier': word_info.get('sizeMultiplier', 1.0) * effect_mods['scale']})
                    
                    x_off, y_off = effect_mods['offset_x'], effect_mods['offset_y']
                    
                    # Adjust X position for drawing word (different for LTR vs RTL)
                    # For RTL, x is the right edge of the current word
                    draw_x = x + x_off
                    anchor = "lm" # Left-middle for LTR
                    if self.is_rtl:
                        anchor = "rm" # Right-middle for RTL
                    
                    # Glow (only if glow_layer exists)
                    if glow_layer and word_color == self.text_color and effect_mods['glow_blur'] > 0:
                        glow_draw.text((draw_x, text_y + y_off), word_text, fill=self.glow_color + (255,), font=word_font, anchor=anchor)
                    
                    # Outline
                    if effect_mods.get('custom_render') and self.settings.get('effect_type') == 'shadow_3d':
                        # Shadow 3D needs anchor update too if it's used
                        self.effects.apply_3d_shadow(overlay_draw, draw_x, text_y + y_off, word_text, word_font, word_color + (255,), anchor=anchor)
                    else:
                        # Draw outline using native stroke if possible, fallback to loop
                        try:
                            overlay_draw.text((draw_x, text_y + y_off), word_text, fill=word_color + (255,), font=word_font, anchor=anchor, stroke_width=2, stroke_fill=self.outline_color + (255,))
                        except TypeError:
                            offsets = [(-2,-2), (2,-2), (-2,2), (2,2)]
                            for dx, dy in offsets:
                                overlay_draw.text((draw_x + dx, text_y + y_off + dy), word_text, fill=self.outline_color + (255,), font=word_font, anchor=anchor)
                    
                    # Dynamic Box
                    if self.settings.get('effect_type') == 'dynamic_box' and is_highlighted:
                        bbox = overlay_draw.textbbox((draw_x, text_y + y_off), word_text, font=word_font, anchor=anchor)
                        pad = 10
                        overlay_draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=self.text_color+(255,))
                        word_color = self._hex_to_rgb(self.settings.get('bgColor', '#000000'))

                    # Progressive Fill (Needs RTL awareness for fill direction)
                    if self.settings.get('effect_type') == 'progressive_fill' and is_highlighted:
                        overlay_draw.text((draw_x, text_y + y_off), word_text, fill=self.primary_color + (255,), font=word_font, anchor=anchor)
                        prog = max(0, min(1, (current_time - w_start) / (w_end - w_start)))
                        bbox = overlay_draw.textbbox((draw_x, text_y + y_off), word_text, font=word_font, anchor=anchor)
                        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                        if tw > 0 and th > 0:
                            word_layer = Image.new("RGBA", (int(tw)+20, int(th)+20), (0,0,0,0))
                            word_draw = ImageDraw.Draw(word_layer)
                            # Draw word on layer for masking
                            word_draw.text((10 if not self.is_rtl else tw + 10, 10 + th/2), word_text, fill=word_color+(255,), font=word_font, anchor=anchor)
                            mask = self.effects.get_progressive_fill_mask(word_layer.width, word_layer.height, prog, rtl=self.is_rtl)
                            overlay_layer.paste(word_layer, (int(bbox[0])-10, int(bbox[1])-10), mask)
                    else:
                        # Final draw with normalized color
                        overlay_draw.text((draw_x, text_y + y_off), word_text, fill=word_color + (255,), font=word_font, anchor=anchor)
                    
                    # Emoji
                    if 'emoji' in effect_mods:
                        emoji_font = self._get_font(int(self.font_size * 1.2), "NotoColorEmoji")
                        overlay_draw.text((draw_x, text_y + y_off - line_height), effect_mods['emoji'], fill=(255,255,255,255), font=emoji_font, anchor="mm")

                    # Update X for next word
                    if self.is_rtl:
                        x -= (word_info['width'] + word_info['space_width'])
                    else:
                        x += (word_info['width'] + word_info['space_width'])

                y_text += line_height
                words_before_line += len(line)

        # Final Compositing
        if glow_layer and self.glow_blur > 0:
            # SMART GLOW: Cache the blur result
            glow_key = f"{subtitle_data}_{highlighted_index}_{self.glow_blur}"
            if glow_key == self._last_glow_key and self._glow_cache is not None:
                glow_layer = self._glow_cache
            else:
                # OPTIMIZATION: Use OpenCV for blur instead of Pillow (5x-10x faster)
                glow_np = np.array(glow_layer)
                k_size = int(self.glow_blur * 2) * 2 + 1 
                glow_np = cv2.GaussianBlur(glow_np, (k_size, k_size), 0)
                glow_layer = Image.fromarray(glow_np)
                self._glow_cache = glow_layer
                self._last_glow_key = glow_key
                
            overlay_layer = Image.alpha_composite(overlay_layer, glow_layer)
        
        pil_image = Image.alpha_composite(pil_image, overlay_layer)
        
        # OPTIMIZATION: Return raw RGB bytes directly for the FFmpeg pipe
        # This skips the extremely slow Numpy array conversion and BGR color swap
        return pil_image.convert("RGB").tobytes()

    def _render_title(self, overlay_layer: Image.Image, title_not_used: str):
        """Render the pre-calculated title surface onto the overlay layer."""
        if hasattr(self, 'title_surface'):
            overlay_layer.paste(self.title_surface, (0, 0), self.title_surface)

    def reset_title_cache(self):
        """Reset title cache when title settings change."""
        self._title_pre_rendered = False
        if hasattr(self, 'title_surface'):
            del self.title_surface

    def _get_word_font(self, word_info: Dict) -> ImageFont.FreeTypeFont:
        """Get the font for a specific word based on its style."""
        size = int(self.font_size * word_info.get('sizeMultiplier', 1.0))
        
        # Word is bold if specifically tagged OR if global font_weight is bold
        is_bold = word_info.get('bold', False) or (self.font_weight == 'bold')
        weight = "Bold" if is_bold else "Regular"
        
        style = "Italic" if word_info.get('italic', False) else ""
        return self._get_font(size, f"{self.font_name}:{weight}{style}")

    def _get_font(self, size: int, font_name: str = None) -> ImageFont.FreeTypeFont:
        """Get font with fallbacks and multi-level caching."""
        if font_name is None: font_name = self.font_name

        # Force Arabic-compatible font for RTL text if default Arial is requested
        if getattr(self, 'is_rtl', False) and font_name.startswith('Arial'):
            font_name = 'DejaVu Sans:style=Bold' if 'Bold' in font_name else 'DejaVu Sans'

        # 1. Check object cache first (fastest)
        cache_key = (font_name, size)
        if cache_key in self._font_obj_cache:
            return self._font_obj_cache[cache_key]
        
        # 2. Check path cache
        font_path = self._font_path_cache.get(font_name)
        
        if not font_path:
            # Try to find the path
            direct_paths = [
                f"/usr/share/fonts/truetype/{font_name}.ttf",
                f"/usr/share/fonts/truetype/{font_name}.ttc",
            ]
            for path in direct_paths:
                if os.path.exists(path):
                    font_path = path
                    break
            
            if not font_path:
                try:
                    result = subprocess.run(['fc-match', '-f', '%{file}', font_name], capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        match_path = result.stdout.strip()
                        if os.path.exists(match_path):
                            font_path = match_path
                except: pass
            
            if not font_path:
                fallbacks = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
                ]
                if ":Bold" in font_name or 'Bold' in font_name:
                    fallbacks = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
                    ] + fallbacks
                for path in fallbacks:
                    if os.path.exists(path):
                        font_path = path
                        break
            
            if font_path:
                self._font_path_cache[font_name] = font_path
        
        # 3. Create font object and cache it
        try:
            if font_path:
                font_obj = ImageFont.truetype(font_path, size)
            else:
                font_obj = ImageFont.load_default()
        except:
            font_obj = ImageFont.load_default()
            
        self._font_obj_cache[cache_key] = font_obj
        return font_obj

    def _get_word_width(self, text: str, font) -> int:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]

    def _get_space_width(self, font) -> int:
        return self._get_word_width(' ', font)

    def release(self):
        if self.cap: self.cap.release()


def render_canvas_karaoke_video(video_path, word_timestamps_path, subtitle_srt_path, output_path, start_time, end_time, settings, progress_callback=None, is_already_trimmed=False):
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
    
    # Standard FPS
    fps = renderer.fps if 0 < renderer.fps <= 120 else 30
    if abs(fps - 23.976) < 0.1: fps = 23.976
    elif abs(fps - 24) < 0.1: fps = 24
    elif abs(fps - 25) < 0.1: fps = 25
    elif abs(fps - 29.97) < 0.1: fps = 29.97
    elif abs(fps - 30) < 0.1: fps = 30
    elif abs(fps - 50) < 0.1: fps = 50
    elif abs(fps - 59.94) < 0.1: fps = 59.94
    elif abs(fps - 60) < 0.1: fps = 60
    
    total_frames = int((end_time - start_time) * fps)
    
    import tempfile, shutil
    temp_dir = Path(tempfile.mkdtemp(prefix='short_renderer_'))
    
    abs_output_path = str(Path(output_path).absolute())
    abs_srt_path = str(Path(subtitle_srt_path).absolute())
    logger.info(f"Final Render Pass: input={video_path}, srt={abs_srt_path}, output={abs_output_path}")
    
    # Use a unique temporary output path to avoid 'Output same as Input' errors
    temp_output_path = temp_dir / f"final_render_{os.getpid()}.mp4"

    # If the video is already trimmed (segment), we don't need to extract it again
    if is_already_trimmed:
        temp_segment = Path(video_path).absolute()
        logger.info(f"Video already trimmed. Skipping extraction. Using: {temp_segment}")
    else:
        # Extract segment and RESIZE to vertical in one go (much faster than OpenCV per-frame)
        temp_segment = temp_dir / "segment.mp4"
        logger.info(f"Extracting and resizing segment {start_time}-{end_time} to vertical {renderer.output_width}x{renderer.output_height}...")
        
        # FFmpeg filter for vertical crop/scale
        vf_filter = f"scale={renderer.output_width}:{renderer.output_height}:force_original_aspect_ratio=increase,crop={renderer.output_width}:{renderer.output_height}:(iw-ow)/2:(ih-oh)/2"
        
        abs_video_path = str(Path(video_path).absolute())

        extract_cmd = [
            'ffmpeg', '-y', '-ss', str(start_time), '-i', abs_video_path,
            '-t', str(end_time - start_time), 
            '-vf', vf_filter,
            '-r', str(fps),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', 
            '-c:a', 'aac', '-b:a', '192k',
            str(temp_segment.absolute())
        ]
        subprocess.run(extract_cmd, capture_output=True, check=True)
    
    segment_renderer = UniversalSubtitleRenderer(str(temp_segment.absolute()), word_timestamps, settings, formatting)
    
    # FFmpeg Pipe
    # If already trimmed, input is just temp_segment, no seeking/duration needed for second input
    if is_already_trimmed:
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{renderer.output_width}x{renderer.output_height}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-i', '-',
            '-i', str(temp_segment.absolute()), # For audio
            '-map', '0:v:0', '-map', '1:a:0?', 
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', 
            str(temp_output_path.absolute())
        ]
    else:
        abs_video_path = str(Path(video_path).absolute())
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{renderer.output_width}x{renderer.output_height}', '-pix_fmt', 'rgb24', '-r', str(fps),
            '-i', '-',
            '-ss', str(start_time), '-i', abs_video_path, # For audio
            '-t', str(end_time - start_time),
            '-map', '0:v:0', '-map', '1:a:0?', 
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', 
            str(temp_output_path.absolute())
        ]
    
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Determine if SRT is relative (starts near 0) or absolute (starts near start_time)
    # Scan first subtitle to check
    is_relative_srt = False
    
    # Heuristic 1: Filename suggests theme-specific (usually relative)
    if 'theme_' in str(subtitle_srt_path).lower() or 'adjust' in str(subtitle_srt_path).lower():
        is_relative_srt = True
        logger.info("Assuming relative SRT based on theme-specific filename")
    
    if subtitles:
        # Heuristic 2: First subtitle starts very early while theme starts much later
        first_sub_start = subtitles[0]['start']
        if first_sub_start < 5.0 and start_time > 10.0:
            # If we are using a 'main' srt, this might still be absolute.
            # But theme SRTs definitely start near 0.
            # Let's verify by checking if ANY subtitle exists near start_time
            has_subs_near_start = any(abs(s['start'] - start_time) < 5.0 for s in subtitles)
            
            if not has_subs_near_start:
                is_relative_srt = True
                logger.info(f"Detected relative SRT timing: first sub at {first_sub_start}s, theme at {start_time}s")
            else:
                is_relative_srt = False
                logger.info(f"Detected absolute SRT timing: found subtitles near theme start {start_time}s")

    # Build a time-indexed subtitle lookup for O(1) lookup instead of O(n)
    subtitle_index = {}  # Maps frame index to subtitle
    subtitles_by_start = sorted(subtitles, key=lambda s: s['start'])

    try:
        for frame_idx in range(total_frames):
            # current_time is absolute time in the ORIGINAL video
            current_time = start_time + (frame_idx / fps)
            # relative_time is time from the start of this segment
            relative_time = frame_idx / fps

            ret, frame = segment_renderer.cap.read()
            if not ret: break

            # Use relative lookup for relative SRTs, absolute for others
            lookup_time = relative_time if is_relative_srt else current_time

            # Fast subtitle lookup using pointer
            subtitle_text, subtitle_start, subtitle_end, subtitle_seq = "", None, None, None

            # Start search from last known position
            for i in range(segment_renderer._last_sub_index, len(subtitles_by_start)):
                sub = subtitles_by_start[i]
                if sub['start'] <= lookup_time <= sub['end']:
                    subtitle_text, subtitle_seq = sub['text'], sub.get('sequence')
                    segment_renderer._last_sub_index = i
                    # SHIFT relative timestamps to absolute for word-lookup
                    if is_relative_srt:
                        subtitle_start, subtitle_end = sub['start'] + start_time, sub['end'] + start_time
                    else:
                        subtitle_start, subtitle_end = sub['start'], sub['end']
                    break
                elif sub['start'] > lookup_time:
                    # Not reached this sub yet
                    break
                elif i == len(subtitles_by_start) - 1 and lookup_time > sub['end']:
                    # Past the last subtitle
                    pass

            # For the renderer, we always use current_time (absolute) 
            # because the word_timestamps file is always absolute.
            rendered_frame = segment_renderer.render_frame(frame, current_time, subtitle_text, subtitle_start, subtitle_end, subtitle_seq)
            
            try:
                process.stdin.write(rendered_frame)
            except (BrokenPipeError, IOError):
                stdout, stderr = process.communicate()
                err_msg = stderr.decode() if stderr else "Unknown FFmpeg failure"
                logger.error(f"FFmpeg Pipe Broken: {err_msg}")
                raise Exception(f"FFmpeg failed during rendering: {err_msg[:200]}")
            
            if progress_callback and frame_idx % 30 == 0:
                progress_callback((frame_idx / total_frames) * 100, "rendering", f"Frame {frame_idx}/{total_frames}")

        process.stdin.close()
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg error: {stderr.decode()}")
            return False
            
        # Move successful render to final output
        if os.path.exists(abs_output_path) and abs_output_path != str(temp_output_path.absolute()):
            os.remove(abs_output_path)
        shutil.move(str(temp_output_path.absolute()), abs_output_path)
        logger.info(f"Final Render Pass complete! File saved to: {abs_output_path}")
            
        return True
    finally:
        if process.poll() is None:
            process.kill()
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
