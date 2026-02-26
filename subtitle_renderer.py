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

        self.fps = 30.0 # Force 30.0 for internal math
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Rendering settings (always 1080x1920 vertical)
        self.output_width = 1080
        self.output_height = 1920
        
        # Scale factors relative to 1080x1920 logical canvas
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        # Pixel-to-Point Correction: PIL fonts use 72 DPI (Points), Browsers use 96 DPI (Pixels).
        # To make "Size 80" in the browser match "Size 80" in PIL, we use 72/96 = 0.75.
        self.pixel_to_pt = 0.75
        
        # Robustly convert numerical settings
        def to_int(val, default):
            try: 
                if val is None or str(val).strip() == "": return default
                return int(float(val))
            except: return default
            
        def to_float(val, default):
            try: 
                if val is None or str(val).strip() == "": return default
                return float(val)
            except: return default
            
        def get_val(keys, default):
            if isinstance(keys, str): keys = [keys]
            for k in keys:
                v = settings.get(k)
                if v is not None and str(v).strip() != "":
                    return v
            return default

        # Use exact values from settings/adjust.md
        self.font_size = to_int(get_val(['fontSize', 'subtitle_font_size'], None), 100)
        self.font_name = get_val(['fontName', 'subtitle_font_name'], 'Avenir Black')
        self.max_subtitle_lines = to_int(get_val('max_subtitle_lines', None), 2)

        # DEBUG: Log what font was loaded
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"DEBUG: CanvasKaraokeRenderer initialized with font_name={self.font_name}, fontSize={self.font_size}")
        
        # Colors
        self.text_color = self._hex_to_rgb(get_val(['textColor', 'textColor'], '#ffff00'))
        self.primary_color = self._hex_to_rgb(get_val(['primaryColor', 'subtitle_primary_color'], '#ffffff'))
        self.past_color = self._hex_to_rgb(get_val(['pastColor', 'pastColor'], '#cccccc'))
        self.outline_color = self._hex_to_rgb(get_val(['outlineColor', 'outlineColor'], '#000000'))
        
        # Glow
        self.glow_color = self._hex_to_rgb(get_val(['glowColor', 'textColor'], '#ffff00'))
        self.glow_blur = to_int(get_val(['glowBlur', 'glowBlur'], None), 0)

        # Background
        self.bg_color = self._hex_to_rgb(get_val(['bgColor', 'subtitle_bg_color'], '#000000'))
        self.bg_opacity = to_float(get_val(['bgOpacity', 'subtitle_bg_opacity'], None), 0.70)
        if self.bg_opacity > 1.0: self.bg_opacity /= 100.0
        
        # Force Normal/Karaoke mode if any highlight effect is chosen
        self.mode = settings.get('mode', 'standard')
        if self.mode == 'standard' and (self.effects.effect_type != 'none' or settings.get('textColor')):
            self.mode = 'normal'
        
        self.font_weight = settings.get('font_weight')
        if not self.font_weight:
            self.font_weight = 'bold' if settings.get('subtitle_bold') else 'normal'

        # Subtitle positioning
        self.subtitle_position = get_val('subtitle_position', 'bottom')
        self.subtitle_left = to_int(get_val('subtitle_left', None), 540)
        self.subtitle_top = to_int(get_val('subtitle_top', None), 1600)
        self.subtitle_h_align = get_val(['subtitle_h_align', 'horizontal_align'], 'center')
        self.subtitle_v_align = get_val(['subtitle_v_align', 'vertical_align'], 'bottom')
        
        self._subtitle_word_cache = {}
        self._overlay_layer = None
        self._font_path_cache = {}
        self._font_obj_cache = {}

        # Title settings
        self.show_title = settings.get('show_title') is True
        self.title_text = get_val('title', '')
        self.title_top = to_int(get_val('title_top', None), 150)
        self.title_text_color = self._hex_to_rgb(get_val('title_text_color', '#00ff9d'))
        self.title_font_weight = str(get_val('title_font_weight', '800'))
        self.title_bg_type = get_val('title_bg_type', 'gradient')
        self.title_all_caps = settings.get('title_all_caps') is not False
        self.title_outline_width = to_float(get_val('title_outline_width', None), 0)
        self.title_outline_color = self._hex_to_rgb(get_val('title_outline_color', '#000000'))
        self.title_font_size_pref = to_int(get_val('title_font_size', None), 67) # Exact match to adjust.md

        # Words
        self.words_by_time = sorted(self.word_timestamps, key=lambda w: w['start'])
        self.is_rtl = self._is_arabic(self.word_timestamps)
        self._title_pre_rendered = False

    def _is_arabic(self, word_timestamps: List[Dict]) -> bool:
        if not word_timestamps: return False
        sample = word_timestamps[:20]
        arabic_chars = 0
        total_chars = 0
        for w in sample:
            txt = w.get('word', '')
            total_chars += len(txt)
            for char in txt:
                if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
                    arabic_chars += 1
        return (arabic_chars > total_chars * 0.3) if total_chars > 0 else False

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        if not hex_color or not str(hex_color).startswith('#'): return (255, 255, 255)
        try:
            hex_color = str(hex_color).lstrip('#')
            if len(hex_color) == 3: hex_color = ''.join([c*2 for c in hex_color])
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except: return (255, 255, 255)

    def _get_font(self, size: int, font_name: str = None) -> ImageFont.FreeTypeFont:
        if font_name is None: font_name = self.font_name
        if self.is_rtl and 'Arial' in font_name: font_name = 'DejaVu Sans'

        family = font_name.split(':')[0]
        cache_key = (family, size)
        if cache_key in self._font_obj_cache: return self._font_obj_cache[cache_key]

        font_path = self._font_path_cache.get(family)
        if not font_path:
            # First, check media/fonts directory for custom fonts
            import os
            # subtitle_renderer.py is in the project root, so just use dirname once
            media_fonts_dir = os.path.join(os.path.dirname(__file__), 'media', 'fonts')
            print(f"[FONT DEBUG] Looking for font '{family}', media_fonts_dir={media_fonts_dir}, exists={os.path.exists(media_fonts_dir)}")

            if os.path.exists(media_fonts_dir):
                # Try exact match
                exact_path = os.path.join(media_fonts_dir, f"{family}.ttf")
                print(f"[FONT DEBUG] Exact path: {exact_path}, exists={os.path.exists(exact_path)}")
                if os.path.exists(exact_path):
                    font_path = exact_path
                else:
                    # Try case-insensitive match
                    try:
                        for f in os.listdir(media_fonts_dir):
                            target = f"{family.lower()}.ttf".replace('_', ' ').replace('-', ' ')
                            current = f.lower().replace('_', ' ').replace('-', ' ')
                            if current == target:
                                font_path = os.path.join(media_fonts_dir, f)
                                print(f"[FONT DEBUG] Found via fuzzy match: {font_path}")
                                break
                    except: pass

            # If not found in media/fonts, try fc-match for system fonts
            if not font_path:
                try:
                    res = subprocess.run(['fc-match', '-f', '%{file}', family], capture_output=True, text=True)
                    if res.returncode == 0 and res.stdout.strip():
                        font_path = res.stdout.strip()
                        print(f"[FONT DEBUG] fc-match returned: {font_path}")
                except: pass

            # Fallback to system fonts
            if not font_path or not os.path.exists(font_path):
                fallbacks = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
                for p in fallbacks:
                    if os.path.exists(p):
                        font_path = p
                        print(f"[FONT DEBUG] Using fallback: {font_path}")
                        break

            if font_path: self._font_path_cache[family] = font_path

        print(f"[FONT DEBUG] Final font path for '{family}': {font_path}")
        try:
            if font_path: font_obj = ImageFont.truetype(font_path, size)
            else: font_obj = ImageFont.load_default()
        except: font_obj = ImageFont.load_default()

        self._font_obj_cache[cache_key] = font_obj
        return font_obj

    def _get_word_font(self, word_info: Dict) -> ImageFont.FreeTypeFont:
        # Scale font by pixel-to-point correction (0.75x) to match browser visual size perfectly.
        render_size = int(self.font_size * word_info.get('sizeMultiplier', 1.0) * self.pixel_to_pt)
        return self._get_font(render_size, self.font_name)

    def _get_lines(self, colored_words: List[Dict], max_w: int = 940) -> List[List[Dict]]:
        """Splits colored words into lines based on width."""
        lines = [[]]
        cur_w = 0
        
        # We need a dummy draw object for measurements
        dummy_img = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        for i, w_info in enumerate(colored_words):
            f = self._get_word_font(w_info)
            bbox = draw.textbbox((0, 0), w_info['text'], font=f)
            w_w = bbox[2] - bbox[0]
            s_w = draw.textbbox((0, 0), " ", font=f)[2] - draw.textbbox((0, 0), " ", font=f)[0]
            
            if not lines[-1] or cur_w + s_w + w_w <= max_w:
                lines[-1].append({**w_info, 'font': f, 'width': w_w, 'space': s_w})
                cur_w += (s_w + w_w)
            else:
                lines.append([{**w_info, 'font': f, 'width': w_w, 'space': s_w}])
                cur_w = w_w
        return lines

    def split_segments_by_max_lines(self, segments: List[Dict], word_timestamps: List[Dict]) -> List[Dict]:
        """Ensures each segment results in no more than self.max_subtitle_lines."""
        if not segments: return []
        
        final_segments = []
        for seg in segments:
            # Get words for this segment
            seg_text = seg.get('text', '')
            words = seg_text.split()
            if not words:
                final_segments.append(seg)
                continue
            
            # Map words to timestamps if available
            seg_words_ts = []
            s_abs = seg['start'] if isinstance(seg['start'], (int, float)) else parse_srt_time(seg['start'])
            e_abs = seg['end'] if isinstance(seg['end'], (int, float)) else parse_srt_time(seg['end'])
            
            for w in word_timestamps:
                if w['start'] < e_abs - 0.02 and w['end'] > s_abs + 0.02:
                    seg_words_ts.append(w)
            
            # Create word info for measurement
            colored_words = [{'text': w, 'sizeMultiplier': 1.0} for w in words]
            lines = self._get_lines(colored_words)
            
            if len(lines) <= self.max_subtitle_lines:
                final_segments.append(seg)
                continue
            
            # Need to split. Let's group lines into chunks of max_subtitle_lines
            for chunk_idx, i in enumerate(range(0, len(lines), self.max_subtitle_lines)):
                chunk = lines[i : i + self.max_subtitle_lines]
                chunk_words = []
                for line in chunk:
                    for w in line:
                        chunk_words.append(w['text'])
                
                first_word_idx = sum(len(l) for l in lines[:i])
                last_word_idx = first_word_idx + sum(len(l) for l in chunk) - 1
                
                s_t = s_abs
                e_t = e_abs
                
                if seg_words_ts:
                    if first_word_idx < len(seg_words_ts):
                        s_t = seg_words_ts[first_word_idx]['start']
                    elif first_word_idx > 0:
                        s_t = seg_words_ts[-1]['end']
                        
                    if last_word_idx < len(seg_words_ts):
                        e_t = seg_words_ts[last_word_idx]['end']
                    
                    if i + self.max_subtitle_lines < len(lines):
                        next_first_idx = last_word_idx + 1
                        if next_first_idx < len(seg_words_ts):
                            e_t = seg_words_ts[next_first_idx]['start']
                else:
                    total_words = len(words)
                    duration = e_abs - s_abs
                    s_t = s_abs + (first_word_idx / total_words) * duration
                    e_t = s_abs + ((last_word_idx + 1) / total_words) * duration
                
                # Format for UI/SRT
                def format_time(secs):
                    h = int(secs // 3600)
                    m = int((secs % 3600) // 60)
                    s = secs % 60
                    return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')

                new_seg = {
                    'start': format_time(s_t) if isinstance(seg['start'], str) else s_t,
                    'end': format_time(e_t) if isinstance(seg['end'], str) else e_t,
                    'text': ' '.join(chunk_words)
                }
                
                # Preserve and adjust sequence number for UI
                if 'sequence' in seg:
                    new_seg['sequence'] = f"{seg['sequence']}.{chunk_idx+1}"
                
                final_segments.append(new_seg)
                
        return final_segments

    def get_words_at_time_for_subtitle(self, current_time: float, subtitle_data: str, subtitle_start: float, subtitle_end: float) -> Tuple[List[Dict], int, List[Dict]]:
        words = subtitle_data.split()
        cache_key = f"{subtitle_start}_{subtitle_end}_{subtitle_data[:50]}"
        if cache_key in self._subtitle_word_cache:
            sub_word_ts = self._subtitle_word_cache[cache_key]
        else:
            sub_word_ts = []
            for w in self.words_by_time:
                if w['start'] < subtitle_end - 0.05 and w['end'] > subtitle_start + 0.05:
                    sub_word_ts.append(w)
            self._subtitle_word_cache[cache_key] = sub_word_ts

        highlighted_index = -1
        for i, ts in enumerate(sub_word_ts):
            if ts['start'] <= current_time < ts['end']:
                highlighted_index = i
                break

        colored_words = []
        for j, word_txt in enumerate(words):
            word_color = self.primary_color
            if self.mode != 'standard' and j < len(sub_word_ts):
                ts = sub_word_ts[j]
                if current_time >= ts['end']:
                    word_color = self.past_color if self.mode == 'cumulative' else self.primary_color
                elif current_time >= ts['start']:
                    word_color = self.text_color
            
            colored_words.append({'text': word_txt, 'color': word_color, 'sizeMultiplier': 1.0})
        return colored_words, highlighted_index, sub_word_ts

    def render_frame(self, frame: np.ndarray, current_time: float, subtitle_text: str, subtitle_start: float = None, subtitle_end: float = None, subtitle_seq: int = None) -> np.ndarray:
        if self.show_title and not self._title_pre_rendered:
            self._precalculate_title_properties()
            self._title_pre_rendered = True

        if frame.shape[1] != 1080 or frame.shape[0] != 1920:
            frame_resized = cv2.resize(frame, (1080, 1920))
        else:
            frame_resized = frame

        pil_image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        if self.show_title and hasattr(self, 'title_surface'):
            overlay.paste(self.title_surface, (0, 0), self.title_surface)

        if subtitle_text and subtitle_text.strip():
            colored_words, highlight_idx, sub_ts = self.get_words_at_time_for_subtitle(current_time, subtitle_text, subtitle_start, subtitle_end)
            
            lines = self._get_lines(colored_words)
            # Hard limit lines to max_subtitle_lines
            if len(lines) > self.max_subtitle_lines:
                lines = lines[:self.max_subtitle_lines]
            
            line_h = self.font_size * self.pixel_to_pt * 1.3
            total_h = len(lines) * line_h
            
            if self.subtitle_position == 'top': start_y = 300
            elif self.subtitle_position == 'middle': start_y = (1920 - total_h) / 2
            elif self.subtitle_position == 'custom': start_y = self.subtitle_top - (total_h / 2)
            else: start_y = 1920 - 300 - total_h
            
            bg_alpha = int(self.bg_opacity * 255)
            if bg_alpha > 0:
                y = start_y
                for line in lines:
                    line_w = sum(w['width'] + w['space'] for w in line) - line[-1]['space']
                    x_c = self.subtitle_left if self.subtitle_position == 'custom' else 540
                    x1 = x_c - line_w if self.subtitle_h_align == 'left' else (x_c if self.subtitle_h_align == 'right' else x_c - (line_w / 2))
                    draw.rectangle([x1 - 25, y, x1 + line_w + 25, y + line_h], fill=self.bg_color + (bg_alpha,))
                    y += line_h

            y = start_y
            word_counter = 0
            for line in lines:
                line_w = sum(w['width'] + w['space'] for w in line) - line[-1]['space']
                x_c = self.subtitle_left if self.subtitle_position == 'custom' else 540
                cur_x = x_c - line_w if self.subtitle_h_align == 'left' else (x_c if self.subtitle_h_align == 'right' else x_c - (line_w / 2))
                if self.is_rtl: cur_x += line_w

                for w in line:
                    is_active = word_counter == highlight_idx
                    eff = self.effects.apply_word_effect(w, current_time, None, None, is_active)
                    w_col = eff['color']
                    if isinstance(w_col, str): w_col = self._hex_to_rgb(w_col)
                    
                    draw_x, draw_y = cur_x + eff['offset_x'], y + (line_h / 2) + eff['offset_y']
                    anchor = "rm" if self.is_rtl else "lm"
                    
                    if eff.get('opacity', 1.0) > 0.1:
                        draw.text((draw_x, draw_y), w['text'], fill=w_col + (255,), font=w['font'], anchor=anchor, stroke_width=3, stroke_fill=self.outline_color + (255,))
                    
                    step = (w['width'] + w['space'])
                    if self.is_rtl: cur_x -= step
                    else: cur_x += step
                    word_counter += 1
                y += line_h

        final = Image.alpha_composite(pil_image.convert("RGBA"), overlay)
        return final.convert("RGB").tobytes()

    def _precalculate_title_properties(self):
        if not self.title_text: return
        img = Image.new("RGBA", (1080, 1920), (0,0,0,0))
        d = ImageDraw.Draw(img)
        f = self._get_font(int(self.title_font_size_pref * self.pixel_to_pt), self.font_name)
        txt = self.title_text.upper() if self.title_all_caps else self.title_text
        bbox = d.textbbox((540, self.title_top), txt, font=f, anchor="mm")
        pad = 30
        d.rounded_rectangle([bbox[0]-pad, bbox[1]-pad/2, bbox[2]+pad, bbox[3]+pad/2], radius=15, fill=(0,0,0,180))
        d.text((540, self.title_top), txt, fill=self.title_text_color+(255,), font=f, anchor="mm", stroke_width=3, stroke_fill=self.title_outline_color+(255,))
        self.title_surface = img

    def release(self):
        if self.cap: self.cap.release()

def generate_dynamic_subtitles(word_timestamps: List[Dict], max_words=8, max_pause=0.8) -> List[Dict]:
    """
    Groups individual words into subtitle-like segments based on count and pauses.
    Provides a high-accuracy fallback when SRT files are incorrect.
    """
    if not word_timestamps: return []
    
    segments = []
    current_segment = []
    
    for i, word in enumerate(word_timestamps):
        current_segment.append(word)
        
        # Check if we should break the segment
        should_break = False
        
        # 1. Word count limit
        if len(current_segment) >= max_words:
            should_break = True
            
        # 2. Significant pause detection
        if not should_break and i < len(word_timestamps) - 1:
            pause = word_timestamps[i+1]['start'] - word['end']
            if pause > max_pause:
                should_break = True
                
        # 3. Punctuation break
        if not should_break and word['word'].strip().endswith(('.', '?', '!', ':')):
            should_break = True
            
        if should_break:
            segments.append({
                'start': current_segment[0]['start'],
                'end': current_segment[-1]['end'],
                'text': ' '.join([w['word'].strip() for w in current_segment])
            })
            current_segment = []
            
    # Add final segment
    if current_segment:
        segments.append({
            'start': current_segment[0]['start'],
            'end': current_segment[-1]['end'],
            'text': ' '.join([w['word'].strip() for w in current_segment])
        })
        
    return segments

def render_canvas_karaoke_video(video_path, word_timestamps_path, subtitle_srt_path, output_path, start_time, end_time, settings, progress_callback=None, is_already_trimmed=False, srt_mode='auto'):
    # 1. Load high-accuracy word data first
    with open(word_timestamps_path, 'r') as f: all_words = json.load(f).get('words', [])
    
    # CRITICAL: Filter words to ONLY the ones within our theme range (+ buffer)
    # Use a larger 0.5s look-back buffer to ensure we catch the first segment's words
    words = [w for w in all_words if (start_time - 0.5) <= w['start'] <= (end_time + 0.5)]
    
    # 2. Load SRT as the primary source for segment boundaries
    # This preserves capitalization and specific word groupings from the original transcription
    subtitles = _parse_srt(subtitle_srt_path)
    
    # Check if we should use dynamic grouping (only if SRT is missing or empty)
    if not subtitles:
        logger.info("SRT file missing or empty. Falling back to dynamic word grouping.")
        subtitles = generate_dynamic_subtitles(words)
        srt_mode = 'absolute'
    else:
        logger.info(f"Using {len(subtitles)} segments from SRT file.")

    # 3. Load manual edits
    edits = {}
    srt_p = Path(subtitle_srt_path)
    t_num = settings.get("theme_number", 0)
    edit_file = srt_p.parent / 'shorts' / f'theme_{int(t_num):03d}_edits.json'
    if not edit_file.exists(): edit_file = srt_p.parent / srt_p.name.replace('.srt', '_edits.json')
    if edit_file.exists():
        with open(edit_file, 'r', encoding='utf-8') as f: edits = json.load(f)

    # Apply manual edits to the segments
    if edits:
        logger.info(f"Applying {len(edits)} manual edits to SRT segments...")
        
        # We'll build a final list of segments by matching edits to SRT blocks
        # or replacing SRT blocks with edited ones.
        final_segments = []
        theme_meta_start = settings.get('theme_meta_start', 0)
        
        # Use a set to track which edited keys we've handled
        applied_edit_keys = set()
        
        for sub in subtitles:
            # Check if this SRT segment matches an edit
            # Edits are keyed by "start_end"
            match_found = False
            for edit_key, edited_text in edits.items():
                e_start, e_end = map(parse_srt_time, edit_key.split('_'))
                
                # Align edit times with SRT times if needed (relative vs absolute)
                sync_base = theme_meta_start if theme_meta_start > 0 else start_time
                if e_start < sync_base - 1.0:
                    e_start_abs = e_start + sync_base
                    e_end_abs = e_end + sync_base
                else:
                    e_start_abs = e_start
                    e_end_abs = e_end
                
                # If SRT segment overlaps significantly with this edit, use the edit
                overlap_start = max(sub['start'], e_start_abs)
                overlap_end = min(sub['end'], e_end_abs)
                if overlap_end > overlap_start + 0.1: # At least 100ms overlap
                    final_segments.append({
                        'start': sub['start'],
                        'end': sub['end'],
                        'text': edited_text,
                        'is_edited': True
                    })
                    applied_edit_keys.add(edit_key)
                    match_found = True
                    break
            
            if not match_found:
                final_segments.append(sub)
        
        # Add any edits that didn't match existing SRT segments (new blocks)
        for edit_key, edited_text in edits.items():
            if edit_key not in applied_edit_keys:
                e_start, e_end = map(parse_srt_time, edit_key.split('_'))
                sync_base = theme_meta_start if theme_meta_start > 0 else start_time
                if e_start < sync_base - 1.0:
                    e_start += sync_base
                    e_end += sync_base
                
                final_segments.append({
                    'start': e_start,
                    'end': e_end,
                    'text': edited_text,
                    'is_edited': True
                })
        
        # Re-sort everything by start time
        subtitles = sorted(final_segments, key=lambda s: s['start'])
    
    renderer = UniversalSubtitleRenderer(video_path, words, settings)
    
    # ENFORCE MAX LINES: Split segments if they result in too many lines
    subtitles = renderer.split_segments_by_max_lines(subtitles, words)
    
    # CRITICAL: If we are using the master (untrimmed) video, we MUST seek to start_time
    if not is_already_trimmed and start_time > 0:
        logger.info(f"Seeking master video to start_time: {start_time}s")
        renderer.cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)
    
    # --- ROBUST TIMING DETECTION ---
    is_rel = (srt_mode == 'relative')
    if srt_mode == 'auto' and subtitles:
        has_subs_at_theme_time = any(abs(s['start'] - start_time) < 10.0 for s in subtitles)
        if has_subs_at_theme_time: is_rel = False
        else: is_rel = True
        logger.info(f"Auto-detected SRT Mode: {'Relative' if is_rel else 'Absolute'}")

    subtitles_by_start = sorted(subtitles, key=lambda s: s['start'])
    temp_dir = Path("/tmp/render_" + str(os.getpid()))
    temp_dir.mkdir(exist_ok=True)
    tmp_out = temp_dir / "out.mp4"
    
    crf = str(settings.get('export_crf', '22'))
    preset = settings.get('export_preset', 'ultrafast')
    
    ffmpeg_cmd = [
        'ffmpeg', '-nostdin', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', '1080x1920', '-pix_fmt', 'rgb24', '-r', '30', '-i', '-',
        '-i', str(Path(video_path).absolute()),
        '-t', str(end_time - start_time), # Limit audio input length
        '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'libx264', '-preset', preset, '-crf', crf,
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '128k', 
        '-t', str(end_time - start_time), # FINAL duration limit for output
        str(tmp_out)
    ]
    
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    total_f = int((end_time - start_time) * 30)
    
    # Determine the reference time for SRT lookup
    # If the SRT is relative, it's relative to theme_meta_start (the original theme bounds)
    srt_ref_start = settings.get('theme_meta_start', start_time)
    
    for f_idx in range(total_f):
        ret, frame = renderer.cap.read()
        if not ret: break
        
        rel_t = f_idx / 30.0
        abs_t = start_time + rel_t
        lookup_t = abs_t - srt_ref_start if is_rel else abs_t
        
        # Best Match Search with a small look-back buffer (0.5s)
        matched = None
        for s in subtitles_by_start:
            if (s['start'] - 0.5) <= lookup_t <= s['end']:
                matched = s
            elif s['start'] > lookup_t + 0.5: break
        
        txt, s_s, s_e = "", 0, 0
        if matched:
            txt = matched['text']
            s_s = matched['start'] + (srt_ref_start if is_rel else 0)
            s_e = matched['end'] + (srt_ref_start if is_rel else 0)

        raw = renderer.render_frame(frame, abs_t, txt, s_s, s_e)
        try: proc.stdin.write(raw)
        except: break
        
        if progress_callback and f_idx % 30 == 0:
            progress_callback((f_idx / total_f) * 100, "rendering", f"Frame {f_idx}/{total_f}")

    proc.stdin.close()
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        err_msg = stderr.decode('utf-8', errors='replace') if stderr else "No error message"
        logger.error(f"FFmpeg rendering failed: {err_msg}")
        raise Exception(f"Subtitle rendering failed: {err_msg}")
    
    out_p = Path(output_path).absolute()
    if os.path.exists(out_p): os.remove(out_p)
    import shutil
    shutil.move(str(tmp_out), str(out_p))
    shutil.rmtree(temp_dir)
    return True

def _parse_srt(path):
    import re
    if not os.path.exists(path): return []
    # Use utf-8-sig to automatically handle BOM if present, then strip it explicitly to be sure
    try:
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f: content = f.read()
    except:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
    
    # Strip BOM and normalize line endings
    content = content.lstrip('\ufeff').replace('\r\n', '\n').strip()
    
    # Split by double newline to get potential SRT blocks
    blocks = re.split(r'\n\s*\n', content)
    subs = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if not lines: continue
        
        # Look for the timing line (e.g., 00:00:01,000 --> 00:00:02,000)
        for i, line in enumerate(lines):
            time_match = re.search(r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})', line)
            if time_match:
                def p(t):
                    h, m, s = map(float, t.replace(',', '.').split(':'))
                    return h * 3600 + m * 60 + s
                
                start = p(time_match.group(1))
                end = p(time_match.group(2))
                
                # Everything after the timing line is the subtitle text
                text = '\n'.join(lines[i+1:]).strip()
                
                subs.append({
                    'start': start,
                    'end': end,
                    'text': text
                })
                break # Move to next block
                
    return subs

def parse_srt_time(ts):
    if not ts: return 0.0
    h, m, s = ts.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)
