import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
import re
from typing import List, Dict, Tuple, Optional

class SubtitleEffects:
    """Handles various subtitle text effects and animations."""
    
    # Basic word-to-emoji mapping for Auto-Emoji effect
    EMOJI_MAP = {
        "fire": "🔥", "lit": "🔥", "hot": "🔥",
        "love": "❤️", "heart": "❤️",
        "laugh": "😂", "funny": "😂", "lol": "😂",
        "sad": "😢", "cry": "😢",
        "angry": "😡", "mad": "😡",
        "wow": "😮", "amazing": "😮", "surprised": "😮",
        "cool": "😎", "chill": "😎",
        "money": "💰", "cash": "💰", "rich": "💰",
        "goal": "⚽", "soccer": "⚽",
        "basketball": "🏀",
        "brain": "🧠", "think": "🧠", "smart": "🧠",
        "light": "💡", "idea": "💡",
        "stop": "🛑", "halt": "🛑",
        "go": "🚀", "fast": "🚀", "speed": "🚀",
        "time": "⏰", "clock": "⏰", "watch": "⏰",
        "danger": "⚠️", "warning": "⚠️",
        "success": "🏆", "winner": "🏆", "win": "🏆",
        "islam": "🕌", "muslim": "🕌", "prayer": "🕌", "dua": "🤲",
        "quran": "📖", "book": "📖", "read": "📖",
        "allah": "☝️", "god": "☝️"
    }

    def __init__(self, settings: Dict):
        self.settings = settings
        self.effect_type = settings.get('effect_type', 'none') # 'none', 'standard', 'pop', 'slide', 'typewriter', etc.
        self.audio_levels = settings.get('audio_levels', None) # Optional volume data
        
        def to_float(val, default):
            try: return float(val)
            except: return default
            
        self.base_time = to_float(settings.get('base_time'), 0.0) # Start time of the segment
        
    @property
    def active_effects(self) -> bool:
        """Returns True if any complex animation/karaoke effect is active."""
        return self.effect_type not in ['standard', 'none']

    def apply_word_effect(self, word_info: Dict, current_time: float, word_start: Optional[float], word_end: Optional[float], is_highlighted: bool = False) -> Dict:
        """
        Calculates modifications to a word's properties based on the active effect.
        Returns a dict with updated: scale, offset_x, offset_y, opacity, color, text, etc.
        """
        def to_float(val, default):
            try: return float(val)
            except: return default

        res = {
            'scale': 1.0,
            'offset_x': 0,
            'offset_y': 0,
            'opacity': 1.0,
            'color': word_info.get('color'),
            'text': word_info.get('text', ''),
            'glow_blur': to_float(self.settings.get('glowBlur'), 0),
            'font_size_multiplier': to_float(word_info.get('sizeMultiplier'), 1.0),
            'custom_render': False # If True, the effect handles its own rendering
        }
        
        highlight_color = self.settings.get('textColor', '#ffff00')
        primary_color = self.settings.get('primaryColor', '#ffffff')
        
        progress = 0
        is_active = is_highlighted

        if word_start is not None and word_end is not None:
            word_duration = word_end - word_start
            if word_duration > 0:
                progress = (current_time - word_start) / word_duration
            
            # Use timing to determine if active if not already flagged by is_highlighted
            if 0 <= progress <= 1:
                is_active = True
        
        # Only apply highlight color if an effect is active
        show_highlight = is_active and self.effect_type != 'none'
        
        if show_highlight:
            res['color'] = highlight_color
        else:
            # Respect word's own color (rich text) or global primary
            res['color'] = word_info.get('color') or primary_color
        
        # 1. Animation Effects
        if self.effect_type == 'pop' and is_active:
            # Word Pop/Bounce: Scale up and bounce
            bounce = math.sin(max(0, min(1, progress)) * math.pi) * 0.3
            res['scale'] = 1.0 + bounce
            res['offset_y'] = -bounce * 30
            
        elif self.effect_type == 'slide' and is_active:
            # Slide Reveal: Slide up into position
            if progress < 0.4:
                slide_prog = progress / 0.4
                res['offset_y'] = 40 * (1.0 - slide_prog)
                res['opacity'] = slide_prog
                
        elif self.effect_type == 'typewriter' and is_active:
            # Typewriter: Show letters one by one
            chars_to_show = int(len(res['text']) * max(0, min(1, progress)))
            res['text'] = res['text'][:chars_to_show] if chars_to_show > 0 else ""

        # 2. Advanced Karaoke Styles
        if self.effect_type == 'flash':
            # Flash (Center Focus): Only one word at a time
            # Hide non-highlighted words
            if not is_active:
                res['opacity'] = 0.0
                res['text'] = ""
            else:
                res['scale'] = 1.5 # Make center word large
        
        if self.effect_type == 'wave':
            # Wave/Wiggle: Continuous sinusoidal movement
            phase = (current_time * 6) + (word_start * 3)
            res['offset_y'] = math.sin(phase) * 8
            if is_active:
                res['offset_y'] *= 1.5
                
        # 3. Visual Styling Effects
        if self.effect_type == 'shadow_3d' and is_active:
            res['custom_render'] = True # Signals renderer to use special shadow method
            
        if self.effect_type == 'dynamic_box' and is_active:
            # Logic handled in renderer mostly, but we can signal it here
            res['glow_blur'] += 5
            
        if self.effect_type == 'neon_glitch' and is_active:
            # Neon Glitch: Occasional RGB split / jitter
            if int(current_time * 30) % 4 == 0:
                res['offset_x'] = np.random.randint(-8, 9)
                base_glow = to_float(self.settings.get('glowBlur'), 10)
                res['glow_blur'] = base_glow * 2.5
                
        # 4. Audio-Reactive Effects
        if self.effect_type == 'volume_shake' and self.audio_levels:
            # Volume Shake: Intensity tied to audio volume
            idx = int((current_time - self.base_time) * 10)
            if 0 <= idx < len(self.audio_levels):
                level = self.audio_levels[idx]
                if level > 0.2:
                    shake = level * 25
                    res['offset_x'] = np.random.randint(-int(shake), int(shake) + 1)
                    res['offset_y'] = np.random.randint(-int(shake), int(shake) + 1)

        if self.effect_type == 'pulsing_glow':
            # Pulsing Glow: Glow brightness pulses
            pulse = (math.sin(current_time * 12) + 1) / 2
            base_glow = to_float(self.settings.get('glowBlur'), 10)
            res['glow_blur'] = int(base_glow * (0.4 + pulse * 1.2))

        # 5. Emphasis/AI-Driven Effects
        if self.settings.get('keyword_scaling', False):
            clean_text = res['text'].strip('.,!?;:"\'')
            if len(clean_text) > 5 or (clean_text and clean_text[0].isupper()):
                res['font_size_multiplier'] *= 1.25

        if self.settings.get('auto_emoji', False) and is_active:
            clean_word = res['text'].lower().strip('.,!?;:"\'')
            if clean_word in self.EMOJI_MAP:
                res['emoji'] = self.EMOJI_MAP[clean_word]

        return res

    def get_progressive_fill_mask(self, width: int, height: int, progress: float, rtl: bool = False) -> Image.Image:
        """Creates a mask for progressive color fill (left to right by default, right to left for RTL)."""
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        fill_width = int(width * progress)
        
        if rtl:
            # Fill from right to left
            draw.rectangle([width - fill_width, 0, width, height], fill=255)
        else:
            # Fill from left to right
            draw.rectangle([0, 0, fill_width, height], fill=255)
        return mask

    def apply_3d_shadow(self, draw: ImageDraw.Draw, x: float, y: float, text: str, font: ImageFont.FreeTypeFont, color: Tuple[int, int, int, int], anchor: str = "lm"):
        """Renders multi-layer shadow for 3D effect."""
        shadow_color = (0, 0, 0, 255)
        for i in range(5, 0, -1):
            draw.text((x + i, y + i), text, fill=shadow_color, font=font, anchor=anchor)
        draw.text((x, y), text, fill=color, font=font, anchor=anchor)
