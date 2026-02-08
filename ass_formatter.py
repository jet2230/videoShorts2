#!/usr/bin/env python3
"""
ASS (Advanced Substation Alpha) Subtitle Formatter

Handles all ASS subtitle generation, formatting, and styling.
Supports HTML to ASS conversion, Arabic RTL, nested styles, and more.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from html import unescape


class ASSFormatter:
    """Handles all ASS subtitle generation and formatting."""

    def __init__(self, settings: Optional[dict] = None):
        """Initialize ASS formatter with optional settings.

        Args:
            settings: Dictionary with settings like resolution, font, etc.
                     If None, uses defaults.
        """
        self.settings = settings or {}
        self.width = int(self.settings.get('video', {}).get('resolution_width', 1080))
        self.height = int(self.settings.get('video', {}).get('resolution_height', 1920))
        self.font_name = self.settings.get('subtitle', {}).get('font_name', 'Arial')
        # Use ass_font_size for ASS files (separate from SRT font_size)
        self.font_size = int(self.settings.get('subtitle', {}).get('ass_font_size', 48))

    # === File Generation ===

    def create_ass_file(self, srt_path: Path, formatting_json_path: Path, output_ass_path: Path) -> bool:
        """Create ASS subtitle file from SRT and formatting JSON.

        Args:
            srt_path: Path to input SRT file
            formatting_json_path: Path to JSON with HTML formatting data
            output_ass_path: Path where ASS file should be written

        Returns:
            True if successful, False otherwise
        """
        # Load formatting data
        with open(formatting_json_path, 'r', encoding='utf-8') as f:
            formatting_data = json.load(f)

        # Check if any cue has a position set - if so, use that for all cues
        # Priority: JSON position > settings.ini defaults
        position_override = None
        for timestamp_key, cue_data in formatting_data.items():
            if isinstance(cue_data, dict) and cue_data.get('position'):
                position_override = cue_data['position']
                break

        # Parse SRT segments
        segments = self._parse_srt_segments(srt_path)

        # Build ASS file
        with open(output_ass_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(self.create_ass_header())
            f.write(self.create_ass_styles(position_override))

            # Write events
            for segment in segments:
                start_time = segment['start']
                end_time = segment['end']

                # Format timestamps for ASS (H:MM:SS.CC)
                start_ass = self._format_timestamp_ass(start_time)
                end_ass = self._format_timestamp_ass(end_time)

                # Find formatting for this timestamp
                # JSON uses "HH:MM:SS.mmm" format (with dot)
                timestamp_json = self._format_timestamp_json(start_time)
                formatting = formatting_data.get(timestamp_json, {})

                # Get text - use html if available, otherwise plain text
                text = formatting.get('html', segment['text'])

                # Parse HTML formatting to ASS tags
                text = self.parse_html_to_ass(text)

                # Escape special ASS characters (but not backslashes in tags)
                text = self._escape_ass_special_chars(text)

                # Write dialogue line
                f.write(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}\n")

        return True

    def create_ass_header(self) -> str:
        """Generate ASS file header."""
        return f"""[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: {self.width}
PlayResY: {self.height}
Timer: 100.0000
WrapStyle: 0
ScaledBorderAndShadow: yes

"""

    def create_ass_styles(self, position_override=None) -> str:
        """Generate ASS style definitions.

        Args:
            position_override: Optional position from JSON ('top', 'middle', 'bottom').
                             If provided, overrides settings.ini defaults.
        """
        # Use position from JSON if available, otherwise use defaults from settings
        alignment = 2  # Default: center
        margin_v = 10   # Default: 10px from bottom

        if position_override:
            # Map position to ASS alignment and margin
            # Alignment: 1=left, 2=center, 3=right, 7=left-top, 8=center-top, 9=right-top, etc.
            # For 9:16 vertical video, we use bottom alignment for subtitles
            if position_override == 'top':
                alignment = 8  # Top-center
                margin_v = 35  # From top
            elif position_override == 'middle':
                alignment = 2  # Center
                margin_v = 0   # Center (no margin)
            elif position_override == 'bottom':
                alignment = 2  # Center
                margin_v = 35  # From bottom (35px default)
        else:
            # Use values from settings.ini
            alignment = int(self.settings.get('subtitle', {}).get('alignment', 2))
            margin_v = int(self.settings.get('subtitle', {}).get('margin_v', 35))

        return f"""[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{self.font_name},{self.font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,0,{alignment},{10},{10},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # === HTML to ASS Conversion ===

    def parse_html_to_ass(self, html_text: str) -> str:
        """Parse HTML formatting and convert to ASS tags.

        For Arabic with partial styling, we reverse the Arabic word order
        BEFORE applying styling, so it displays correctly in RTL.

        Args:
            html_text: HTML string with <span> tags containing styling

        Returns:
            ASS-formatted string with override tags
        """
        # Check if we have Arabic with styling
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in html_text)
        has_styling = '<span' in html_text or '<strong>' in html_text or '<b>' in html_text

        if has_arabic and has_styling:
            # Step 1: Get plain text and words
            plain_text = unescape(re.sub(r'<[^>]+>', '', html_text))
            words = plain_text.split()

            # Step 2: Find Arabic word indices
            arabic_indices = [i for i, w in enumerate(words) if any('\u0600' <= c <= '\u06FF' for c in w)]

            # Step 3: Reverse Arabic words for RTL
            arabic_words_rev = [words[i] for i in arabic_indices][::-1]
            new_words = words[:]
            for idx, rev_word in zip(arabic_indices, arabic_words_rev):
                new_words[idx] = rev_word

            # Step 4: Extract ALL styled segments from HTML
            # Maps: original_word_text -> {color, bold, italic, size}
            word_styles = self._extract_styles_from_html(html_text)

            # Step 5: Map styled words to their NEW positions after reversal
            position_styles = self._map_styled_positions(words, new_words, arabic_indices, word_styles)

            # Step 6: Build result with styles applied to correct positions
            result_parts = []
            for i, word in enumerate(new_words):
                if i in position_styles:
                    styles = position_styles[i]
                    tags = self.get_style_tags(styles)
                    if tags:
                        # Note: No {\r} reset - let styles persist until end of dialogue
                        result_parts.append(''.join(tags) + word)
                    else:
                        result_parts.append(word)
                else:
                    result_parts.append(word)

            return ' '.join(result_parts)

        # Default: simple HTML to ASS conversion (no Arabic)
        return self._simple_html_to_ass(html_text)

    def _extract_styles_from_html(self, html_text: str) -> Dict[str, dict]:
        """Extract all styles from HTML spans.

        Returns a mapping of text -> styles dict.
        For nested spans, only innermost color applies, but other styles stack.
        """
        word_styles = {}

        # Find all <span> tags with style
        span_pattern = r'<span[^>]*style="([^"]*)"[^>]*>(.*?)</span>'
        for match in re.finditer(span_pattern, html_text, re.DOTALL):
            style_content = match.group(1)
            span_text = unescape(re.sub(r'<[^>]+>', '', match.group(2)))  # Handle nested spans

            # Extract styles from this span
            color = None
            size = None
            bold = False
            italic = False

            # Color - handle both hex and rgb formats
            color_match = re.search(r'color:\s*rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', style_content)
            if color_match:
                r, g, b = int(color_match.group(1)), int(color_match.group(2)), int(color_match.group(3))
                color = f'&H00{b:02X}{g:02X}{r:02X}'
            else:
                # Handle hex colors like #ffff00
                hex_match = re.search(r'color:\s*#([0-9a-fA-F]{6})', style_content)
                if hex_match:
                    hex_color = hex_match.group(1)
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    color = f'&H00{b:02X}{g:02X}{r:02X}'

            # Font size
            size_match = re.search(r'font-size:\s*([\d.]+)em', style_content)
            if size_match:
                size = round(self.font_size * float(size_match.group(1)))

            # Bold/italic
            if re.search(r'font-weight:\s*bold', style_content) or re.search(r'font-weight:\s*700', style_content):
                bold = True
            if re.search(r'font-style:\s*italic', style_content):
                italic = True

            # For nested spans with colors, only INNERMOST (last) color applies
            # But other styles (bold, italic, size) can stack
            # Split multi-word styled text into individual words
            span_words = span_text.split()
            for word in span_words:
                if word in word_styles:
                    existing = word_styles[word]
                    # Update color (innermost takes precedence)
                    if color is not None:
                        existing['color'] = color
                    # Properly stack other styles (bold, italic, size)
                    if bold:
                        existing['bold'] = True
                    if italic:
                        existing['italic'] = True
                    if size is not None:
                        existing['size'] = size
                else:
                    word_styles[word] = {
                        'color': color,
                        'size': size,
                        'bold': bold,
                        'italic': italic
                    }

        # Find all <b> tags (bold)
        bold_pattern = r'<b[^>]*>(.*?)</b>'
        for match in re.finditer(bold_pattern, html_text, re.DOTALL):
            bold_text = unescape(re.sub(r'<[^>]+>', '', match.group(1)))
            # Split multi-word styled text into individual words
            bold_words = bold_text.split()
            for word in bold_words:
                if word in word_styles:
                    word_styles[word]['bold'] = True
                else:
                    word_styles[word] = {'color': None, 'size': None, 'bold': True, 'italic': False}

        # Find all <i> tags (italic)
        italic_pattern = r'<i[^>]*>(.*?)</i>'
        for match in re.finditer(italic_pattern, html_text, re.DOTALL):
            italic_text = unescape(re.sub(r'<[^>]+>', '', match.group(1)))
            # Split multi-word styled text into individual words
            italic_words = italic_text.split()
            for word in italic_words:
                if word in word_styles:
                    word_styles[word]['italic'] = True
                else:
                    word_styles[word] = {'color': None, 'size': None, 'bold': False, 'italic': True}

        return word_styles

    def _map_styled_positions(self, original_words: list, reversed_words: list,
                              arabic_indices: list, word_styles: dict) -> dict:
        """Map styled words to their positions after Arabic word reversal.

        Returns: position -> styles dict
        """
        position_styles = {}

        for orig_idx, word in enumerate(original_words):
            if word in word_styles:
                # Find where this word ended up after reversal
                if orig_idx in arabic_indices:
                    # Arabic word was reversed - find new position
                    for new_idx, new_word in enumerate(reversed_words):
                        if new_word == word:
                            position_styles[new_idx] = word_styles[word]
                            break
                else:
                    # Non-Arabic word, position unchanged
                    position_styles[orig_idx] = word_styles[word]

        return position_styles

    def get_style_tags(self, styles: dict) -> List[str]:
        """Convert styles dict to list of ASS override tags.

        Args:
            styles: Dict with keys 'color', 'size', 'bold', 'italic'

        Returns:
            List of ASS tag strings like '{\\c&H00FF00}', '{\\b1}'
        """
        tags = []

        if styles.get('color') is not None:
            tags.append(f'{{\\c{styles["color"]}}}')
        if styles.get('bold'):
            tags.append('{\\b1}')
        if styles.get('italic'):
            tags.append('{\\i1}')
        if styles.get('size') is not None:
            tags.append(f'{{\\fs{styles["size"]}}}')

        return tags

    def _simple_html_to_ass(self, html_text: str) -> str:
        """Simple HTML to ASS conversion (no Arabic RTL handling)."""
        decoded = unescape(html_text)
        result = decoded

        # CSS font-size keywords to em multipliers
        size_keywords = {
            'xx-small': 0.5,
            'x-small': 0.7,
            'small': 0.9,
            'medium': 1.0,
            'large': 1.2,
            'x-large': 1.4,
            'xx-large': 1.6,
            'xxx-large': 1.8,
            'smaller': 0.8,
            'larger': 1.2
        }

        # Font size attribute (1-7) to em multipliers
        font_size_map = {
            '1': 0.6,
            '2': 0.8,
            '3': 1.0,
            '4': 1.2,
            '5': 1.4,
            '6': 1.6,
            '7': 1.8
        }

        def hex_to_rgb(hex_color):
            """Convert hex color #RRGGBB to rgb(r, g, b)."""
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f'rgb({r}, {g}, {b})'
            return None

        def convert_span_styles(match):
            """Convert a span with combined styles to ASS tags."""
            style_content = match.group(1)
            text = match.group(2)

            tags = []
            closing_tags = []

            # Extract color - handle both rgb() and hex formats
            color_match = re.search(r'color:\s*(rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)|#[0-9a-fA-F]{6})', style_content)
            if color_match:
                color_str = color_match.group(1)
                if color_str.startswith('#'):
                    # Convert hex to rgb
                    rgb_str = hex_to_rgb(color_str)
                    if rgb_str:
                        rgb_match = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', rgb_str)
                        if rgb_match:
                            r, g, b = int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
                            tags.append(f'{{\\c&H00{b:02X}{g:02X}{r:02X}}}')
                            closing_tags.insert(0, '{\\c&H00FFFFFF}')
                else:
                    # Already rgb format
                    rgb_match = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color_str)
                    if rgb_match:
                        r, g, b = int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
                        tags.append(f'{{\\c&H00{b:02X}{g:02X}{r:02X}}}')
                        closing_tags.insert(0, '{\\c&H00FFFFFF}')

            # Extract font-weight (bold)
            if re.search(r'font-weight:\s*(bold|700)', style_content):
                tags.append('{\\b1}')
                closing_tags.insert(0, '{\\b0}')

            # Extract font-style (italic)
            if re.search(r'font-style:\s*italic', style_content):
                tags.append('{\\i1}')
                closing_tags.insert(0, '{\\i0}')

            # Extract font-size
            size_match = re.search(r'font-size:\s*([\d.]+)\s*(em|px|%)|font-size:\s*(\w+)', style_content)
            if size_match:
                if size_match.group(1):  # Has unit (em, px, %)
                    value = float(size_match.group(1))
                    unit = size_match.group(2)
                    if unit == 'em':
                        size_em = value
                    elif unit == 'px':
                        size_em = value / self.font_size
                    elif unit == '%':
                        size_em = value / 100
                elif size_match.group(3) in size_keywords:  # CSS keyword
                    size_em = size_keywords[size_match.group(3)]
                else:
                    size_em = 1.0

                size_px = round(self.font_size * size_em)
                tags.append(f'{{\\fs{size_px}}}')
                closing_tags.insert(0, '{\\r}')  # Font size needs full reset

            # Build result with opening tags, text, and closing tags
            return ''.join(tags) + text + ''.join(closing_tags)

        # Convert <font color="#..."> tags to ASS color
        def convert_font_color(m):
            hex_color = m.group(1)
            text = m.group(2)
            rgb_str = hex_to_rgb(hex_color)
            if rgb_str:
                rgb_match = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', rgb_str)
                if rgb_match:
                    r, g, b = int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
                    return f'{{\\c&H00{b:02X}{g:02X}{r:02X}}}{text}{{\\c&H00FFFFFF}}'
            return text

        # Convert <font color="#..."> tags to ASS color
        def convert_font_color(m):
            hex_color = m.group(1)
            text = m.group(2)
            rgb_str = hex_to_rgb(hex_color)
            if rgb_str:
                rgb_match = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', rgb_str)
                if rgb_match:
                    r, g, b = int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
                    return f'{{\\c&H00{b:02X}{g:02X}{r:02X}}}{text}{{\\c&H00FFFFFF}}'
            return text

        # Convert <font size="1-7"> tags to ASS font size
        def convert_font_size(m):
            size_value = m.group(1)
            size_em = font_size_map.get(size_value, 1.0)
            size_px = round(self.font_size * size_em)
            return f'{{\\fs{size_px}}}{m.group(2)}{{\\r}}'

        # Convert <font color="..." size="..."> combined tags (handles both attributes)
        def convert_font_combined(m):
            attributes = m.group(1)
            text = m.group(2)

            tags = []
            closers = []

            # Extract color
            color_match = re.search(r'color="([^"]+)"', attributes)
            if color_match:
                color_val = color_match.group(1)
                rgb_str = hex_to_rgb(color_val)
                if rgb_str:
                    rgb_match = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', rgb_str)
                    if rgb_match:
                        r, g, b = int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
                        tags.append(f'{{\\c&H00{b:02X}{g:02X}{r:02X}}}')
                        closers.insert(0, '{\\c&H00FFFFFF}')

            # Extract size
            size_match = re.search(r'size="([1-7])"', attributes)
            if size_match:
                size_val = size_match.group(1)
                size_em = font_size_map.get(size_val, 1.0)
                size_px = round(self.font_size * size_em)
                tags.append(f'{{\\fs{size_px}}}')
                closers.insert(0, '{\\r}')

            # Build result with tags, text, and closers
            return ''.join(tags) + text + ''.join(closers)

        # Process combined <font color="..." size="..."> tags first
        result = re.sub(r'<font\s+(color="[^"]*"\s+size="[^"]*")\s*>(.*?)</font>', convert_font_combined, result, flags=re.DOTALL)

        # Then process remaining <font> tags with only color
        result = re.sub(r'<font\s+color="(#?[0-9a-fA-F]+)">(.*?)</font>', convert_font_color, result, flags=re.DOTALL)

        # Convert <font size="1-7"> tags to ASS font size
        def convert_font_size(m):
            size_value = m.group(1)
            size_em = font_size_map.get(size_value, 1.0)
            size_px = round(self.font_size * size_em)
            return f'{{\\fs{size_px}}}{m.group(2)}{{\\r}}'

        result = re.sub(r'<font\s+size="([1-7])">(.*?)</font>', convert_font_size, result, flags=re.DOTALL)

        # Convert all <span style="..."> tags with any style
        span_pattern = r'<span\s+style="([^"]*)">(.*?)</span>'
        result = re.sub(span_pattern, convert_span_styles, result, flags=re.DOTALL)

        # Convert <strong> and <b> to ASS bold tags
        result = re.sub(r'<strong>(.*?)</strong>', r'{\\b1}\1{\\b0}', result, flags=re.DOTALL)
        result = re.sub(r'<b>(.*?)</b>', r'{\\b1}\1{\\b0}', result, flags=re.DOTALL)

        # Convert <em> and <i> to ASS italic tags
        result = re.sub(r'<em>(.*?)</em>', r'{\\i1}\1{\\i0}', result, flags=re.DOTALL)
        result = re.sub(r'<i>(.*?)</i>', r'{\\i1}\1{\\i0}', result, flags=re.DOTALL)

        # Remove any remaining HTML tags
        result = re.sub(r'<[^>]+>', '', result)

        return result

    # === Color Conversion ===

    def rgb_to_ass(self, rgb_string: str) -> str:
        """Convert RGB string like 'rgb(255, 0, 0)' to ASS color format '&H00BBGGRR'.

        Args:
            rgb_string: RGB string in format 'rgb(r, g, b)'

        Returns:
            ASS color string in format '&H00BBGGRR' (BGR order with alpha prefix)
        """
        match = re.match(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', rgb_string)
        if not match:
            return '&H00FFFFFF'  # Default white

        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        # ASS color format: &H00BBGGRR (BGR order with alpha prefix)
        return f'&H00{b:02X}{g:02X}{r:02X}'

    def hex_to_ass(self, hex_string: str) -> str:
        """Convert hex color like '#FF0000' to ASS color format '&H00BBGGRR'.

        Args:
            hex_string: Hex color in format '#RRGGBB' or 'RRGGBB'

        Returns:
            ASS color string in format '&H00BBGGRR' (BGR order with alpha prefix)
        """
        # Remove # if present
        hex_string = hex_string.lstrip('#')

        if len(hex_string) != 6:
            return '&H00FFFFFF'  # Default white

        r = int(hex_string[0:2], 16)
        g = int(hex_string[2:4], 16)
        b = int(hex_string[4:6], 16)

        return f'&H00{b:02X}{g:02X}{r:02X}'

    # === Text Processing ===

    def process_arabic_rtl(self, text: str, has_styling: bool) -> str:
        """Process Arabic text for RTL display.

        Note: ASS/libass handles RTL automatically for plain text.
        Word reversal is only needed when partial styling is applied.

        Args:
            text: Text to process
            has_styling: Whether text has partial HTML styling

        Returns:
            Processed text
        """
        # ASS handles RTL automatically - just return text as-is
        # Word reversal is handled in parse_html_to_ass when needed
        return text

    def _escape_ass_special_chars(self, text: str) -> str:
        """Escape special ASS characters, but preserve ASS tags.

        Args:
            text: Text that may contain ASS tags

        Returns:
            Text with special chars escaped, tags preserved
        """
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

        return temp_text

    # === Timestamp Formatting ===

    def _format_timestamp_ass(self, seconds: float) -> str:
        """Format seconds to ASS timestamp (H:MM:SS.CC)."""
        start_h = int(seconds // 3600)
        start_m = int((seconds % 3600) // 60)
        start_s = seconds % 60
        return f"{start_h}:{start_m:02d}:{start_s:05.2f}"

    def _format_timestamp_json(self, seconds: float) -> str:
        """Format seconds to JSON timestamp key (HH:MM:SS.mmm)."""
        return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}.{int((seconds % 1) * 1000):03d}"

    # === SRT Parsing ===

    def _parse_srt_segments(self, srt_path: Path) -> List[dict]:
        """Parse SRT file into list of segments.

        Returns:
            List of dicts with 'start', 'end', 'text' keys
        """
        segments = []

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by double newlines to get subtitle blocks
        blocks = re.split(r'\n\s*\n', content.strip())

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # First line is subtitle number (skip)
                # Second line is timestamp range
                timestamp_line = lines[1]
                # Rest is subtitle text
                text = '\n'.join(lines[2:])

                # Parse timestamp: "00:00:00,000 --> 00:00:02,000"
                match = re.match(r'(\d+):(\d+):([\d,]+)\s*-->\s*(\d+):(\d+):([\d,]+)', timestamp_line)
                if match:
                    start_h, start_m, start_s = match.group(1), match.group(2), match.group(3).replace(',', '.')
                    end_h, end_m, end_s = match.group(4), match.group(5), match.group(6).replace(',', '.')

                    start_time = int(start_h) * 3600 + int(start_m) * 60 + float(start_s)
                    end_time = int(end_h) * 3600 + int(end_m) * 60 + float(end_s)

                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text.strip()
                    })

        return segments
