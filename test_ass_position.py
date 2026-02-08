#!/usr/bin/env python3
"""
Test ASS subtitle positioning without processing full video.

Creates a 5-second test video with the ASS subtitles burned in
to verify positioning is correct.
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
ASS_FILE = "videos/001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6/shorts/theme_003.ass"
OUTPUT_FILE = "test_subtitles.mp4"
WIDTH = 1080
HEIGHT = 1920
DURATION = 5
FONT_SIZE = 48

def create_test_video():
    """Create a 5-second test video with ASS subtitles."""

    # Check if ASS file exists
    if not os.path.exists(ASS_FILE):
        print(f"❌ ASS file not found: {ASS_FILE}")
        print("Please update the ASS_FILE path in the script.")
        return False

    # Get absolute path for ASS file (ffmpeg needs absolute paths)
    ass_path = os.path.abspath(ASS_FILE)

    print(f"📹 Creating test video...")
    print(f"   ASS file: {ass_path}")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Resolution: {WIDTH}x{HEIGHT}")
    print(f"   Duration: {DURATION}s")

    # FFmpeg command to create test video with subtitles
    # Using color source (black background) for 5 seconds
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c=black:s={WIDTH}x{HEIGHT}:d={DURATION}',
        '-vf', f'subtitles={ass_path}:force_style=\'FontSize={FONT_SIZE},Alignment=2,MarginV=10,FontName=Arial\'',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-t', str(DURATION),
        '-y',
        OUTPUT_FILE
    ]

    print(f"\n🔧 Running FFmpeg...")
    print(f"   Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Test video created successfully!")
        print(f"   Output: {OUTPUT_FILE}")
        print(f"\n🎬 Play the video to check subtitle positioning:")
        print(f"   vlc {OUTPUT_FILE}")
        print(f"   ffplay {OUTPUT_FILE}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error:")
        print(f"   {e.stderr}")
        return False

def test_without_force_style():
    """Test without force_style to use ASS file's internal positioning."""

    ass_path = os.path.abspath(ASS_FILE)
    output_file = "test_subtitles_no_force.mp4"

    print(f"\n📹 Creating test video WITHOUT force_style (using ASS internal positioning)...")

    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c=black:s={WIDTH}x{HEIGHT}:d={DURATION}',
        '-vf', f'subtitles={ass_path}',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-t', str(DURATION),
        '-y',
        output_file
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Created: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ASS Subtitle Positioning Test")
    print("=" * 60)

    # Test with force_style
    create_test_video()

    # Test without force_style (uses ASS file positioning)
    test_without_force_style()

    print("\n" + "=" * 60)
    print("Compare the two videos:")
    print("  1. test_subtitles.mp4 - with force_style")
    print("  2. test_subtitles_no_force.mp4 - without force_style (ASS positioning)")
    print("=" * 60)
