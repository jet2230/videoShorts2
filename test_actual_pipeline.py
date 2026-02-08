#!/usr/bin/env python3
"""
Test the actual shorts creation pipeline to debug subtitle positioning.

Simulates: scale → crop → subtitles (with force_style)
"""

import subprocess
import os

# Configuration
ASS_FILE = "videos/001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6/shorts/theme_003.ass"
SOURCE_VIDEO = "/home/obo/playground/videoShorts2/videos/001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6/The Hereafter ｜ Shaikh Abul Abbaas ｜ Unbreakable ｜ Session 6.mp4"
OUTPUT_1 = "test_pipeline_scale_crop_sub.mp4"
OUTPUT_2 = "test_pipeline_with_source.mp4"
WIDTH = 1080
HEIGHT = 1920
DURATION = 5
alignment = 2
margin_v = 10

ass_path = os.path.abspath(ASS_FILE)
subtitle_file_for_ffmpeg = ass_path.replace(':', '\\:').replace('\\', '\\\\').replace("'", "\\'")

print("=" * 70)
print("Testing Actual Shorts Creation Pipeline")
print("=" * 70)

# Test 1: Simulate the ACTUAL pipeline (like shorts_creator.py)
print("\n📹 Test 1: Scale → Crop → Subtitles (with force_style)")
print(f"   This simulates the actual shorts creation process")

vf_filter = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}:(iw-ow)/2:(ih-oh)/2,subtitles={subtitle_file_for_ffmpeg}:force_style='Alignment={alignment},MarginV={margin_v}'"

print(f"   Filter: {vf_filter}")

cmd = [
    'ffmpeg',
    '-ss', '0',
    '-i', SOURCE_VIDEO,
    '-t', str(DURATION),
    '-vf', vf_filter,
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-crf', '23',
    '-y',
    OUTPUT_1
]

try:
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"   ✅ Created: {OUTPUT_1}")

    # Extract frame
    subprocess.run([
        'ffmpeg', '-i', OUTPUT_1, '-ss', '0:00:01', '-vframes', 1,
        'test_pipeline_1_frame.png', '-y'
    ], capture_output=True)
    print(f"   📸 Frame: test_pipeline_1_frame.png")

except subprocess.CalledProcessError as e:
    print(f"   ❌ Error: {e.stderr}")

# Test 2: Check what intermediate dimensions are after scale
print("\n📐 Test 2: Check intermediate dimensions after scale")

vf_check = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}:(iw-ow)/2:(ih-oh)/2"

cmd_check = [
    'ffmpeg',
    '-ss', '0',
    '-i', SOURCE_VIDEO,
    '-t', '1',
    '-vf', vf_check,
    '-f', 'null',
    '-'
]

try:
    result = subprocess.run(cmd_check, capture_output=True, text=True)
    # Parse output to find dimensions
    for line in result.stderr.split('\n'):
        if 'Output' in line or 'Stream' in line or 'x' in line:
            print(f"   {line}")
except:
    pass

# Test 3: Try with FontName in force_style
print("\n📹 Test 3: Add FontName to force_style")

vf_filter_3 = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}:(iw-ow)/2:(ih-oh)/2,subtitles={subtitle_file_for_ffmpeg}:force_style='Alignment={alignment},MarginV={margin_v},FontName=Arial'"

cmd_3 = [
    'ffmpeg',
    '-ss', '0',
    '-i', SOURCE_VIDEO,
    '-t', str(DURATION),
    '-vf', vf_filter_3,
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-crf', '23',
    '-y',
    'test_pipeline_with_fontname.mp4'
]

try:
    subprocess.run(cmd_3, check=True, capture_output=True, text=True)
    print(f"   ✅ Created: test_pipeline_with_fontname.mp4")

    subprocess.run([
        'ffmpeg', '-i', 'test_pipeline_with_fontname.mp4', '-ss', '0:00:01', '-vframes', 1,
        'test_pipeline_fontname_frame.png', '-y'
    ], capture_output=True)
    print(f"   📸 Frame: test_pipeline_fontname_frame.png")

except subprocess.CalledProcessError as e:
    print(f"   ❌ Error: {e.stderr}")

print("\n" + "=" * 70)
print("Compare these images:")
print("  1. test_pipeline_1_frame.png - current pipeline")
print("  2. test_pipeline_fontname_frame.png - with FontName added")
print("=" * 70)
