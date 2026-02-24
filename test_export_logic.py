import sys
import os
import json
from pathlib import Path
from video_processor import VideoProcessor
from subtitle_renderer import render_canvas_karaoke_video

# Setup test paths
folder = Path("videos/001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6")
master_video = str(list(folder.glob("*.mp4"))[0])
srt_path = str(folder / "shorts" / "theme_002.srt")
word_timestamps = str(list(folder.glob("*_word_timestamps.json"))[0])
output_video = str(folder / "edited_shorts" / "test_manual_export.mp4")

os.makedirs(os.path.dirname(output_video), exist_ok=True)

settings = {
    "folder_number": "001",
    "theme_number": "2",
    "trim": {"start": "00:01:40", "end": "00:01:50"}, # 10s test
    "subtitles": {
        "reburn": True,
        "fontSize": 80,
        "subtitle_position": "bottom",
        "subtitle_top": 1405,
        "title": "TEST TITLE",
        "title_top": 150,
        "show_title": True
    },
    "effects": {},
    "lighting": {"brightness": 100, "contrast": 100, "exposure": 0, "saturation": 100},
    "animations": {}
}

print(f"--- STARTING TEST EXPORT ---")
print(f"Input: {master_video}")
print(f"Output: {output_video}")

try:
    processor = VideoProcessor(master_video)
    # Pass 1 & 2
    intermediate = processor.apply_effects(output_video, settings, log_callback=print)
    print(f"Intermediate file created: {intermediate}")
    
    # Pass 3
    success = render_canvas_karaoke_video(
        intermediate,
        word_timestamps,
        srt_path,
        output_video,
        100.0, # Start time 00:01:40
        110.0, # End time 00:01:50
        settings["subtitles"],
        progress_callback=lambda p, s, m: print(f"{s}: {p:.1f}% - {m}"),
        is_already_trimmed=True
    )
    
    if success and os.path.exists(output_video):
        size = os.path.getsize(output_video)
        print(f"--- TEST SUCCESS ---")
        print(f"Final file size: {size} bytes")
    else:
        print(f"--- TEST FAILED: File not created ---")

except Exception as e:
    import traceback
    print(f"--- TEST CRASHED ---")
    traceback.print_exc()
