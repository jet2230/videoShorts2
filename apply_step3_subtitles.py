
import sys
import os
import json
import re
from pathlib import Path
sys.path.append(os.getcwd())

from subtitle_renderer import render_canvas_karaoke_video

def apply_step3():
    folder_name = "001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6"
    folder_path = Path(f"videos/{folder_name}")
    input_path = folder_path / "shorts/edited_shorts/step2_effects.mp4"
    output_path = folder_path / "shorts/edited_shorts/step3_final.mp4"
    srt_path = folder_path / "shorts/theme_005.srt"
    word_timestamps_path = next(folder_path.glob("*_word_timestamps.json"))
    
    print(f"🚀 Starting Step 3: Burning subtitles onto {input_path}...")
    
    # Define settings exactly from theme_005_adjust.md and previous successful logs
    settings = {
        "folder_number": "001",
        "theme_number": 5,
        "show_title": True,
        "title": "Srât: Physical Bridge to Hell?",
        "title_top": 150,
        "title_text_color": "#00ff9d",
        "title_font_weight": "800",
        "title_font_size": 67,
        "title_all_caps": False,
        "subtitle_position": "custom",
        "subtitle_left": 546,
        "subtitle_top": 1355,
        "subtitle_h_align": "center",
        "subtitle_v_align": "middle",
        "fontSize": 80,
        "fontName": "Arial",
        "primaryColor": "#ffffff",
        "textColor": "#3366ff",  # Blue highlight from adjust.md
        "glowColor": "#0044ff",
        "glowBlur": 15,
        "mode": "normal",
        "bgOpacity": 0.0,
        "font_weight": "bold"
    }
    
    # Metadata sync times
    start_time = 598.0 # 00:09:58
    end_time = 647.0   # 00:10:47
    
    success = render_canvas_karaoke_video(
        str(input_path),
        str(word_timestamps_path),
        str(srt_path),
        str(output_path),
        start_time,
        end_time,
        settings,
        is_already_trimmed=True,
        srt_mode='relative' # theme_005.srt is 0-based
    )
    
    if success:
        print(f"✅ Step 3 complete. Final output: {output_path}")
    else:
        print(f"❌ Step 3 failed.")

if __name__ == "__main__":
    apply_step3()
