
import sys
import os
from pathlib import Path
sys.path.append(os.getcwd())

from video_processor import VideoProcessor

def apply_step2():
    folder = "001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6"
    input_path = f"videos/{folder}/shorts/edited_shorts/step1_cropped.mp4"
    output_path = f"videos/{folder}/shorts/edited_shorts/step2_effects.mp4"
    
    print(f"🚀 Starting Step 2: Applying effects to {input_path}...")
    
    settings = {
        "folder_number": "001",
        "theme_number": 5,
        "trim": {
            "start": "00:00:00",
            "end": "00:00:49" 
        },
        "subtitles": {
            "reburn": False  # IMPORTANT: No subs in Step 2
        },
        "audio": {
            "file": "nasheed2.mp3",
            "volume": 30,
            "originalVolume": 100
        },
        "images": [
            {
                "name": "an-naseeha_logo.png",
                "markers": [{
                    "start_time": 0,
                    "end_time": 49,
                    "x": 50,
                    "y": 10,
                    "scale": 0.5
                }]
            }
        ],
        "broll_markers": [
            {
                "name": "broll_justice.mp4",
                "start_time": 2,
                "end_time": 7,
                "transition": "fade"
            }
        ],
        "effects": {},
        "lighting": {"brightness": 100, "contrast": 100},
        "animations": {}
    }
    
    processor = VideoProcessor(input_path)
    result = processor.apply_effects(output_path, settings)
    print(f"✅ Step 2 complete. Output: {result}")

if __name__ == "__main__":
    apply_step2()
