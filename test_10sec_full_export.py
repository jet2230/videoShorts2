
import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:5000"
FOLDER = "001"
THEME = 5
# Pretend we are editing the theme clip from the UI
VIDEO_PATH = "001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6/The Hereafter ｜ Shaikh Abul Abbaas ｜ Unbreakable ｜ Session 6.mp4"

def run_10sec_full_export():
    print(f"🚀 Starting 10-second FULL (Logo, Audio, B-roll) Export for Theme {THEME}...")
    
    payload = {
        "video_path": VIDEO_PATH,
        "settings": {
            "folder_number": FOLDER,
            "theme_number": THEME,
            "trim": {
                "start": "00:09:58",
                "end": "00:10:08" 
            },
            "subtitles": {
                "reburn": True,
                "style": "default",
                "offset": 0,
                "show_title": True
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
                        "end_time": 10,
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
            "lighting": {"brightness": 100, "contrast": 100}
        }
    }
    
    try:
        res = requests.post(f"{BASE_URL}/api/process-edit", json=payload)
        if res.status_code != 200:
            print(f"❌ Request failed: {res.text}")
            return
            
        data = res.json()
        edit_id = data.get("edit_id")
        print(f"⏳ Export started: {edit_id}. Polling status...")
        
        start_time = time.time()
        while True:
            try:
                prog_res = requests.get(f"{BASE_URL}/api/process-edit/{edit_id}/status")
                prog_data = prog_res.json()
                status = prog_data.get("status")
                progress = prog_data.get("progress", 0)
                
                print(f"\rProgress: {progress}% | [{status}]", end="")
                
                if status == "completed":
                    duration = time.time() - start_time
                    print(f"\n✅ 10s Full Export Finished in {duration:.2f} seconds.")
                    print(f"📁 Output file: {prog_data.get('output_path')}")
                    return True
                elif status in ["error", "failed"]:
                    print(f"\n❌ Export Failed")
                    print(f"LOG: {prog_data.get('log')}")
                    return False
            except Exception as inner_e:
                print(f"\n⚠️ Poll error: {inner_e}")
                
            time.sleep(2)
            
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    run_10sec_full_export()
