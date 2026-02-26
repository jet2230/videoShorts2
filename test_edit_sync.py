
import requests
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:5000"
FOLDER = "001"
THEME = 5 # TARGETING THEME 5
VIDEO_PATH = "001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6/The Hereafter ｜ Shaikh Abul Abbaas ｜ Unbreakable ｜ Session 6.mp4"

def test_theme5_sync():
    print(f"🚀 Starting 10-second Theme 5 Sync Test...")
    
    # Theme 5 starts at 09:58
    payload = {
        "video_path": VIDEO_PATH,
        "settings": {
            "folder_number": FOLDER,
            "theme_number": THEME,
            "trim": {
                "start": "00:09:58",
                "end": "00:10:08" # 10 seconds
            },
            "subtitles": {
                "reburn": True,
                "style": "default"
            },
            "effects": {},
            "lighting": {"brightness": 100, "contrast": 100}
        }
    }
    
    try:
        res = requests.post(f"{BASE_URL}/api/process-edit", json=payload)
        data = res.json()
        edit_id = data.get("edit_id")
        print(f"⏳ Export started: {edit_id}. Polling...")
        
        while True:
            prog_res = requests.get(f"{BASE_URL}/api/process-edit/{edit_id}/status")
            prog_data = prog_res.json()
            status = prog_data.get("status")
            print(f"\rProgress: {prog_data.get('progress')}% [{status}]", end="")
            
            if status == "completed":
                print(f"\n✅ Finished! Path: {prog_data.get('output_path')}")
                return True
            elif status in ["error", "failed"]:
                print(f"\n❌ Failed: {prog_data.get('log')}")
                return False
            time.sleep(2)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_theme5_sync()
