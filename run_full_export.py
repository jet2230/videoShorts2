
import requests
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:5000"
FOLDER = "001"
THEME = 5 # SWITCHED TO THEME 5
VIDEO_PATH = "001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6/The Hereafter ｜ Shaikh Abul Abbaas ｜ Unbreakable ｜ Session 6.mp4"

def run_full_theme_export():
    print(f"🚀 Starting FULL Export for Theme {THEME}...")
    
    # Using exact times from theme_005_adjust.md
    payload = {
        "video_path": VIDEO_PATH,
        "settings": {
            "folder_number": FOLDER,
            "theme_number": THEME,
            "trim": {
                "start": "00:09:58",
                "end": "00:10:47" 
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
        if res.status_code != 200:
            print("❌ Request failed:", res.text)
            return
            
        data = res.json()
        edit_id = data.get("edit_id")
        print(f"⏳ Export started: {edit_id}. Polling status...")
        
        start_time = time.time()
        while True:
            prog_res = requests.get(f"{BASE_URL}/api/process-edit/{edit_id}/status")
            prog_data = prog_res.json()
            status = prog_data.get("status")
            progress = prog_data.get("progress", 0)
            
            print(f"\rProgress: {progress}% [{status}]", end="")
            
            if status == "completed":
                duration = time.time() - start_time
                print(f"\n✅ Full Export Finished in {duration:.2f} seconds.")
                print(f"📁 Output file: {prog_data.get('output_path')}")
                return True
            elif status in ["error", "failed"]:
                print(f"\n❌ Export Failed: {prog_data.get('log')}")
                return False
                
            time.sleep(3)
            
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    run_full_theme_export()
