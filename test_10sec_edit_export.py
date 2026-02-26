import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:5000"
FOLDER = "001"
THEME = 5
VIDEO_PATH = "001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6/The Hereafter ｜ Shaikh Abul Abbaas ｜ Unbreakable ｜ Session 6.mp4"

def run_10sec_edit_export():
    print(f"🚀 Starting 10-second EDIT Export for Theme {THEME}...")
    
    # 10 seconds starting from theme start (00:09:58 to 00:10:08)
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
            try:
                prog_res = requests.get(f"{BASE_URL}/api/process-edit/{edit_id}/status")
                prog_data = prog_res.json()
                status = prog_data.get("status")
                progress = prog_data.get("progress", 0)
                
                # Print stage and progress
                log = prog_data.get('log', '')
                stage = "Processing"
                if "STAGE:" in log:
                    try:
                        stage = log.split("STAGE:")[1].split("\n")[0]
                    except:
                        pass
                
                print(f"\rProgress: {progress}% | Stage: {stage} | [{status}]", end="")
                
                if status == "completed":
                    duration = time.time() - start_time
                    print(f"\n✅ 10s Edit Export Finished in {duration:.2f} seconds.")
                    print(f"📁 Output file: {prog_data.get('output_path')}")
                    return True
                elif status in ["error", "failed"]:
                    print(f"\n❌ Export Failed: {prog_data.get('log')}")
                    return False
            except Exception as inner_e:
                print(f"\n⚠️ Poll error: {inner_e}")
                
            time.sleep(2)
            
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    run_10sec_edit_export()
