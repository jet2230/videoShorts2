import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:5000"
FOLDER = "001"
THEME = 5

def run_10sec_text_export():
    print(f"🚀 Starting 10-second TEXT (Canvas Karaoke) Export for Theme {THEME}...")
    
    # 10 seconds starting from theme start (00:09:58 to 00:10:08)
    payload = {
        "folder": FOLDER,
        "theme": THEME,
        "themeStart": "00:09:58",
        "themeEnd": "00:10:08",
        "settings": {
            "mode": "normal",
            "fontSize": 80,
            "primaryColor": "#ffffff",
            "textColor": "#ffff00",
            "output_dir": "edited_shorts"
        }
    }
    
    try:
        res = requests.post(f"{BASE_URL}/api/export-canvas-karaoke", json=payload)
        if res.status_code != 200:
            print(f"❌ Request failed: {res.text}")
            return
            
        data = res.json()
        job_id = data.get("job_id")
        print(f"⏳ Export started: {job_id}. Polling status...")
        
        start_time = time.time()
        while True:
            try:
                prog_res = requests.get(f"{BASE_URL}/api/export-canvas-karaoke/{job_id}/status")
                prog_data = prog_res.json()
                status = prog_data.get("status")
                progress = prog_data.get("progress", 0)
                
                print(f"\rProgress: {progress}% | [{status}]", end="")
                
                if status == "completed":
                    duration = time.time() - start_time
                    print(f"\n✅ 10s Text Export Finished in {duration:.2f} seconds.")
                    print(f"📁 Output file: {prog_data.get('output_path')}")
                    return True
                elif status in ["error", "failed"]:
                    print(f"\n❌ Export Failed: {prog_data.get('error')}")
                    return False
            except Exception as inner_e:
                print(f"\n⚠️ Poll error: {inner_e}")
                
            time.sleep(2)
            
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    run_10sec_text_export()
