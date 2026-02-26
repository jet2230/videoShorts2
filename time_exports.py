import requests
import time
import sys
import os

BASE_URL = "http://localhost:5000"
FOLDER = "001"
THEME = 2
START_SEC = 100.0
END_SEC = 153.0

def time_adjust_export():
    print("\n🚀 Starting Adjust Export (Canvas Karaoke)...")
    start_time = time.time()
    
    payload = {
        "folder": FOLDER,
        "theme": THEME,
        "themeStart": START_SEC,
        "themeEnd": END_SEC,
        "settings": {
            "effect_type": "standard",
            "fontSize": 80,
            "fontName": "Arial"
        }
    }
    
    try:
        res = requests.post(f"{BASE_URL}/api/export-canvas-karaoke", json=payload)
        data = res.json()
        job_id = data.get("job_id")
        
        if not job_id:
            print("❌ Adjust export failed to start:", data)
            return None
            
        print(f"⏳ Polling Adjust job {job_id}...")
        while True:
            prog_res = requests.get(f"{BASE_URL}/api/canvas-karaoke-progress/{job_id}")
            prog_data = prog_res.json()
            status = prog_data.get("status")
            progress = prog_data.get("progress", 0)
            
            print(f"\rAdjust Progress: {progress}% [{status}]", end="")
            
            if status == "complete":
                duration = time.time() - start_time
                print(f"\n✅ Adjust Export Finished in {duration:.2f} seconds.")
                return duration
            elif status == "error":
                print("\n❌ Adjust Export Error:", prog_data.get("error"))
                return None
                
            time.sleep(2)
    except Exception as e:
        print(f"\n❌ Adjust Request Error: {e}")
        return None

def time_edit_export():
    print("\n🚀 Starting Edit Export (Process Edit)...")
    start_time = time.time()
    
    # EXACT FILENAME discovered via ls
    video_path = "001_The_Hereafter_Shaikh_Abul_Abbaas_Unbreakable_Session_6/The Hereafter ｜ Shaikh Abul Abbaas ｜ Unbreakable ｜ Session 6.mp4"
    
    payload = {
        "video_path": video_path,
        "settings": {
            "folder_number": FOLDER,
            "theme_number": THEME,
            "trim": {"start": "00:01:40", "end": "00:02:33"},
            "subtitles": {"reburn": True},
            "lighting": {"brightness": 100, "contrast": 100},
            "effects": {}
        }
    }
    
    try:
        res = requests.post(f"{BASE_URL}/api/process-edit", json=payload)
        data = res.json()
        edit_id = data.get("edit_id")
        
        if not edit_id:
            print("❌ Edit export failed to start:", data)
            return None
            
        print(f"⏳ Polling Edit job {edit_id}...")
        while True:
            prog_res = requests.get(f"{BASE_URL}/api/process-edit/{edit_id}/status")
            prog_data = prog_res.json()
            status = prog_data.get("status")
            progress = prog_data.get("progress", 0)
            
            print(f"\rEdit Progress: {progress}% [{status}]", end="")
            
            if status == "completed":
                duration = time.time() - start_time
                print(f"\n✅ Edit Export Finished in {duration:.2f} seconds.")
                return duration
            elif status in ["error", "failed"]:
                print(f"\n❌ Edit Export Error: {prog_data.get('log')}")
                return None
                
            time.sleep(2)
    except Exception as e:
        print(f"\n❌ Edit Request Error: {e}")
        return None

if __name__ == "__main__":
    # Check if server is up
    try:
        requests.get(BASE_URL)
    except:
        print(f"❌ Server not found at {BASE_URL}. Please start server.py first.")
        sys.exit(1)
        
    adjust_dur = time_adjust_export()
    edit_dur = time_edit_export()
    
    print("\n" + "="*30)
    print("📊 FINAL EXPORT TIME RESULTS")
    print("="*30)
    if adjust_dur: print(f"Adjust (Multi-Pass): {adjust_dur:.2f}s")
    if edit_dur:   print(f"Edit (Standard):     {edit_dur:.2f}s")
    print("="*30)
