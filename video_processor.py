#!/usr/bin/env python3
"""
Video processor with face tracking and effects.
Uses ffmpeg for processing to preserve audio.
"""

import os

# Disable hardware acceleration for FFmpeg backend to fix AV1 decoding issues
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;none"

import cv2
import numpy as np
from pathlib import Path
import json
import subprocess
import tempfile


class VideoProcessor:
    """Process videos with effects including face tracking."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
            
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(str(video_path))
            if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def find_media(self, filename):
        """Recursively find a media file in the media directory."""
        if not filename: return None
        media_root = Path('media')
        
        # 1. Try direct path
        p = media_root / filename
        if p.exists(): return p
        
        # 2. Try recursive search
        try:
            matches = list(media_root.rglob(filename))
            if matches: return matches[0]
        except: pass
        
        # 3. Try absolute or relative to CWD
        p = Path(filename)
        if p.exists(): return p
        
        return None

    def apply_effects(self, output_path: str, settings: dict, cancel_flag=None, log_callback=None):
        """
        Apply time-based effects using a single-pass FFmpeg complex filter.
        """
        import subprocess
        import tempfile
        import os
        import shutil

        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        trim_settings = settings.get('trim', {})
        effect_markers = settings.get('effect_markers', [])
        image_settings = settings.get('images', [])
        broll_markers = settings.get('broll_markers', [])
        audio_settings = settings.get('audio', {})
        outro_settings = settings.get('outro', {})
        global_effects = settings.get('effects', {})
        lighting = settings.get('lighting', {})
        animations = settings.get('animations', {})
        
        log(f"Received animations: {animations}")

        log("Starting optimized video processing...")

        def to_sec(ts):
            if ts is None: return 0.0
            if isinstance(ts, (int, float)): return float(ts)
            if ':' in str(ts):
                parts = list(map(float, str(ts).replace(',', '.').split(':')))
                return parts[-1] + (parts[-2] * 60 if len(parts) > 1 else 0) + (parts[-3] * 3600 if len(parts) > 2 else 0)
            return float(ts)

        # Parse trim settings
        start_time = trim_settings.get('start', '0')
        end_time = trim_settings.get('end')

        # If we have subsequent passes, we MUST use a temporary file for the first pass result.
        has_face_tracking = bool(global_effects.get('faceTracking'))
        reburn_subs = settings.get('subtitles', {}).get('reburn', True)
        
        # Check if we should enforce a theme duration from metadata
        # BUT only if user didn't explicitly provide an end_time in trim settings
        f_num = settings.get('folder_number')
        t_num = settings.get('theme_number')
        
        if not end_time and f_num and t_num:
            # Look up metadata...
            pass
        
        # ALWAYS use a temporary file for the first pass result to ensure Step 3 
        # (SubtitleRenderer) reads the frames with effects/B-roll already applied.
        # Use a local .tmp directory instead of /tmp (tmpfs) to save RAM
        temp_root = self.video_path.parent / '.tmp'
        temp_root.mkdir(exist_ok=True)
        temp_dir_for_cleanup = tempfile.mkdtemp(dir=str(temp_root))
        
        # --- NEW TWO-PASS STRATEGY TO PREVENT OOM ---
        # If we are seeking deep into a large file, FFmpeg often hits OOM if complex filters are attached.
        # We first extract a clean clip of JUST the portion we need.
        # We use re-encoding with 'ultrafast' to ensure the clip is ACCURATELY trimmed at the start/end,
        # which is critical for subtitle sync in Pass 2.
        clean_clip = os.path.join(temp_dir_for_cleanup, 'source_clip.mp4')
        log(f"Pass 0: Extracting source clip ({start_time} to {end_time})")
        
        extract_cmd = ['ffmpeg', '-nostdin', '-y']
        if start_time != '0':
            # Use input seeking for speed
            extract_cmd.extend(['-ss', str(to_sec(start_time))])
        if end_time:
            dur = to_sec(end_time) - to_sec(start_time)
            if dur > 0:
                extract_cmd.extend(['-t', str(dur)])
        
        # Accurate extraction with re-encoding (Fast preset for speed)
        # CRITICAL: Scale to 1080p HERE to reduce memory usage in subsequent passes
        extract_cmd.extend([
            '-i', str(self.video_path.absolute()),
            '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(iw-ow)/2:(ih-oh)/2',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
            '-c:a', 'aac', '-b:a', '192k',
            '-threads', '4',
            clean_clip
        ])
        
        try:
            log(f"Starting Pass 0 accurately...")
            # Use _run_with_cancel for Pass 0 to get progress and handle cancellation
            # We give Pass 0 a 10% weight of the total progress
            self._run_with_cancel(extract_cmd, cancel_flag, log_callback, progress_offset=0, progress_scale=0.1, stage_name="Preparing Source Clip")
            source_for_effects = clean_clip
            # When using the clean clip, the internal time 't' starts at 0
            # relative to the clip, which is exactly what we want for filters.
            internal_start = 0
        except Exception as e:
            log(f"Fast extraction failed, falling back to direct seek: {e}")
            source_for_effects = str(self.video_path.absolute())
            internal_start = to_sec(start_time)

        intermediate_output = os.path.join(temp_dir_for_cleanup, 'intermediate.mp4')
        
        has_face_tracking = bool(global_effects.get('faceTracking'))
        reburn_subs = settings.get('subtitles', {}).get('reburn', True)
        
        # START BUILDING THE SINGLE-PASS COMMAND
        cmd = ['ffmpeg', '-nostdin', '-y']
        
        # If Pass 0 failed and we fell back to the original file, we MUST seek/trim here.
        if internal_start != 0:
            # Use input seeking for speed and memory efficiency
            cmd.extend(['-ss', str(internal_start)])
        
        # Main input (0:v and 0:a)
        cmd.extend(['-i', source_for_effects])
        
        # If Pass 0 failed, we also need to limit the duration of this input
        if source_for_effects != clean_clip and end_time:
            dur = to_sec(end_time) - to_sec(start_time)
            if dur > 0:
                cmd.extend(['-t', str(dur)])
        
        # Input index for subsequent inputs
        input_index = 1

        # Add additional input for custom audio EARLY to keep indices stable
        custom_audio_index = None
        if audio_settings.get('file'):
            audio_path = self.find_media(audio_settings['file'])
            if audio_path and audio_path.exists():
                # Optimize audio input: only read the required duration
                # Use same seek logic as main video for sync
                if internal_start != 0:
                    cmd.extend(['-ss', str(internal_start)])
                
                if end_time:
                    dur = to_sec(end_time) - to_sec(start_time)
                    if dur > 0:
                        cmd.extend(['-t', str(dur)])
                
                cmd.extend(['-i', str(audio_path.absolute())])
                custom_audio_index = input_index
                input_index += 1

        # Calculate reference duration for percentage mapping
        # Prefer display_duration from settings (passed by server for theme clips)
        # otherwise fall back to video duration.
        display_duration = settings.get('display_duration')
        if display_duration is None:
            display_duration = self.total_frames / (self.fps if self.fps > 0 else 30)

        log(f"Using display_duration={display_duration:.2f}s for marker time mapping")

        # Add additional inputs for images
        image_inputs = []
        log(f"DEBUG_VP: Processing {len(image_settings)} image inputs")
        for i, img in enumerate(image_settings):
            img_path = self.find_media(img['name'])
            
            if img_path and img_path.exists():
                log(f"DEBUG_VP: Found image {i} '{img['name']}' at {img_path} (Absolute: {img_path.absolute()})")
                # Loop image for duration
                cmd.extend(['-loop', '1', '-i', str(img_path.absolute())])
                
                # Apply internal_start offset if we fell back to original source
                # because markers are 0-based relative to the trim.
                updated_markers = []
                for m_idx, m in enumerate(img.get('markers', [])):
                    # Resolve timing: prefer start_time (seconds), fall back to start (percentage)
                    s = m.get('start_time')
                    e = m.get('end_time')
                    
                    if s is None and 'start' in m:
                        s = m['start'] * display_duration
                    if e is None and 'end' in m:
                        e = m['end'] * display_duration
                    
                    # Markers are 0-based relative to the trim.
                    s = (s if s is not None else 0)
                    e = (e if e is not None else display_duration)
                    log(f"  - Marker {m_idx}: {s:.2f}s to {e:.2f}s")
                    updated_markers.append({**m, 'start_time': s, 'end_time': e})

                image_inputs.append({
                    **img,
                    'markers': updated_markers,
                    'input_index': input_index
                })
                input_index += 1
            else:
                log(f"WARNING_VP: Image {i} '{img['name']}' not found in media root.")

        # Add additional inputs for B-roll
        broll_inputs = []
        log(f"DEBUG_VP: Processing {len(broll_markers)} B-roll markers")
        for i, marker in enumerate(broll_markers):
            br_path = self.find_media(marker['name'])

            if br_path and br_path.exists():
                log(f"DEBUG_VP: Found B-roll {i} '{marker['name']}' at {br_path} (Absolute: {br_path.absolute()})")
                cmd.extend(['-i', str(br_path.absolute())])

                # Resolve timing: prefer start_time (seconds), fall back to start (percentage)
                s = marker.get('start_time')
                e = marker.get('end_time')

                if s is None and 'start' in marker:
                    s = marker['start'] * display_duration
                if e is None and 'end' in marker:
                    e = marker['end'] * display_duration

                # Use original marker timing (relative to theme/trim)
                s = (s if s is not None else 0)
                e = (e if e is not None else display_duration)
                log(f"  - B-roll {i} Timing: {s:.2f}s to {e:.2f}s")

                broll_inputs.append({
                    **marker,
                    'start_time': s,
                    'end_time': e,
                    'input_index': input_index
                })
                input_index += 1
            else:
                log(f"WARNING_VP: B-roll {i} '{marker['name']}' not found in media root.")


        # Build filter complex
        vf_filters = []
        
        # 0. Global Lighting (Multiplicative model to match UI CSS filters)
        b_ui = float(lighting.get('brightness', 100)) / 100.0
        e_ui = float(lighting.get('exposure', 0))
        c_ui = float(lighting.get('contrast', 100)) / 100.0
        s_ui = float(lighting.get('saturation', 100)) / 100.0
        
        # Base multiplier from UI (Brightness * Exposure approximation)
        base_M = b_ui * (1.0 + e_ui * 0.2)
        safe_M = max(0.001, base_M)
        
        # Global Flash multiplier
        global_flash_expr = "1.0"
        cam_glare_global = animations.get('cameraGlare') or animations.get('cameraFlash')
        if cam_glare_global:
            intensity = 5
            if isinstance(cam_glare_global, dict):
                intensity = int(cam_glare_global.get('intensity', 5))
            
            num_flashes = max(1, intensity)
            flash_width = min(0.05, 0.4 / num_flashes)
            spikes = []
            for i in range(num_flashes):
                s = i / num_flashes
                e = s + flash_width
                v = 0.7 if i % 2 == 0 else 0.4
                spikes.append((s, e, v))
            spike_exprs = [f"{v}*between(mod(t,1),{s:.3f},{e:.3f})" for s, e, v in spikes]
            global_flash_expr = "1.0+" + "+".join(spike_exprs)
        
        # Formula to simulate multiplicative brightness M followed by centered contrast C using FFmpeg's eq filter:
        # contrast = M * C
        # brightness = 0.5 * (M - 1)
        # Apply global lighting + global flash
        
        # We simplify the expression for FFmpeg parser safety
        # M_base is the base multiplier (brightness * exposure)
        M_base = safe_M * c_ui
        B_base = 0.5 * c_ui * (safe_M - 1)
        
        if global_flash_expr != "1.0":
            c_expr_global = f"({M_base}*({global_flash_expr}))"
            b_expr_global = f"({B_base}*({global_flash_expr}))"
        else:
            c_expr_global = f"{M_base}"
            b_expr_global = f"{B_base}"
        
        # Force vertical 1080x1920 output as the base for all effects
        vf_filters.append(f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(iw-ow)/2:(ih-oh)/2")
        
        # Use eval=init for better performance since these are global constants
        vf_filters.append(f"eq=contrast='{c_expr_global}':brightness='{b_expr_global}':saturation={s_ui}:eval=init")

        # Apply each active timeline effect
        for marker in effect_markers:
            etype = marker['type']
            s = marker['start_time']
            e = marker['end_time']
            enable = f"between(t,{s},{e})"
            
            if etype == 'cameraGlare' or etype == 'cameraFlash':
                # Local flash on top of global!
                intensity = 5
                if 'intensity' in marker:
                    intensity = int(marker['intensity'])
                
                num_flashes = max(1, intensity)
                flash_width = min(0.05, 0.4 / num_flashes)
                
                spikes = []
                for i in range(num_flashes):
                    spike_s = i / num_flashes
                    spike_e = spike_s + flash_width
                    v = 0.7 if i % 2 == 0 else 0.4
                    spikes.append((spike_s, spike_e, v))
                
                spike_exprs = [f"{v}*between(mod(t,1),{ss:.3f},{ee:.3f})" for ss, ee, v in spikes]
                flash_mult = "1.0+" + "+".join(spike_exprs)
                
                vf_filters.append(f"eq=contrast='{flash_mult}':brightness='0.5*({flash_mult}-1)':enable='{enable}':eval=frame")
            elif etype == 'mirror':
                vf_filters.append(f"hflip=enable='{enable}'")
            elif etype == 'grayscale':
                vf_filters.append(f"hue=s=0:enable='{enable}'")
            elif etype == 'sepia':
                vf_filters.append(f"colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131:enable='{enable}'")
            elif etype == 'blur':
                vf_filters.append(f"gblur=sigma=5:enable='{enable}'")
            elif etype == 'glitch':
                vf_filters.append(f"rgbashift=rh=4:gh=-4:enable='{enable}'")
            elif etype == 'vhs':
                vf_filters.append(f"noise=alls=15:allf=t+u,hue=s=0.4,gblur=sigma=1:enable='{enable}'")
            elif etype == 'neon':
                vf_filters.append(f"edgedetect=low=0.1:high=0.4,hue=h=120:s=2:enable='{enable}'")
            elif etype == 'vignette':
                vf_filters.append(f"vignette='PI/4':enable='{enable}'")
            elif etype == 'cinematic':
                vf_filters.append(f"drawbox=y=0:h=ih*0.1:t=fill:c=black:enable='{enable}'")
                vf_filters.append(f"drawbox=y=ih*0.9:h=ih*0.1:t=fill:c=black:enable='{enable}'")
            elif etype == 'vibrance':
                vf_filters.append(f"vibrance=intensity=0.8:enable='{enable}'")
            elif etype == 'shake':
                vf_filters.append(f"crop=w=iw-40:h=ih-40:x='20+20*sin(2*PI*t*10)':y='20+20*cos(2*PI*t*13)':enable='{enable}',scale=1080:1920")
            elif etype == 'pixelate':
                vf_filters.append(f"boxblur=20:enable='{enable}'")
            elif etype == 'zoom':
                vf_filters.append(f"zoompan=z='if({enable},1.2,1)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:enable='{enable}'")
            elif etype == 'pulse':
                # Expanding centered rectangle to simulate pulse
                vf_filters.append(f"drawbox=x=(iw-iw*0.5)/2:y=(ih-ih*0.3)/2:w=iw*0.5:h=ih*0.3:t='2+8*(1-mod(t,2)/2)':c=0x00ccff@'0.8*(1-mod(t,2)/2)':enable='{enable}'")
            elif etype == 'neonBorder':
                # Layered boxes for glow
                vf_filters.append(f"drawbox=w=iw:h=ih:t=4:c=0x00ff9d@0.8:enable='{enable}'")
                vf_filters.append(f"drawbox=w=iw-4:h=ih-4:x=2:y=2:t=8:c=0x00ff9d@0.3:enable='{enable}'")
            elif etype == 'vectorGrid':
                # Moving grid
                vf_filters.append(f"drawgrid=w=60:h=60:t=2:c=0x00ff9d@0.4:x='40*t/20':y='40*t/20':enable='{enable}'")

        # 1.5 Global Animations & Styles
        style = settings.get('style', 'none')
        style_levels = settings.get('style_levels', {})
        
        def get_level(name, default):
            try:
                return float(style_levels.get(name, default))
            except:
                return float(default)

        if style == 'pixels':
            # Fast pixelation: scale down then up
            px_val = int(get_level('pixels', 8))
            vf_filters.append(f"scale=iw/{px_val}:ih/{px_val}:flags=neighbor,scale=1080:1920:flags=neighbor")
        elif style == 'paint':
            # Smoothness level 1-10 (default 5)
            smooth = get_level('paint', 5)
            vf_filters.append(f"bilateral=sigmaS={smooth}:sigmaR=0.1,unsharp=3:3:1.5:3:3:0.5")
        elif style == 'pencil':
            # Sensitivity level 1-10 (default 5)
            sens = get_level('pencil', 5) / 10.0
            vf_filters.append(f"format=gray,edgedetect=low={sens*0.1}:high={sens*0.2},negate")
        elif style == 'neon':
            # Glow intensity 1-10 (default 5)
            glow = get_level('neon', 5)
            vf_filters.append(f"edgedetect=low=0.1:high={0.1 + (11-glow)*0.05},hue=h=120:s={1.0 + glow/5}")
        elif style == 'poster':
            # Color levels 2-10 (default 5)
            levels = int(get_level('poster', 5))
            vf_filters.append(f"posterize=levels={levels}")
        elif style == 'retro':
            # Vintage tint 1-10 (default 5)
            tint = get_level('retro', 5)
            vf_filters.append(f"noise=alls={tint*2}:allf=t,hue=s={1.0 - tint/20},vignette")

        # Global Animations
        if animations.get('vhs'):
            vf_filters.append("noise=alls=20:allf=t+u,hue=s=0.5,gblur=sigma=1")
        if animations.get('kenBurns'):
            vf_filters.append(f"zoompan=z='min(zoom+0.0005,1.2)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920")
        if animations.get('scanlines'):
            vf_filters.append("drawgrid=w=iw:h=4:t=1:c=black@0.3")
        if animations.get('heartbeat'):
            vf_filters.append("vignette='PI/4+0.1*sin(2*PI*t/2)'")
        if animations.get('tiltShift'):
            vf_filters.append("boxblur=20:enable='between(y,0,ih*0.15)+between(y,ih*0.85,ih)'")
        if animations.get('audioBorder'):
            vf_filters.append(f"drawbox=w=iw:h=ih:t=5:c=0x00ff9d@'0.5+0.5*sin(2*PI*t)':enable='1'")
        if animations.get('progressBar'):
            vf_filters.append(f"drawbox=y=ih-10:w='iw*t/{self.total_frames/self.fps}':h=10:t=fill:c=0x00ff9d")
        if animations.get('pulse'):
            vf_filters.append(f"drawbox=x=(iw-iw*0.5)/2:y=(ih-ih*0.3)/2:w=iw*0.5:h=ih*0.3:t='2+8*(1-mod(t,2)/2)':c=0x00ccff@'0.8*(1-mod(t,2)/2)':enable='1'")
        if animations.get('neonBorder'):
            vf_filters.append(f"drawbox=w=iw:h=ih:t=4:c=0x00ff9d@0.8:enable='1'")
            vf_filters.append(f"drawbox=w=iw-4:h=ih-4:x=2:y=2:t=8:c=0x00ff9d@0.3:enable='1'")
        if animations.get('vectorGrid'):
            vf_filters.append(f"drawgrid=w=60:h=60:t=2:c=0x00ff9d@0.4:x='40*t/20':y='40*t/20':enable='1'")
        if animations.get('particles'):
            vf_filters.append(f"drawgrid=w=50:h=50:t=1:c=white@0.2:x='500*t/20':y='1000*t/20':enable='1'")

        # 2. Add image overlay filters
        # Use filter_complex for multiple inputs
        filter_complex = []
        if vf_filters:
            # Join video filters and label result [v_base]
            filter_complex.append("[0:v]setpts=PTS-STARTPTS," + ",".join(vf_filters) + "[v_base]")
            last_v = "[v_base]"
        else:
            filter_complex.append("[0:v]setpts=PTS-STARTPTS[v_base]")
            last_v = "[v_base]"
            
        overlay_count = 0
        
        # 2. Add B-roll overlays
        for i, marker in enumerate(broll_inputs):
            # Each B-roll marker uses its own input once. 
            # If multiple markers used same B-roll file, they have different input_index.
            input_label = f"[{marker['input_index']}:v]"
            br_stream_label = f"[br_stream_{i}]"
            
            s = marker['start_time']
            e = marker['end_time']
            
            # If hideLogo is enabled, clamp end time to main video duration
            if outro_settings.get('hideLogo', False) and end_time:
                main_dur = to_sec(end_time) - to_sec(start_time)
                if e > main_dur:
                    e = main_dur
            
            duration = max(0.1, e - s)
            transition = marker.get('transition', 'fade')
            trans_dur = min(0.4, duration / 2) # Match CSS 0.4s, but cap at half duration
            
            enable = f"between(t,{s},{e})"
            
            # Base scaling for B-roll - MUST MATCH MAIN VIDEO (1080x1920)
            br_filters = [
                f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(iw-ow)/2:(ih-oh)/2,format=rgba",
                "settb=1/30,setpts=PTS-STARTPTS"
            ]
            
            if transition == 'fade':
                br_filters.append(f"fade=t=in:st=0:d={trans_dur}:alpha=1")
                br_filters.append(f"fade=t=out:st={duration-trans_dur}:d={trans_dur}:alpha=1")
            elif transition == 'zoom':
                br_filters.append(f"zoompan=z='min(zoom+0.001,1.5)':d=1:s=1080x1920:fps=30")
            
            # Shift PTS to start time
            br_filters.append(f"setpts=PTS+{s}*30")
            
            filter_complex.append(f"{input_label}{','.join(br_filters)}{br_stream_label}")
            
            # Slide transitions are handled by dynamic x/y in the overlay filter
            x_pos = "0"
            y_pos = "0"
            
            if transition == 'slide-left':
                x_pos = f"if(between(t,{s},{s+trans_dur}),W-(t-{s})/{trans_dur}*W,if(between(t,{e-trans_dur},{e}),-(t-({e-trans_dur}))/{trans_dur}*W,0))"
            elif transition == 'slide-right':
                x_pos = f"if(between(t,{s},{s+trans_dur}),-W+(t-{s})/{trans_dur}*W,if(between(t,{e-trans_dur},{e}),(t-({e-trans_dur}))/{trans_dur}*W,0))"
            elif transition == 'slide-up':
                y_pos = f"if(between(t,{s},{s+trans_dur}),H-(t-{s})/{trans_dur}*H,if(between(t,{e-trans_dur},{e}),-(t-({e-trans_dur}))/{trans_dur}*H,0))"
            elif transition == 'slide-down':
                y_pos = f"if(between(t,{s},{s+trans_dur}),-H+(t-{s})/{trans_dur}*H,if(between(t,{e-trans_dur},{e}),(t-({e-trans_dur}))/{trans_dur}*H,0))"
            
            # Overlay B-roll over the LAST video stream
            current_ov_label = f"[v_ov_br{i}]"
            filter_complex.append(f"{last_v}{br_stream_label}overlay=x='{x_pos}':y='{y_pos}':enable='{enable}'{current_ov_label}")
            last_v = current_ov_label
            
        # 3. Add image overlay filters
        for i, img_data in enumerate(image_inputs):
            markers = img_data.get('markers', [])
            if not markers:
                continue
                
            # Each marker for this image needs its own split from the input
            if len(markers) > 1:
                split_labels = "".join([f"[img_{i}_m{m_idx}]" for m_idx in range(len(markers))])
                filter_complex.append(f"[{img_data['input_index']}:v]split={len(markers)}{split_labels}")
            else:
                filter_complex.append(f"[{img_data['input_index']}:v]null[img_{i}_m0]")
            
            for m_idx, marker in enumerate(markers):
                output_label = f"[v_ov{overlay_count}]"
                input_m_label = f"[img_{i}_m{m_idx}]"
                
                # Resolve timing: prefer start_time (seconds), fall back to start (percentage)
                s = marker.get('start_time')
                e = marker.get('end_time')
                
                if s is None and 'start' in marker:
                    s = marker['start'] * display_duration
                if e is None and 'end' in marker:
                    e = marker['end'] * display_duration
                
                s = s if s is not None else 0
                e = e if e is not None else display_duration
                
                # If hideLogo is enabled, we clamp the end time to the main video duration
                # to prevent logos/images from showing over the outro clip
                if outro_settings.get('hideLogo', False) and end_time:
                    main_dur = to_sec(end_time) - to_sec(start_time)
                    if e > main_dur:
                        e = main_dur
                
                x_pct = marker.get('x', 50)
                y_pct = marker.get('y', 50)
                scale_factor = marker.get('scale', 1.0)
                stretch_width = marker.get('stretch_width', False)
                
                # Center the overlay by default
                # If stretch_width is true, x_pos should be 0 for a 1080px wide overlay
                if stretch_width:
                    x_pos = "0"
                else:
                    x_pos = f"(main_w*{x_pct}/100-overlay_w/2)"
                
                y_pos = f"(main_h*{y_pct}/100-overlay_h/2)"
                
                enable = f"between(t,{s},{e})"
                
                # Scale image based on 1080 reference width
                scaled_marker_label = f"[img_s_ov{overlay_count}]"
                if stretch_width:
                    target_w = "1080" # Force full width for logo stretching
                else:
                    target_w = f"(1080*0.4*{scale_factor})"
                
                # IMPORTANT: Reset timebase and PTS for images too
                filter_complex.append(f"{input_m_label}scale=w='{target_w}':h='-1',settb=1/30,setpts=PTS-STARTPTS+({s}*30){scaled_marker_label}")
                
                filter_complex.append(f"{last_v}{scaled_marker_label}overlay=x='{x_pos}':y='{y_pos}':enable='{enable}'{output_label}")
                last_v = output_label
                overlay_count += 1

        # 5. Handle Audio Mixing
        orig_vol = float(audio_settings.get('originalVolume', 100)) / 100.0
        custom_vol = float(audio_settings.get('volume', 100)) / 100.0
        
        last_a = "0:a"
        if custom_audio_index is not None:
            # Apply volume to original and custom audio then mix
            filter_complex.append(f"[0:a]asetpts=PTS-STARTPTS,volume={orig_vol}[a_orig]")
            filter_complex.append(f"[{custom_audio_index}:a]asetpts=PTS-STARTPTS,volume={custom_vol}[a_custom]")
            filter_complex.append(f"[a_orig][a_custom]amix=inputs=2:duration=first:dropout_transition=0[a_mixed]")
            last_a = "[a_mixed]"
        else:
            # Just apply volume to original audio
            filter_complex.append(f"[0:a]asetpts=PTS-STARTPTS,volume={orig_vol}[a_orig]")
            last_a = "[a_orig]"

        if filter_complex:
            # Check if we should add final pass for titles/subtitles over all overlays
            # We skip this if it's a theme clip and reburn is False to avoid double titles
            is_theme_clip = "theme_" in str(self.video_path.name)
            final_vf = []
            
            subs = settings.get('subtitles', {})
            if subs.get('show_title') and subs.get('title') and (subs.get('native_burn') or not reburn_subs):
                if not (is_theme_clip and not reburn_subs):
                    title_text = subs['title'].replace("'", "'\\\\\\''").replace(":", "\\:")
                    t_size = subs.get('title_font_size', 80)
                    t_top = subs.get('title_top', 150)
                    t_color = subs.get('title_text_color', '#00ff9d').replace('#', '0x')
                    final_vf.append(f"drawtext=text='{title_text}':fontcolor={t_color}:fontsize={t_size}:x=(w-text_w)/2:y={t_top}:box=1:boxcolor=black@0.6:boxborderw=10")

            if subs.get('native_burn') and subs.get('srt_path'):
                if not (is_theme_clip and not reburn_subs):
                    srt_path = subs['srt_path'].replace('\\', '/').replace(':', '\\:')
                    f_size = subs.get('fontSize', 80)
                    p_color_raw = subs.get('primaryColor', '#ffffff').lstrip('#')
                    p_color = p_color_raw[4:6] + p_color_raw[2:4] + p_color_raw[0:2] if len(p_color_raw) == 6 else "ffffff"
                    final_vf.append(f"subtitles='{srt_path}':force_style='FontSize={f_size},PlayResY=1920,PrimaryColour=&H00{p_color.upper()},OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=100'")

            if final_vf:
                filter_complex.append(f"{last_v}" + ",".join(final_vf) + "[v_final]")
                last_v = "[v_final]"

            # Add Outro Effect (Fade out, Slide out, etc)
            outro_type = outro_settings.get('effect', 'none')
            # If "Hide Video" is checked with "Fade Smooth" in UI, treat it as a fade-out even if no explicit effect
            if outro_type == 'none' and outro_settings.get('fadeVideo') and outro_settings.get('fadeVideoSmooth'):
                outro_type = 'fade-out'

            if outro_type != 'none' and end_time:
                main_dur = to_sec(end_time) - to_sec(start_time)
                outro_dur = float(outro_settings.get('duration', 1.0))
                s = max(0, main_dur - outro_dur)
                enable = f"between(t,{s},{main_dur})"
                
                if outro_type == 'fade-out':
                    filter_complex.append(f"{last_v}fade=t=out:st={s}:d={outro_dur}[v_outro_fade]")
                    last_v = "[v_outro_fade]"
                elif outro_type == 'slide-out-left':
                    filter_complex.append(f"{last_v}overlay=x='-w*(t-{s})/{outro_dur}':y=0:enable='{enable}'[v_outro_slide]")
                    last_v = "[v_outro_slide]"
                elif outro_type == 'slide-out-right':
                    filter_complex.append(f"{last_v}overlay=x='w*(t-{s})/{outro_dur}':y=0:enable='{enable}'[v_outro_slide]")
                    last_v = "[v_outro_slide]"
                elif outro_type == 'zoom-out':
                    filter_complex.append(f"{last_v}zoompan=z='1.0+(0.2*(1-(t-{s})/{outro_dur}))':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920:enable='{enable}'[v_outro_zoom]")
                    last_v = "[v_outro_zoom]"
                elif outro_type == 'blur-out':
                    filter_complex.append(f"{last_v}boxblur=luma_radius='20*(t-{s})/{outro_dur}':enable='{enable}'[v_outro_blur]")
                    last_v = "[v_outro_blur]"

            # Add audio fade out if enabled (MUST be in filter_complex if we already used it for mixing)
            if outro_settings.get('fadeAudio') and end_time:
                main_dur = to_sec(end_time) - to_sec(start_time)
                fade_mode = outro_settings.get('fadeMode', 'after')
                
                if fade_mode == 'before':
                    # Use the specified outro duration for the audio fade leading up to the cut
                    audio_fade_dur = float(outro_settings.get('duration', 1.0))
                    fade_st = max(0, main_dur - audio_fade_dur)
                    filter_complex.append(f"{last_a}afade=t=out:st={fade_st}:d={audio_fade_dur}[a_faded]")
                    last_a = "[a_faded]"
                else:
                    # 'after' mode: Main video audio stays at 100% until the cut
                    # (The background audio will continue and fade in the concatenated outro clip)
                    pass

            cmd.extend(['-filter_complex', ";".join(filter_complex)])
            # Map the last result
            cmd.extend(['-map', f"{last_v}", '-map', last_a])
        elif vf_filters:
            # Skip native burns here too for consistency
            is_theme_clip = "theme_" in str(self.video_path.name)
            subs = settings.get('subtitles', {})
            if not (is_theme_clip and not reburn_subs):
                if subs.get('show_title') and subs.get('title') and (subs.get('native_burn') or not reburn_subs):
                    title_text = subs['title'].replace("'", "'\\\\\\''").replace(":", "\\:")
                    t_size = subs.get('title_font_size', 80)
                    t_top = subs.get('title_top', 150)
                    t_color = subs.get('title_text_color', '#00ff9d').replace('#', '0x')
                    vf_filters.append(f"drawtext=text='{title_text}':fontcolor={t_color}:fontsize={t_size}:x=(w-text_w)/2:y={t_top}:box=1:boxcolor=black@0.6:boxborderw=10")
                if subs.get('native_burn') and subs.get('srt_path'):
                    srt_path = subs['srt_path'].replace('\\', '/').replace(':', '\\:')
                    f_size = subs.get('fontSize', 80)
                    p_color_raw = subs.get('primaryColor', '#ffffff').lstrip('#')
                    p_color = p_color_raw[4:6] + p_color_raw[2:4] + p_color_raw[0:2] if len(p_color_raw) == 6 else "ffffff"
                    vf_filters.append(f"subtitles='{srt_path}':force_style='FontSize={f_size},PlayResY=1920,PrimaryColour=&H00{p_color.upper()},OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=100'")
            
            cmd.extend(['-vf', ",".join(vf_filters)])

        crf = str(settings.get('export_crf', '22'))
        preset = settings.get('export_preset', 'ultrafast')
        log(f"Using dynamic quality settings: CRF={crf}, Preset={preset}")

        cmd.extend([
            '-c:v', 'libx264', '-preset', preset, '-crf', crf,
            '-pix_fmt', 'yuv420p',
            '-r', '30',
            '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
            '-async', '1',
            '-movflags', '+faststart',
            '-threads', '4'
        ])
        
        # ADDED: Force output duration to match the intended trim to prevent infinite loops 
        # with -stream_loop or -loop inputs.
        if end_time:
            dur = to_sec(end_time) - to_sec(start_time)
            if dur > 0:
                cmd.extend(['-t', str(dur)])
        
        cmd.append(str(Path(intermediate_output).absolute()))

        log(f"Running FFmpeg command: {' '.join(cmd)}")
        log(f"Running single-pass effects processing...")
        # Split progress across all passes (Pass 0 took 10%)
        # The remaining 90% is shared by subsequent passes
        has_face_tracking = bool(global_effects.get('faceTracking'))
        reburn_subs = settings.get('subtitles', {}).get('reburn', True)
        
        pass_count = 1
        if has_face_tracking: pass_count += 1
        if reburn_subs: pass_count += 1
        
        p_scale = 0.9 / pass_count
        current_offset = 10.0
        
        # Use a unique temp file for Pass 1 to avoid 'Output same as Input' errors
        # if output_path is accidentally pointing to the source folder
        pass1_temp = os.path.join(temp_dir_for_cleanup or tempfile.gettempdir(), f"pass1_{os.getpid()}.mp4")
        
        # Update command to use temp output
        cmd[-1] = str(Path(pass1_temp).absolute())
        
        # Decide first pass stage name
        pass1_msg = "Applying Edit Effects & Overlays"
        if start_time != '0' or end_time:
            pass1_msg = "Trimming & Applying Edit Effects"

        self._run_with_cancel(cmd, cancel_flag, log_callback, progress_offset=current_offset, progress_scale=p_scale, stage_name=pass1_msg)
        
        # Move temp to intermediate_output
        if os.path.exists(intermediate_output) and intermediate_output != pass1_temp:
            os.remove(intermediate_output)
        import shutil
        shutil.move(pass1_temp, intermediate_output)
        
        current_offset += (100 * p_scale)

        # Apply global effects (Face Tracking) if needed
        if has_face_tracking:
            log("Starting Pass: Face Tracking and Zooming...")
            # If we also have reburn, we need a temp file here
            face_output = output_path
            if reburn_subs:
                face_output = os.path.join(temp_dir_for_cleanup or tempfile.gettempdir(), 'face_tracked.mp4')
            
            self._apply_face_tracking(intermediate_output, face_output, global_effects, cancel_flag, log_callback, progress_offset=current_offset, progress_scale=p_scale)
            intermediate_output = face_output
            current_offset += (100 * p_scale)

        # FINAL STEP: Ensure the result is at the requested output_path
        if intermediate_output != output_path:
            log(f"Moving final result to {output_path}")
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(intermediate_output, output_path)

        # Return the path of the processed video so server.py can do the final pass
        log("Video effects pass complete!")
        return output_path

    def _format_time(self, seconds):
        """Format seconds to HH:MM:SS."""
        h = int(seconds / 3600)
        m = int((seconds % 3600) / 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _run_with_cancel(self, cmd, cancel_flag=None, log_callback=None, progress_offset=0, progress_scale=1.0, stage_name="Processing"):
        """Run subprocess with cancellation and progress reporting."""
        import time
        import re
        import os
        import selectors
        import subprocess

        # Ensure we capture stderr for progress
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )

        # Set stderr to non-blocking
        fd = process.stderr.fileno()
        import fcntl
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        duration_sec = 0
        try:
            if '-t' in cmd:
                t_idx = cmd.index('-t')
                t_val = cmd[t_idx+1]
                if ':' in t_val:
                    parts = list(map(float, t_val.split(':')))
                    duration_sec = parts[-1] + (parts[-2] * 60 if len(parts) > 1 else 0) + (parts[-3] * 3600 if len(parts) > 2 else 0)
                else:
                    duration_sec = float(t_val)
            elif '-to' in cmd:
                to_idx = cmd.index('-to')
                to_val = cmd[to_idx+1]
                # If -ss is also present, duration is to - ss
                start_sec = 0
                if '-ss' in cmd:
                    ss_val = cmd[cmd.index('-ss')+1]
                    if ':' in ss_val:
                        parts = list(map(float, ss_val.split(':')))
                        start_sec = parts[-1] + (parts[-2] * 60 if len(parts) > 1 else 0) + (parts[-3] * 3600 if len(parts) > 2 else 0)
                    else:
                        start_sec = float(ss_val)
                
                if ':' in to_val:
                    parts = list(map(float, to_val.split(':')))
                    to_sec = parts[-1] + (parts[-2] * 60 if len(parts) > 1 else 0) + (parts[-3] * 3600 if len(parts) > 2 else 0)
                    duration_sec = to_sec - start_sec
                else:
                    duration_sec = float(to_val) - start_sec
            
            if duration_sec <= 0:
                duration_sec = self.total_frames / self.fps if self.fps > 0 else 0
        except:
            duration_sec = 0

        sel = selectors.DefaultSelector()
        sel.register(process.stderr, selectors.EVENT_READ)
        stderr_buffer = ""

        try:
            while process.poll() is None:
                if cancel_flag and cancel_flag():
                    process.terminate()
                    time.sleep(0.2)
                    if process.poll() is None: process.kill()
                    raise Exception("Cancelled by user")
                
                events = sel.select(timeout=0.1)
                for key, mask in events:
                    try:
                        chunk = process.stderr.read(4096)
                        if not chunk: continue
                        stderr_buffer += chunk
                        
                        if '\r' in stderr_buffer or '\n' in stderr_buffer:
                            lines = re.split(r'[\r\n]+', stderr_buffer)
                            stderr_buffer = lines.pop()
                            
                            for line in lines:
                                if not line.strip(): continue
                                if log_callback:
                                    # Support 2 or 3 decimal places in FFmpeg time output
                                    time_match = re.search(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d+)", line)
                                    if time_match and duration_sec > 0:
                                        h, m, s, ms_str = time_match.groups()
                                        ms = float("0." + ms_str)
                                        current_time = int(h) * 3600 + int(m) * 60 + int(s) + ms
                                        percent = min(99, int((current_time / duration_sec) * 100))
                                        # Scale and offset for two-pass progress
                                        scaled_percent = int(progress_offset + (percent * progress_scale))
                                        log_callback(f"PROGRESS:{scaled_percent}%|{stage_name}")
                                    
                                    if "time=" not in line and "frame=" not in line:
                                        log_callback(line)
                    except (BlockingIOError, TypeError):
                        pass
                time.sleep(0.01)
        finally:
            sel.unregister(process.stderr)
            sel.close()

        stdout, stderr_remainder = process.communicate()
        if process.returncode != 0:
            # Note: process.communicate()'s stderr will be empty if we've read from the pipe
            # So we rely on our accumulated buffer + any remainder
            full_stderr = (stderr_buffer + (stderr_remainder if stderr_remainder else "")).strip()
            err_msg = full_stderr if full_stderr else "Process ended abruptly. Check logs for details."
            if log_callback:
                log_callback(f"FFmpeg Error: {err_msg}")
            raise subprocess.CalledProcessError(process.returncode, cmd, err_msg)
        return stdout

    def _copy_video_with_trim(self, output_path, start_time, end_time, settings, cancel_flag=None, log_callback=None):
        """Copy video with optional trim, no effects."""
        ffmpeg_cmd = ['ffmpeg', '-i', str(self.video_path)]

        if start_time != '0':
            ffmpeg_cmd.extend(['-ss', start_time])
        if end_time:
            ffmpeg_cmd.extend(['-t', end_time])

        ffmpeg_cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-y',
            str(output_path)
        ])

        if cancel_flag:
            self._run_with_cancel(ffmpeg_cmd, cancel_flag)
        else:
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

    def _create_segment(self, start_sec, end_sec, output_path, settings, cancel_flag=None, log_callback=None):
        """Create a video segment with no effects."""
        self._copy_video_with_trim(output_path, self._format_time(start_sec), self._format_time(end_sec), settings, cancel_flag)

    def _create_effect_segment(self, start_sec, end_sec, output_path, effect_type, settings, cancel_flag=None, log_callback=None):
        """Create a video segment with specific effect applied."""
        effect_filters = {
            'mirror': 'hflip',
            'grayscale': 'format=gray',
            'sepia': 'colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131',
            'blur': 'gblur=sigma=5',
            'zoom': 'scale=1.2*iw:-1,crop=iw/1.2:ih/1.2'
        }

        ffmpeg_cmd = ['ffmpeg', '-i', str(self.video_path),
                        '-ss', self._format_time(start_sec),
                        '-t', self._format_time(end_sec - start_sec)]

        if effect_filters.get(effect_type):
            ffmpeg_cmd.extend(['-vf', effect_filters[effect_type]])

        ffmpeg_cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-y',
            str(output_path)
        ])

        if cancel_flag:
            self._run_with_cancel(ffmpeg_cmd, cancel_flag)
        else:
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

    def _concatenate_segments(self, segment_files, output_path, settings, cancel_flag=None, log_callback=None):
        """Concatenate video segments using ffmpeg concat demuxer."""
        import os

        # Create concat file
        concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)

        for segment in segment_files:
            concat_file.write(f"file '{segment}'\n")

        concat_file.close()

        # Build ffmpeg concat command
        ffmpeg_cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', concat_file.name,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-y',
            str(output_path)
        ]

        if cancel_flag:
            self._run_with_cancel(ffmpeg_cmd, cancel_flag)
        else:
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

        # Clean up concat file
        os.unlink(concat_file.name)

    def _apply_face_tracking(self, input_path: str, output_path: str, effects: dict, cancel_flag=None, log_callback=None, progress_offset=0, progress_scale=1.0):
        """Apply face tracking with zoom and pan - Optimized version."""
        import os
        import tempfile
        import shutil
        import time

        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        log(f"Starting optimized face tracking processing...")
        
        # ... rest of function using progress_offset and progress_scale ...

        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Open video
        cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
        if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
            
        if not cap.isOpened():
            cap = cv2.VideoCapture(input_path)

        # Verify readability
        ret, test_frame = cap.read()
        if not ret:
            log("OpenCV failed to read video directly. Attempting automatic transcode fallback...")
            cap.release()
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                proxy_path = tmp.name
            transcode_cmd = ['ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'copy', proxy_path]
            subprocess.run(transcode_cmd, capture_output=True)
            input_path = proxy_path
            cap = cv2.VideoCapture(input_path)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        log(f"Video info: {total_frames} frames at {fps}fps, {width}x{height}")

        zoom_level = float(effects.get('faceZoomLevel', 1.5))
        smoothing = effects.get('faceSmoothing', 'medium')
        smooth_factor = {'low': 0.05, 'high': 0.25}.get(smoothing, 0.12)

        # OPTIMIZATION: Setup direct FFmpeg pipe instead of cv2.VideoWriter + final pass
        crf = str(settings.get('export_crf', '23'))
        preset = settings.get('export_preset', 'fast')
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', # Stdin
            '-i', input_path, # For audio
            '-c:v', 'libx264', '-preset', preset, '-crf', crf,
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0', '-map', '1:a:0?',
            '-movflags', '+faststart',
            '-threads', '4',
            output_path
        ]
        
        # Start FFmpeg process
        pipe_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        target_width = int(width / zoom_level)
        target_height = int(height / zoom_level)
        smooth_x = float(width - target_width) / 2.0
        smooth_y = float(height - target_height) / 2.0
        prev_valid_x, prev_valid_y = smooth_x, smooth_y
        
        # Detection optimizations
        detect_interval = 5 # Detect every 5 frames
        detect_scale = 0.4 # Downscale for detection
        last_faces = []
        no_face_count = 0
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                if cancel_flag and cancel_flag():
                    raise Exception("Cancelled by user")

                ret, frame = cap.read()
                if not ret: break

                target_x, target_y = smooth_x, smooth_y

                # Only detect periodically
                if frame_count % detect_interval == 0:
                    # OPTIMIZATION: Convert and downscale only for detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    small_gray = cv2.resize(gray, (0,0), fx=detect_scale, fy=detect_scale)
                    
                    faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(50*detect_scale), int(50*detect_scale)))
                    last_faces = faces
                else:
                    faces = last_faces

                if len(faces) > 0:
                    # Map back to original scale
                    face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = [int(v / detect_scale) for f in [face] for v in f]

                    face_center_x, face_center_y = x + w // 2, y + h // 2
                    new_target_x = float(face_center_x - target_width // 2)
                    new_target_y = float(face_center_y - target_height // 3)

                    max_jump = max(target_width, target_height) * 0.4
                    if abs(new_target_x - prev_valid_x) < max_jump and abs(new_target_y - prev_valid_y) < max_jump:
                        target_x, target_y = new_target_x, new_target_y
                        prev_valid_x, prev_valid_y = target_x, target_y
                        no_face_count = 0
                else:
                    no_face_count += 1
                    if no_face_count > 30:
                        target_x, target_y = (width - target_width) / 2.0, (height - target_height) / 2.0

                # Clamp and smooth
                target_x = max(0, min(target_x, width - target_width))
                target_y = max(0, min(target_y, height - target_height))
                smooth_x += (target_x - smooth_x) * smooth_factor
                smooth_y += (target_y - smooth_y) * smooth_factor

                # Extract and resize
                sx, sy = int(smooth_x), int(smooth_y)
                cropped = frame[sy:sy + target_height, sx:sx + target_width]
                resized = cv2.resize(cropped, (width, height))

                # Write to pipe
                pipe_proc.stdin.write(resized.tobytes())

                frame_count += 1
                if frame_count % 50 == 0:
                    percent = int((frame_count / total_frames) * 100)
                    scaled_percent = int(progress_offset + (percent * progress_scale))
                    if log_callback:
                        log_callback(f"PROGRESS:{scaled_percent}%")
                    else:
                        log(f"Processing... {percent}% ({frame_count}/{total_frames})")

            # Finish pipe
            pipe_proc.stdin.close()
            pipe_proc.wait()
            log(f"Face tracking complete! Total time: {int(time.time() - start_time)}s")

        finally:
            cap.release()
            if pipe_proc.poll() is None:
                pipe_proc.terminate()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: video_processor.py <input_video> <output_video>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]

    processor = VideoProcessor(input_video)

    # Example: Apply face tracking with default settings
    settings = {
        'effects': {
            'faceTracking': True,
            'faceZoomLevel': 1.5,
            'faceSmoothing': 'medium'
        }
    }

    processor.apply_effects(output_video, settings)
