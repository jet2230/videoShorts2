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

    def apply_effects(self, output_path: str, settings: dict, cancel_flag=None, log_callback=None):
        """
        Apply time-based effects using a single-pass FFmpeg complex filter.
        """
        import subprocess
        import tempfile
        import os

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
        global_effects = settings.get('effects', {})
        lighting = settings.get('lighting', {})
        animations = settings.get('animations', {})
        
        log(f"Received animations: {animations}")

        log("Starting optimized video processing...")

        # Parse trim settings
        start_time = trim_settings.get('start', '0')
        end_time = trim_settings.get('end')

        # If we have subsequent passes, we MUST use a temporary file for the first pass result.
        has_face_tracking = bool(global_effects.get('faceTracking'))
        reburn_subs = settings.get('subtitles', {}).get('reburn', True)
        
        intermediate_output = output_path
        temp_dir_for_cleanup = tempfile.mkdtemp()
        
        if has_face_tracking or reburn_subs:
            intermediate_output = os.path.join(temp_dir_for_cleanup, 'intermediate.mp4')

        # Build FFmpeg command
        cmd = ['ffmpeg', '-y']
        
        # Fast input seeking for trim
        if start_time != '0':
            cmd.extend(['-ss', start_time])
        
        abs_input = str(self.video_path.absolute())
        cmd.extend(['-i', abs_input])
        
        input_index = 1
        
        # Add additional inputs ... (keeping existing logic)
        
        # Add additional inputs for images
        image_inputs = []
        for img_data in image_settings:
            img_path = Path('media') / img_data['name']
            if img_path.exists():
                cmd.extend(['-i', str(img_path.absolute())])
                img_data['input_index'] = input_index
                image_inputs.append(img_data)
                input_index += 1
            else:
                log(f"Warning: Image not found: {img_path}")
        
        # Add additional inputs for B-roll
        broll_inputs = []
        for marker in broll_markers:
            broll_path = Path('media') / marker['name']
            if broll_path.exists():
                cmd.extend(['-i', str(broll_path.absolute())])
                marker['input_index'] = input_index
                broll_inputs.append(marker)
                input_index += 1
            else:
                log(f"Warning: B-roll clip not found: {broll_path}")
        
        # Add additional input for custom audio
        custom_audio_index = None
        if audio_settings.get('file'):
            audio_path = Path('media') / audio_settings['file']
            if audio_path.exists():
                cmd.extend(['-i', str(audio_path.absolute())])
                custom_audio_index = input_index
                input_index += 1
            else:
                log(f"Warning: Audio file not found: {audio_path}")
        
        if end_time:
            # We use duration for -t to be safe
            try:
                def to_sec(ts):
                    if ':' in str(ts):
                        parts = list(map(float, str(ts).replace(',', '.').split(':')))
                        return parts[-1] + (parts[-2] * 60 if len(parts) > 1 else 0) + (parts[-3] * 3600 if len(parts) > 2 else 0)
                    return float(ts)
                
                dur = to_sec(end_time) - to_sec(start_time)
                if dur > 0:
                    cmd.extend(['-t', str(dur)])
            except:
                # Fallback to -to if duration calculation fails
                cmd.extend(['-to', end_time])

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
        # brightness = 0.5 * C * (M - 1)
        # Apply global lighting + global flash
        c_expr_global = f"({safe_M}*({global_flash_expr})*{c_ui})"
        b_expr_global = f"(0.5*{c_ui}*({safe_M}*({global_flash_expr})-1))"
        
        # Force vertical 1080x1920 output as the base for all effects
        vf_filters.append(f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(iw-ow)/2:(ih-oh)/2")
        
        vf_filters.append(f"eq=contrast='{c_expr_global}':brightness='{b_expr_global}':saturation={s_ui}:eval=frame")

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
                vf_filters.append(f"crop=w=iw-40:h=ih-40:x='20+20*sin(2*PI*t*10)':y='20+20*cos(2*PI*t*13)':enable='{enable}',scale={self.width}:{self.height}")
            elif etype == 'pixelate':
                vf_filters.append(f"boxblur=20:enable='{enable}'")
            elif etype == 'zoom':
                vf_filters.append(f"zoompan=z='if({enable},1.2,1)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={self.width}x{self.height}:enable='{enable}'")

        # 1.5 Global Animations & Styles
        style = settings.get('style', 'none')
        if style == 'pixels':
            # Fast pixelation: scale down then up
            vf_filters.append(f"scale=iw/8:ih/8:flags=neighbor,scale={self.width}:{self.height}:flags=neighbor")
        elif style == 'paint':
            vf_filters.append("bilateral=sigmaS=5:sigmaR=0.1,unsharp=3:3:1.5:3:3:0.5")
        elif style == 'pencil':
            vf_filters.append("format=gray,edgedetect=low=0.1:high=0.2,negate")
        elif style == 'neon':
            vf_filters.append("edgedetect=low=0.1:high=0.4,hue=h=120:s=2")
        elif style == 'poster':
            vf_filters.append("colorlevels=rimin=0.05:gimin=0.05:bimin=0.05:rimax=0.95:gimax=0.95:bimax=0.95")
        elif style == 'retro':
            vf_filters.append("noise=alls=10:allf=t,hue=s=0.3,vignette")

        # Global Animations
        if animations.get('vhs'):
            vf_filters.append("noise=alls=20:allf=t+u,hue=s=0.5,gblur=sigma=1")
        if animations.get('kenBurns'):
            vf_filters.append(f"zoompan=z='min(zoom+0.0005,1.2)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={self.width}x{self.height}")
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

        # 2. Add image overlay filters
        # Use filter_complex for multiple inputs
        filter_complex = []
        if vf_filters:
            # Join video filters and label result [v_base]
            filter_complex.append("[0:v]" + ",".join(vf_filters) + "[v_base]")
            last_v = "[v_base]"
        else:
            last_v = "[0:v]"
            
        overlay_count = 0
        
        # 2. Add B-roll overlays
        for i, marker in enumerate(broll_inputs):
            input_label = f"[{marker['input_index']}:v]"
            output_label = f"[v_br{overlay_count}]"
            
            s = marker['start_time']
            e = marker['end_time']
            duration = e - s
            transition = marker.get('transition', 'fade')
            trans_dur = 0.4 # Match CSS 0.4s
            
            enable = f"between(t,{s},{e})"
            
            # Base scaling for B-roll - MUST MATCH MAIN VIDEO (1080x1920)
            br_filters = [f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(iw-ow)/2:(ih-oh)/2"]
            
            # Apply transition effects to the B-roll input before overlaying
            if transition == 'fade':
                br_filters.append(f"fade=t=in:st={s}:d={trans_dur}:alpha=1")
                br_filters.append(f"fade=t=out:st={e-trans_dur}:d={trans_dur}:alpha=1")
            elif transition == 'zoom':
                # Use standard 1080x1920 for zoompan output
                br_filters.append(f"zoompan=z='if(between(t,{s},{s+trans_dur}),1+(t-{s})/{trans_dur}*0.2,if(between(t,{e-trans_dur},{e}),1.2-({trans_dur}-(t-({e-trans_dur})))/{trans_dur}*0.2,1.2))':d=1:s=1080x1920")
            
            filter_complex.append(f"{input_label}{','.join(br_filters)}{output_label}")
            
            # Slide transitions are handled by dynamic x/y in the overlay filter
            x_pos = "0"
            y_pos = "0"
            
            if 'slide-left' in transition:
                # Slide from right (width) to 0, then stay, then slide to left (-width)
                x_pos = f"if(between(t,{s},{s+trans_dur}),W-(t-{s})/{trans_dur}*W,if(between(t,{e-trans_dur},{e}),-(t-({e-trans_dur}))/{trans_dur}*W,0))"
            elif 'slide-right' in transition:
                # Slide from left (-width) to 0, then stay, then slide to right (width)
                x_pos = f"if(between(t,{s},{s+trans_dur}),-W+(t-{s})/{trans_dur}*W,if(between(t,{e-trans_dur},{e}),(t-({e-trans_dur}))/{trans_dur}*W,0))"
            elif 'slide-up' in transition:
                y_pos = f"if(between(t,{s},{s+trans_dur}),H-(t-{s})/{trans_dur}*H,if(between(t,{e-trans_dur},{e}),-(t-({e-trans_dur}))/{trans_dur}*H,0))"
            elif 'slide-down' in transition:
                y_pos = f"if(between(t,{s},{s+trans_dur}),-H+(t-{s})/{trans_dur}*H,if(between(t,{e-trans_dur},{e}),(t-({e-trans_dur}))/{trans_dur}*H,0))"
            
            # Overlay B-roll over main video
            final_ov_label = f"[v_ov{overlay_count}]"
            filter_complex.append(f"{last_v}{output_label}overlay=x='{x_pos}':y='{y_pos}':enable='{enable}'{final_ov_label}")
            last_v = final_ov_label
            overlay_count += 1
            
        # 3. Add image overlay filters
        for i, img_data in enumerate(image_inputs):
            input_label = f"[{img_data['input_index']}:v]"
            
            for marker in img_data.get('markers', []):
                output_label = f"[v_ov{overlay_count}]"
                
                s = marker['start_time']
                e = marker['end_time']
                x_pct = marker.get('x', 50)
                y_pct = marker.get('y', 50)
                scale_factor = marker.get('scale', 1.0)
                stretch_width = marker.get('stretch_width', False)
                
                # Proper centering: main_w*x_pct/100 - overlay_w/2
                x_pos = f"(main_w*{x_pct}/100-overlay_w/2)"
                y_pos = f"(main_h*{y_pct}/100-overlay_h/2)"
                
                enable = f"between(t,{s},{e})"
                
                # Scale image based on video width and scale factor
                scaled_marker_label = f"[img_s_ov{overlay_count}]"
                if stretch_width:
                    target_w = str(self.width)
                else:
                    target_w = f"({self.width}*0.4*{scale_factor})"
                
                filter_complex.append(f"{input_label}scale=w='{target_w}':h='-1'{scaled_marker_label}")
                
                filter_complex.append(f"{last_v}{scaled_marker_label}overlay=x='{x_pos}':y='{y_pos}':enable='{enable}'{output_label}")
                last_v = output_label
                overlay_count += 1

        # 5. Handle Audio Mixing
        orig_vol = float(audio_settings.get('originalVolume', 100)) / 100.0
        custom_vol = float(audio_settings.get('volume', 100)) / 100.0
        
        last_a = "0:a"
        if custom_audio_index is not None:
            # Apply volume to original and custom audio then mix
            filter_complex.append(f"[0:a]volume={orig_vol}[a_orig]")
            filter_complex.append(f"[{custom_audio_index}:a]volume={custom_vol}[a_custom]")
            filter_complex.append(f"[a_orig][a_custom]amix=inputs=2:duration=first[a_mixed]")
            last_a = "[a_mixed]"
        else:
            # Just apply volume to original audio
            filter_complex.append(f"[0:a]volume={orig_vol}[a_orig]")
            last_a = "[a_orig]"

        if filter_complex:
            cmd.extend(['-filter_complex', ";".join(filter_complex)])
            # Map the last result
            cmd.extend(['-map', f"{last_v}", '-map', last_a])
        elif vf_filters:
            cmd.extend(['-vf', ",".join(vf_filters)])

        cmd.extend([
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart',
            str(Path(intermediate_output).absolute())
        ])

        log(f"Running single-pass effects processing...")
        # Split progress across all passes
        has_face_tracking = bool(global_effects.get('faceTracking'))
        reburn_subs = settings.get('subtitles', {}).get('reburn', True)
        
        # Check if we actually have any FFmpeg effects to run
        # If reburn is enabled, we ALWAYS want to run Pass 1 to get a vertical trimmed source
        has_ffmpeg_effects = bool(vf_filters or filter_complex or start_time != '0' or end_time or reburn_subs)
        
        if has_ffmpeg_effects:
            pass_count = 1
            if has_face_tracking: pass_count += 1
            if reburn_subs: pass_count += 1
            
            p_scale = 1.0 / pass_count
            current_offset = 0
            
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
        else:
            log("No FFmpeg effects requested, skipping Pass 1.")
            # Use original as intermediate
            intermediate_output = str(self.video_path)
            pass_count = 1
            if has_face_tracking: pass_count += 1
            if reburn_subs: pass_count += 1
            p_scale = 1.0 / pass_count
            current_offset = (100 * p_scale) # The 'skip' counts as 1 pass done

        # Apply global effects (Face Tracking) if needed
        if has_face_tracking:
            log("Starting Pass: Face Tracking and Zooming...")
            # If we also have reburn, we need a temp file here
            face_output = output_path
            if reburn_subs:
                face_output = os.path.join(temp_dir_for_cleanup or tempfile.mkdtemp(), 'face_tracked.mp4')
            
            self._apply_face_tracking(intermediate_output, face_output, global_effects, cancel_flag, log_callback, progress_offset=current_offset, progress_scale=p_scale)
            intermediate_output = face_output
            current_offset += (100 * p_scale)

        # Return the path of the processed video so server.py can do the final pass
        log("Video effects pass complete!")
        return intermediate_output

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

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            # Note: stderr is already partially read if we were tracking progress
            err_msg = stderr if stderr else "Process ended abruptly. Check logs for details."
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
            transcode_cmd = ['ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20', '-c:a', 'copy', proxy_path]
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
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', # Stdin
            '-i', input_path, # For audio
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0', '-map', '1:a:0?',
            '-movflags', '+faststart',
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
