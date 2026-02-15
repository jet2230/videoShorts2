#!/usr/bin/env python3
"""
Whisper Transcription Class

Encapsulates Whisper transcription with GPU management, progress tracking,
and word-level timestamp generation for karaoke subtitles.
"""

import subprocess
import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Callable


class TranscribeWhisper:
    """Encapsulates Whisper transcription with GPU management and progress tracking."""

    def __init__(self, model_size='small', language='en'):
        """Initialize TranscribeWhisper with model and language settings.

        Args:
            model_size: Whisper model size ('base', 'small', 'medium', 'large')
            language: Language code ('en', 'ar', 'es', etc.) or None for auto-detect
        """
        self.model_size = model_size
        self.language = language
        self.device = None
        self.model = None

    def prepare_resources(self) -> None:
        """Stop Ollama and clear CUDA cache to free GPU memory."""
        # Stop Ollama to free GPU memory
        try:
            subprocess.run(
                ['ollama', 'stop', 'llama3'],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
            print("\033[91mStopped Ollama llama3 model\033[0m")
        except Exception:
            pass

        # Clear PyTorch CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("\033[91mCleared CUDA cache\033[0m")
        except Exception:
            pass

    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """Load Whisper model with GPU/CPU auto-fallback.

        Args:
            progress_callback: Optional callback for progress updates
        """
        def _log_msg(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        import whisper as whisper_module
        import torch

        _log_msg(f"Loading Whisper model ({self.model_size})...")
        print(f"DEBUG: Starting whisper.load_model('{self.model_size}')")

        # Check if CUDA is available and has memory
        use_cuda = False
        if torch.cuda.is_available():
            try:
                gpu_memory_free = (
                    torch.cuda.get_device_properties(0).total_memory -
                    torch.cuda.memory_allocated(0)
                )
                _log_msg(f"GPU memory free: {gpu_memory_free / 1024**2:.0f} MB")
                # Only use CUDA if at least 2GB free
                if gpu_memory_free > 2 * 1024**3:
                    use_cuda = True
            except Exception as e:
                _log_msg(f"GPU memory check failed: {e}")

        self.device = "cuda" if use_cuda else "cpu"
            
        _log_msg(f"Using device: {self.device}")

        try:
            self.model = whisper_module.load_model(self.model_size, device=self.device)
            _log_msg(f"Model loaded on {self.device}")
        except RuntimeError as e:
            if self.device == "cuda" and ("CUDA out of memory" in str(e) or "out of memory" in str(e)):
                _log_msg("GPU out of memory. Falling back to CPU...")
                self.device = "cpu"
                self.model = whisper_module.load_model(self.model_size, device="cpu")
                _log_msg(f"Model loaded on CPU")
            else:
                raise

    def transcribe(
        self,
        video_path: Path,
        output_folder: Path,
        task: str = 'transcribe',
        progress_callback: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> str:
        """Transcribe video with word-level timestamps.

        Args:
            video_path: Path to video file
            output_folder: Path to folder for output files (SRT and JSON)
            task: Whisper task ('transcribe' or 'translate')
            progress_callback: Optional callback for progress updates
            cancel_check: Optional function that returns True if task should be cancelled

        Returns:
            Path to generated SRT file
        """
        def _log_msg(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        # Handle language parameter
        language = self.language
        if language and language.lower() in ('auto', 'none', ''):
            language = None

        _log_msg(f"Transcribing with Whisper ({self.model_size})...")

        # Import whisper dependencies
        import whisper as whisper_module
        from whisper.utils import get_writer

        # Start background progress simulator
        stop_progress = threading.Event()

        def progress_simulator():
            for i in range(0, 101, 10):
                if stop_progress.is_set():
                    break
                # Check for cancellation within simulator too
                if cancel_check and cancel_check():
                    break
                _log_msg(f"Progress: {i}%")
                time.sleep(2)

        progress_thread = threading.Thread(target=progress_simulator, daemon=True)
        progress_thread.start()

        try:
            # Check for cancellation before starting long task
            if cancel_check and cancel_check():
                raise Exception("Cancelled by user")

            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                str(video_path),
                language=language,
                task=task,
                word_timestamps=True,
                verbose=False,
                fp16=(self.device == "cuda")
            )
            
            # Check for cancellation after transcription
            if cancel_check and cancel_check():
                raise Exception("Cancelled by user")
        finally:
            stop_progress.set()
            progress_thread.join()
            
            # If we were cancelled, don't report 100%
            if not (cancel_check and cancel_check()):
                _log_msg("Progress: 100%")

        # Save SRT file
        base_name = Path(video_path).stem
        srt_path = Path(output_folder) / f"{base_name}.srt"

        writer = get_writer('srt', str(output_folder))
        writer(result, srt_path)

        # Save word-level timestamps for karaoke
        word_timestamps_path = Path(output_folder) / f"{base_name}_word_timestamps.json"
        self._save_word_timestamps(result, word_timestamps_path)
        if word_timestamps_path.exists():
            _log_msg(f"Created word timestamps: {word_timestamps_path}")

        if srt_path.exists():
            _log_msg(f"Created subtitles: {srt_path}")
            _log_msg("Subtitles: Complete")
        else:
            raise FileNotFoundError(f"Whisper did not create expected subtitle file: {srt_path}")

        return str(srt_path)

    def _save_word_timestamps(self, result: dict, output_path: Path) -> None:
        """Save word-level timestamps from Whisper result to JSON file.

        Args:
            result: Whisper result dict with word timestamps
            output_path: Path to save JSON file
        """
        word_data = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word_data.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"]
                })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "language": result.get("language"),
                "duration": result.get("duration"),
                "words": word_data
            }, f, indent=2, ensure_ascii=False)

    @staticmethod
    def get_available_models() -> list:
        """Return list of available Whisper models."""
        return ['base', 'small', 'medium', 'large']
