#!/usr/bin/env python3
"""
Optimize all existing videos with ffmpeg faststart for better seeking.
This script processes main video files (not shorts/edited_shorts).
"""

import subprocess
import sys
from pathlib import Path

def optimize_video(video_path: Path) -> bool:
    """Optimize a single video with faststart."""
    print(f"\n{'='*60}")
    print(f"Optimizing: {video_path.name}")
    print(f"{'='*60}")

    # Check if video exists
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return False

    # Create temp file path
    optimized_path = video_path.parent / f"{video_path.stem}_temp.mp4"

    # Run ffmpeg faststart optimization
    try:
        print(f"Running ffmpeg faststart...")
        result = subprocess.run([
            'ffmpeg', '-i', str(video_path),
            '-c', 'copy',  # Copy streams without re-encoding (fast)
            '-movflags', 'faststart',
            '-y',  # Overwrite output file if exists
            str(optimized_path)
        ], capture_output=True, text=True)

        if result.returncode == 0:
            # Get file sizes
            original_size = video_path.stat().st_size / (1024 * 1024)  # MB
            optimized_size = optimized_path.stat().st_size / (1024 * 1024)  # MB

            # Replace original with optimized
            video_path.unlink()
            optimized_path.rename(video_path)

            print(f"✅ Optimization complete!")
            print(f"   Original size: {original_size:.1f} MB")
            print(f"   Optimized size: {optimized_size:.1f} MB")
            return True
        else:
            print(f"❌ Optimization failed!")
            print(f"   Error: {result.stderr}")
            # Clean up temp file if it exists
            if optimized_path.exists():
                optimized_path.unlink()
            return False

    except FileNotFoundError:
        print(f"❌ ffmpeg not found! Please install ffmpeg.")
        print(f"   Ubuntu/Debian: sudo apt install ffmpeg")
        print(f"   macOS: brew install ffmpeg")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        # Clean up temp file if it exists
        if optimized_path.exists():
            optimized_path.unlink()
        return False

def main():
    """Find and optimize all main videos."""
    videos_dir = Path("videos")

    if not videos_dir.exists():
        print(f"❌ Videos directory not found: {videos_dir}")
        sys.exit(1)

    # Find all main MP4 files (exclude shorts and edited_shorts)
    video_files = []
    for video_path in videos_dir.rglob("*.mp4"):
        # Skip if in shorts or edited_shorts directories
        if "/shorts/" in str(video_path) or "\\shorts\\" in str(video_path):
            continue
        if "/edited_shorts/" in str(video_path) or "\\edited_shorts\\" in str(video_path):
            continue
        video_files.append(video_path)

    if not video_files:
        print("No main video files found to optimize.")
        return

    print(f"Found {len(video_files)} main video(s) to optimize:")
    for i, v in enumerate(video_files, 1):
        print(f"  {i}. {v.parent.name} / {v.name}")

    # Ask for confirmation (skip if --yes flag is provided)
    if '--yes' not in sys.argv:
        try:
            response = input("\nProceed with optimization? (y/n): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return
        except (EOFError, KeyboardInterrupt):
            print("\nNon-interactive mode, proceeding...")
            print("(Use --yes flag to skip this prompt in the future)")

    # Optimize each video
    success_count = 0
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]", end=" ")
        if optimize_video(video_path):
            success_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Optimization complete!")
    print(f"  Success: {success_count}/{len(video_files)}")
    print(f"  Failed: {len(video_files) - success_count}/{len(video_files)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
