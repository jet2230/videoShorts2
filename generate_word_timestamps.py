#!/usr/bin/env python3
"""
Generate word timestamps JSON files for videos that don't have them.
Uses the existing transcription functionality from shorts_creator.py.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/home/obo/playground/videoShorts2')

from shorts_creator import YouTubeShortsCreator, load_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    # Load settings
    settings = load_settings()
    base_dir = Path(settings.get('video', 'output_dir'))

    if not base_dir.exists():
        logger.error(f"Videos directory not found: {base_dir}")
        sys.exit(1)

    # Find all videos (both with and without word timestamps)
    all_folders = []
    folders_without_ts = []

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        # Check for video files
        video_files = list(folder.glob('*.mp4'))
        if not video_files:
            continue

        video_file = video_files[0]
        size_mb = video_file.stat().st_size / (1024*1024)

        # Check for word timestamps
        word_ts_files = list(folder.glob('*_word_timestamps.json'))
        has_word_ts = len(word_ts_files) > 0

        folder_info = {
            'folder': folder,
            'video': video_file,
            'name': folder.name,
            'size_mb': size_mb,
            'has_word_ts': has_word_ts
        }

        all_folders.append(folder_info)
        if not has_word_ts:
            folders_without_ts.append(folder_info)

    if not all_folders:
        logger.info("No video folders found!")
        return

    # Display all videos
    logger.info("=" * 80)
    logger.info("VIDEO LIBRARY")
    logger.info("=" * 80)
    logger.info("\nFormat: [index] Folder Name (Size - has word timestamps?)\n")
    logger.info(f"Total videos: {len(all_folders)}")
    logger.info(f"Videos without word timestamps: {len(folders_without_ts)}")
    logger.info("")

    for i, item in enumerate(all_folders, 1):
        status = "✅ has timestamps" if item['has_word_ts'] else "❌ missing timestamps"
        logger.info(f"{i:2}. {item['name']}")
        logger.info(f"    Size: {item['size_mb']:.0f} MB | {status}")

    # Selection prompt
    logger.info("\n" + "=" * 80)
    logger.info("SELECT VIDEOS TO PROCESS")
    logger.info("=" * 80)
    logger.info("\nEnter video numbers to process (comma-separated, e.g., 1,3,5 or 1-5)")
    logger.info("  - 'all' to process all videos without timestamps")
    logger.info("  - 'list' to show only videos without timestamps")
    logger.info("  - 'q' to quit")

    while True:
        response = input("\nSelect videos: ").strip()

        if response.lower() == 'q':
            logger.info("Cancelled.")
            return

        if response.lower() == 'list':
            logger.info("\nVideos WITHOUT word timestamps:")
            for i, item in enumerate(folders_without_ts, 1):
                logger.info(f"{i:2}. {item['name']} ({item['size_mb']:.0f} MB)")
            logger.info(f"\nTotal: {len(folders_without_ts)} videos")
            continue

        if response.lower() == 'all':
            # Process all videos without timestamps
            folders_to_process = folders_without_ts
            break

        # Parse selection (numbers or ranges)
        try:
            selected_indices = []
            parts = response.split(',')

            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Range (e.g., 1-5)
                    start, end = part.split('-')
                    selected_indices.extend(range(int(start), int(end) + 1))
                else:
                    # Single number
                    selected_indices.append(int(part))

            # Get selected folders
            folders_to_process = []
            total_size_mb = 0

            for idx in selected_indices:
                if 1 <= idx <= len(all_folders):
                    folder_info = all_folders[idx - 1]
                    folders_to_process.append(folder_info)
                    total_size_mb += folder_info['size_mb']
                else:
                    logger.warning(f"Invalid index: {idx}")

            if not folders_to_process:
                logger.error("No valid videos selected. Try again.")
                continue

            break

        except ValueError as e:
            logger.error(f"Invalid input: {response}")
            logger.error("Use format: 1,3,5 or 1-5 or 'all' or 'list'")
            continue

    if not folders_to_process:
        logger.info("No videos selected.")
        return

    # Show selection summary
    logger.info("\n" + "=" * 80)
    logger.info("SELECTED VIDEOS FOR TRANSCRIPTION")
    logger.info("=" * 80)
    logger.info(f"\nTotal: {len(folders_to_process)} videos, {total_size_mb:.0f} MB ({total_size_mb/1024:.1f} GB)")
    logger.info(f"Estimated time: {total_size_mb/1024*10:.0f}-{total_size_mb/1024*20:.0f} minutes\n")

    for i, item in enumerate(folders_to_process, 1):
        status = "(has timestamps - will regenerate)" if item['has_word_ts'] else "(missing timestamps)"
        logger.info(f"{i}. {item['name']} {status}")

    # Confirm
    logger.info("\n" + "=" * 80)
    response = input("Proceed with transcription? (y/n): ")
    if response.lower() != 'y':
        logger.info("Cancelled.")
        return

    # Create shorts creator instance
    creator = YouTubeShortsCreator()

    # Process each video
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRANSCRIPTION...")
    logger.info("=" * 80 + "\n")

    success_count = 0
    failed = []

    for i, item in enumerate(folders_to_process, 1):
        logger.info(f"\n[{i}/{len(folders_to_process)}] Processing: {item['name']}")
        logger.info("-" * 80)

        try:
            # Use the existing transcription method
            creator.transcribe_video(str(item['video']), item['folder'])
            success_count += 1
            logger.info(f"✅ Completed: {item['name']}")
        except Exception as e:
            logger.error(f"❌ Failed: {item['name']} - {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed.append(item['name'])

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRANSCRIPTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total processed: {len(folders_to_process)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"❌ Failed: {len(failed)}")

    if failed:
        logger.info("\nFailed videos:")
        for name in failed:
            logger.info(f"  - {name}")

    logger.info("\n✅ Done!")

if __name__ == '__main__':
    main()
