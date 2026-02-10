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

    # Find videos that need word timestamps
    folders_to_process = []

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        # Check for video files
        video_files = list(folder.glob('*.mp4'))
        if not video_files:
            continue

        # Check for word timestamps
        word_ts_files = list(folder.glob('*_word_timestamps.json'))

        if not word_ts_files:
            video_file = video_files[0]
            folders_to_process.append({
                'folder': folder,
                'video': video_file,
                'name': folder.name
            })

    if not folders_to_process:
        logger.info("✅ All videos already have word timestamps!")
        return

    logger.info(f"Found {len(folders_to_process)} videos needing word timestamps:\n")
    total_size_mb = 0
    for i, item in enumerate(folders_to_process, 1):
        size_mb = item['video'].stat().st_size / (1024*1024)
        total_size_mb += size_mb
        logger.info(f"{i}. {item['name']}")
        logger.info(f"   Video: {item['video'].name}")
        logger.info(f"   Size: {size_mb:.0f} MB")

    logger.info(f"\nTotal size: {total_size_mb:.0f} MB ({total_size_mb/1024:.1f} GB)")
    logger.info(f"\nEstimated time: {total_size_mb/1024*10:.0f}-{total_size_mb/1024*20:.0f} minutes")

    # Confirm
    logger.info("\n" + "=" * 60)
    response = input("Proceed with transcription? (y/n): ")
    if response.lower() != 'y':
        logger.info("Cancelled.")
        return

    # Create shorts creator instance
    creator = YouTubeShortsCreator()

    # Process each video
    logger.info("\n" + "=" * 60)
    logger.info("Starting transcription...")
    logger.info("=" * 60 + "\n")

    success_count = 0
    failed = []

    for i, item in enumerate(folders_to_process, 1):
        logger.info(f"\n[{i}/{len(folders_to_process)}] Processing: {item['name']}")
        logger.info("-" * 60)

        try:
            # Use the existing transcription method
            creator.transcribe_video(str(item['video']), item['folder'])
            success_count += 1
            logger.info(f"✅ Completed: {item['name']}")
        except Exception as e:
            logger.error(f"❌ Failed: {item['name']} - {e}")
            failed.append(item['name'])

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRANSCRIPTION SUMMARY")
    logger.info("=" * 60)
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
