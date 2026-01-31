# YouTube Shorts Creator

Automatically download YouTube videos, generate subtitles, and identify themes for creating YouTube Shorts.

## Features

- **Download videos** in highest quality using yt-dlp
- **Auto-numbered folder structure** (001_, 002_, etc.)
- **Multi-lingual subtitle generation** using Whisper AI
- **Multiple subtitle formats** (SRT, VTT, TXT)
- **Smart theme identification** for 30sec-4min shorts
- **Context-aware titles** based on content analysis

## Installation

### Requirements

- Python 3.9+
- ffmpeg (system package)

### Install ffmpeg

**Fedora/Nobara:**
```bash
sudo dnf install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic usage

```bash
python shorts_creator.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### With specific Whisper model

```bash
python shorts_creator.py "URL" --model small
```

Available models: `tiny`, `base`, `small`, `medium`, `large`
- `tiny` - Fastest, less accurate
- `base` - Good balance (default)
- `small` - Better accuracy
- `medium` - Very accurate
- `large` - Best accuracy, slowest

### Custom output directory

```bash
python shorts_creator.py "URL" --output-dir my_videos
```

## Output Structure

```
videos/
└── 001_Video_Title/
    ├── video_name.mp4              # Downloaded video
    ├── video_name.srt              # SubRip subtitles
    ├── video_name.vtt              # WebVTT subtitles
    ├── video_name_subtitles.txt    # Plain text transcript
    ├── video info.txt              # Video metadata
    └── themes.md                   # Identified shorts themes
```

## themes.md Format

The `themes.md` file contains:

1. **Summary Table** - Quick overview of all identified themes
2. **Detailed Descriptions** - Full context for each theme including:
   - Time range
   - Duration
   - Why it works as a short
   - Transcript preview

## How Theme Selection Works

The algorithm identifies segments that:
- Are between 30 seconds and 4 minutes
- Have engaging hooks (questions, stories, key insights)
- Contain self-contained messages
- Match topic patterns (prayer, grave, hereafter, etc.)

## Example Output

```markdown
## Theme Summary

| # | Theme | Duration | Time Range |
|---|-------|----------|------------|
| 1 | No Silence in Prayer | 0m 32s | 00:17:41 - 00:18:13 |
| 2 | Grave Questioning Explained | 0m 35s | 00:19:01 - 00:19:36 |
```

## Tips for Best Results

1. **Use longer videos** (30+ minutes) for more theme options
2. **Choose the right Whisper model** - `small` is usually best for balance
3. **Review themes.md** to select the best segments for your audience
4. **Edit video clips** using the timestamps provided

## Dependencies

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube video downloading
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech-to-text transcription
- ffmpeg - Audio/video processing

## License

MIT License
