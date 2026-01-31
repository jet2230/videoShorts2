# YouTube Shorts Creator

Automatically download YouTube videos, generate subtitles, and identify themes for creating YouTube Shorts.

## Features

- **Download videos** in highest quality using yt-dlp
- **Process local video files** with automatic folder organization
- **Auto-numbered folder structure** (001_, 002_, etc.)
- **Multi-lingual subtitle generation** using Whisper AI
- **Multiple subtitle formats** (SRT, VTT, TXT)
- **Smart theme identification** for 20sec-4min shorts
- **Pattern-based titles** or **AI-generated titles** (Llama 3)
- **Auto-detects URLs vs local files** - no flags needed

## Installation

### Requirements

- Python 3.9+
- ffmpeg (system package)
- **AI theme generation (optional):** [Ollama](https://ollama.ai/download) with Llama 3

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

### (Optional) Install Ollama for AI theme titles

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# In another terminal, download Llama 3
ollama pull llama3

# Test connection
python test_llama_connection.py
```

## Usage

### Basic usage (pattern-based titles)

```bash
python shorts_creator.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### With AI-generated titles (better, requires Ollama)

```bash
python shorts_creator.py "URL" --ai
```

**AI vs Pattern-based comparison:**

| Aspect | Pattern-based | AI (Llama 3) |
|--------|--------------|---------------|
| Speed | Fast | Slower (~1-2 sec per theme) |
| Quality | Good, generic keywords | Excellent, context-aware |
| Example | "Why Your Dua Isn't Answered" | "When Dua's Go Unanswered?" |

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

### Process local video file

```bash
python shorts_creator.py "/path/to/video.mp4"
```

The script automatically detects if the input is a URL or a local file path.

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
- Are between **20 seconds and 4 minutes**
- Have engaging hooks (questions, stories, key insights)
- Contain self-contained messages
- Match topic patterns (prayer, dua, grave, hereafter, etc.)

**Pattern-based mode:**
- Uses keyword matching and predefined topics
- Fast but generic titles
- Good for quick processing

**AI mode (--ai flag):**
- Uses Llama 3 to analyze transcript context
- Generates catchy, click-worthy titles
- Understands nuance and generates question-style titles
- Auto-corrects common Islamic term typos
- Generates better "why this works" reasons

## Example Output

**Pattern-based titles:**
```markdown
| # | Theme | Duration | Time Range |
|---|-------|----------|------------|
| 1 | Why Your Dua Isn't Answered | 0m 29s | 00:02:17 - 00:02:46 |
| 2 | Self-Reflection: Know Yourself | 0m 25s | 00:07:03 - 00:07:28 |
```

**AI-generated titles:**
```markdown
| # | Theme | Duration | Time Range |
|---|-------|----------|------------|
| 1 | When Dua's Go Unanswered? | 0m 29s | 00:02:17 - 00:02:46 |
| 2 | Faults You Never Admit | 0m 25s | 00:07:03 - 00:07:28 |
```

## Tips for Best Results

1. **Use longer videos** (30+ minutes) for more theme options
2. **Use --ai flag** for the best titles (requires Ollama)
3. **Choose the right Whisper model** - `small` is usually best for balance
4. **Review themes.md** to select the best segments for your audience
5. **Edit video clips** using the timestamps provided

## Dependencies

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube video downloading
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech-to-text transcription
- [Ollama](https://ollama.ai) (optional) - Local LLM for AI theme generation
- ffmpeg - Audio/video processing

## License

MIT License
