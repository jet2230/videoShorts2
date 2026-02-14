# YouTube Shorts Creator

Automatically download YouTube videos, generate subtitles, and identify themes for creating YouTube Shorts.

## Features

- **Download videos** in highest quality using yt-dlp
- **Process local video files** with automatic folder organization
- **Auto-numbered folder structure** (001_, 002_, etc.)
- **Multi-lingual subtitle generation** using Whisper AI
- **Arabic subtitles in proper Arabic script** (default) - transliteration disabled
- **High-performance Web UI** with 100% WYSIWYG consistency
- **Unified Canvas Rendering** for server-side and browser-side consistency
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

### Creating Shorts from Themes

Once a video has been processed and themes identified, you can create actual short video clips:

```bash
# Create a single short (theme 2 from video 001)
python shorts_creator.py 001 --theme=2

# Create multiple shorts (themes 1, 2, and 5)
python shorts_creator.py 001 --theme=1,2,5

# Create shorts for all themes
python shorts_creator.py 001 --theme=all
```

The created shorts are saved in a `shorts/` subdirectory within the video folder:
```
videos/001_Video_Title/shorts/
├── theme_001_Theme_Title.mp4
├── theme_002_Another_Theme.mp4
└── ...
```

## Output Structure

```
videos/
└── 001_Video_Title/
    ├── video_name.mp4              # Downloaded video
    ├── video_name.srt              # SubRip subtitles
    ├── video_name_subtitles.txt    # Plain text transcript
    ├── video info.txt              # Video metadata
    ├── themes.md                   # Identified shorts themes
    └── shorts/                     # Created short clips
        ├── theme_001_Theme_Title.mp4
        └── theme_001_Theme_Title_adjust.md # Visual adjustments
```

## Browser-based Adjustment & Export

The project includes a powerful web interface for fine-tuning your shorts:

- **WYSIWYG Subtitle Editor:** Drag and drop subtitles directly on the video preview.
- **Unified Rendering:** The same engine powers the browser preview and final export.
- **Style Controls:** Adjust font size, colors, and karaoke modes (normal, cumulative).
- **Client-side Export:** Uses `ffmpeg.wasm` to export adjusted videos directly in your browser.
- **Server-side Consistency:** Bulk creation from the dashboard respects all browser adjustments.

## themes.md Format

The `themes.md` file contains:

1. **Summary Table** - Quick overview of all identified themes
2. **Detailed Descriptions** - Full context for each theme including:
   - Time range
   - Duration
   - Why it works as a short
   - Transcript preview

## Subtitle Settings

**Default Language:** Arabic (`language='ar'`)

The script is configured to generate subtitles with **proper Arabic script** by default. This ensures:
- Arabic words appear as **بسم الله الرحمن الرحيم** instead of "Bismillah" or "al-Rahman"
- Mixed Arabic/English content preserves Arabic script for Arabic portions
- Consistent with Islamic lecture video requirements

**Note:** This setting was chosen because the primary use case is Islamic content with Arabic terms and Quranic verses. If you need subtitles in a different language, modify the `language` parameter in `shorts_creator.py`.

## How Theme Selection Works

The algorithm identifies segments that:
- Are between **20 seconds and 4 minutes**
- Have engaging hooks (questions, stories, key insights)
- Contain self-contained messages
- Match topic patterns (prayer, dua, grave, hereafter, etc.)

### Title Generation Methods

**Pattern-based (default):**
- Uses **keyword matching** against predefined topic patterns
- Matches content against categories like:
  - Islamic concepts (salah, dua, grave, barzakh, jannah, etc.)
  - Emotional indicators (questions, exclamations, stories)
  - Structured patterns ("number one", "second", "the point is")
- Falls back to generic titles if no pattern matches
- **Fast, no external dependencies**

**AI mode (--ai flag):**
- Uses **Llama 3** (via Ollama) to analyze full transcript context
- Generates **context-aware, catchy titles**
- Creates question-style titles that hook viewers
- Understands nuance and subtext
- Auto-corrects common Islamic term typos (Duaats → Dua's)
- Generates better "why this works" reasons
- **Slower, requires Ollama + Llama 3**

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
5. **Create shorts automatically** using `--theme` flag after processing
6. **Shorts are created with fast stream copy** - no re-encoding, preserving quality

## Dependencies

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube video downloading
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech-to-text transcription
- [Ollama](https://ollama.ai) (optional) - Local LLM for AI theme generation
- ffmpeg - Audio/video processing

## License

MIT License
