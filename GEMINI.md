# YouTube Shorts Creator - Developer Guide

## Project Overview
The **YouTube Shorts Creator** is an automated pipeline for transforming long-form videos (from YouTube or local files) into engaging YouTube Shorts. It leverages AI for transcription (Whisper) and theme identification (LLM/Ollama), providing both a CLI interface and a web-based WYSIWYG editor for fine-tuning.

### Key Technologies
- **Python 3.9+**: Core logic and processing scripts.
- **Flask**: Powers the web-based adjustment and export UI.
- **OpenAI Whisper**: High-accuracy speech-to-text transcription.
- **Ollama (Llama 3)**: (Optional) AI-driven theme boundary detection and title generation.
- **FFmpeg**: Handles all video and audio manipulation.
- **Playwright**: End-to-end testing for the web interface.
- **ffmpeg.wasm**: Enables client-side video export in the browser.

### Architecture
- `shorts_creator.py`: The primary CLI entry point for downloading, transcribing, and identifying themes.
- `server.py`: Flask server providing the web UI (`index.html`, `edit.html`, `adjust.html`).
- `transcribe_whisper.py`: Wrapper for Whisper-based transcription.
- `ai_theme_generator.py`: Integration with Ollama for intelligent theme extraction.
- `subtitle_renderer.py` & `subtitle_effects.py`: Logic for styling and rendering subtitles.
- `video_processor.py`: Utilities for video clipping and formatting.

## Building and Running

### Prerequisites
- **System**: `ffmpeg` must be installed and available in the system PATH.
- **Python**: Version 3.9 or higher.
- **AI (Optional)**: [Ollama](https://ollama.ai/) installed with the `llama3` model pulled (`ollama pull llama3`).

### Installation
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Install Node.js dependencies for testing:
   ```bash
   npm install
   ```

### Running the CLI
Process a YouTube video or local file:
```bash
# Basic processing
python shorts_creator.py "https://www.youtube.com/watch?v=VIDEO_ID"

# With AI-generated titles
python shorts_creator.py "URL" --ai

# Create shorts from identified themes (e.g., video folder 001, theme 2)
python shorts_creator.py 001 --theme=2
```

### Running the Web UI
Start the Flask server for visual adjustments:
```bash
python server.py
```
Access the dashboard at `http://localhost:5000`.

### Testing
Execute the Playwright test suite:
```bash
npm test
```

## Development Conventions

### Configuration
- All project settings are managed via `settings.ini`. 
- Use `shorts_creator.load_settings()` to access configuration values through `configparser`.
- Key sections: `[whisper]`, `[video]`, `[subtitle]`, `[theme]`, `[folder]`.

### Subtitles & Localization
- **Default Language**: Arabic (`language = ar`). The project is optimized for proper Arabic script rendering.
- **Styles**: Subtitle styles (font, color, alignment) are defined in `settings.ini` and rendered using ASS format logic in `subtitle_renderer.py`.

### Project Structure
- `videos/`: Main output directory. Each processed video gets a numbered subfolder (e.g., `001_Title/`).
- `media/`: Storage for assets like logos, backgrounds, and B-roll.
- `static/`: Frontend assets (JavaScript, CSS) for the Flask UI.

### Logging
- `server.log`: Captures Flask and background task logs using a `RotatingFileHandler`.
- Logging level is typically set to `INFO`.

### Coding Style
- Follow PEP 8 guidelines for Python code.
- Use type hinting for function signatures where possible.
- Ensure all new features are compatible with the Unified Canvas Rendering system for consistency between browser and server exports.
