#!/bin/bash

# Whisper Video Transcription Script with Word Timestamps
# Usage: ./transcribe_video.sh

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Config file for last run
CONFIG_FILE="$HOME/.whisper_transcribe_last.conf"

# Get output directory from settings.ini
OUTPUT_DIR=$(grep -A 10 "\[video\]" /home/obo/playground/videoShorts2/settings.ini | grep "output_dir" | cut -d'=' -f2 | xargs)
BASE_DIR="/home/obo/playground/videoShorts2/$OUTPUT_DIR"

# Initialize default for word timestamps
GENERATE_WORD_TS="true"

# Function to save config
save_config() {
    cat > "$CONFIG_FILE" << EOF
SELECTED_FOLDER="$1"
MODEL="$2"
LANGUAGE="$3"
DURATION="$4"
OUTPUT_BASE="$5"
GENERATE_WORD_TS="$6"
EOF
}

# Function to load config
load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        source "$CONFIG_FILE"
        return 0
    else
        return 1
    fi
}

# Function to show last config
show_last_config() {
    if load_config; then
        echo ""
        echo "============================================"
        echo "  LAST TRANSCRIPTION"
        echo "============================================"
        echo ""
        echo "  Video: $(basename "$SELECTED_FOLDER")"
        echo "  Model: $MODEL"
        echo "  Language: ${LANGUAGE:-auto-detect}"
        echo "  Duration: $([ "$DURATION" -eq 0 ] && echo "full" || echo "${DURATION}s")"
        echo "  Output: $OUTPUT_BASE.srt"
        echo "  Word timestamps: $([ "$GENERATE_WORD_TS" = "true" ] && echo "Yes" || echo "No")"
        echo ""
        return 0
    else
        return 1
    fi
}

# Function to re-run with different settings
rerun_last() {
    if ! load_config; then
        echo -e "${RED}No previous transcription found${NC}"
        return 1
    fi

    echo ""
    echo "============================================"
    echo "  RE-RUN WITH DIFFERENT SETTINGS"
    echo "============================================"
    echo ""
    echo "Current settings:"
    echo "  Model: $MODEL"
    echo "  Language: ${LANGUAGE:-auto-detect}"
    echo ""

    # Ask what to change
    echo "What do you want to change?"
    echo "  [1] Model only"
    echo "  [2] Language only"
    echo "  [3] Both model and language"
    echo ""
    read -n 1 -p "Select [1-3]: " change_choice
    echo ""

    case $change_choice in
        1)
            select_model
            ;;
        2)
            select_language
            ;;
        3)
            select_model
            select_language
            ;;
        *)
            echo "Invalid choice"
            return 1
            ;;
    esac

    # Update output name with new settings
    timestamp=$(date +%Y%m%d_%H%M%S)
    local lang_id="${LANGUAGE:-auto}"
    OUTPUT_BASE="transcribe_${MODEL}_${lang_id}_${timestamp}"

    # Show new settings and confirm
    echo ""
    echo "============================================"
    echo "  NEW SETTINGS"
    echo "============================================"
    echo ""
    echo "  Video: $(basename "$SELECTED_FOLDER")"
    echo "  Model: $MODEL"
    echo "  Language: ${LANGUAGE:-auto-detect}"
    echo "  Duration: $([ "$DURATION" -eq 0 ] && echo "full" || echo "${DURATION}s")"
    echo "  Output: $OUTPUT_BASE.srt"
    echo "  Word timestamps: $([ "$GENERATE_WORD_TS" = "true" ] && echo "Yes" || echo "No")"
    echo ""
    read -p "Proceed? (Y/n): " confirm
    if [[ "$confirm" =~ ^[Nn]$ ]]; then
        echo "Cancelled."
        return 0
    fi

    # Run transcription
    transcribe "$SELECTED_FOLDER" "$MODEL" "$LANGUAGE" "$DURATION" "$OUTPUT_BASE"

    # Save new config
    save_config "$SELECTED_FOLDER" "$MODEL" "$LANGUAGE" "$DURATION" "$OUTPUT_BASE"

    echo ""
    echo "Done!"
    exit 0
}

# Function to display videos
select_video() {
    echo ""
    echo "============================================"
    echo "  SELECT VIDEO TO TRANSCRIBE"
    echo "============================================"
    echo ""

    local folders=()
    local i=1

    for folder in "$BASE_DIR"/*; do
        if [ -d "$folder" ]; then
            folder_name=$(basename "$folder")
            video_files=("$folder"/*.mp4)

            if [ -f "${video_files[0]}" ]; then
                size_mb=$(du -h "${video_files[0]}" | cut -f1)
                echo "  [$i] $folder_name ($size_mb)"
                folders+=("$folder")
                ((i++))
            fi
        fi
    done

    # Determine number of digits needed based on video count
    local num_videos=${#folders[@]}
    local num_digits=${#num_videos}

    echo ""
    read -n $num_digits -p "Enter video number (or 'q' to quit): " choice
    echo ""

    if [[ $choice == 'q' ]]; then
        echo "Cancelled."
        exit 0
    fi

    if ! [[ $choice =~ ^[0-9]+$ ]] || [ $choice -lt 1 ] || [ $choice -gt $num_videos ]; then
        echo -e "${RED}Invalid selection${NC}"
        exit 1
    fi

    SELECTED_FOLDER="${folders[$((choice-1))]}"
}

# Function to select model
select_model() {
    echo ""
    echo "============================================"
    echo "  SELECT WHISPER MODEL"
    echo "============================================"
    echo ""
    echo "  [1] tiny     (fastest, least accurate)"
    echo "  [2] base     (fast, good for quick tests)"
    echo "  [3] small    (recommended, good balance)"
    echo "  [4] medium   (slower, more accurate)"
    echo "  [5] large    (slowest, most accurate)"
    echo ""

    read -n 1 -p "Select model [1-5]: " model_choice
    echo ""

    case $model_choice in
        1) MODEL="tiny" ;;
        2) MODEL="base" ;;
        3) MODEL="small" ;;
        4) MODEL="medium" ;;
        5) MODEL="large" ;;
        *)
            echo -e "${YELLOW}Using default: base${NC}"
            MODEL="base"
            ;;
    esac

    echo "Selected: $MODEL"
}

# Function to select language
select_language() {
    echo ""
    echo "============================================"
    echo "  SELECT LANGUAGE"
    echo "============================================"
    echo ""
    echo "  [1] auto (auto-detect)"
    echo "  [2] en   (English)"
    echo "  [3] ar   (Arabic)"
    echo "  [4] es   (Spanish)"
    echo "  [5] fr   (French)"
    echo "  [6] de   (German)"
    echo "  [7] other (enter manually)"
    echo ""

    read -n 1 -p "Select language [1-7]: " lang_choice
    echo ""

    case $lang_choice in
        1) LANGUAGE="" ;;
        2) LANGUAGE="en" ;;
        3) LANGUAGE="ar" ;;
        4) LANGUAGE="es" ;;
        5) LANGUAGE="fr" ;;
        6) LANGUAGE="de" ;;
        7)
            read -p "Enter language code (e.g., en, ar, es): " LANGUAGE
            ;;
        *)
            echo -e "${YELLOW}Using default: ar${NC}"
            LANGUAGE="ar"
            ;;
    esac

    if [ -z "$LANGUAGE" ]; then
        echo "Selected: auto-detect"
    else
        echo "Selected: $LANGUAGE"
    fi
}

# Function to select duration
select_duration() {
    echo ""
    echo "============================================"
    echo "  SELECT DURATION"
    echo "============================================"
    echo ""
    echo "  [1] 2 minutes (test, default)"
    echo "  [2] 5 minutes"
    echo "  [3] 10 minutes"
    echo "  [4] 30 minutes"
    echo "  [5] Full video"
    echo "  [6] Custom (enter seconds)"
    echo ""

    read -n 1 -p "Select duration [1-6]: " dur_choice
    echo ""

    case $dur_choice in
        1) DURATION=120 ;;
        2) DURATION=300 ;;
        3) DURATION=600 ;;
        4) DURATION=1800 ;;
        5) DURATION=0 ;;
        6)
            read -p "Enter duration in seconds (0 for full): " DURATION
            ;;
        *)
            echo -e "${YELLOW}Using default: 2 minutes${NC}"
            DURATION=120
            ;;
    esac

    if [ "$DURATION" -eq 0 ]; then
        echo "Selected: Full video"
    else
        mins=$((DURATION / 60))
        secs=$((DURATION % 60))
        echo "Selected: ${mins}m ${secs}s"
    fi
}

# Function to select word timestamps option
select_word_timestamps() {
    echo ""
    echo "============================================"
    echo "  WORD TIMESTAMPS"
    echo "============================================"
    echo ""
    echo "Generate word timestamps JSON file?"
    echo "  [1] Yes (default)"
    echo "  [2] No"
    echo ""
    read -n 1 -p "Select [1-2]: " wt_choice
    echo ""

    case $wt_choice in
        2) GENERATE_WORD_TS="false" ;;
        *) GENERATE_WORD_TS="true" ;;
    esac

    if [ "$GENERATE_WORD_TS" = "true" ]; then
        echo "Selected: Yes (will generate word timestamps)"
    else
        echo "Selected: No"
    fi
}

# Function to get output filename
get_output_name() {
    echo ""
    read -p "Output filename (without extension, or press Enter for default): " output_name

    if [ -z "$output_name" ]; then
        timestamp=$(date +%Y%m%d_%H%M%S)
        local lang_id="${LANGUAGE:-auto}"
        OUTPUT_BASE="transcribe_${MODEL}_${lang_id}_${timestamp}"
    else
        OUTPUT_BASE="$output_name"
    fi
}

# Function to transcribe
transcribe() {
    local video_folder="$1"
    local model="$2"
    local language="$3"
    local duration="$4"
    local output_base="$5"

    # Find video file
    video_file=$(ls "$video_folder"/*.mp4 2>/dev/null | head -1)

    if [ -z "$video_file" ]; then
        echo -e "${RED}No video file found in $video_folder${NC}"
        exit 1
    fi

    # Calculate clip end time if duration is specified
    local clip_timestamps=""
    if [ "$duration" -gt 0 ]; then
        # clip_timestamps expects comma-separated timestamps in seconds
        clip_timestamps="--clip_timestamps 0,$duration"
    fi

    echo ""
    echo "============================================"
    echo "  TRANSCRIBING"
    echo "============================================"
    echo ""
    echo "  Video: $(basename "$video_file")"
    echo "  Model: $model"
    echo "  Language: ${language:-auto-detect}"
    echo "  Duration: $([ "$duration" -eq 0 ] && echo "full" || echo "${duration}s")"
    echo ""

    # Build whisper command
    cmd="whisper \"$video_file\" --model $model --output_dir \"$video_folder\" --output_format srt --task transcribe --verbose False"

    # Add language if specified
    if [ -n "$language" ]; then
        cmd="$cmd --language $language"
    fi

    # Add clip timestamps if duration specified
    if [ -n "$clip_timestamps" ]; then
        cmd="$cmd $clip_timestamps"
    fi

    # Show command
    echo -e "${BLUE}Running:${NC}"
    echo "$cmd"
    echo ""

    # Run transcribe
    eval "$cmd"

    # Find the generated SRT file (whisper uses video filename)
    local video_basename=$(basename "$video_file" .mp4)
    local output_file="$video_folder/${video_basename}.srt"

    # Rename if output_base is different from video name
    if [ "$output_base" != "$video_basename" ]; then
        mv "$output_file" "$video_folder/${output_base}.srt"
        output_file="$video_folder/${output_base}.srt"
    fi

    if [ -f "$output_file" ]; then
        echo ""
        echo -e "${GREEN}✅ Transcription complete!${NC}"
        echo "   Output: $output_file"

        # Show some stats
        line_count=$(grep -c "^$" "$output_file" || echo "0")
        segment_count=$((line_count / 4 + 1))
        file_size=$(du -h "$output_file" | cut -f1)

        echo "   Segments: ~$segment_count"
        echo "   File size: $file_size"
        echo ""

        # Generate word timestamps if requested
        if [ "$GENERATE_WORD_TS" = "true" ]; then
            echo -e "${BLUE}Generating word timestamps...${NC}"

            # Create Python script for word timestamps
            local wt_script="/tmp/gen_word_ts_$$.py"

            cat > "$wt_script" << 'EOFPY'
import sys
import whisper
import json
from pathlib import Path

video_path = sys.argv[1]
model_name = sys.argv[2]
language = sys.argv[3] if sys.argv[3] != "None" else None
duration = int(sys.argv[4])
output_path = sys.argv[5]

print(f"Loading Whisper model ({model_name})...")
model = whisper.load_model(model_name)

print("Transcribing with word timestamps...")
kwargs = {
    "word_timestamps": True,
    "verbose": False
}

if language:
    kwargs["language"] = language

result = model.transcribe(str(video_path), **kwargs)

# Filter words by duration if specified
if duration > 0:
    print(f"Filtering to first {duration} seconds...")
    result["words"] = [w for w in result.get("words", []) if w["start"] < duration]

# Save word timestamps
output_data = {
    "language": result.get("language"),
    "duration": duration if duration > 0 else result.get("duration"),
    "words": result.get("words", [])
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(output_data['words'])} words to {output_path}")
EOFPY

            local lang_arg="$language"
            if [ -z "$lang_arg" ]; then
                lang_arg="None"
            fi

            local wt_output="$video_folder/${output_base}_word_timestamps.json"

            python3 "$wt_script" "$video_file" "$model" "$lang_arg" "$duration" "$wt_output"

            rm -f "$wt_script"

            if [ -f "$wt_output" ]; then
                local word_count=$(python3 -c "import json; print(len(json.load(open('$wt_output'))['words']))")
                local wt_size=$(du -h "$wt_output" | cut -f1)
                echo -e "${GREEN}✅ Word timestamps complete!${NC}"
                echo "   Output: $wt_output"
                echo "   Words: $word_count"
                echo "   File size: $wt_size"
                echo ""
            fi
        fi

        # Open in Kate
        if command -v kate &> /dev/null; then
            echo "Opening in Kate..."
            kate "$output_file" &
        else
            echo -e "${YELLOW}Kate not found, skipping open${NC}"
        fi
    else
        echo ""
        echo -e "${RED}❌ Transcription failed${NC}"
        exit 1
    fi
}

# Main flow
clear
echo "============================================"
echo "  WHISPER VIDEO TRANSCRIBER"
echo "============================================"

# Check if whisper is installed
if ! command -v whisper &> /dev/null; then
    echo -e "${RED}Error: whisper command not found${NC}"
    echo "Install with: pip install openai-whisper"
    exit 1
fi

# Check if there's a previous run
if show_last_config; then
    echo ""
    echo "What would you like to do?"
    echo "  [1] New transcription"
    echo "  [2] Re-run last with different model/language"
    echo ""
    read -n 1 -p "Select [1-2]: " menu_choice
    echo ""

    case $menu_choice in
        2)
            rerun_last
            exit 0
            ;;
        *)
            echo ""
            echo "Starting new transcription..."
            ;;
    esac
fi

# Get user choices for new transcription
select_video
select_model
select_language
select_duration
get_output_name
select_word_timestamps

# Confirm
echo ""
echo "============================================"
echo "  CONFIRM SETTINGS"
echo "============================================"
echo ""
echo "  Video: $(basename "$SELECTED_FOLDER")"
echo "  Model: $MODEL"
echo "  Language: ${LANGUAGE:-auto-detect}"
echo "  Duration: $([ "$DURATION" -eq 0 ] && echo "full" || echo "${DURATION}s")"
echo "  Output: $OUTPUT_BASE.srt"
echo "  Word timestamps: $([ "$GENERATE_WORD_TS" = "true" ] && echo "Yes" || echo "No")"
echo ""
read -p "Proceed? (Y/n): " confirm
if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Run transcription
transcribe "$SELECTED_FOLDER" "$MODEL" "$LANGUAGE" "$DURATION" "$OUTPUT_BASE"

# Save config for next time
save_config "$SELECTED_FOLDER" "$MODEL" "$LANGUAGE" "$DURATION" "$OUTPUT_BASE"

echo ""
echo "Done!"
