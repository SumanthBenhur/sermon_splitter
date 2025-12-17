# Sermon Splitter

This project is a Python application designed to process sermon videos. It provides a set of tools to cut, concatenate, reformat, and add subtitles to video files, making it easier to create clips for social media or other platforms.

## Features

- **Clip Extraction**: Cut specific segments from a larger video file.
- **Video Concatenation**: Combine multiple video clips into a single file.
- **Face-Tracked Vertical Video**: Automatically converts a standard horizontal video into a vertical format by tracking the speaker's face to keep them in the frame.
- **Automatic Transcription**: Uses OpenAI's Whisper model to generate subtitles for the video.
- **Subtitle Burn-in**: Burns the generated subtitles directly into the video.
- **Web Interface**: A simple web-based UI built with Streamlit for easy interaction.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SumanthBenhur/sermon_splitter.git
    cd sermon_splitter
    ```

2.  **Install dependencies:**
    This project uses `uv` for package management. If you don't have `uv`, you can install it with `pip`:
    ```bash
    pip install uv
    ```
    Then, install the project dependencies:
    ```bash
    uv sync
    ```

3.  **Install FFmpeg:**
    This project relies on FFmpeg for video processing. You must have FFmpeg installed and available in your system's PATH. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).

## Configuration

To ensure the code runs reliably across different environments, use the `PROJECT_ROOT` and `VIDEOS_DIR` constants from `config.py` instead of hardcoding absolute paths.

```python
from config import PROJECT_ROOT, VIDEOS_DIR

# Example usage
input_video = VIDEOS_DIR / "my_video.mp4"
```

## Development

### Pre-commit Hooks

This project uses `pre-commit` to ensure code quality. Before committing any changes, please install the git hooks:

```bash
pre-commit install
```

This will automatically run checks (like `rules` and `ruff`) whenever you try to commit changes.

## Usage

### Deprecated Functionality

> **Note:** The previous Streamlit app and core script have been moved to the `deprecated/` directory and will eventually be removed.

If you still need to run the old application, you can execute it from the project root:

**Web Application:**
```bash
streamlit run deprecated/app.py
```
