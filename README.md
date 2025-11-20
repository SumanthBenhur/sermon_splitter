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

## Usage

### Web Application

The easiest way to use the Sermon Splitter is through the Streamlit web interface.

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser:**
    Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Process your video:**
    - Enter the absolute path to your local video file.
    - Specify the number of clips you want to create.
    - For each clip, provide the start and end times in `HH:MM:SS` format.
    - Enter an output filename for the final video.
    - Click "Process Video" to start.

### Command-Line Interface

For more direct control, you can use the `main.py` script. Note that you will need to modify the script to change the input file and clip timings.

1.  **Edit `main.py`:**
    Open `main.py` and change the `source_video_path`, `clips_data`, and `output_filename` variables as needed.

2.  **Run the script:**
    ```bash
    python main.py
    ```

## Dependencies

This project uses the following major libraries:

- [Streamlit](https://streamlit.io/): For the web-based user interface.
- [OpenCV](https://opencv.org/): For video manipulation and frame processing.
- [MediaPipe](https://mediapipe.dev/): For face detection and tracking.
- [Transformers](https://huggingface.co/docs/transformers/index) (by Hugging Face): For using the Whisper model for transcription.
- [PyTorch](https://pytorch.org/): As a backend for the Whisper model.
- [FFmpeg](https://ffmpeg.org/): For all heavy-lifting video processing tasks (cutting, concatenating, etc.).

## Project Structure

- `app.py`: The main entry point for the Streamlit web application.
- `main.py`: A command-line script for running the video processing pipeline.
- `sermon_splitter.py`: The core module containing all the logic for video processing, transcription, and subtitle generation.
- `videos/`: A directory for storing input videos and generated artifacts.
- `pyproject.toml`: The project's dependency and metadata file.
