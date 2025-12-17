import os
from pathlib import Path
from pytubefix import YouTube

def download_youtube_video(url: str, output_directory: str):
    """
    Downloads a YouTube video using the yt-dlp Python API.

    Args:
        url: The URL of the video (e.g., 'https://www.youtube.com/watch?v=dQw4w9WgXcQ').
        output_directory: The local folder where the video will be saved.
    """
    # 1. Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    video_download = YouTube(url).streams.get_highest_resolution()
    audio_download = YouTube(url).streams.get_audio_only()

    entry = YouTube(url).title
    print(f"\nVideo found: {entry}\n")

    print(f"Downloading Video...")
    video_download.download(filename=f"{entry}.mp4")

    print("Downloading Audio...")
    audio_download.download(filename=f"{entry}.mp3")

# Replace with the actual YouTube video URL you want to download
VIDEO_URL = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' 
# Replace with your desired save location
SAVE_FOLDER = './yt_videos/test.mp4'

download_youtube_video(VIDEO_URL, SAVE_FOLDER)