from pathlib import Path
from sermon_splitter import SermonSplitterApp

if __name__ == "__main__":
    # The user must have a video file at this path for the script to work.
    # This hardcoded path is based on the original script's context.
    # For a more robust application, this should be a command-line argument.
    source_video_path = "/Users/sumanthbenhur/Desktop/sermon_splitter/videos/test1/ashish_vertical.mp4"
    
    if not Path(source_video_path).exists():
        print(f"Error: Video file not found at '{source_video_path}'")
        print("Please ensure the video file exists at the specified path.")
    else:
        app = SermonSplitterApp(source_path=source_video_path)
        app.run()