
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
        # Example of how to call the refactored run method
        num_clips = 1
        clips_data = [{"start_time": "00:00:10", "end_time": "00:00:30"}]
        output_filename = "my_awesome_clip.mp4"
        app.run(num_clips, clips_data, output_filename)