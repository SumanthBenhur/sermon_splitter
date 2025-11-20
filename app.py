import streamlit as st
from sermon_splitter import SermonSplitterApp, VideoUtils, download_video
from pathlib import Path

st.title("Sermon Splitter")

input_source = st.text_input("Enter a YouTube URL or the absolute path to a local video file:")

source_video_path = ""

if input_source:
    # Check if it's a URL
    if input_source.strip().startswith("http"):
        try:
            with st.spinner("Downloading video from YouTube... This may take a moment."):
                # The download function will save it to a specific folder
                source_video_path = download_video(input_source)
            st.success(f"Video downloaded successfully to: {source_video_path}")
        except Exception as e:
            st.error(f"Failed to download or process YouTube video: {e}")
            st.error("Please ensure it's a valid YouTube URL.")
            source_video_path = ""  # Reset on failure
    # Check if it's a local file
    else:
        source_path = Path(input_source)
        if not source_path.is_file():
            st.error("The provided path is not a valid file. Please check the path and try again.")
            st.info("If you are providing a local file, make sure to provide the absolute path.")
            source_video_path = ""  # Reset on failure
        else:
            source_video_path = str(source_path)

if source_video_path:
    st.video(str(source_video_path))

    num_clips = st.number_input("Number of clips", min_value=1, value=1, step=1)

    clips_data = []
    for i in range(num_clips):
        st.subheader(f"Clip {i+1}")
        start_time = st.text_input(f"Start time (HH:MM:SS)", key=f"start_{i}")
        end_time = st.text_input(f"End time (HH:MM:SS)", key=f"end_{i}")
        clips_data.append({"start_time": start_time, "end_time": end_time})

    output_filename = st.text_input("Output file name (e.g., my_clip.mp4)", "clip.mp4")

    if st.button("Process Video"):
        if any(not clip["start_time"] or not clip["end_time"] for clip in clips_data):
            st.warning("Please provide start and end times for all clips.")
        else:
            try:
                with st.spinner("Processing video... This may take a while."):
                    app = SermonSplitterApp(source_path=str(source_video_path))
                    final_output_path = app.run(num_clips, clips_data, output_filename)
                    st.success(f"Video processing complete!")
                    st.info(f"Output file: {final_output_path}")

                    with open(final_output_path, "rb") as f:
                        st.download_button(
                            label="Download Processed Video",
                            data=f,
                            file_name=Path(final_output_path).name,
                            mime="video/mp4"
                        )

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")