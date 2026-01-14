from pytube import YouTube

def download_with_pytube(url: str, output_dir):
    yt = YouTube(url)

    video_stream = (
        yt.streams
        .filter(adaptive=True, file_extension="mp4", only_video=True)
        .order_by("resolution")
        .desc()
        .first()
    )

    audio_stream = (
        yt.streams
        .filter(adaptive=True, file_extension="mp4", only_audio=True)
        .order_by("abr")
        .desc()
        .first()
    )

    if not video_stream or not audio_stream:
        raise Exception("No suitable streams found")

    video_stream.download(output_path=output_dir, filename="video.mp4")
    audio_stream.download(output_path=output_dir, filename="audio.mp4")