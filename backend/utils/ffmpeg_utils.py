import ffmpeg


def import_video(Path):
    file = ffmpeg.input(Path)
    return file  # ,FMT


def export_video(file, Outpath):
    # Default video_bitrate is ~2400kbps
    Output = ffmpeg.output(file, Outpath)
    ffmpeg.run(Output)
