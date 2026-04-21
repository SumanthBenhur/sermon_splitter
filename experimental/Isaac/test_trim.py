import ffmpeg


def trim_file(Video, Starttime, Endtime, Audio=None):
    """
    This Function trims the Video and return Audio and Video (as a tuple)

    :param Video: The Main File or Video Component of the file
    :param Starttime: The Trimming Start Time
    :param Endtime: The Trimming Stop Time
    :param Audio: (Optional) Audio Component of the file
    """
    if Audio:
        Trimmed_video = Video.filter("trim", start=Starttime, end=Endtime)
        Trimmed_audio = Audio.filter("atrim", start=Starttime, end=Endtime)
    else:
        Trimmed_video = Video.video.filter("trim", start=Starttime, end=Endtime)
        Trimmed_audio = Video.audio.filter("atrim", start=Starttime, end=Endtime)
    return Trimmed_audio, Trimmed_video


test_file = ffmpeg.input("experimental/Isaac/test_video.mp4")
ffmpeg.output(
    *trim_file(test_file.video, "00:00:00", "00:00:10", Audio=test_file.audio),
    "experimental/Isaac/test_export1.mp4",
).run()
ffmpeg.output(
    *trim_file(test_file, "00:00:10", "00:00:11"),
    "experimental/Isaac/test_export2.mp4",
).run()
