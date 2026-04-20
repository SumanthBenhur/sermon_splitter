from pathlib import Path

from backend.utils.ffmpeg_manager import FfmpegManager


def cut_video_clip(
    ffmpeg_manager: FfmpegManager,
    source_mp4: Path,
    start_time: str,
    end_time: str,
    out_file: Path,
) -> None:
    """Accurate re-encode cut with audio."""
    args = [
        "-y",
        "-i",
        str(source_mp4),
        "-ss",
        start_time,
        "-to",
        end_time,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(out_file),
    ]
    ffmpeg_manager.run_command(args)
