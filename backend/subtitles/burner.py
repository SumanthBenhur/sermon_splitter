import shutil
from pathlib import Path

from backend.utils.ffmpeg_manager import FfmpegManager


def burn_subtitles_into_video(
    ffmpeg_manager: FfmpegManager,
    input_mp4: Path,
    ass_path: Path,
    out_path: Path,
) -> None:
    """Burn ASS subtitles into a video."""
    # Copy ASS to CWD for ffmpeg compatibility on Windows paths.
    cwd_ass_path = Path("temp_subtitles.ass")
    shutil.copy(ass_path, cwd_ass_path)

    vf = f"subtitles=filename='{cwd_ass_path.name}'"
    args = [
        "-y",
        "-i",
        str(input_mp4),
        "-vf",
        vf,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    ffmpeg_manager.run_command(args)

    # Cleanup: remove the temporary ASS file from CWD.
    if cwd_ass_path.exists():
        cwd_ass_path.unlink()
