from pathlib import Path

from backend.utils.ffmpeg_manager import FfmpegManager


def concatenate_video_clips(ffmpeg_manager: FfmpegManager, inputs, out_file: Path) -> None:
    """Re-encode concat (safe)."""
    fc_in = "".join([f"[{i}:v:0][{i}:a:0]" for i in range(len(inputs))])
    fc = f"{fc_in}concat=n={len(inputs)}:v=1:a=1[v][a]"
    args = ["-y"]
    for p in inputs:
        args += ["-i", str(p)]
    args += [
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-map",
        "[a]",
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
