from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import subprocess


class Ffmeg:
    def __init__(self, ffmpeg_path: Optional[str] = None) -> None:
        """
        Initialize an FFmpeg helper.

        Args:
            ffmpeg_path: Optional path to the ffmpeg executable.
                         If not provided, ffmpeg is expected to be available on PATH.
        """
        pass

    def run_command(self, args: List[str]) -> subprocess.CompletedProcess:
        """
        Execute an ffmpeg command.

        Args:
            args: Command-line arguments to pass to ffmpeg
                  (excluding the ffmpeg executable itself).

        Returns:
            A CompletedProcess containing stdout, stderr, and return code.

        Raises:
            RuntimeError: If the ffmpeg command fails.
        """
        pass

    def get_ffmpeg_path(self) -> str:
        """
        Get the resolved path to the ffmpeg executable.

        Returns:
            Absolute path to ffmpeg as a string.
        """
        pass


def sanitize_mp4_filename(name: str, default: str = "clip.mp4") -> str:
    """
    Clean and normalize a filename for MP4 output.

    - Removes unsafe characters
    - Ensures no directory traversal
    - Guarantees a `.mp4` extension

    Args:
        name: Input filename.
        default: Filename to use if input is empty or invalid.

    Returns:
        A safe MP4 filename.
    """
    pass


def build_concat_filter(inputs: List[Path]) -> str:
    """
    Create an FFmpeg filter string for concatenating videos.

    This produces a filter suitable for safe re-encoded concatenation
    of multiple video and audio streams.

    Args:
        inputs: List of video file paths to concatenate.

    Returns:
        FFmpeg `filter_complex` concat expression.
    """
    pass


def cut_video_clip(
    source_mp4: Path, start_time: str, end_time: str, out_file: Path
) -> None:
    """
    Extract a specific time segment from a video.

    The clip is re-encoded to ensure frame-accurate cutting
    and audio-video synchronization.

    Args:
        source_mp4: Source video file.
        start_time: Start timestamp (HH:MM:SS or seconds).
        end_time: End timestamp.
        out_file: Output path for the clipped video.
    """
    pass


def extract_audio_to_wav(
    input_mp4: Path, wav_path: Path, sample_rate: int = 16000
) -> None:
    """
    Extract audio from a video file and save it as a WAV file.

    Audio is converted to mono and resampled to the specified rate.

    Args:
        input_mp4: Source video file.
        wav_path: Destination WAV file.
        sample_rate: Output audio sample rate.
    """
    pass


def burn_subtitles_into_video(input_mp4: Path, srt_path: Path, out_path: Path) -> None:
    """
    Permanently render subtitles into a video.

    Subtitles become part of the video frames and
    cannot be disabled in the output file.

    Args:
        input_mp4: Input video file.
        srt_path: Subtitle (.srt) file.
        out_path: Output video with burned-in subtitles.
    """
    pass


def create_face_tracked_vertical_video(
    input_mp4: Path,
    output_mp4: Path,
    out_w: int = 1080,
    out_h: int = 1920,
    smooth: float = 0.98,
) -> None:
    """
    Generate a vertical (portrait) video focused on a subject's face.

    The video is cropped and framed to keep the dominant face centered,
    producing a social-media–friendly vertical format.

    Args:
        input_mp4: Source video file.
        output_mp4: Output vertical video file.
        out_w: Output width in pixels.
        out_h: Output height in pixels.
        smooth: Smoothing factor for camera movement.
    """
    pass
