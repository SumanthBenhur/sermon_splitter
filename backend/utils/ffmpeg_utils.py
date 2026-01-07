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
        cmd = ["ffmpeg", *args]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg command failed with exit code {e.returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDERR:\n{e.stderr}"
            ) from e

    def get_ffmpeg_path(self) -> str:
        """
        Get the resolved path to the ffmpeg executable.

        Returns:
            Absolute path to ffmpeg as a string.
        """
        pass

    def merge_audio_video(
        self, video_path: Path, audio_path: Path, output_path: Path
    ) -> None:
        """
        Combine a video file with an audio file into a single output video.

        Args:
            video_path: Path to the input video file.
            audio_path: Path to the input audio file.
            output_path: Path to the output merged video file.
        """

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path.resolve()}")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path.resolve()}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            "-y",
            str(output_path),
        ]

        self.run_command(args)


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
