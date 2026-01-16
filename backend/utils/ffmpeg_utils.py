import ffmpeg
from pathlib import Path
from typing import List, Optional


class Ffmeg:
    def __init__(self, ffmpeg_path: Optional[str] = None) -> None:
        self.ffmpeg_path = ffmpeg_path

    def get_ffmpeg_path(self) -> str:
        """Returns the path to the ffmpeg binary."""
        return self.ffmpeg_path or "ffmpeg"

    def merge_audio_video(
        self, video_path: Path, audio_path: Path, output_path: Path
    ) -> None:
        if not video_path.exists() or not audio_path.exists():
            raise FileNotFoundError("Inputs missing.")

        output_path = output_path.with_suffix(".mkv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            v = ffmpeg.input(str(video_path))
            a = ffmpeg.input(str(audio_path))

            (
                ffmpeg.output(
                    v["v"],
                    a["a"],
                    str(output_path),
                    vcodec="copy",
                    acodec="aac",
                    audio_bitrate="192k",
                    shortest=None,
                )
                .overwrite_output()
                .run(
                    cmd=self.get_ffmpeg_path(), capture_stdout=True, capture_stderr=True
                )
            )
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg failed:\n{error_msg}") from e

    # Add this back if you need it for other parts of your code
    def build_concat_filter(self, inputs: List[Path]) -> str:
        # implementation...
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
