# Add test cases for the FFmpegUtils class
import pytest
import ffmpeg
import config
from backend.utils.ffmpeg_utils import Ffmeg


def test_merge_and_convert_to_mp4():
    ffmpeg_utils = Ffmeg()
    samples_dir = config.VIDEOS_DIR / "samples"

    video_path = samples_dir / "video_without_audio.mp4"
    audio_path = samples_dir / "audio.mp3"

    mkv_path = samples_dir / "merged.mkv"
    final_mp4_path = samples_dir / "merged.mp4"
    if not video_path.exists():
        pytest.fail(f"Test aborted: Missing {video_path}")
    if not audio_path.exists():
        pytest.fail(f"Test aborted: Missing {audio_path}")
    for path in [mkv_path, final_mp4_path]:
        if path.exists():
            path.unlink()
    try:
        ffmpeg_utils.merge_audio_video(video_path, audio_path, mkv_path)
    except Exception as e:
        pytest.fail(f"Merge to MKV failed: {e}")

    assert mkv_path.exists(), "MKV file was not created"
    try:
        ffmpeg_utils.convert_to_mp4(mkv_path, final_mp4_path)
    except Exception as e:
        pytest.fail(f"Conversion to MP4 failed: {e}")
    assert final_mp4_path.exists(), "Final MP4 file was not created"
    assert final_mp4_path.suffix == ".mp4"
    probe = ffmpeg.probe(str(final_mp4_path))
    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )

    assert video_stream is not None, "Final MP4 is missing video!"
    assert audio_stream is not None, "Final MP4 is missing audio!"
    assert video_stream["codec_name"] == "h264", "Video should be re-encoded to h264"
