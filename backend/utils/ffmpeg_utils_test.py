import pytest
from unittest.mock import Mock
from ffmpeg_utils import Ffmpeg
import ffmpeg
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# Add test cases for the FFmpegUtils class


def test_trim_file_with_explicit_audio():
    # The Setup
    mock_video_component = Mock(name="video_component")
    mock_audio_component = Mock(name="audio_component")
    # The Assertions
    output_video = Mock(name="trimmed_video")
    output_audio = Mock(name="trimmed_audio")
    mock_audio_component.filter.return_value = output_audio
    mock_video_component.filter.return_value = output_video
    starttime, endtime = "00:00:00", "00:00:10"
    # Call the Function and test
    result_audio, result_video = Ffmpeg.trim_file(
        starttime=starttime,
        endtime=endtime,
        video=mock_video_component,
        audio=mock_audio_component,
    )
    # Making Sure the Function were Called Properly
    mock_audio_component.filter.assert_called_once_with(
        "atrim", start=starttime, end=endtime
    )
    mock_video_component.filter.assert_called_once_with(
        "trim", start=starttime, end=endtime
    )
    # Checking the Output
    assert result_audio == output_audio
    assert result_video == output_video


def test_trim_file_without_explicit_audio():
    """
    Test the scenario where only the main Video file is provided,
    and audio must be extracted from it.
    """
    # 1. Setup Mocks
    mock_main_file = Mock()
    # The function accesses mock_main_file.video and mock_main_file.audio
    mock_video_component = Mock()
    mock_audio_component = Mock()
    mock_main_file.video = mock_video_component
    mock_main_file.audio = mock_audio_component
    # Setup expected returns for the filter calls
    expected_video_out = Mock(name="trimmed_video_stream")
    expected_audio_out = Mock(name="trimmed_audio_stream")
    mock_video_component.filter.return_value = expected_video_out
    mock_audio_component.filter.return_value = expected_audio_out
    start_time = 5
    end_time = 15
    # 2. Call the function (No Audio arg provided)
    result_audio, result_video = Ffmpeg.trim_file(
        video=mock_main_file, starttime=start_time, endtime=end_time
    )
    # 3. Assertions
    # Ensure filter was called on the .video component
    mock_video_component.filter.assert_called_once_with(
        "trim", start=start_time, end=end_time
    )
    # Ensure filter was called on the .audio component
    mock_audio_component.filter.assert_called_once_with(
        "atrim", start=start_time, end=end_time
    )
    # Validate return values
    assert result_video == expected_video_out
    assert result_audio == expected_audio_out


def test_trim_file_actual():
    samples_dir = config.VIDEOS_DIR / "samples"

    video_path = samples_dir / "video_without_audio.mp4"
    audio_path = samples_dir / "audio.mp3"
    output_path = samples_dir / "trimmed"
    trimmed_audio, trimmed_video = Ffmpeg.trim_file(
        video=str(video_path),
        audio=str(audio_path),
        starttime="00:00:00",
        endtime="00:00:10",
    )

    ffmpeg.output(
        trimmed_audio, trimmed_video, str(output_path.with_suffix(".mp4"))
    ).run()
    assert output_path.with_suffix(".mp4").exists()


def test_merge_and_convert_to_mp4():
    ffmpeg_utils = Ffmpeg()
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
