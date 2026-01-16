# Add test cases for the FFmpegUtils class
import pytest
import ffmpeg
from config import VIDEOS_DIR
from backend.utils.ffmpeg_utils import Ffmeg


def test_merge_audio_video():
    ffmpeg_utils = Ffmeg()
    samples_dir = VIDEOS_DIR / "samples"

    video_path = samples_dir / "VideoWithoutAudio.mp4"
    audio_path = samples_dir / "audio.mp3"

    # 1. Verification of Input Files
    if not video_path.exists():
        pytest.fail(f"Test aborted: Missing {video_path}")
    if not audio_path.exists():
        pytest.fail(f"Test aborted: Missing {audio_path}")

    # 2. MATCH THE EXTENSION: Ensure this matches what your Ffmeg class produces
    # If you switched to MKV in the backend, change this to "merged.mkv"
    output_path = samples_dir / "merged.mkv"

    if output_path.exists():
        output_path.unlink()

    # 3. Run Merge with enhanced error catching
    print(f"\n🚀 Merging {video_path.name} and {audio_path.name}...")
    try:
        ffmpeg_utils.merge_audio_video(video_path, audio_path, output_path)
    except Exception as e:
        pytest.fail(f"FFmpeg execution failed: {e}")

    # 4. Assertions
    assert output_path.exists(), (
        f"Output file {output_path} was not created. Check FFmpeg logs."
    )
    assert output_path.stat().st_size > 0, "Output file exists but is 0 bytes."

    # 5. Metadata verification
    try:
        probe = ffmpeg.probe(str(output_path))
    except ffmpeg.Error as e:
        pytest.fail(f"Could not probe output file: {e.stderr.decode()}")

    print(f"\n📊 Stream info for {output_path.name}:")
    for stream in probe["streams"]:
        codec_type = stream.get("codec_type")
        codec_name = stream.get("codec_name")
        print(f"  - {codec_type}: {codec_name}")

    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )

    assert video_stream is not None, "Merged file is missing a video track!"
    assert audio_stream is not None, "Merged file is missing an audio track!"

    print(
        f"\n✅ Success! Video: {video_stream['codec_name']}, Audio: {audio_stream['codec_name']}"
    )
