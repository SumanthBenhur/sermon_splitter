# Add test cases for the FFmpegUtils class
import pytest
import ffmpeg
from config import VIDEOS_DIR
from backend.utils.ffmpeg_utils import Ffmeg


def test_merge_and_convert_to_mp4():
    ffmpeg_utils = Ffmeg()
    samples_dir = VIDEOS_DIR / "samples"

    video_path = samples_dir / "VideoWithoutAudio.mp4"
    audio_path = samples_dir / "audio.mp3"

    # Define intermediate and final paths
    mkv_path = samples_dir / "merged.mkv"
    final_mp4_path = samples_dir / "merged.mp4"

    # 1. Verification of Input Files
    if not video_path.exists():
        pytest.fail(f"Test aborted: Missing {video_path}")
    if not audio_path.exists():
        pytest.fail(f"Test aborted: Missing {audio_path}")

    # Cleanup old files
    for path in [mkv_path, final_mp4_path]:
        if path.exists():
            path.unlink()

    # 2. Step 1: Merge to MKV
    print(f"\n🚀 Step 1: Merging into {mkv_path.name}...")
    try:
        ffmpeg_utils.merge_audio_video(video_path, audio_path, mkv_path)
    except Exception as e:
        pytest.fail(f"Merge to MKV failed: {e}")

    assert mkv_path.exists(), "MKV file was not created"

    # 3. Step 2: Convert MKV to MP4
    print(f"🔄 Step 2: Converting {mkv_path.name} to {final_mp4_path.name}...")
    try:
        ffmpeg_utils.convert_to_mp4(mkv_path, final_mp4_path)
    except Exception as e:
        pytest.fail(f"Conversion to MP4 failed: {e}")

    # 4. Final Assertions
    assert final_mp4_path.exists(), "Final MP4 file was not created"
    assert final_mp4_path.suffix == ".mp4"

    # 5. Metadata verification on final file
    probe = ffmpeg.probe(str(final_mp4_path))

    print("\n📊 Final MP4 Stream info:")
    for stream in probe["streams"]:
        print(f"  - {stream.get('codec_type')}: {stream.get('codec_name')}")

    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )

    assert video_stream is not None, "Final MP4 is missing video!"
    assert audio_stream is not None, "Final MP4 is missing audio!"
    assert video_stream["codec_name"] == "h264", "Video should be re-encoded to h264"

    print(f"\n✅ Success! Final file ready at: {final_mp4_path}")
