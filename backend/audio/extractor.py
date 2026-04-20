from pathlib import Path

from backend.utils.ffmpeg_manager import FfmpegManager


def extract_audio_to_wav(
	ffmpeg_manager: FfmpegManager,
	input_mp4: Path,
	wav_path: Path,
	sample_rate: int = 16000,
) -> None:
	"""Extracts audio from a video file to a mono 16kHz WAV file."""
	args = [
		"-y",
		"-i",
		str(input_mp4),
		"-vn",
		"-acodec",
		"pcm_s16le",
		"-ac",
		"1",
		"-ar",
		str(sample_rate),
		str(wav_path),
	]
	ffmpeg_manager.run_command(args)
