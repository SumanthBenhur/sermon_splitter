import contextlib
from datetime import timedelta
import logging
import os
from pathlib import Path
import sys

import srt as srtlib

from backend.audio.extractor import extract_audio_to_wav
from backend.utils.ffmpeg_manager import FfmpegManager


class Transcriber:
	def __init__(self, ffmpeg_manager: FfmpegManager):
		self.ffmpeg_manager = ffmpeg_manager

	def transcribe_video_with_whisper(self, video_path: Path, srt_path: Path) -> None:
		"""Transcribes video using Whisper and saves SRT."""
		try:
			from transformers import pipeline
		except ImportError:
			print("[ERR] `transformers` and `torch` are required for Whisper. Please install them.")
			print("      pip install transformers torch")
			sys.exit(1)

		model_name = "openai/whisper-base"
		print(f"\n[STEP] Transcribing with Whisper ({model_name})...")

		wav_tmp = video_path.with_suffix(".wav")
		print(f"   Extracting audio to '{wav_tmp}'...")
		extract_audio_to_wav(self.ffmpeg_manager, video_path, wav_tmp, 16000)

		# Set FFMPEG_BINARY for soundfile/librosa used by transformers
		os.environ["FFMPEG_BINARY"] = self.ffmpeg_manager.ffmpeg_path
		# Disable HuggingFace progress bars and set verbosity to suppress console output
		os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
		os.environ["TRANSFORMERS_VERBOSITY"] = "error"

		print("   Loading Whisper model...")
		# Suppress verbose output from transformers
		logging.getLogger("transformers").setLevel(logging.ERROR)

		# Redirect stdout/stderr to devnull during model loading and transcription
		with open(os.devnull, "w") as devnull:
			with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
				transcriber = pipeline("automatic-speech-recognition", model=model_name, device=-1)
				transcription_result = transcriber(
					str(wav_tmp),
					chunk_length_s=30,
					return_timestamps=True,
				)

		print("\nTranscription complete.")

		subs = []
		for i, chunk in enumerate(transcription_result["chunks"]):
			start, end = chunk["timestamp"]
			if start is None or end is None:
				continue
			subs.append(
				srtlib.Subtitle(
					index=i + 1,
					start=timedelta(seconds=start),
					end=timedelta(seconds=end),
					content=chunk["text"].strip(),
				)
			)

		srt_content = srtlib.compose(subs)
		srt_path.write_text(srt_content, encoding="utf-8")
		print(f"[OK] SRT saved -> {srt_path}")

		if wav_tmp.exists():
			wav_tmp.unlink()
			print(f"   Cleanup: Removed temporary file '{wav_tmp}'")
