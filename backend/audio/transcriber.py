from dataclasses import asdict, dataclass
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any
from uuid import uuid4
import wave

import srt as srtlib

from backend.audio.extractor import extract_audio_to_wav
from backend.audio.script_detection import get_script_stats, is_mostly_roman_text
from backend.utils.ffmpeg_manager import FfmpegManager


logger = logging.getLogger(__name__)


DEFAULT_HINGLISH_MODEL = "Oriserve/Whisper-Hindi2Hinglish-Swift"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_CACHE_DIR = PROJECT_ROOT / "ai_cache"
HF_CACHE_DIR = AI_CACHE_DIR / "huggingface"
TORCH_CACHE_DIR = AI_CACHE_DIR / "torch"
MODEL_CACHE_DIR = AI_CACHE_DIR / "models"
PROJECT_TEMP_DIR = PROJECT_ROOT / "temp"
MIN_MODEL_CACHE_FREE_BYTES = 2 * 1024 * 1024 * 1024

_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}

PUNCTUATION_BREAK_RE = re.compile(r"[.!?,;:\u0964]$")


@dataclass(frozen=True)
class SubtitleTimingConfig:
    max_words_per_chunk: int = 4
    min_words_per_chunk: int = 2
    pause_threshold: float = 0.45
    max_chunk_duration: float = 1.8
    minimum_subtitle_duration: float = 0.35
    minimum_gap_between_subtitles: float = 0.0

    @property
    def max_duration_per_subtitle(self) -> float:
        return self.max_chunk_duration

    @property
    def min_duration_per_subtitle(self) -> float:
        return self.minimum_subtitle_duration


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class SubtitleChunk:
    text: str
    start: float
    end: float
    words: list[WordTimestamp]


def ensure_project_storage_dirs() -> dict[str, Path]:
    """Create and return all project-local AI/cache/temp directories."""
    dirs = {
        "ai_cache": AI_CACHE_DIR,
        "huggingface": HF_CACHE_DIR,
        "torch": TORCH_CACHE_DIR,
        "models": MODEL_CACHE_DIR,
        "temp": PROJECT_TEMP_DIR,
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def configure_project_cache_environment() -> dict[str, Path]:
    """Force model/runtime caches into the Sermon Splitter project on D drive."""
    dirs = ensure_project_storage_dirs()

    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
    os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
    os.environ["XDG_CACHE_HOME"] = str(AI_CACHE_DIR)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("AI cache directory: %s", AI_CACHE_DIR)
    logger.info("HuggingFace home: %s", HF_CACHE_DIR)
    logger.info("Transformers/model cache: %s", MODEL_CACHE_DIR)
    logger.info("Torch cache: %s", TORCH_CACHE_DIR)
    logger.info("Project temp directory: %s", PROJECT_TEMP_DIR)
    return dirs


class Transcriber:
    """Generate Roman Hinglish SRT subtitles with a HuggingFace Whisper ASR model."""

    def __init__(
        self,
        ffmpeg_manager: FfmpegManager,
        model_name: str = DEFAULT_HINGLISH_MODEL,
        chunk_length_s: int = 20,
        batch_size: int = 1,
        timing_config: SubtitleTimingConfig | None = None,
    ):
        self.ffmpeg_manager = ffmpeg_manager
        self.model_name = model_name
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        self.timing_config = timing_config or SubtitleTimingConfig()
        self.cache_dirs = self._configure_runtime()
        self._pipeline = None

    def transcribe_video_with_whisper(self, video_path: Path, srt_path: Path) -> None:
        """Backward-compatible entry point used by the sermon pipeline."""
        self.transcribe_video_to_srt(video_path, srt_path)

    def transcribe_video_to_srt(self, video_path: Path, srt_path: Path) -> None:
        """Transcribe a video directly to Roman Hinglish SRT while preserving timestamps."""
        self._configure_runtime()
        os.environ["FFMPEG_BINARY"] = self.ffmpeg_manager.ffmpeg_path
        try:
            self._validate_ffmpeg()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "FFmpeg was not found. Install FFmpeg or configure FfmpegManager with a valid ffmpeg binary."
            ) from exc

        logger.info("[STEP] Transcribing directly to Roman Hinglish: %s", self.model_name)
        logger.info("Source video: %s", video_path)
        logger.info("Destination SRT: %s", srt_path)
        print(f"\n[STEP] Transcribing directly to Roman Hinglish ({self.model_name})...")

        wav_tmp = self._temporary_wav_path(video_path)
        total_started = time.perf_counter()
        try:
            extraction_started = time.perf_counter()
            logger.info("Extracting audio to 16 kHz WAV: %s", wav_tmp)
            extract_audio_to_wav(self.ffmpeg_manager, video_path, wav_tmp, 16000)
            audio_duration = self._get_wav_duration(wav_tmp)
            logger.info("Extracted audio duration: %.3fs", audio_duration)
            logger.info("Audio extraction completed in %.2fs", time.perf_counter() - extraction_started)

            transcriber = self._load_pipeline()
            transcription_started = time.perf_counter()
            transcription_result = self._run_transcription(transcriber, wav_tmp)
            logger.info("Transcription completed in %.2fs", time.perf_counter() - transcription_started)

            subtitle_started = time.perf_counter()
            subtitles = self._result_to_subtitles(transcription_result, srt_path, audio_duration)

            if not subtitles:
                raise RuntimeError("Transcription completed, but no timestamped subtitle chunks were produced.")

            srt_path.parent.mkdir(parents=True, exist_ok=True)
            srt_path.write_text(srtlib.compose(subtitles), encoding="utf-8")
            logger.info(
                "Generated %s subtitle entries in %.2fs",
                len(subtitles),
                time.perf_counter() - subtitle_started,
            )

            combined_text = "\n".join(sub.content for sub in subtitles)
            stats = get_script_stats(combined_text)
            logger.info(
                "Transcription script stats: roman=%s devanagari=%s roman_ratio=%.2f",
                stats.roman_chars,
                stats.devanagari_chars,
                stats.roman_ratio,
            )
            if is_mostly_roman_text(combined_text):
                logger.info("Transcription output is mostly Roman script; transliteration should be skipped.")
            elif stats.devanagari_chars:
                logger.warning("Transcription output contains Devanagari; fallback transliteration may be needed.")

            print(f"[OK] Hinglish SRT saved -> {srt_path}")
        except OSError as exc:
            raise RuntimeError(f"Transcription failed while accessing project-local files: {exc}") from exc
        finally:
            if wav_tmp.exists():
                wav_tmp.unlink()
                logger.info("Removed temporary audio file: %s", wav_tmp)
            logger.info("Subtitle transcription step finished in %.2fs", time.perf_counter() - total_started)

    @staticmethod
    def _configure_runtime() -> dict[str, Path]:
        dirs = configure_project_cache_environment()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        return dirs

    def _load_pipeline(self) -> Any:
        self._configure_runtime()
        self._ensure_model_cache_has_space()

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError:
            print("[ERR] `transformers` and `torch` are required for HuggingFace Whisper transcription.")
            print("      pip install transformers torch accelerate")
            sys.exit(1)

        device, pipeline_device, torch_dtype = self._detect_device(torch)
        cache_key = (self.model_name, device)
        if cache_key in _PIPELINE_CACHE:
            logger.info("Reusing cached ASR pipeline for %s on %s", self.model_name, device)
            self._pipeline = _PIPELINE_CACHE[cache_key]
            return self._pipeline

        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading ASR model: %s", self.model_name)
        logger.info("Inference device: %s, dtype: %s", device, torch_dtype)
        logger.info("Model files will be read/downloaded from: %s", MODEL_CACHE_DIR)
        print(f"   Loading ASR model on {device}...")

        started = time.perf_counter()
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                cache_dir=str(MODEL_CACHE_DIR),
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(device)
            model.eval()

            processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(MODEL_CACHE_DIR),
            )

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=pipeline_device,
                generate_kwargs=self._generate_kwargs(),
            )
        except ImportError as exc:
            raise RuntimeError(
                "Could not load the ASR model with low_cpu_mem_usage=True. "
                "Install the `accelerate` package in this environment."
            ) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Could not load or download the HuggingFace ASR model. "
                f"Check your connection and available space under {MODEL_CACHE_DIR}. "
                "The project-local cache prevents fallback to the default user cache."
            ) from exc

        _PIPELINE_CACHE[cache_key] = self._pipeline
        logger.info("ASR model loaded in %.2fs", time.perf_counter() - started)
        return self._pipeline

    @staticmethod
    def _detect_device(torch_module: Any) -> tuple[str, int, Any]:
        if torch_module.cuda.is_available():
            gpu_name = torch_module.cuda.get_device_name(0)
            logger.info("CUDA available; using GPU: %s", gpu_name)
            return "cuda", 0, torch_module.float16

        logger.info("CUDA unavailable or not installed; using CPU transcription with float32.")
        return "cpu", -1, torch_module.float32

    @staticmethod
    def _ensure_model_cache_has_space() -> None:
        usage = shutil.disk_usage(MODEL_CACHE_DIR)
        logger.info("Free space available for model cache: %.2f GB", usage.free / (1024**3))
        if usage.free < MIN_MODEL_CACHE_FREE_BYTES:
            raise RuntimeError(
                "Low disk space for model cache. "
                f"Free at least {MIN_MODEL_CACHE_FREE_BYTES / (1024**3):.1f} GB under {AI_CACHE_DIR} "
                "before loading the HuggingFace model."
            )

    def _validate_ffmpeg(self) -> None:
        ffmpeg_path = self.ffmpeg_manager.ffmpeg_path
        if Path(ffmpeg_path).exists() or shutil.which(ffmpeg_path):
            logger.info("Using FFmpeg binary: %s", ffmpeg_path)
            return
        raise FileNotFoundError(ffmpeg_path)

    @staticmethod
    def _temporary_wav_path(video_path: Path) -> Path:
        PROJECT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in video_path.stem)
        return PROJECT_TEMP_DIR / f"{safe_stem}_{os.getpid()}_{uuid4().hex[:8]}.wav"

    @staticmethod
    def _get_wav_duration(wav_path: Path) -> float:
        with wave.open(str(wav_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)

    @staticmethod
    def _generate_kwargs() -> dict[str, str]:
        # This model is trained to emit Hindi/Hinglish speech in Latin script.
        # The English language token is the convention used by the Oriserve examples
        # to keep the decoder in Roman-script Hinglish mode.
        return {
            "task": "transcribe",
            "language": "en",
        }

    def _run_transcription(self, transcriber: Any, wav_path: Path) -> dict[str, Any]:
        logger.info(
            "Running ASR with chunk_length_s=%s, batch_size=%s, return_timestamps=word",
            self.chunk_length_s,
            self.batch_size,
        )

        attempts = [
            {"return_timestamps": "word", "batch_size": self.batch_size},
            {"return_timestamps": "word"},
            {"return_timestamps": True},
        ]
        last_error: Exception | None = None
        for attempt in attempts:
            call_kwargs = {
                "chunk_length_s": self.chunk_length_s,
                "return_timestamps": attempt["return_timestamps"],
                "generate_kwargs": self._generate_kwargs(),
            }
            if attempt.get("batch_size"):
                call_kwargs["batch_size"] = attempt["batch_size"]

            try:
                return transcriber(str(wav_path), **call_kwargs)
            except (TypeError, ValueError) as exc:
                last_error = exc
                logger.warning("ASR attempt failed with %s. Retrying with safer timestamp settings.", exc)

        raise RuntimeError("ASR transcription failed for word and segment timestamp modes.") from last_error

    def _result_to_subtitles(
        self,
        transcription_result: dict[str, Any],
        srt_path: Path,
        audio_duration: float,
    ) -> list[srtlib.Subtitle]:
        self._write_raw_transcription_debug_output(srt_path, transcription_result)

        words = self._extract_word_timestamps(transcription_result)
        if words:
            logger.info("Using %s word-level timestamps for subtitle grouping.", len(words))
        else:
            logger.warning("No word-level timestamps found; estimating word timings from ASR segment timestamps.")
            words = self._estimate_words_from_segments(transcription_result)

        if not words:
            full_text = (transcription_result.get("text") or "").strip()
            if full_text:
                logger.warning("No segment timestamps found; estimating word timings across full audio duration.")
                words = self._estimate_words_from_text(full_text, start=0.0, end=audio_duration or 10.0)

        if words:
            chunks = self._group_words_for_reels(words)
            chunks = self._balance_short_subtitle_chunks(chunks)
            chunks = self._smooth_subtitle_chunks(chunks)
            self._write_timing_debug_outputs(srt_path, words, chunks)
            return self._chunks_to_subtitles(chunks)

        logger.warning("No transcription words could be derived; no subtitles generated.")
        self._write_timing_debug_outputs(srt_path, [], [])
        return []

    @staticmethod
    def _extract_word_timestamps(transcription_result: dict[str, Any]) -> list[WordTimestamp]:
        words: list[WordTimestamp] = []
        for chunk in transcription_result.get("chunks") or []:
            text = (chunk.get("word") or chunk.get("text") or "").strip()
            timestamp = chunk.get("timestamp")
            start = chunk.get("start")
            end = chunk.get("end")

            if timestamp and len(timestamp) == 2:
                start, end = timestamp

            if not text or start is None or end is None:
                continue

            start_f = float(start)
            end_f = float(end)
            if end_f <= start_f:
                continue

            split_words = text.split()
            if len(split_words) != 1:
                continue

            words.append(WordTimestamp(word=split_words[0], start=start_f, end=end_f))

        return words

    @staticmethod
    def _estimate_words_from_segments(transcription_result: dict[str, Any]) -> list[WordTimestamp]:
        estimated: list[WordTimestamp] = []
        for chunk in transcription_result.get("chunks") or []:
            text = (chunk.get("text") or "").strip()
            timestamp = chunk.get("timestamp") or ()
            if not text or len(timestamp) != 2:
                continue

            start, end = timestamp
            if start is None or end is None or end <= start:
                continue
            segment_words = text.split()
            if not segment_words:
                continue

            estimated.extend(
                Transcriber._estimate_words_from_text(
                    text,
                    start=float(start),
                    end=float(end),
                )
            )

        return estimated

    @staticmethod
    def _estimate_words_from_text(text: str, start: float, end: float) -> list[WordTimestamp]:
        words = text.split()
        if not words:
            return []

        safe_start = max(0.0, float(start))
        safe_end = max(safe_start + 0.05, float(end))
        duration = safe_end - safe_start
        total_weight = sum(Transcriber._word_timing_weight(word) for word in words) or 1.0
        seconds_per_weight = duration / total_weight

        estimated: list[WordTimestamp] = []
        cursor = safe_start
        for word in words:
            word_duration = max(0.08, Transcriber._word_timing_weight(word) * seconds_per_weight)
            word_end = min(safe_end, cursor + word_duration)
            if word_end <= cursor:
                word_end = min(safe_end, cursor + 0.05)
            estimated.append(WordTimestamp(word=word, start=cursor, end=word_end))
            cursor = word_end

        return estimated

    @staticmethod
    def _word_timing_weight(word: str) -> float:
        return max(1.0, min(2.5, len(word.strip()) / 4.0))

    def _group_words_for_reels(self, words: list[WordTimestamp]) -> list[SubtitleChunk]:
        cfg = self.timing_config
        chunks: list[SubtitleChunk] = []
        current: list[WordTimestamp] = []

        for word in words:
            if not current:
                current = [word]
                continue

            previous = current[-1]
            gap = max(0.0, word.start - previous.end)
            would_duration = word.end - current[0].start
            previous_ends_sentence = bool(PUNCTUATION_BREAK_RE.search(previous.word))
            should_split = (
                gap >= cfg.pause_threshold
                or len(current) >= cfg.max_words_per_chunk
                or would_duration > cfg.max_duration_per_subtitle
                or (previous_ends_sentence and len(current) >= 2)
            )

            if should_split:
                chunks.append(self._make_subtitle_chunk(current))
                current = [word]
            else:
                current.append(word)

        if current:
            chunks.append(self._make_subtitle_chunk(current))

        logger.info(
            "Grouped %s words into %s reel-style subtitle chunks "
            "(max_words=%s, max_duration=%.2fs, pause_threshold=%.2fs, min_gap=%.2fs).",
            len(words),
            len(chunks),
            cfg.max_words_per_chunk,
            cfg.max_duration_per_subtitle,
            cfg.pause_threshold,
            cfg.minimum_gap_between_subtitles,
        )
        return chunks

    def _balance_short_subtitle_chunks(self, chunks: list[SubtitleChunk]) -> list[SubtitleChunk]:
        cfg = self.timing_config
        balanced: list[SubtitleChunk] = []

        for chunk in chunks:
            word_count = len(chunk.words)
            if (
                word_count < cfg.min_words_per_chunk
                and balanced
                and len(balanced[-1].words) + word_count <= cfg.max_words_per_chunk
                and chunk.end - balanced[-1].start <= cfg.max_chunk_duration
            ):
                previous_words = [*balanced[-1].words, *chunk.words]
                balanced[-1] = self._make_subtitle_chunk(previous_words)
            else:
                balanced.append(chunk)

        rebalance_index = 0
        while rebalance_index < len(balanced) - 1:
            chunk = balanced[rebalance_index]
            next_chunk = balanced[rebalance_index + 1]
            if (
                len(chunk.words) < cfg.min_words_per_chunk
                and len(chunk.words) + len(next_chunk.words) <= cfg.max_words_per_chunk
                and next_chunk.end - chunk.start <= cfg.max_chunk_duration
            ):
                balanced[rebalance_index] = self._make_subtitle_chunk([*chunk.words, *next_chunk.words])
                del balanced[rebalance_index + 1]
                continue
            rebalance_index += 1

        return balanced

    @staticmethod
    def _make_subtitle_chunk(words: list[WordTimestamp]) -> SubtitleChunk:
        text = " ".join(word.word.strip() for word in words if word.word.strip())
        return SubtitleChunk(
            text=text,
            start=max(0.0, words[0].start),
            end=max(words[-1].end, words[0].start),
            words=list(words),
        )

    def _smooth_subtitle_chunks(self, chunks: list[SubtitleChunk]) -> list[SubtitleChunk]:
        if not chunks:
            return []

        cfg = self.timing_config
        smoothed: list[SubtitleChunk] = []
        for index, chunk in enumerate(chunks):
            start = max(0.0, chunk.start)
            end = max(chunk.end, start)
            next_start = chunks[index + 1].start if index + 1 < len(chunks) else None

            max_end = start + cfg.max_duration_per_subtitle
            if next_start is not None:
                max_end = min(max_end, max(start, next_start - cfg.minimum_gap_between_subtitles))

            readable_end = start + cfg.min_duration_per_subtitle
            if end < readable_end and readable_end <= max_end:
                end = min(readable_end, max_end)
            else:
                end = min(end, max_end)

            if end <= start:
                end = min(start + 0.05, start + cfg.max_duration_per_subtitle)

            if smoothed and start < smoothed[-1].end + cfg.minimum_gap_between_subtitles:
                previous = smoothed[-1]
                clipped_previous_end = max(
                    previous.start + 0.05,
                    start - cfg.minimum_gap_between_subtitles,
                )
                smoothed[-1] = SubtitleChunk(
                    text=previous.text,
                    start=previous.start,
                    end=min(previous.end, clipped_previous_end),
                    words=previous.words,
                )

            smoothed.append(SubtitleChunk(text=chunk.text, start=start, end=end, words=chunk.words))

        return smoothed

    @staticmethod
    def _chunks_to_subtitles(chunks: list[SubtitleChunk]) -> list[srtlib.Subtitle]:
        return [
            srtlib.Subtitle(
                index=index,
                start=timedelta(seconds=round(chunk.start, 3)),
                end=timedelta(seconds=round(chunk.end, 3)),
                content=chunk.text,
            )
            for index, chunk in enumerate(chunks, 1)
            if chunk.text.strip() and chunk.end > chunk.start
        ]

    @staticmethod
    def _segments_to_subtitles(transcription_result: dict[str, Any]) -> list[srtlib.Subtitle]:
        words = Transcriber._estimate_words_from_segments(transcription_result)
        if not words and transcription_result.get("text"):
            words = Transcriber._estimate_words_from_text(transcription_result["text"], start=0.0, end=10.0)

        if not words:
            return []

        config = SubtitleTimingConfig()
        chunks = []
        for index in range(0, len(words), config.max_words_per_chunk):
            chunk_words = words[index : index + config.max_words_per_chunk]
            chunks.append(Transcriber._make_subtitle_chunk(chunk_words))

        return Transcriber._chunks_to_subtitles(chunks)

    @staticmethod
    def _subtitles_to_debug_chunks(subtitles: list[srtlib.Subtitle]) -> list[SubtitleChunk]:
        return [
            SubtitleChunk(
                text=subtitle.content,
                start=subtitle.start.total_seconds(),
                end=subtitle.end.total_seconds(),
                words=[],
            )
            for subtitle in subtitles
        ]

    @staticmethod
    def _write_timing_debug_outputs(
        srt_path: Path,
        words: list[WordTimestamp],
        chunks: list[SubtitleChunk],
    ) -> None:
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        raw_words_path = srt_path.with_name(f"{srt_path.stem}_raw_word_timestamps.json")
        grouped_chunks_path = srt_path.with_name(f"{srt_path.stem}_grouped_subtitle_chunks.json")
        final_srt_debug_path = srt_path.with_name(f"{srt_path.stem}_timed_final.srt")

        raw_words_path.write_text(
            json.dumps([asdict(word) for word in words], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        grouped_chunks_path.write_text(
            json.dumps(
                [
                    {
                        "text": chunk.text,
                        "start": round(chunk.start, 3),
                        "end": round(chunk.end, 3),
                        "duration": round(chunk.end - chunk.start, 3),
                        "words": [asdict(word) for word in chunk.words],
                    }
                    for chunk in chunks
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        final_srt_debug_path.write_text(
            srtlib.compose(Transcriber._chunks_to_subtitles(chunks)),
            encoding="utf-8",
        )
        logger.info("Saved raw word timestamp debug file: %s", raw_words_path)
        logger.info("Saved grouped subtitle debug file: %s", grouped_chunks_path)
        logger.info("Saved final timed SRT debug copy: %s", final_srt_debug_path)

    @staticmethod
    def _write_raw_transcription_debug_output(srt_path: Path, transcription_result: dict[str, Any]) -> None:
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        raw_transcription_path = srt_path.with_name(f"{srt_path.stem}_raw_transcription.json")
        raw_transcription_path.write_text(
            json.dumps(Transcriber._json_safe(transcription_result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved raw transcription debug file: %s", raw_transcription_path)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): Transcriber._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [Transcriber._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)
