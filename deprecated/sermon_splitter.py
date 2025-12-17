import sys
import subprocess
from pathlib import Path
import cv2
import mediapipe as mp
import srt as srtlib
from datetime import timedelta

# ---------- Config (tweakable) ----------
OUT_W, OUT_H = 1080, 1920
SMOOTH = 0.88
FFMPEG = "ffmpeg"  # assumes ffmpeg is on PATH


class FfmpegManager:
    def __init__(self, ffmpeg_path: str = FFMPEG):
        self.ffmpeg_path = ffmpeg_path

    def run_command(self, args: list):
        """Executes an FFmpeg command and raises an exception if it fails."""
        proc = subprocess.run(
            [self.ffmpeg_path, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")
        return proc


class VideoUtils:
    @staticmethod
    def sanitize_mp4_filename(name: str, default: str = "clip.mp4") -> str:
        """Sanitizes a string to be a valid filename and ensures it ends with .mp4."""
        name = (name or default).strip()
        if not name.lower().endswith(".mp4"):
            name += ".mp4"
        for bad in r'<>:"/|?*':
            name = name.replace(bad, "_")
        return name


class VideoProcessor:
    def __init__(self, ffmpeg_manager: FfmpegManager):
        self.ffmpeg = ffmpeg_manager

    def cut_video_clip(
        self, source_mp4: Path, start_time: str, end_time: str, out_file: Path
    ):
        """Accurate re-encode cut with audio."""
        args = [
            "-y",
            "-i",
            str(source_mp4),
            "-ss",
            start_time,
            "-to",
            end_time,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out_file),
        ]
        self.ffmpeg.run_command(args)

    def concatenate_video_clips(self, inputs, out_file: Path):
        """Re-encode concat (safe)."""
        fc_in = "".join([f"[{i}:v:0][{i}:a:0]" for i in range(len(inputs))])
        fc = f"{fc_in}concat=n={len(inputs)}:v=1:a=1[v][a]"
        args = ["-y"]
        for p in inputs:
            args += ["-i", str(p)]
        args += [
            "-filter_complex",
            fc,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out_file),
        ]
        self.ffmpeg.run_command(args)

    def create_face_tracked_vertical_video(
        self,
        input_mp4: str,
        output_mp4: str,
        out_w: int = OUT_W,
        out_h: int = OUT_H,
        smooth: float = SMOOTH,
    ):
        """Center the largest face; pipe frames to ffmpeg; copy audio from original."""

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        cap = cv2.VideoCapture(input_mp4)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_mp4}")

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps < 1:
            fps = 30.0

        ff_cmd = [
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{out_w}x{out_h}",
            "-r",
            f"{fps:.3f}",
            "-i",
            "-",
            "-i",
            input_mp4,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            output_mp4,
        ]
        proc = subprocess.Popen(
            [self.ffmpeg.ffmpeg_path, *ff_cmd], stdin=subprocess.PIPE
        )

        cx_s, cy_s = src_w / 2, src_h / 2
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4
        ) as mp_fd:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mp_fd.process(rgb)

                if res.detections:
                    best = None
                    for d in res.detections:
                        bb = d.location_data.relative_bounding_box
                        x, y, w, h = bb.xmin, bb.ymin, bb.width, bb.height
                        area = w * h
                        if best is None or area > best[0]:
                            best = (area, x, y, w, h)
                    _, x, y, w, h = best
                    cx = (x + w / 2) * src_w
                    cy = (y + h / 2) * src_h
                    cx_s = smooth * cx_s + (1 - smooth) * cx
                    cy_s = smooth * cy_s + (1 - smooth) * cy
                else:
                    cx_s = smooth * cx_s + (1 - smooth) * (src_w / 2)
                    cy_s = smooth * cy_s + (1 - smooth) * (src_h / 2)

                scale = max(out_h / src_h, out_w / src_w)
                new_w = max(out_w, int(round(src_w * scale)))
                new_h = max(out_h, int(round(src_h * scale)))
                frame_resized = cv2.resize(
                    frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC
                )

                cx_scaled = cx_s * scale
                cy_scaled = cy_s * scale
                x0 = int(round(cx_scaled - out_w / 2))
                y0 = int(round(cy_scaled - out_h / 2))
                x0 = clamp(x0, 0, new_w - out_w)
                y0 = clamp(y0, 0, new_h - out_h)
                x1 = x0 + out_w
                y1 = y0 + out_h
                crop = frame_resized[y0:y1, x0:x1]
                if crop is None or crop.shape[1] != out_w or crop.shape[0] != out_h:
                    crop = cv2.resize(
                        frame_resized, (out_w, out_h), interpolation=cv2.INTER_CUBIC
                    )

                try:
                    proc.stdin.write(crop.tobytes())
                except BrokenPipeError:
                    break

        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    def extract_audio_to_wav(self, input_mp4: Path, wav_path: Path, sample_rate=16000):
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
        self.ffmpeg.run_command(args)

    def burn_subtitles_into_video(
        self, input_mp4: Path, srt_path: Path, out_path: Path
    ):
        """Burn SRT with a proper opaque black box behind each line."""
        FORCE_STYLE = (
            "FontName=Arial,"
            "FontSize=12,"
            "PrimaryColour=&H00FFFFFF&,"
            "BackColour=&H00000000&,"
            "BorderStyle=3,"
            "Outline=0,Shadow=0,"
            "WrapStyle=2,Alignment=2,MarginV=80,"
            "ScaleBorderAndShadow=yes"
        )

        vf = f"subtitles=filename='{srt_path}':force_style='{FORCE_STYLE}'"
        args = [
            "-y",
            "-i",
            str(input_mp4),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        self.ffmpeg.run_command(args)


class Transcriber:
    def __init__(self, video_processor: VideoProcessor):
        self.video_processor = video_processor

    def transcribe_video_with_whisper(self, video_path: Path, srt_path: Path):
        """Transcribes video using Whisper and saves SRT."""
        try:
            from transformers import pipeline
        except ImportError:
            print(
                "[ERR] `transformers` and `torch` are required for Whisper. Please install them."
            )
            print("      pip install transformers torch")
            sys.exit(1)

        MODEL_NAME = "openai/whisper-base.en"
        print(f"\n[STEP] Transcribing with Whisper ({MODEL_NAME})...")

        wav_tmp = video_path.with_suffix(".wav")
        print(f"   Extracting audio to '{wav_tmp}'...")
        self.video_processor.extract_audio_to_wav(video_path, wav_tmp, 16000)

        print("   Loading Whisper model...")
        transcriber = pipeline(
            "automatic-speech-recognition", model=MODEL_NAME, device=-1
        )
        print("   Model loaded. Starting transcription (this may take a while)...")

        # CUSTOM_PROMPT = "The sermon Genesis 20, where Abhraham calls Sarah his sister. "

        transcription_result = transcriber(
            str(wav_tmp),
            chunk_length_s=30,
            return_timestamps=True,
            # generate_kwargs={
            #     "task": "transcribe",
            #     # UNCOMMENT THIS LINE and use your custom prompt
            #     "prompt_ids": transcriber.tokenizer.encode(CUSTOM_PROMPT, add_special_tokens=False)
            # }
        )

        full_text = transcription_result["text"].strip()
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

    def refit_srt(
        self,
        in_srt_path: Path,
        out_srt_path: Path,
        max_chars: int = 30,
        max_duration: float = 4.0,
        max_lines: int = 2,
        min_duration: float = 0.8,
        safety_gap: float = 0.05,
    ):
        """
        Re-wrap and split SRT entries so each subtitle:
          - has at most `max_lines` visual lines
          - each line is <= `max_chars` (greedy word-wrap)
          - each subtitle segment lasts <= `max_duration` seconds
        When splitting, duration is distributed ~proportionally to text length.
        """

        def norm_spaces(s: str) -> str:
            return " ".join(s.strip().split())

        def greedy_wrap(words, line_limit):
            """Greedy wrap by words with max_chars per line."""
            lines, line = [], ""
            for w in words:
                if not line:
                    cand = w
                else:
                    cand = line + " " + w
                if len(cand) <= line_limit:
                    line = cand
                else:
                    if line:
                        lines.append(line)
                    line = w
            if line:
                lines.append(line)
            return lines

        def chunk_text_to_lines(text: str, max_chars: int, max_lines: int):
            """
            Returns:
              chunks: list[str] where each element is up to max_lines of wrapped lines joined by '\n'.
            """
            words = norm_spaces(text).split()
            chunks = []
            i = 0
            while i < len(words):
                # Build up to max_lines worth of wrapped lines
                block_lines = []
                j = i
                while j < len(words) and len(block_lines) < max_lines:
                    # Fill one wrapped line
                    # Grow line greedily until max_chars or words end
                    k = j
                    line_words = []
                    cur_len = 0
                    while k < len(words):
                        w = words[k]
                        add_len = len(w) if cur_len == 0 else (1 + len(w))
                        if cur_len + add_len <= max_chars:
                            line_words.append(w)
                            cur_len += add_len
                            k += 1
                        else:
                            break
                    if not line_words:  # single long token fallback
                        line_words = [words[k]]
                        k = k + 1
                    block_lines.append(" ".join(line_words))
                    j = k

                chunks.append("\n".join(block_lines))
                i = j
            return chunks

        def seconds(td: timedelta) -> float:
            return td.total_seconds()

        def td(sec: float) -> timedelta:
            return timedelta(seconds=max(0.0, sec))

        # ---- load
        src = in_srt_path.read_text(encoding="utf-8", errors="ignore")
        items = list(srtlib.parse(src))

        new_items = []
        idx = 1
        for it in items:
            text = norm_spaces(it.content)
            if not text:
                continue

            vis_chunks = chunk_text_to_lines(
                text, max_chars=max_chars, max_lines=max_lines
            )
            if not vis_chunks:
                continue

            dur = max(0.0, seconds(it.end - it.start))
            base_dur = max(dur, min_duration)

            total_chars = sum(len(c.replace("\n", " ")) for c in vis_chunks) or 1
            dur_per_char = base_dur / total_chars

            cur_start = seconds(it.start)

            final_chunks = []
            for ch in vis_chunks:
                est_dur = (len(ch.replace("\n", " ")) or 1) * dur_per_char

                if est_dur > max_duration:
                    # This chunk is too long. Split it proportionally by character count.
                    num_splits = int(est_dur / max_duration) + 1
                    words = ch.replace("\n", " ").split()
                    words_per_split = (len(words) + num_splits - 1) // num_splits

                    # First, create all the sub-chunks of text
                    sub_chunks_text = []
                    for i in range(0, len(words), words_per_split):
                        sub_words = words[i : i + words_per_split]
                        if not sub_words:
                            continue

                        sub_text = "\n".join(
                            greedy_wrap(sub_words, max_chars)[:max_lines]
                        )
                        if not sub_text.strip():
                            continue
                        sub_chunks_text.append(sub_text)

                    # Then, calculate total characters of the new sub-chunks
                    total_sub_chars = (
                        sum(len(s.replace("\n", " ")) for s in sub_chunks_text) or 1
                    )

                    # Finally, distribute the original estimated duration proportionally
                    for sub_text in sub_chunks_text:
                        sub_chars = len(sub_text.replace("\n", " "))
                        proportional_sub_dur = est_dur * (sub_chars / total_sub_chars)
                        final_chunks.append(
                            {"text": sub_text, "duration": proportional_sub_dur}
                        )
                else:
                    final_chunks.append({"text": ch, "duration": est_dur})

            # Create SRT items from the final chunks
            for chunk in final_chunks:
                seg_dur = max(min_duration, chunk["duration"])
                st = cur_start
                en = st + seg_dur

                new_items.append(
                    srtlib.Subtitle(
                        index=idx, start=td(st), end=td(en), content=chunk["text"]
                    )
                )
                idx += 1
                cur_start = en  # Chain contiguously

        # Post-process to sort and fix any overlaps
        if new_items:
            new_items.sort(key=lambda x: x.start)

            for i in range(len(new_items) - 1):
                current_sub = new_items[i]
                next_sub = new_items[i + 1]

                # Add safety gap between subtitles that were not originally contiguous
                # This is a heuristic: if the gap is very small, they were likely split
                if next_sub.start - current_sub.end < timedelta(seconds=safety_gap * 2):
                    next_sub.start = current_sub.end + timedelta(seconds=safety_gap)

                if current_sub.end > next_sub.start:
                    current_sub.end = next_sub.start - timedelta(microseconds=1)

                if current_sub.end <= current_sub.start:
                    current_sub.end = current_sub.start + timedelta(
                        seconds=min_duration
                    )

        # Re-index cleanly
        for i, it in enumerate(new_items, 1):
            it.index = i

        out_srt_path.write_text(srtlib.compose(new_items), encoding="utf-8")
        print(
            f"[OK] Refitted SRT -> {out_srt_path}  (max_chars={max_chars}, max_lines={max_lines}, max_duration={max_duration}s)"
        )


class SermonSplitterApp:
    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
        self.artifacts_dir = self.source_path.parent / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        self.ffmpeg_manager = FfmpegManager()
        self.video_processor = VideoProcessor(self.ffmpeg_manager)
        self.transcriber = Transcriber(self.video_processor)
        self.video_utils = VideoUtils()

    def run(self, num_clips, clips_data, output_filename):
        """The main command-line interface workflow for processing video clips."""
        n = num_clips

        clips = []
        for i in range(n):
            print(f"\n--- Clip {i} ---")
            start_time = clips_data[i]["start_time"]
            end_time = clips_data[i]["end_time"]
            if n == 1:
                out_name = self.video_utils.sanitize_mp4_filename(
                    output_filename or "clip.mp4"
                )
            else:
                out_name = self.video_utils.sanitize_mp4_filename(f"part_{i}.mp4")
            out_path = self.artifacts_dir / out_name
            self.video_processor.cut_video_clip(
                self.source_path, start_time, end_time, out_path
            )
            print(f"Saved: {out_path}")
            clips.append(out_path)

        if len(clips) == 1:
            final_clip = clips[0]
        else:
            combo_name = self.video_utils.sanitize_mp4_filename(
                output_filename or "combined.mp4"
            )
            final_clip = self.artifacts_dir / combo_name
            self.video_processor.concatenate_video_clips(clips, final_clip)
            print(f"Concatenated file saved: {final_clip}")

        print("\nMaking vertical 1080x1920 with face tracking...")
        vert_out = final_clip.with_name(final_clip.stem + "_vertical.mp4")
        self.video_processor.create_face_tracked_vertical_video(
            str(final_clip), str(vert_out)
        )
        final_clip = vert_out
        print(f"Vertical saved: {final_clip}")

        print("\nAdding ENGLISH subtitles (Whisper) and burning them in...")
        srt_out = final_clip.with_suffix(".srt")
        subbed_out = final_clip.with_name(final_clip.stem + "_subbed.mp4")
        self.transcriber.transcribe_video_with_whisper(final_clip, srt_out)

        print("[STEP] Refitting subtitles for better readability...")
        refit_srt_out = final_clip.with_suffix(".refit.srt")
        self.transcriber.refit_srt(srt_out, refit_srt_out)

        print("[STEP] Burning subtitles (white text on black box)...")
        self.video_processor.burn_subtitles_into_video(
            final_clip, refit_srt_out, subbed_out
        )
        print(f"\n Done. Output: {subbed_out}")
        return subbed_out
