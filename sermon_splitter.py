import os
import sys
import subprocess
from pathlib import Path
import cv2
import numpy as np
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
        proc = subprocess.run([self.ffmpeg_path, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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

    def cut_video_clip(self, source_mp4: Path, start_time: str, end_time: str, out_file: Path):
        """Accurate re-encode cut with audio."""
        args = [
            "-y",
            "-i", str(source_mp4),
            "-ss", start_time, "-to", end_time,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_file)
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
            "-filter_complex", fc,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_file)
        ]
        self.ffmpeg.run_command(args)

    def create_face_tracked_vertical_video(self, input_mp4: str, output_mp4: str,
                                           out_w: int = OUT_W, out_h: int = OUT_H,
                                           smooth: float = SMOOTH):
        """Center the largest face; pipe frames to ffmpeg; copy audio from original."""
        def clamp(v, lo, hi): return max(lo, min(hi, v))

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
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}",
            "-r", f"{fps:.3f}",
            "-i", "-",
            "-i", input_mp4,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            "-movflags", "+faststart",
            output_mp4
        ]
        proc = subprocess.Popen([self.ffmpeg.ffmpeg_path, *ff_cmd], stdin=subprocess.PIPE)

        cx_s, cy_s = src_w / 2, src_h / 2
        with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4) as mp_fd:
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
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

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
                    crop = cv2.resize(frame_resized, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

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
            "-y", "-i", str(input_mp4),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", str(sample_rate),
            str(wav_path)
        ]
        self.ffmpeg.run_command(args)

    def burn_subtitles_into_video(self, input_mp4: Path, srt_path: Path, out_path: Path):
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
            "-i", str(input_mp4),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(out_path)
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
            print("[ERR] `transformers` and `torch` are required for Whisper. Please install them.")
            print("      pip install transformers torch")
            sys.exit(1)

        MODEL_NAME = "openai/whisper-base.en"
        print(f"\n[STEP] Transcribing with Whisper ({MODEL_NAME})...")

        wav_tmp = video_path.with_suffix(".wav")
        print(f"   Extracting audio to '{wav_tmp}'...")
        self.video_processor.extract_audio_to_wav(video_path, wav_tmp, 16000)

        print(f"   Loading Whisper model...")
        transcriber = pipeline("automatic-speech-recognition", model=MODEL_NAME, device=-1)
        print(f"   Model loaded. Starting transcription (this may take a while)...")

        transcription_result = transcriber(
            str(wav_tmp),
            chunk_length_s=30,
            return_timestamps=True,
        )

        full_text = transcription_result["text"].strip()
        print("\nTranscription complete.")

        subs = []
        for i, chunk in enumerate(transcription_result["chunks"]):
            start, end = chunk["timestamp"]
            if start is None or end is None:
                continue
            subs.append(srtlib.Subtitle(
                index=i + 1,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                content=chunk["text"].strip()
            ))

        srt_content = srtlib.compose(subs)
        srt_path.write_text(srt_content, encoding="utf-8")
        print(f"[OK] SRT saved -> {srt_path}")

        if wav_tmp.exists():
            wav_tmp.unlink()
            print(f"   Cleanup: Removed temporary file '{wav_tmp}'")


class SermonSplitterApp:
    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
        self.artifacts_dir = self.source_path.parent / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        self.ffmpeg_manager = FfmpegManager()
        self.video_processor = VideoProcessor(self.ffmpeg_manager)
        self.transcriber = Transcriber(self.video_processor)
        self.video_utils = VideoUtils()

    def run(self):
        """The main command-line interface workflow for processing video clips."""
        try:
            n = int(input("How many clips? (1 for single): ").strip() or "1")
        except ValueError:
            n = 1

        clips = []
        for i in range(1, n + 1):
            print(f"\n--- Clip {i} ---")
            start_time = input("Start time (HH:MM:SS): ").strip()
            end_time = input("End time   (HH:MM:SS): ").strip()
            if n == 1:
                out_name = self.video_utils.sanitize_mp4_filename(
                    input("Output file name (e.g., my_clip.mp4): ").strip() or "clip.mp4")
            else:
                out_name = self.video_utils.sanitize_mp4_filename(f"part_{i}.mp4")
            out_path = self.artifacts_dir / out_name
            self.video_processor.cut_video_clip(self.source_path, start_time, end_time, out_path)
            print(f"Saved: {out_path}")
            clips.append(out_path)

        if len(clips) == 1:
            final_clip = clips[0]
        else:
            combo_name = self.video_utils.sanitize_mp4_filename(
                input("\nName for concatenated file (e.g., combined.mp4): ").strip() or "combined.mp4")
            final_clip = self.artifacts_dir / combo_name
            self.video_processor.concatenate_video_clips(clips, final_clip)
            print(f"Concatenated file saved: {final_clip}")

        print("\nMaking vertical 1080x1920 with face tracking...")
        vert_out = final_clip.with_name(final_clip.stem + "_vertical.mp4")
        self.video_processor.create_face_tracked_vertical_video(str(final_clip), str(vert_out))
        final_clip = vert_out
        print(f"Vertical saved: {final_clip}")

        print("\nAdding ENGLISH subtitles (Whisper) and burning them in...")
        srt_out = final_clip.with_suffix(".srt")
        subbed_out = final_clip.with_name(final_clip.stem + "_subbed.mp4")
        self.transcriber.transcribe_video_with_whisper(final_clip, srt_out)
        print("[STEP] Burning subtitles (white text on black box)...")
        self.video_processor.burn_subtitles_into_video(final_clip, srt_out, subbed_out)
        print(f"\n Done. Output: {subbed_out}")
