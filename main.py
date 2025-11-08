import os
import sys
import subprocess
from pathlib import Path
import json
import shutil
import wave
import contextlib
import time

import cv2
import numpy as np
import mediapipe as mp
from yt_dlp import YoutubeDL

import srt as srtlib
from datetime import timedelta
from transformers import pipeline



# ---------- Config (tweakable) ----------
OUT_W, OUT_H = 1080, 1920
SMOOTH = 0.88
FFMPEG = "ffmpeg"  # assumes ffmpeg.exe is on PATH from setup BAT or system PATH

# --- Subtitle style (ASS/libass) ---
# Colors are &HAABBGGRR (AA = alpha, BBGGRR = BGR)
# White text, semi-transparent black box, bottom-center, sane margins/wrap.
SUB_FONT = "Arial"
SUB_FONTSIZE = 42
FORCE_STYLE = (
    f"FontName={SUB_FONT},"
    f"FontSize={SUB_FONTSIZE},"
    f"PrimaryColour=&H00FFFFFF&, "     # opaque white
    f"BackColour=&H7F000000&,"        # ~50% black box
    f"BorderStyle=3,"                 # boxed background (tight to glyphs)
    f"Outline=0,Shadow=0,"            # no outline/shadow (box handles contrast)
    f"Alignment=2,"                   # bottom-center
    f"MarginV=60,MarginL=60,MarginR=60,"
    f"WrapStyle=2,"                   # smart wrapping
    f"ScaleX=100,ScaleY=100"
)


# ---------- Utils ----------
def sanitize_mp4_filename(name: str, default: str = "clip.mp4") -> str:
    """Sanitizes a string to be a valid filename and ensures it ends with .mp4."""
    name = (name or default).strip()
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    for bad in r'<>:"/|?*':
        name = name.replace(bad, "_")
    return name


def run_ffmpeg_command(args: list):
    """Executes an FFmpeg command and raises an exception if it fails."""
    proc = subprocess.run([FFMPEG, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")
    return proc

def download_video_from_url(url: str, outname: str = "source.mp4") -> Path:
    """Best available (falls back automatically inside yt-dlp). Merges to MP4."""
    ydl_opts = {
        "format": "bv*+ba/b",
        "outtmpl": "source.%(ext)s",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "retries": 10,
        "fragment_retries": 10,
        "ignoreerrors": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    if filename.name != outname:
        Path(filename).replace(outname)
    return Path(outname).resolve()

def cut_video_clip(source_mp4: Path, start_time: str, end_time: str, out_file: Path):
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
    run_ffmpeg_command(args)

def concatenate_video_clips(inputs, out_file: Path):
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
    run_ffmpeg_command(args)

def create_face_tracked_vertical_video(input_mp4: str, output_mp4: str,
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
        "-i", "-",                # stdin video
        "-i", input_mp4,          # original for audio
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        "-movflags", "+faststart",
        output_mp4
    ]
    proc = subprocess.Popen([FFMPEG, *ff_cmd], stdin=subprocess.PIPE)

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

def extract_audio_to_wav(input_mp4: Path, wav_path: Path, sample_rate=16000):
    """Extracts audio from a video file to a mono 16kHz WAV file."""
    args = [
        "-y", "-i", str(input_mp4),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        str(wav_path)
    ]
    run_ffmpeg_command(args)

def transcribe_video_with_whisper(video_path: Path, srt_path: Path):
    """Transcribes video using Whisper and saves SRT."""
    # Lazy import as transformers is a heavy optional dependency
    try:
        from transformers import pipeline
    except ImportError:
        print("[ERR] `transformers` and `torch` are required for Whisper. Please install them.")
        print("      pip install transformers torch")
        # Or depending on pytorch.org for your platform, e.g. with CUDA
        sys.exit(1)

    from datetime import timedelta
    import srt as srtlib

    MODEL_NAME = "openai/whisper-base.en"  # Using a base model for broad compatibility

    print(f"\n[STEP] Transcribing with Whisper ({MODEL_NAME})...")

    # 1. Extract audio to a temporary WAV file
    wav_tmp = video_path.with_suffix(".wav")
    print(f"   Extracting audio to '{wav_tmp}'...")
    extract_audio_to_wav(video_path, wav_tmp, 16000)

    # 2. Load model and transcribe
    print(f"   Loading Whisper model...")
    # Use device=0 for GPU if available, or device=-1 for CPU
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        device=-1  # Set to 0 if running on a machine with a GPU
    )
    print(f"   Model loaded. Starting transcription (this may take a while)...")

    # Run transcription
    transcription_result = transcriber(
        str(wav_tmp),
        chunk_length_s=30,  # Process in 30-second chunks
        return_timestamps=True, # Essential for SRT generation
        # generate_kwargs={
        #     "task": "transcribe",
        #     # Example of how to provide context to Whisper for better accuracy
        #     # "prompt_ids": transcriber.tokenizer.encode("A sermon about Jesus Christ.", add_special_tokens=False)
        # }
    )

    full_text = transcription_result["text"].strip()
    print("\n3. Transcription Complete:")
    print("--------------------------------------------------")
    print(full_text)
    print("--------------------------------------------------")

    # 4. Convert transcription chunks to SRT format
    subs = []
    for i, chunk in enumerate(transcription_result["chunks"]):
        start, end = chunk["timestamp"]
        # Whisper can sometimes return None for timestamps
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

    # 5. Cleanup
    if wav_tmp.exists():
        wav_tmp.unlink()
        print(f"   Cleanup: Removed temporary file '{wav_tmp}'")


def _format_path_for_subtitle_filter(p: Path) -> str:
    """
    Make a filename safe inside -vf subtitles=... on Windows.
    - Use forward slashes
    - Escape colon (:) and single quote (')
    """
    s = p.as_posix()
    s = s.replace(':', r'\:').replace("'", r"'\'")
    return s

def burn_subtitles_into_video(input_mp4: Path, srt_path: Path, out_path: Path):
    """
    Burn SRT with a proper opaque black box behind each line,
    white text, bottom-center, smart wrapping, decent margins.
    """
    # Force style for libass:
    # - BorderStyle=3 => opaque box per line
    # - BackColour=&H00000000& => opaque black (AA=00, BB=00, GG=00, RR=00)
    # - PrimaryColour=&H00FFFFFF& => opaque white
    # - WrapStyle=2 => smart line wrapping
    # - Alignment=2 => bottom-center
    # - MarginV=80 => lift a bit above the very bottom
    # - ScaleBorderAndShadow=yes keeps things proportional if scaled
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

    subfile = _format_path_for_subtitle_filter(srt_path)
    vf = f"subtitles=filename='{subfile}':force_style='{FORCE_STYLE}'"

    cmd = [
        FFMPEG, "-y",
        "-i", str(input_mp4),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path)
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")
    print(f"[OK] Burned-in video -> {out_path}")

# ---------- Main ----------
def main_cli_workflow():
    """The main command-line interface workflow for processing video clips."""
    source_path = Path("/Users/sumanthbenhur/Desktop/sermon_splitter/videos/test1/ashish_vertical.mp4")
    
    artifacts_dir = source_path.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    try:
        n = int(input("How many clips? (1 for single): ").strip() or "1")
    except ValueError:
        n = 1

    clips = []
    for i in range(1, n + 1):
        print(f"\n--- Clip {i} ---")
        start_time = input("Start time (HH:MM:SS): ").strip()
        end_time   = input("End time   (HH:MM:SS): ").strip()
        if n == 1:
            out_name = sanitize_mp4_filename(input("Output file name (e.g., my_clip.mp4): ").strip() or "clip.mp4")
        else:
            out_name = sanitize_mp4_filename(f"part_{i}.mp4")
        out_path = artifacts_dir / out_name
        cut_video_clip(source_path, start_time, end_time, out_path)
        print(f"Saved: {out_path}")
        clips.append(out_path)

    if len(clips) == 1:
        final_clip = clips[0]
    else:
        combo_name = sanitize_mp4_filename(input("\nName for concatenated file (e.g., combined.mp4): ").strip() or "combined.mp4")
        final_clip = artifacts_dir / combo_name
        concatenate_video_clips(clips, final_clip)
        print(f"Concatenated file saved: {final_clip}")

    # Always make vertical and add subtitles
    print("\nMaking vertical 1080x1920 with face tracking...")
    vert_out = final_clip.with_name(final_clip.stem + "_vertical.mp4")
    create_face_tracked_vertical_video(str(final_clip), str(vert_out))
    final_clip = vert_out
    print(f"Vertical saved: {final_clip}")

    print("\nAdding ENGLISH subtitles (Whisper) and burning them in...")
    srt_out = final_clip.with_suffix(".srt")
    subbed_out = final_clip.with_name(final_clip.stem + "_subbed.mp4")            # Transcribe using Whisper
    transcribe_video_with_whisper(final_clip, srt_out)
    print("[STEP] Burning subtitles (white text on black box)...")
    burn_subtitles_into_video(final_clip, srt_out, subbed_out)
    print(f"\n Done. Output: {subbed_out}")

    
if __name__ == "__main__":
    main_cli_workflow()
