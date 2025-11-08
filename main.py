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

# from vosk import Model, KaldiRecognizer
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
    f"PrimaryColour=&H00FFFFFF&,"     # opaque white
    f"BackColour=&H7F000000&,"        # ~50% black box
    f"BorderStyle=3,"                 # boxed background (tight to glyphs)
    f"Outline=0,Shadow=0,"            # no outline/shadow (box handles contrast)
    f"Alignment=2,"                   # bottom-center
    f"MarginV=60,MarginL=60,MarginR=60,"
    f"WrapStyle=2,"                   # smart wrapping
    f"ScaleX=100,ScaleY=100"
)

VOSK_LANG = "en"  # English only
VOSK_MODEL_DIR = Path("models") / "vosk-en"

# ---------- Utils ----------
def ensure_mp4_name(name: str, default: str = "clip.mp4") -> str:
    name = (name or default).strip()
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    for bad in r'<>:"/\|?*':
        name = name.replace(bad, "_")
    return name

def _ffmpeg_filter_escape_path(p: Path) -> str:
    # Use forward slashes, then escape the drive-colon and single quotes
    s = p.as_posix()
    s = s.replace(":", r"\:").replace("'", r"\'")
    return f"'{s}'"  # wrap for ffmpeg


def run_ffmpeg(args: list):
    proc = subprocess.run([FFMPEG, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")
    return proc

def download_video(url: str, outname: str = "source.mp4") -> Path:
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

def cut_from_source(source_mp4: Path, start_time: str, end_time: str, out_file: Path):
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
    run_ffmpeg(args)

def concat_mp4s(inputs, out_file: Path):
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
    run_ffmpeg(args)

def face_tracked_vertical(input_mp4: str, output_mp4: str,
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

# ---------- Vosk transcription ----------
def ensure_vosk_model():
    """Download a small English model if missing."""
    if VOSK_MODEL_DIR.exists():
        return
    VOSK_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    # Small model URL (stable & lightweight)
    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    zip_path = VOSK_MODEL_DIR.parent / "vosk_en_small.zip"
    print("[INFO] Downloading Vosk EN model (small)...")
    import urllib.request, zipfile
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(VOSK_MODEL_DIR.parent)
    # Move/rename to VOSK_MODEL_DIR
    extracted = VOSK_MODEL_DIR.parent / top
    if VOSK_MODEL_DIR.exists():
        shutil.rmtree(VOSK_MODEL_DIR)
    extracted.rename(VOSK_MODEL_DIR)
    zip_path.unlink(missing_ok=True)
    print("[INFO] Vosk model ready at", VOSK_MODEL_DIR)

def extract_wav(input_mp4: Path, wav_path: Path, sample_rate=16000):
    args = [
        "-y", "-i", str(input_mp4),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        str(wav_path)
    ]
    run_ffmpeg(args)

def transcribe_vosk(wav_path: Path, srt_path: Path):
    ensure_vosk_model()
    model = Model(str(VOSK_MODEL_DIR))
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    subs = []
    idx = 1

    with wave.open(str(wav_path), "rb") as wf:
        total = wf.getnframes()
        chunk_size = 4000
        frames = 0
        t0 = time.time()
        while True:
            data = wf.readframes(chunk_size)
            if len(data) == 0:
                break
            frames += chunk_size
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                subs += _result_to_srt(res, idx)
                idx = len(subs) + 1
            # else: partials ignored (we’ll handle in FinalResult)

        res = json.loads(rec.FinalResult())
        subs += _result_to_srt(res, idx)

    srt_text = srtlib.compose(subs)
    srt_path.write_text(srt_text, encoding="utf-8")
    print(f"[OK] SRT saved -> {srt_path}")

def _result_to_srt(res_json, start_idx):
    import srt as srtlib
    from datetime import timedelta
    subs = []
    if not res_json or "result" not in res_json:
        return subs
    words = res_json["result"]
    if not words:
        return subs

    # group words into ~5 sec chunks for readability
    group = []
    grp_start = None
    max_span = 5.0
    idx = start_idx
    for w in words:
        st = float(w["start"])
        et = float(w["end"])
        if grp_start is None:
            grp_start = st
        group.append(w["word"])
        if (et - grp_start) >= max_span:
            text = " ".join(group)
            subs.append(srtlib.Subtitle(
                index=idx,
                start=timedelta(seconds=grp_start),
                end=timedelta(seconds=et),
                content=text
            ))
            idx += 1
            group = []
            grp_start = None
    if group:
        et = float(words[-1]["end"])
        st = grp_start if grp_start is not None else float(words[0]["start"])
        text = " ".join(group)
        subs.append(srtlib.Subtitle(
            index=idx,
            start=timedelta(seconds=st),
            end=timedelta(seconds=et),
            content=text
        ))
    return subs

def transcribe_whisper(video_path: Path, srt_path: Path):
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
    extract_wav(video_path, wav_tmp, 16000)

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


def _escape_for_subtitles_filter(p: Path) -> str:
    """
    Make a filename safe inside -vf subtitles=... on Windows.
    - Use forward slashes
    - Escape colon (:) and single quote (')
    """
    s = p.as_posix()
    s = s.replace(':', r'\:').replace("'", r"\'")
    return s

def burn_in_subs(input_mp4: Path, srt_path: Path, out_path: Path):
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

    subfile = _escape_for_subtitles_filter(srt_path)
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
def main():
    # print("Input type? [1] YouTube URL  [2] Local file path")
    # mode = input("Choose 1/2: ").strip()
    # if mode == "1":
    #     url = input("Enter YouTube URL: ").strip()
    #     source_path = download_video(url)
    # else:
    #     path = input("Enter path to local video file: ").strip().strip('"')
    #     source_path = Path(path).resolve()
    #     if not source_path.exists():
    #         print("[ERR] File not found.")
    #         sys.exit(1)

    source_path = "/Users/sumanthbenhur/Desktop/sermon_splitter/ashish_vertical.mp4"
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
            out_name = ensure_mp4_name(input("Output file name (e.g., my_clip.mp4): ").strip() or "clip.mp4")
        else:
            out_name = ensure_mp4_name(f"part_{i}.mp4")
        out_path = Path(out_name).resolve()
        cut_from_source(source_path, start_time, end_time, out_path)
        print(f"Saved: {out_path}")
        clips.append(out_path)

    if len(clips) == 1:
        final_clip = clips[0]
    else:
        combo_name = ensure_mp4_name(input("\nName for concatenated file (e.g., combined.mp4): ").strip() or "combined.mp4")
        final_clip = Path(combo_name).resolve()
        concat_mp4s(clips, final_clip)
        print(f"Concatenated file saved: {final_clip}")

    # Optional vertical
    make_vert = input("\nMake vertical 1080x1920 with face tracking? (y/n): ").strip().lower()
    if make_vert in ("y", "yes"):
        vert_out = final_clip.with_name(final_clip.stem + "_vertical.mp4")
        face_tracked_vertical(str(final_clip), str(vert_out))
        final_clip = vert_out
        print(f"Vertical saved: {final_clip}")

        # Optional subtitles

        do_subs = input("\nAdd ENGLISH subtitles (Whisper) and burn-in? (y/n): ").strip().lower()

        if do_subs in ("y", "yes"):

            srt_out = final_clip.with_suffix(".srt")

            subbed_out = final_clip.with_name(final_clip.stem + "_subbed.mp4")

    

            # The old Vosk-based transcription is now replaced by Whisper.

            # You can uncomment these lines and comment out the whisper call to switch back.

            # wav_tmp = final_clip.with_suffix(".wav")

            # print("[STEP] Extracting WAV (16k mono)...")

            # extract_wav(final_clip, wav_tmp, 16000)

            # print("[STEP] Transcribing with Vosk (EN only)...")

            # transcribe_vosk(wav_tmp, srt_out)

            # try:

            #     wav_tmp.unlink()

            # except Exception:

            #     pass

    

            # Transcribe using Whisper

            transcribe_whisper(final_clip, srt_out)

    

            print("[STEP] Burning subtitles (white text on black box)...")

            burn_in_subs(final_clip, srt_out, subbed_out)

    

            print(f"\n Done. Output: {subbed_out}")

        else:

            print("\n Done. Output:", final_clip)

    
if __name__ == "__main__":
    main()

    