from datetime import timedelta
from pathlib import Path

import srt as srtlib


def _to_ass_timestamp(value: timedelta) -> str:
    total_seconds = max(0.0, value.total_seconds())
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    centiseconds = int(round((total_seconds - int(total_seconds)) * 100))
    if centiseconds == 100:
        seconds += 1
        centiseconds = 0
    if seconds == 60:
        minutes += 1
        seconds = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _escape_ass_text(text: str) -> str:
    # Escape ASS control characters and preserve explicit line breaks.
    safe = text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    lines = [ln.strip().upper() for ln in safe.splitlines() if ln.strip()]
    if not lines:
        return ""
    return r"\N".join(lines[:2])


def convert_srt_to_styled_ass(in_srt_path: Path, out_ass_path: Path) -> None:
    """Convert SRT subtitles into social-media styled ASS subtitles."""
    src = in_srt_path.read_text(encoding="utf-8", errors="ignore")
    subs = list(srtlib.parse(src))

    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: SocialCaption,Arial Black,54,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,3,12,0,2,80,80,150,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for sub in subs:
        text = _escape_ass_text(sub.content)
        if not text:
            continue
        start = _to_ass_timestamp(sub.start)
        end = _to_ass_timestamp(sub.end)
        ass_lines.append(f"Dialogue: 0,{start},{end},SocialCaption,,0,0,0,,{text}")

    out_ass_path.write_text("\n".join(ass_lines) + "\n", encoding="utf-8")
    print(f"[OK] Styled ASS saved -> {out_ass_path}")
