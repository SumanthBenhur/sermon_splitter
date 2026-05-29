from dataclasses import dataclass
from pathlib import Path
import re

import srt as srtlib


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ROMAN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class ScriptStats:
    roman_chars: int
    devanagari_chars: int

    @property
    def total_script_chars(self) -> int:
        return self.roman_chars + self.devanagari_chars

    @property
    def roman_ratio(self) -> float:
        total = self.total_script_chars
        if total == 0:
            return 0.0
        return self.roman_chars / total

    @property
    def devanagari_ratio(self) -> float:
        total = self.total_script_chars
        if total == 0:
            return 0.0
        return self.devanagari_chars / total


def get_script_stats(text: str) -> ScriptStats:
    return ScriptStats(
        roman_chars=len(ROMAN_RE.findall(text)),
        devanagari_chars=len(DEVANAGARI_RE.findall(text)),
    )


def contains_devanagari_text(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text))


def is_mostly_roman_text(text: str, roman_threshold: float = 0.8) -> bool:
    stats = get_script_stats(text)
    if stats.total_script_chars == 0:
        return False
    return stats.roman_ratio >= roman_threshold


def read_srt_text(srt_path: Path) -> str:
    source = srt_path.read_text(encoding="utf-8", errors="ignore")
    try:
        subtitles = list(srtlib.parse(source))
    except srtlib.SRTParseError:
        return source

    return "\n".join(subtitle.content for subtitle in subtitles)


def srt_contains_devanagari(srt_path: Path) -> bool:
    return contains_devanagari_text(read_srt_text(srt_path))


def is_srt_mostly_roman(srt_path: Path, roman_threshold: float = 0.8) -> bool:
    return is_mostly_roman_text(read_srt_text(srt_path), roman_threshold=roman_threshold)


def get_srt_script_stats(srt_path: Path) -> ScriptStats:
    return get_script_stats(read_srt_text(srt_path))
