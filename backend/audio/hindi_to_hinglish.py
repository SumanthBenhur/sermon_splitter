from pathlib import Path
import json
import os
import re
import urllib.error
import urllib.request

import srt as srtlib
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from dotenv import load_dotenv


load_dotenv()


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
ROMAN_WORD_RE = re.compile(r"[A-Za-z]+")


URDU_CHAR_MAP = {
    "\u0627": "a",  # alif
    "\u0622": "aa",  # alif madd
    "\u0628": "b",
    "\u067E": "p",
    "\u062A": "t",
    "\u0679": "t",
    "\u062B": "s",
    "\u062C": "j",
    "\u0686": "ch",
    "\u062D": "h",
    "\u062E": "kh",
    "\u062F": "d",
    "\u0688": "d",
    "\u0630": "z",
    "\u0631": "r",
    "\u0691": "r",
    "\u0632": "z",
    "\u0698": "zh",
    "\u0633": "s",
    "\u0634": "sh",
    "\u0635": "s",
    "\u0636": "z",
    "\u0637": "t",
    "\u0638": "z",
    "\u0639": "a",
    "\u063A": "gh",
    "\u0641": "f",
    "\u0642": "q",
    "\u06A9": "k",
    "\u06AF": "g",
    "\u0644": "l",
    "\u0645": "m",
    "\u0646": "n",
    "\u06BA": "n",  # noon ghunna
    "\u0648": "w",
    "\u06C1": "h",
    "\u0647": "h",
    "\u06BE": "h",
    "\u0621": "",
    "\u06CC": "y",
    "\u06D2": "e",
    "\u0626": "y",
    "\u0649": "a",
}


URDU_WORD_MAP = {
    "yhan": "yahan",
    "pr": "par",
    "wh": "wo",
    "dykh": "dekh",
    "rhy": "rahe",
    "rhe": "rahe",
    "kh": "ke",
    "hm": "hum",
    "mnwsh": "manush",
}


WORD_MAP = {
    "prmyshwr": "parmeshwar",
    "parameshvara": "parmeshwar",
    "khe": "keh",
    "kaha": "keh",
    "rha": "raha",
    "hum": "hum",
    "hyn": "hain",
    "apny": "apne",
    "myn": "mein",
    "pyta": "pita",
    "pita": "pita",
    "rhtma": "atma",
    "swrwb": "swaroop",
    "loga": "log",
    "aura": "aur",
    "karate": "karte",
}


ENGLISH_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_.+-]*\b")


class HinglishRewriter:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("HINGLISH_REWRITE_MODEL", "gpt-4o-mini")
        self.enabled = bool(self.api_key)
        self._cache: dict[str, str] = {}
        if not self.api_key:
            print("WARNING: OPENAI_API_KEY not found. Falling back to transliteration mode.")

    def rewrite_line(self, text: str) -> str:
        if not text.strip():
            return text

        cached = self._cache.get(text)
        if cached is not None:
            return cached

        masked_text, token_map = self._mask_english_words(text)

        rewritten = None
        if self.enabled:
            rewritten = self._rewrite_with_llm(masked_text)

        if not rewritten:
            rewritten = _fallback_rewrite_line(masked_text)

        rewritten = self._unmask_english_words(rewritten, token_map)
        rewritten = self._remove_leftover_placeholders(rewritten)
        rewritten = _collapse_spaces_keep_newlines(rewritten)
        rewritten = _capitalize_first_letter(rewritten)

        self._cache[text] = rewritten
        return rewritten

    def _rewrite_with_llm(self, text: str) -> str | None:
        prompt = (
            "Convert the following Hindi or Urdu sentence into natural Hinglish written in Roman script. "
            "Keep all tokens like [[EN_0]], [[EN_1]] exactly unchanged. "
            "Keep punctuation unchanged and do not change meaning. "
            "Return only the rewritten sentence, no explanation.\n\n"
            f"Input: {text}"
        )

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You rewrite subtitles into clean, readable Hinglish in Roman script.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            return content.strip('"')
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError):
            return None

    @staticmethod
    def _mask_english_words(text: str) -> tuple[str, dict[str, str]]:
        token_map: dict[str, str] = {}
        counter = 0

        def repl(match: re.Match[str]) -> str:
            nonlocal counter
            original = match.group(0)
            token = f"[[EN_{counter}]]"
            token_map[token] = original
            counter += 1
            return token

        return ENGLISH_WORD_RE.sub(repl, text), token_map

    @staticmethod
    def _unmask_english_words(text: str, token_map: dict[str, str]) -> str:
        restored = text
        for token, original in token_map.items():
            restored = restored.replace(token, original)
            restored = restored.replace(token.lower(), original)
            restored = restored.replace(token.upper(), original)
        return restored

    @staticmethod
    def _remove_leftover_placeholders(text: str) -> str:
        cleaned = re.sub(r"\[\[\s*EN[_\-\s]?\d+\s*\]\]", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[\s*EN[_\-\s]?\d+\s*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bEN[_\-\s]?\d+\b", "", cleaned, flags=re.IGNORECASE)
        return cleaned


def _capitalize_first_letter(text: str) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.upper()
            break
    return "".join(chars)


def _apply_word_map(text: str) -> str:
    # Replace only alpha tokens so punctuation/spacing remain unchanged.
    def replace_word(match: re.Match[str]) -> str:
        word = match.group(0)
        return WORD_MAP.get(word.lower(), word.lower())

    return ROMAN_WORD_RE.sub(replace_word, text)


def _collapse_spaces_keep_newlines(text: str) -> str:
    lines = text.splitlines() or [text]
    cleaned = [re.sub(r"\s+", " ", line).strip() for line in lines]
    return "\n".join(cleaned).strip()


def _normalize_roman(text: str) -> str:
    text = text.replace("M", "n").replace("N", "n")
    text = text.replace(".h", "h")
    return text.strip()


def _romanize_urdu_line(text: str) -> str:
    # Remove Arabic diacritics/tatweel that do not help Roman subtitle readability.
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)

    out = []
    for ch in text:
        if ch in URDU_CHAR_MAP:
            out.append(URDU_CHAR_MAP[ch])
        else:
            out.append(ch)

    converted = "".join(out)
    converted = re.sub(r"\b([a-z]+)\b", lambda m: URDU_WORD_MAP.get(m.group(1), m.group(1)), converted)
    converted = _normalize_roman(converted)
    converted = _apply_word_map(converted)
    return _capitalize_first_letter(converted)


def _transliterate_line(text: str) -> str:
    converted = text

    if DEVANAGARI_RE.search(converted):
        converted = transliterate(converted, sanscript.DEVANAGARI, sanscript.ITRANS)

    if ARABIC_RE.search(converted):
        converted = _romanize_urdu_line(converted)
    else:
        converted = re.sub(r"\baja\b", "aaj", converted, flags=re.IGNORECASE)
        converted = re.sub(r"\bhama\b", "hum", converted, flags=re.IGNORECASE)
        converted = re.sub(r"\bbata\b", "baat", converted, flags=re.IGNORECASE)
        converted = _normalize_roman(converted)
        converted = _apply_word_map(converted)
        converted = _capitalize_first_letter(converted)

    return converted


def _fallback_rewrite_line(text: str) -> str:
    # Deterministic fallback for environments without an LLM API key.
    return _transliterate_line(text)


def convert_srt_to_hinglish(input_srt: Path, output_srt: Path) -> None:
    """Convert Hindi/Urdu SRT text to Roman Hinglish while preserving timestamps."""
    source = input_srt.read_text(encoding="utf-8", errors="ignore")
    subtitles = list(srtlib.parse(source))
    rewriter = HinglishRewriter()

    converted_subtitles = []
    for subtitle in subtitles:
        lines = subtitle.content.splitlines() or [subtitle.content]
        converted_content = "\n".join(
            rewriter.rewrite_line(line) if line.strip() else line
            for line in lines
        ).strip()

        converted_subtitles.append(
            srtlib.Subtitle(
                index=subtitle.index,
                start=subtitle.start,
                end=subtitle.end,
                content=converted_content,
            )
        )

    output_srt.parent.mkdir(parents=True, exist_ok=True)
    output_srt.write_text(srtlib.compose(converted_subtitles), encoding="utf-8")