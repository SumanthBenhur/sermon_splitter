from datetime import timedelta
from pathlib import Path

import srt as srtlib


def refit_srt(
    in_srt_path: Path,
    out_srt_path: Path,
    max_chars: int = 30,
    max_duration: float = 4.0,
    max_lines: int = 2,
    min_duration: float = 1.0,
    safety_gap: float = 0.05,
) -> None:
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

        vis_chunks = chunk_text_to_lines(text, max_chars=max_chars, max_lines=max_lines)
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

                    sub_text = "\n".join(greedy_wrap(sub_words, max_chars)[:max_lines])
                    if not sub_text.strip():
                        continue
                    sub_chunks_text.append(sub_text)

                # Then, calculate total characters of the new sub-chunks
                total_sub_chars = sum(len(s.replace("\n", " ")) for s in sub_chunks_text) or 1

                # Finally, distribute the original estimated duration proportionally
                for sub_text in sub_chunks_text:
                    sub_chars = len(sub_text.replace("\n", " "))
                    proportional_sub_dur = est_dur * (sub_chars / total_sub_chars)
                    final_chunks.append({"text": sub_text, "duration": proportional_sub_dur})
            else:
                final_chunks.append({"text": ch, "duration": est_dur})

        # Create SRT items from the final chunks
        for chunk in final_chunks:
            seg_dur = max(min_duration, chunk["duration"])
            st = cur_start
            en = st + seg_dur

            new_items.append(
                srtlib.Subtitle(
                    index=idx,
                    start=td(st),
                    end=td(en),
                    content=chunk["text"],
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
                current_sub.end = current_sub.start + timedelta(seconds=min_duration)

    # Re-index cleanly
    for i, it in enumerate(new_items, 1):
        it.index = i

    out_srt_path.write_text(srtlib.compose(new_items), encoding="utf-8")
    print(
        f"[OK] Refitted SRT -> {out_srt_path}  (max_chars={max_chars}, max_lines={max_lines}, max_duration={max_duration}s)"
    )
