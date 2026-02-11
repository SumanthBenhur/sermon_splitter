from faster_whisper import WhisperModel
from pathlib import Path
from datetime import timedelta
import csv
import srt


class Transcriber:
    # def pick_model(self):
    #     models = ["large-v3","base","medium","small","tiny"]
    def __init__(self, size, device="auto"):
        print("Loading up the Model")
        self.model = WhisperModel(size, device=device)

    def transcribe(
        self,
        path: str,
        log: bool = True,
        language: str = None,
    ):
        segments, info = self.model.transcribe(
            path, language=language, log_progress=log
        )

        return segments

    def export_to_csv(self, segments, name):
        filename = name + ".csv"
        with open(filename, "x") as file:
            Writer = csv.writer(file)
            for segment in segments:
                Writer.writerow((segment.start, segment.end, segment.text))

    def export_to_srt(self, segments, name):
        subs = []
        for segment in segments:
            subs.append(
                srt.Subtitle(
                    index=segment.id,
                    start=timedelta(seconds=segment.start),
                    end=timedelta(seconds=segment.end),
                    content=segment.text,
                )
            )
        srt_path = Path(f"{name}.srt")
        srt_content = srt.compose(subs)
        srt_path.write_text(srt_content, encoding="utf-8")


if __name__ == "__main__":
    Buck = Transcriber("tiny", "cuda")
    Text = Buck.transcribe("experimental\Isaac\sermon.mp3", language="en")
    Buck.export_to_srt(Text, "translation")
