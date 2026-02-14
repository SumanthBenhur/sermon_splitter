from faster_whisper import WhisperModel
from pathlib import Path
from datetime import timedelta
import csv
import srt

"""This Module deals with transcribing audio files and writing it to a srt file (or a csv file)

    Note -: In the Example Listener var name of the Object Instance of Transcriber class (Listener = Transriber("tiny"))
"""


class Transcriber:
    # def pick_model(self):
    #     models = ["large-v3","base","medium","small","tiny"]
    def __init__(self, size="small", device="auto"):
        """
        This Function is called while creating the Transcriber

        :param self: Description
        :param size: Description
        :param device: Description
        """

        print("Loading up the Model")
        self.model = WhisperModel(size, device=device)

    def transcribe(
        self,
        path: str,
        log: bool = True,
        language: str = None,
    ):
        """
        The Function takes the model then transribe the audio using the model and returns a iterable with
        transription (including info like confidence).

        :param path: The Input Path of the audio file
        :type path: str
        :param log: (optional) Toggle logging while transcribing (It will appear will exporting it)
        :type log: bool
        :param language: (optional) The Language used by the speaker in the audio example : English = "en"
        :type language: str

        Example:
            Listener.transcribe("inputfile.mp3") #Simply returns the iterable/segments
            Listener.transcribe("inputfile.mp3",log = True) #You get a progress bas while exporting
            Listener.transcribe("inputfile.mp3",log = True, language="en") #Now Transcription in that language (not translation)
        """
        segments, info = self.model.transcribe(
            path, language=language, log_progress=log
        )

        return segments

    def export_to_csv(self, segments, OutputName: str):
        """
        This Functions takes the iterable from the transcribe write it on a csv file

        :param segments: The Iterable file which is returned by the transcription function
        :param OutputName: The name of the new file along with the path (without the extension)

        Example:
            Exports a file v1-transcribe.csv in current directory
            Listener.export_to_csv(segments,"v1-transcribe")

            Exports a file v1-transcribe.csv in specified directory
            Listener.export_to_csv(segments,"experimental/v1-transcribe")

            Exports a file v1-transcribe.csv using abs path
            Listener.export_to_csv(segments,"C:/Imports/translation")
        """
        csv_path = Path(OutputName).with_suffix(".csv")
        with csv_path.open("x") as file:
            Writer = csv.writer(file)
            for segment in segments:
                Writer.writerow((segment.start, segment.end, segment.text))

    def export_to_srt(self, segments, name):
        """
        This Functions takes the iterable from the transcribe write it on a srt file

        :param segments: The Iterable file which is returned by the transcription function
        :param OutputName: The name of the new file along with the path

        Example:
            Exports a file v1-transcribe.csv in current directory
            Listener.export_to_srt(segments,"v1-transcribe")

            Exports a file v1-transcribe.csv in specified directory
            Listener.export_to_srt(segments,"experimental/v1-transcribe")

            Exports a file v1-transcribe.csv using abs path
            Listener.export_to_srt(segments,"C:/Imports/translation")

        """

        subs = []
        srt_path = Path(name).with_suffix(".srt")
        for segment in segments:
            subs.append(
                srt.Subtitle(
                    index=segment.id,
                    start=timedelta(seconds=segment.start),
                    end=timedelta(seconds=segment.end),
                    content=segment.text,
                )
            )
        srt_content = srt.compose(subs)
        srt_path.write_text(srt_content, encoding="utf-8")


if __name__ == "__main__":
    Buck = Transcriber("tiny", "cuda")
    Text = Buck.transcribe("experimental/Isaac/sermon.mp3", language="en", log=True)
    Buck.export_to_srt(Text, "C:/Imports/translation")
    # Text = Buck.transcribe("experimental/Isaac/sermon.mp3", language="en",log=True)
    # Buck.export_to_csv(Text,"experimental/Isaac/translation")
