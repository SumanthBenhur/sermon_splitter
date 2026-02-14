from faster_whisper import WhisperModel
from pathlib import Path
from datetime import timedelta
import csv
import srt

"""This Module deals with transcribing audio files and writing it to a srt file (or a csv file)

    Note -: In the Example Listener var name of the Object Instance of Transcriber class (Listener = Transriber("tiny"))
"""


class Transcriber:
    def __init__(self, size: str = "small", device: str = "auto"):
        """
        This Function is called while creating the Transcriber

        :param size: This Parameter lets choose which model to load e.g. - "base","medium","small","tiny","large-v3"
        :param device: This parameter lets you choose which device to use ("cpu","cuda")
        """

        # print("Loading up the Model")
        self.model = WhisperModel(size, device=device)

    def transcribe(
        self,
        path: str,
        log: bool = True,
        language: str = None,
        chunk: int = 10,
        no_repeat=0,
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
        :param chunk: This parameter controls the length of audio transcribed per segment (5 -> 5 seconds per segment)
        :type chunk: int
        :param no_repeat: This parameter controls the repetition error (0 is disabled). Use it when changing chunk
        :type no_repeat: int
        Example:
            Simply returns the iterable/segments
            Listener.transcribe("inputfile.mp3")

            You get a progress bar while exporting
            Listener.transcribe("inputfile.mp3",log = True)

            Now Transcription in that language (not translation)
            Listener.transcribe("inputfile.mp3",log = True, language="en")

            Now the segments will be longer (while no_repeat prevents from repetition)
            Listener.transcribe("inputfile.mp3",chunk =7,no_repeat=3)
        """
        segments, info = self.model.transcribe(
            path,
            language=language,
            log_progress=log,
            chunk_length=chunk,
            no_repeat_ngram_size=no_repeat,
        )

        return segments

    def export_to_file(self, segments, outputname: str, filetype: str = ".srt"):
        """
        This Functions takes the iterable from the transcribe write it on  file (srt or csv)

        :param segments: The Iterable file which is returned by the transcription function
        :param outputname: The name of the new file along with the path

        Example:
            Exports a file v1-transcribe.srt in current directory
            Listener.export_to_file(segments,"v1-transcribe")

            Exports a file v1-transcribe.csv in specified directory
            Listener.export_to_srt(segments,"experimental/v1-transcribe",filetype=".csv")

            Exports a file v1-transcribe.srt using abs path
            Listener.export_to_srt(segments,"C:/Imports/translation")

        """

        if filetype == ".srt":
            srt_path = Path(outputname).with_suffix(filetype)
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
            srt_content = srt.compose(subs)
            srt_path.write_text(srt_content, encoding="utf-8")
        elif filetype == ".csv":
            csv_path = Path(outputname).with_suffix(filetype)
            with csv_path.open("x") as file:
                Writer = csv.writer(file)
                for segment in segments:
                    Writer.writerow((segment.start, segment.end, segment.text))
        else:
            raise ValueError("filetype parameter only supports '.srt' and '.csv'")
