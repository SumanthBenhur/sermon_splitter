from faster_whisper import WhisperModel
from pathlib import Path
from datetime import timedelta
import csv
import srt

"""This Module deals with transcribing audio files and writing it to a srt file (or a csv file)

    Note -: In the Example Listener var name of the Object Instance of Transcriber class (Listener = Transcriber("tiny"))
"""


class Transcriber:
    def __init__(
        self, size: str = "small", device: str = "auto", compute_type: str = "auto"
    ):
        """
        This Function is called while creating the Transcriber

        :param size: This Parameter lets choose which model to load e.g. - "base","medium","small","tiny","large-v3"
        :param device: This parameter lets you choose which device to use ("cpu","cuda")
        :type device: str
        :param compute_type: This Parameter allows you to change compute_size (basically) (int8 for CPU, float16 for GPU)
        :type compute_type: str
        """

        # print("Loading up the Model")
        self.model = WhisperModel(size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        path: str,
        log_progress: bool = True,
        language: str = None,
        chunk_length: int = 10,
        no_repeat: int = 0,
        initial_prompt: str = None,
        vad_filter: bool = True,
    ):
        """
        The Function takes the model then transribe the audio using the model and returns a iterable with
        transription (including info like confidence).

        :param path: The Input Path of the audio file
        :type path: str
        :param log_progress: (optional) Toggle logging while transcribing (It will appear will exporting it)
        :type log_progress: bool
        :param language: (optional) The Language used by the speaker in the audio example : English = "en"
        :type language: str
        :param chunk_length: This parameter controls the length of audio transcribed per segment (5 -> 5 seconds per segment)
        :type chunk_length: int
        :param no_repeat: This parameter controls the repetition error (0 is disabled). Use it when changing chunk
        :type no_repeat: int
        Example:
            Simply returns the iterable/segments
            Listener.transcribe("inputfile.mp3")

            You get a progress bar while exporting
            Listener.transcribe("inputfile.mp3",log_progress = True)

            Now Transcription in that language (not translation)
            Listener.transcribe("inputfile.mp3",log_progress = True, language="en")

            Now the segments will be longer (while no_repeat prevents from repetition)
            Listener.transcribe("inputfile.mp3",chunk_length =7,no_repeat=3)
        """
        segments, info = self.model.transcribe(
            path,
            language=language,
            log_progress=log_progress,
            chunk_length=chunk_length,
            no_repeat_ngram_size=no_repeat,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
        )

        return segments

    def export_to_file(
        self, segments, outputname: str, filetype: str = ".srt", overwrite: bool = False
    ):
        """
        This Functions takes the iterable from the transcribe write it on  file (srt or csv)

        :param segments: The Iterable file which is returned by the transcription function
        :param outputname: The name of the new file along with the path
        :param overwrite: This Parameter lets you overwrite files
        Example:
            Exports a file v1-transcribe.srt in current directory
            Listener.export_to_file(segments,"v1-transcribe")

            Exports a file v1-transcribe.csv in specified directory
            Listener.export_to_srt(segments,"experimental/v1-transcribe",filetype=".csv")

            Exports a file v1-transcribe.srt using abs path
            Listener.export_to_srt(segments,"C:/Imports/translation")

        """
        path = Path(outputname).with_suffix(filetype)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path.name} already exists")
        if filetype == ".srt":
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
            path.write_text(srt_content, encoding="utf-8")
        elif filetype == ".csv":
            with path.open("w") as file:
                writer = csv.writer(file)
                for segment in segments:
                    writer.writerow((segment.start, segment.end, segment.text))
        else:
            raise ValueError("filetype parameter only supports '.srt' and '.csv'")
