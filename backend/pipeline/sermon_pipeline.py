from pathlib import Path

from backend.audio.hindi_to_hinglish import convert_srt_to_hinglish
from backend.audio.transcriber import Transcriber
from backend.subtitles.ass_styler import convert_srt_to_styled_ass
from backend.subtitles.burner import burn_subtitles_into_video
from backend.subtitles.refitter import refit_srt
from backend.utils.ffmpeg_manager import FfmpegManager
from backend.utils.file_utils import VideoUtils
from backend.video.concatenator import concatenate_video_clips
from backend.video.cutter import cut_video_clip
from backend.video.face_tracker import create_face_tracked_vertical_video


class SermonPipeline:
    def __init__(self, source_path: str, base_work_dir: Path = Path(".")):
        self.source_path = Path(source_path)
        self.base_work_dir = base_work_dir
        self.artifacts_dir = self.source_path.parent / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        self.ffmpeg_manager = FfmpegManager()
        self.transcriber = Transcriber(self.ffmpeg_manager)
        self.video_utils = VideoUtils()

    def run(self, num_clips, clips_data, output_filename):
        """The main workflow for processing sermon clips."""
        n = num_clips

        clips = []
        for i in range(n):
            print(f"\n--- Clip {i} ---")
            start_time = clips_data[i]["start_time"]
            end_time = clips_data[i]["end_time"]
            if n == 1:
                out_name = self.video_utils.sanitize_mp4_filename(output_filename or "clip.mp4")
            else:
                out_name = self.video_utils.sanitize_mp4_filename(f"part_{i}.mp4")
            out_path = self.artifacts_dir / out_name
            cut_video_clip(self.ffmpeg_manager, self.source_path, start_time, end_time, out_path)
            print(f"Saved: {out_path}")
            clips.append(out_path)

        if len(clips) == 1:
            final_clip = clips[0]
        else:
            combo_name = self.video_utils.sanitize_mp4_filename(output_filename or "combined.mp4")
            final_clip = self.artifacts_dir / combo_name
            concatenate_video_clips(self.ffmpeg_manager, clips, final_clip)
            print(f"Concatenated file saved: {final_clip}")

        print("\nMaking vertical 1080x1920 with face tracking...")
        vert_out = final_clip.with_name(final_clip.stem + "_vertical.mp4")
        create_face_tracked_vertical_video(self.ffmpeg_manager.ffmpeg_path, str(final_clip), str(vert_out))
        if not vert_out.exists() or vert_out.stat().st_size == 0:
            raise RuntimeError("Vertical video generation failed")
        final_clip = vert_out
        print(f"Vertical saved: {final_clip}")

        print("\nAdding HINDI subtitles (Whisper) and burning them in...")
        srt_out = final_clip.with_suffix(".srt")
        ass_out = self.artifacts_dir / "temp_subtitles.ass"
        subbed_out = final_clip.with_name(final_clip.stem + "_subbed.mp4")
        self.transcriber.transcribe_video_with_whisper(final_clip, srt_out)

        print("[STEP] Refitting subtitles for better readability...")
        refit_srt_out = self.artifacts_dir / "temp_subtitles.srt"
        refit_srt(srt_out, refit_srt_out)

        print("[STEP] Converting Hindi subtitles to Hinglish...")
        hinglish_srt = self.artifacts_dir / "hinglish_subtitles.srt"
        convert_srt_to_hinglish(refit_srt_out, hinglish_srt)

        print("[STEP] Converting SRT to styled ASS...")
        convert_srt_to_styled_ass(hinglish_srt, ass_out)

        print("[STEP] Burning subtitles...")
        burn_subtitles_into_video(self.ffmpeg_manager, final_clip, ass_out, subbed_out)
        print(f"\n Done. Output: {subbed_out}")
        return subbed_out
