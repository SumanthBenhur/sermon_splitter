import shutil
import subprocess

import imageio_ffmpeg


class FfmpegManager:
    def __init__(self, ffmpeg_path: str = None):
        if ffmpeg_path:
            self.ffmpeg_path = ffmpeg_path
        else:
            # Try to get FFmpeg from the system path first
            ffmpeg_executable = shutil.which("ffmpeg")
            if ffmpeg_executable:
                self.ffmpeg_path = ffmpeg_executable
                print(f"[INFO] Using FFmpeg from system PATH: {self.ffmpeg_path}")
            else:
                # Fallback to imageio-ffmpeg if not in PATH
                self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"[INFO] Using imageio-ffmpeg binary at: {self.ffmpeg_path}")

    def run_command(self, args: list):
        """Executes an FFmpeg command and raises an exception if it fails."""
        proc = subprocess.run([self.ffmpeg_path, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")
        return proc