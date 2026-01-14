from pytube import YouTube
import yt_dlp


def get_metadata_with_pytube(url: str):
    yt = YouTube(url)
    return yt.title, yt.length


def get_metadata_with_ytdlp(url: str):
    ydl_opts = {"quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["title"], info["duration"]
