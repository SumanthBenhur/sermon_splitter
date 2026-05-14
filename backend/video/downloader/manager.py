from utils.folder_utils import create_download_folder_from_title
from utils.metadata_utils import (
    get_metadata_with_pytube,
    get_metadata_with_ytdlp,
)
from .pytube_downloader import download_with_pytube
from .ytdlp_downloader import download_with_ytdlp


def download_video(url):
    try:
        print("Attempting download with Pytube...")
        title, duration = get_metadata_with_pytube(url)
        folder = create_download_folder_from_title(title)

        download_with_pytube(url, folder)
        print("Pytube download successful!")
        return folder, "pytube"

    except Exception as e:
        print(f"Pytube failed with error: {e}. Falling back to yt-dlp...")
        title, duration = get_metadata_with_ytdlp(url)
        folder = create_download_folder_from_title(title)

        download_with_ytdlp(url, folder)
        return folder, "yt-dlp"
