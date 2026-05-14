import re
from config import VIDEOS_DIR

MAX_TITLE_LENGTH = 40


def sanitize_title(title: str) -> str:
    title = re.sub(r'[<>:"/\\|?*]', "", title)  # Windows safe
    title = title.strip()
    return title[:MAX_TITLE_LENGTH]


def create_download_folder_from_title(title: str):
    safe_title = sanitize_title(title)
    folder = VIDEOS_DIR / safe_title
    folder.mkdir(parents=True, exist_ok=True)
    return folder
