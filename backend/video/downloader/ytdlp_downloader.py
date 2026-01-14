import yt_dlp


def download_with_ytdlp(url: str, output_dir):
    ydl_opts = {
        "outtmpl": str(output_dir / "%(f_id)s.%(ext)s"),
        "format": "bestvideo,bestaudio",
        "allow_unmerged_formats": True,
        "merge_output_format": None,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
