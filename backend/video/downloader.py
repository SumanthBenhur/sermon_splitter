from pathlib import Path
import re

from pytubefix import YouTube


def download_video(url: str) -> Path:
	"""
	Downloads a YouTube video and returns the path to the downloaded file.
	It downloads a progressive stream (video + audio).
	The file is saved in 'videos/youtube_downloads/'.
	"""
	yt = YouTube(url)
	print(f"Downloading video from URL: {url}")
	print(f"Video Title: {yt.title}")

	stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
	if not stream:
		raise RuntimeError("No progressive mp4 stream found for this video.")

	download_dir = Path("videos/youtube_downloads")
	download_dir.mkdir(parents=True, exist_ok=True)

	# Sanitize filename from video title
	title = yt.title
	sanitized_title = re.sub(r"[^\w\s-]", "", title).strip()
	sanitized_title = re.sub(r"[-\s]+", "-", sanitized_title)
	output_filename = f"{sanitized_title}.mp4"

	output_path = download_dir / output_filename

	print(f"Selected stream: {stream}")
	print(f"Downloading to: {output_path}...")

	stream.download(output_path=str(download_dir), filename=output_filename)

	print("Download complete.")
	return output_path.resolve()
