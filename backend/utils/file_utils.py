class VideoUtils:
    @staticmethod
    def sanitize_mp4_filename(name: str, default: str = "clip.mp4") -> str:
        """Sanitizes a string to be a valid filename and ensures it ends with .mp4."""
        name = (name or default).strip()
        if not name.lower().endswith(".mp4"):
            name += ".mp4"
        for bad in r'<>:"/|?*':
            name = name.replace(bad, "_")
        return name
