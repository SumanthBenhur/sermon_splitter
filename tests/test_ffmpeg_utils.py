import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# 1. Add the project root directory to the python path
# This allows us to see the 'backend' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. Import from the correct location
from backend.utils import ffmpeg_utils


class TestFfmpegUtils(unittest.TestCase):
    # NOTE: We patch 'backend.utils.ffmpeg_utils.ffmpeg' because that is where
    # the ffmpeg module is imported IN YOUR CODE.
    @patch("backend.utils.ffmpeg_utils.ffmpeg")
    def test_import_video_success(self, mock_ffmpeg):
        """
        Test that import_video calls ffmpeg.input with the correct path and format.
        """
        # Setup
        video_path = "videos/test_video.mp4"

        # Call the function
        result = ffmpeg_utils.import_video(video_path)

        # Assertions
        mock_ffmpeg.input.assert_called_once_with(video_path)
        self.assertEqual(result, mock_ffmpeg.input.return_value)

    @patch("backend.utils.ffmpeg_utils.ffmpeg")
    def test_export_video_success(self, mock_ffmpeg):
        """
        Test that export_video calls ffmpeg.output correctly.
        """
        # Setup
        mock_file_stream = (
            MagicMock()
        )  # This would contain the file with Audio and video
        output_path = "videos/Exports/output.mp4"

        mock_output_stream = MagicMock()
        mock_ffmpeg.output.return_value = mock_output_stream
        # Call the function
        ffmpeg_utils.export_video(mock_file_stream, output_path)

        # Assertions
        # Ensures Video is Exported to Correct Path
        mock_ffmpeg.output.assert_called_once_with(mock_file_stream, output_path)

        # Ensuring the Video is Exported
        mock_ffmpeg.run.assert_called_once_with(mock_output_stream)


if __name__ == "__main__":
    unittest.main()
