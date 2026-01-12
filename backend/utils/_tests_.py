import unittest
from unittest.mock import Mock
from ffmpeg_utils import Ffmeg


class TestFfmpegTrim(unittest.TestCase):
    def test_trim_file_with_separate_audio_and_video(self):
        """
        Tests trimming when video and audio are separate streams.
        """
        # Arrange
        mock_video = Mock()
        mock_audio = Mock()
        starttime = "00:01:00"
        endtime = "00:02:00"

        # Act
        trimmed_audio, trimmed_video = Ffmeg.trim_file(
            Video=mock_video, Starttime=starttime, Endtime=endtime, Audio=mock_audio
        )

        # Assert
        mock_video.filter.assert_called_once_with("trim", start=starttime, end=endtime)
        mock_audio.filter.assert_called_once_with("atrim", start=starttime, end=endtime)
        self.assertEqual(trimmed_video, mock_video.filter.return_value)
        self.assertEqual(trimmed_audio, mock_audio.filter.return_value)

    def test_trim_file_with_combined_video(self):
        """
        Tests trimming when video and audio are part of the same input.
        """
        # Arrange
        mock_video_combined = Mock()
        starttime = "00:03:00"
        endtime = "00:04:00"

        # Act
        trimmed_audio, trimmed_video = Ffmeg.trim_file(
            Video=mock_video_combined, Starttime=starttime, Endtime=endtime, Audio=None
        )

        # Assert
        mock_video_combined.video.filter.assert_called_once_with(
            "trim", start=starttime, end=endtime
        )
        mock_video_combined.audio.filter.assert_called_once_with(
            "atrim", start=starttime, end=endtime
        )
        self.assertEqual(trimmed_video, mock_video_combined.video.filter.return_value)
        self.assertEqual(trimmed_audio, mock_video_combined.audio.filter.return_value)


if __name__ == "__main__":
    unittest.main()
