# Add test cases for the FFmpegUtils class

import pytest
from unittest.mock import Mock
from ffmpeg_utils import Ffmeg

# Assuming your function is in a file named 'media_utils.py'


class TestTrimFile:
    def test_trim_file_with_explicit_audio(self):
        # The Setup
        mock_video_component = Mock(name="video_component")
        mock_audio_component = Mock(name="audio_component")

        # The Assertions
        output_video = Mock(name="trimmed_video")
        output_audio = Mock(name="trimmed_audio")
        mock_audio_component.filter.return_value = output_audio
        mock_video_component.filter.return_value = output_video

        starttime, endtime = "00:00:00", "00:00:10"

        # Call the Function and test
        result_audio, result_video = Ffmeg.trim_file(
            starttime=starttime,
            endtime=endtime,
            video=mock_video_component,
            audio=mock_audio_component,
        )
        # Making Sure the Function were Called Properly
        mock_audio_component.filter.assert_called_once_with(
            "atrim", start=starttime, end=endtime
        )
        mock_video_component.filter.assert_called_once_with(
            "trim", start=starttime, end=endtime
        )

        # Checking the Output
        assert result_audio == output_audio
        assert result_video == output_video

    def test_trim_file_without_explicit_audio(self):
        """
        Test the scenario where only the main Video file is provided,
        and audio must be extracted from it.
        """
        # 1. Setup Mocks
        mock_main_file = Mock()

        # The function accesses mock_main_file.video and mock_main_file.audio
        mock_video_component = Mock()
        mock_audio_component = Mock()

        mock_main_file.video = mock_video_component
        mock_main_file.audio = mock_audio_component

        # Setup expected returns for the filter calls
        expected_video_out = Mock(name="trimmed_video_stream")
        expected_audio_out = Mock(name="trimmed_audio_stream")

        mock_video_component.filter.return_value = expected_video_out
        mock_audio_component.filter.return_value = expected_audio_out

        start_time = 5
        end_time = 15

        # 2. Call the function (No Audio arg provided)
        result_audio, result_video = Ffmeg.trim_file(
            video=mock_main_file, starttime=start_time, endtime=end_time
        )

        # 3. Assertions
        # Ensure filter was called on the .video component
        mock_video_component.filter.assert_called_once_with(
            "trim", start=start_time, end=end_time
        )

        # Ensure filter was called on the .audio component
        mock_audio_component.filter.assert_called_once_with(
            "atrim", start=start_time, end=end_time
        )

        # Validate return values
        assert result_video == expected_video_out
        assert result_audio == expected_audio_out


pytest.main()
