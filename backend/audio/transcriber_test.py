import pytest
from unittest.mock import MagicMock, patch
from transcriber import Transcriber


@pytest.fixture
# Mocking Segments
def mock_segments():
    seg1 = MagicMock()
    seg1.id, seg1.text = 1, "Hello World"
    seg1.start, seg1.end = 0.0, 2.6

    seg2 = MagicMock()
    seg2.id, seg2.text = 2, "This is a Test"
    seg2.start, seg2.end = 2.6, 3.1

    return [seg1, seg2]


@patch("transcriber.WhisperModel")
def test_init(MockWhisperModel):
    transcriber = Transcriber(size="base", device="cpu")
    MockWhisperModel.assert_called_once_with("base", device="cpu")
    assert transcriber.model == MockWhisperModel.return_value


@patch("transcriber.WhisperModel")
def test_transcribe(MockWhisperModel, mock_segments):
    transcriber = Transcriber(size="base", device="cpu")
    mock_model_instance = MockWhisperModel.return_value
    mock_model_instance.transcribe.return_value = (mock_segments, {"language": "en"})

    transcriber = Transcriber()
    result = transcriber.transcribe(
        "dummy_audio.mp3", language="en", log=False, chunk=20, no_repeat=2
    )

    mock_model_instance.transcribe.assert_called_once_with(
        "dummy_audio.mp3",
        language="en",
        log_progress=False,
        chunk_length=20,
        no_repeat_ngram_size=2,
    )

    assert result == mock_segments


@patch("transcriber.WhisperModel")
def test_srt_export(MockWhisperModel, mock_segments, tmp_path):
    transcriber = Transcriber()
    outputbase = tmp_path / "test_output"
    transcriber.export_to_file(segments=mock_segments, outputname=outputbase)

    expected_file = tmp_path / "test_output.srt"
    assert expected_file.exists()

    content = expected_file.read_text(encoding="utf-8")
    assert "1" in content
    assert "00:00:00,000" in content
    assert "Hello World" in content
    assert "This is a Test" in content


@patch("transcriber.WhisperModel")
def test_csv_export(MockWhisperModel, mock_segments, tmp_path):
    transcriber = Transcriber()
    outputbase = tmp_path / "test_output"
    transcriber.export_to_file(
        segments=mock_segments, outputname=outputbase, filetype=".csv"
    )

    expected_file = tmp_path / "test_output.csv"
    assert expected_file.exists()

    content = expected_file.read_text(encoding="utf-8")
    assert "0.0,2.6,Hello World" in content
    assert "2.6,3.1,This is a Test" in content


@patch("transcriber.WhisperModel")
def test_invalid_filetype(MockWhisperModel, mock_segments):
    transcriber = Transcriber()

    with pytest.raises(
        ValueError, match="filetype parameter only supports '.srt' and '.csv'"
    ):
        transcriber.export_to_file(mock_segments, "dummy_output", filetype=".xyz")


@patch("transcriber.WhisperModel")
def test_overwrite(MockWhisperModel, mock_segments, tmp_path):
    transcriber = Transcriber()
    Name = "test_file"
    path = MagicMock()
    path.name = f"{Name}.srt"
    temp_file = tmp_path / f"{Name}.srt"
    temp_file.touch()
    with pytest.raises(FileExistsError, match=f"{path.name} already exists"):
        transcriber.export_to_file(mock_segments, temp_file)
