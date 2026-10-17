import io
import zipfile
from unittest.mock import patch
import pytest
from diabetes_screening.download import download_source
from diabetes_screening.schema import FILENAME


def archive_bytes(names):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, "csv content")
    buffer.seek(0)
    return buffer


def test_download_selects_exact_member(tmp_path):
    with patch("urllib.request.urlopen", return_value=archive_bytes(["nested/" + FILENAME, "other.csv"])):
        path = download_source(tmp_path / "raw" / FILENAME)
    assert path.read_text() == "csv content"
    with pytest.raises(FileExistsError):
        download_source(path)


@pytest.mark.parametrize("names", [["wrong.csv"], [FILENAME, "nested/" + FILENAME]])
def test_wrong_archive_is_rejected(names, tmp_path):
    with patch("urllib.request.urlopen", return_value=archive_bytes(names)):
        with pytest.raises(ValueError, match="exactly one"):
            download_source(tmp_path / FILENAME)
