"""Fetch only the required binary member from the public UCI archive."""
import io
from pathlib import Path
import urllib.request
import zipfile

from .schema import FILENAME

SOURCE_URL = "https://archive.ics.uci.edu/static/public/891/cdc+diabetes+health+indicators.zip"


def download_source(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == FILENAME]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {FILENAME}; found {len(matches)}")
        contents = archive.read(matches[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(contents)
    return destination
