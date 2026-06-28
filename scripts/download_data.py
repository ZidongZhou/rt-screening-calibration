"""Download the five public Zenodo CSV files and verify their MD5 checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


RECORD_ID = "10423537"
EXPECTED = {
    "demographic.csv": "7a567dc5a9cf1c6a7ea2b70f166dbb57",
    "gad7.csv": "7025495755569edd7dd5561a706178fb",
    "isi.csv": "b9976f71c4db473dd31e4b12b40cf592",
    "phq9.csv": "585000bbbec153efde405b3442eda479",
    "pss.csv": "91d7c288095213d50f62adcfad15117e",
}
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    request = Request(
        f"https://zenodo.org/api/records/{RECORD_ID}",
        headers={"User-Agent": "rt-screening-reproduction/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        record = json.load(response)
    links = {entry["key"].lower(): entry["links"]["content"] for entry in record["files"]}
    DEST.mkdir(parents=True, exist_ok=True)
    for name, checksum in EXPECTED.items():
        if name not in links:
            raise RuntimeError(f"{name} is absent from Zenodo record {RECORD_ID}")
        path = DEST / name
        if not path.exists() or md5(path) != checksum:
            print(f"Downloading {name}...")
            req = Request(links[name], headers={"User-Agent": "rt-screening-reproduction/1.0"})
            with urlopen(req, timeout=180) as response, path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        actual = md5(path)
        if actual != checksum:
            raise RuntimeError(f"Checksum mismatch for {name}: {actual} != {checksum}")
        print(f"Verified {name}: {actual}")


if __name__ == "__main__":
    main()
