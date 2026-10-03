"""Extract only TreeFinder GeoTIFF tiles, ignoring macOS metadata in the ZIP."""

from pathlib import Path
import shutil
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "datasets" / "tiles224_v3.zip"
DEST = ROOT / "datasets" / "tiles224_v3"
CSV = ROOT / "datasets" / "tile_info224_v3.csv"


def main() -> None:
    expected = set(pd.read_csv(CSV).FileName)
    DEST.mkdir(parents=True, exist_ok=True)
    extracted = set()
    with zipfile.ZipFile(ZIP) as archive:
        for info in archive.infolist():
            parts = Path(info.filename.replace("\\", "/")).parts
            if len(parts) != 3 or parts[:2] != ("tiles224_v3", "tiles224_v3"):
                continue
            name = parts[2]
            if name not in expected:
                continue
            target = DEST / name
            if not target.exists() or target.stat().st_size != info.file_size:
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.add(name)
    missing = expected - extracted
    assert not missing, f"Missing {len(missing)} tiles, e.g. {sorted(missing)[:5]}"
    print(f"Validated {len(extracted)} tiles in {DEST}")


if __name__ == "__main__":
    main()
