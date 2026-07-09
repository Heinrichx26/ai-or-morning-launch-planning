from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


BTS_PREZIP = "https://transtats.bts.gov/PREZIP"
ZIP_NAME = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"


def download_month(year: int, month: int, output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_name = ZIP_NAME.format(year=year, month=month)
    zip_path = output_dir / zip_name
    if not zip_path.exists() or force:
        urlretrieve(f"{BTS_PREZIP}/{zip_name}", zip_path)
    extract_dir = output_dir / "extracted" / f"{year}{month:02d}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".complete"
    if force or not marker.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        marker.write_text("ok\n", encoding="utf-8")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BTS 2025 on-time performance monthly zip files.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_us_bts"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for month in args.months:
        path = download_month(args.year, month, args.output_dir, force=args.force)
        print(path)


if __name__ == "__main__":
    main()

