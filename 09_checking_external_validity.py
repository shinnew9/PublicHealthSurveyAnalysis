"""
TEDS-A data preparation (Section 8, external validity check).

Extracts the TEDS-A Stata zip files (2021-2023) and converts them to CSV,
producing the input files that 14_teds_subgroup_validation.py reads from
TEDS_A_2021_2023_csv/.
"""
import os
import zipfile
from pathlib import Path

import pandas as pd


def extract_selected_teds_stata_zips(base_dir, overwrite=False):
    """Extract the TEDS-A Stata zip files for 2021-2023."""
    base_dir = Path(base_dir)

    zip_files = {
        2021: "TEDS-A-2021-DS0001-bndl-data-stata_v3.zip",
        2022: "TEDS-A-2022-DS0001-bndl-data-stata_v2.zip",
        2023: "teds-a-2023-ds0001-bndl-data-stata_v1.zip",
    }

    summary = {}
    for year, zip_name in zip_files.items():
        zip_path = base_dir / zip_name
        extract_dir = base_dir / f"TEDS_A_{year}_stata_extracted"
        print(f"\n====== {year} ======")

        if not zip_path.exists():
            print(f"[ERROR] Zip file not found: {zip_path}")
            summary[year] = {
                "status": "missing_zip", "zip_path": zip_path, "extract_dir": extract_dir,
                "dta_files": [], "do_files": [], "all_files": [],
            }
            continue

        if extract_dir.exists() and not overwrite:
            print(f"[INFO] Extract folder already exists, skipping extraction: {extract_dir.resolve()}")
        else:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"[OK] Extracted to: {extract_dir.resolve()}")

        all_files = sorted([p for p in extract_dir.glob("*") if p.is_file()])
        dta_files = sorted([p for p in all_files if p.suffix.lower() == ".dta"])
        do_files = sorted([p for p in all_files if p.suffix.lower() == ".do"])

        print(f"[INFO] Total files: {len(all_files)}")
        print(f"[INFO] DTA files: {len(dta_files)}")
        for f in dta_files:
            print(" -", f)
        print(f"[INFO] DO files: {len(do_files)}")
        for f in do_files:
            print(" -", f)

        summary[year] = {
            "status": "ok", "zip_path": zip_path, "extract_dir": extract_dir,
            "dta_files": dta_files, "do_files": do_files, "all_files": all_files,
        }

    return summary


def save_teds_dta_as_csv(summary, output_dir):
    """Convert extracted TEDS-A .dta files to year-specific CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_summary = {}
    for year, info in summary.items():
        print(f"\n========== {year} ==========")
        dta_files = info.get("dta_files", [])
        if len(dta_files) == 0:
            print(f"[WARNING] No .dta file found for {year}")
            csv_summary[year] = {"status": "no_dta", "csv_path": None}
            continue

        dta_path = dta_files[0]
        print(f"[INFO] Reading DTA file: {dta_path}")
        df = pd.read_stata(dta_path)
        print(f"[INFO] Data shape: {df.shape}")

        csv_path = output_dir / f"TEDS_A_{year}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved CSV: {csv_path}")

        csv_summary[year] = {"status": "ok", "dta_path": dta_path, "csv_path": csv_path, "shape": df.shape}

    return csv_summary


if __name__ == "__main__":
    BASE_DIR = Path(os.environ.get("NSDUH_DATA_DIR", "."))

    summary = extract_selected_teds_stata_zips(base_dir=BASE_DIR, overwrite=False)
    csv_summary = save_teds_dta_as_csv(summary, output_dir=BASE_DIR / "TEDS_A_2021_2023_csv")

    for year, info in csv_summary.items():
        print(year, info)

    for year, info in csv_summary.items():
        if info["status"] == "ok":
            df_test = pd.read_csv(info["csv_path"], nrows=5)
            print(f"\n========== {year} ==========")
            print(df_test.shape)
            print(df_test.head())
