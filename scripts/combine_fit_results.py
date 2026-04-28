#!/usr/bin/env python3
"""Combine per-spectrum fit outputs into master and summary tables."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fit_pipeline_common import (
    DEFAULT_BASE_DIR,
    DEFAULT_OUTPUT_SUBDIR,
    DEFAULT_RUN_LABEL,
    DEFAULT_SNAP,
    LINE_MAP,
    RESULT_FIELDNAMES,
    STATUS_FIELDNAMES,
    delimiter_for_path,
    output_root,
    read_sid_file,
    write_dict_rows,
)


def optional_pandas():
    try:
        import pandas as pd  # noqa: PLC0415

        return pd
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-spectrum fit CSVs and summarize pipeline status."
    )
    parser.add_argument("--snap", type=int, default=DEFAULT_SNAP)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--sid-file", default=None)
    parser.add_argument(
        "--output",
        default=os.path.join(
            DEFAULT_BASE_DIR,
            "combined_fit_results_snap99_L4Rvir_snr10_bin3.csv",
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=os.path.join(
            DEFAULT_BASE_DIR,
            "combined_fit_summary_snap99_L4Rvir_snr10_bin3.csv",
        ),
    )
    return parser.parse_args()


def candidate_sids(args: argparse.Namespace) -> List[int]:
    if args.sid_file:
        return read_sid_file(args.sid_file)

    pattern = os.path.join(args.base_dir, "sid*")
    sids: List[int] = []
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        if base.startswith("sid") and base[3:].isdigit():
            sids.append(int(base[3:]))
    return sorted(sids)


def find_result_files(args: argparse.Namespace) -> List[str]:
    files: List[str] = []
    for sid in candidate_sids(args):
        root = output_root(args.base_dir, sid, args.snap, args.run_label, args.output_subdir)
        pattern = os.path.join(root, "per_spectrum", "*", "alpha*", "*_fits.csv")
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def find_status_files(args: argparse.Namespace) -> List[str]:
    files: List[str] = []
    for sid in candidate_sids(args):
        root = output_root(args.base_dir, sid, args.snap, args.run_label, args.output_subdir)
        pattern = os.path.join(root, "logs", "per_spectrum", "*", "alpha*", "*_status.csv")
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter_for_path(path)))


def read_many_csvs(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for row in read_csv_rows(path):
            rows.append(row)
    return rows


def to_int(value: Any, default: int = -1) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def write_master_outputs(args: argparse.Namespace, result_files: List[str]) -> None:
    pd = optional_pandas()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if pd is None:
        rows = read_many_csvs(result_files)
        write_dict_rows(args.output, rows, RESULT_FIELDNAMES)
        print("pandas is unavailable; wrote CSV only.")
        return

    if result_files:
        frames = [pd.read_csv(path) for path in result_files]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=RESULT_FIELDNAMES)

    df.to_csv(args.output, index=False)
    print(f"Wrote combined CSV: {args.output}")

    parquet_path = os.path.splitext(args.output)[0] + ".parquet"
    hdf5_path = os.path.splitext(args.output)[0] + ".h5"

    wrote_binary = False
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Wrote combined Parquet: {parquet_path}")
        wrote_binary = True
    except Exception as exc:
        print(f"[WARN] Could not write Parquet: {exc}")

    try:
        df.to_hdf(hdf5_path, key="fits", mode="w")
        print(f"Wrote combined HDF5: {hdf5_path}")
        wrote_binary = True
    except Exception as exc:
        print(f"[WARN] Could not write HDF5: {exc}")

    if not wrote_binary:
        print("[WARN] No binary combined table was written; optional dependencies are missing.")


def make_summary(result_rows: List[Dict[str, Any]], status_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, int, str, str, int], Dict[str, Any]] = {}

    for row in status_rows:
        key = (
            to_int(row.get("SID")),
            to_int(row.get("snap")),
            str(row.get("run_label", "")),
            str(row.get("mode", "")),
            to_int(row.get("alpha")),
        )
        groups.setdefault(key, {"status_rows": [], "result_rows": []})["status_rows"].append(row)

    for row in result_rows:
        key = (
            to_int(row.get("SID")),
            to_int(row.get("snap")),
            str(row.get("run_label", "")),
            str(row.get("mode", "")),
            to_int(row.get("alpha")),
        )
        groups.setdefault(key, {"status_rows": [], "result_rows": []})["result_rows"].append(row)

    summary: List[Dict[str, Any]] = []
    success_statuses = {"succeeded", "succeeded_with_line_errors"}

    for key in sorted(groups):
        sid, snap, run_label, mode, alpha = key
        status_group = groups[key]["status_rows"]
        result_group = groups[key]["result_rows"]

        attempted = len(status_group) if status_group else 0
        succeeded = sum(1 for row in status_group if str(row.get("status", "")) in success_statuses)
        skipped = sum(1 for row in status_group if str(row.get("status", "")) == "skipped")
        failed = sum(1 for row in status_group if str(row.get("status", "")) == "failed")

        for line_label in LINE_MAP.keys():
            line_rows = [row for row in result_group if str(row.get("saved_line_label", "")) == line_label]
            line_sources = {row.get("source_spectrum_path", "") for row in line_rows}
            if attempted == 0:
                attempted_line = len(line_sources)
                succeeded_line = len(line_sources)
            else:
                attempted_line = attempted
                succeeded_line = succeeded

            detections = sum(
                1
                for row in line_rows
                if not to_bool(row.get("UpLim", ""))
                and str(row.get("fit_status", "")) != "line_failed"
                and str(row.get("logN", "")).strip() != ""
            )
            upper_limits = sum(1 for row in line_rows if to_bool(row.get("UpLim", "")))
            line_failed = sum(1 for row in line_rows if str(row.get("fit_status", "")) == "line_failed")

            summary.append(
                {
                    "SID": sid,
                    "snap": snap,
                    "run_label": run_label,
                    "mode": mode,
                    "alpha": alpha,
                    "saved_line_label": line_label,
                    "pygad_ion_key": LINE_MAP[line_label],
                    "number_spectra_attempted": attempted_line,
                    "number_succeeded": succeeded_line,
                    "number_skipped": skipped,
                    "number_failed": failed,
                    "number_line_failed": line_failed,
                    "number_detections": detections,
                    "number_upper_limits": upper_limits,
                    "number_fit_rows": len(line_rows),
                }
            )

    return summary


def main() -> int:
    args = parse_args()
    result_files = find_result_files(args)
    status_files = find_status_files(args)

    print(f"Found result CSVs: {len(result_files)}")
    print(f"Found status CSVs: {len(status_files)}")

    write_master_outputs(args, result_files)

    result_rows = read_many_csvs(result_files)
    status_rows = read_many_csvs(status_files)
    summary_rows = make_summary(result_rows, status_rows)
    summary_fields = [
        "SID",
        "snap",
        "run_label",
        "mode",
        "alpha",
        "saved_line_label",
        "pygad_ion_key",
        "number_spectra_attempted",
        "number_succeeded",
        "number_skipped",
        "number_failed",
        "number_line_failed",
        "number_detections",
        "number_upper_limits",
        "number_fit_rows",
    ]
    write_dict_rows(args.summary_output, summary_rows, summary_fields)
    print(f"Wrote summary CSV: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

