#!/usr/bin/env python3
"""Generate one-spectrum-per-task TSV files for individual-line fitting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from fit_individual_line_pipeline_common import (
    DEFAULT_BASE_DIR,
    DEFAULT_OUTPUT_SUBDIR,
    DEFAULT_RUN_LABEL,
    DEFAULT_SID_FILE,
    DEFAULT_SNAP,
    DEFAULT_TASK_LIST,
    TASK_FIELDNAMES,
    discover_spectrum_files,
    normalize_alpha,
    normalize_mode,
    parse_spectrum_filename,
    per_spectrum_output_paths,
    read_sid_file,
    write_dict_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover saved spectra and write a SLURM task list."
    )
    parser.add_argument("--sid-file", default=DEFAULT_SID_FILE)
    parser.add_argument("--snap", type=int, default=DEFAULT_SNAP)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--mode", choices=["all", "flip", "noflip"], default="all")
    parser.add_argument("--alpha", default="all")
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--output", default=DEFAULT_TASK_LIST)
    parser.add_argument("--max-files-per-sid", type=int, default=None)
    parser.add_argument("--max-sids", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sids = read_sid_file(args.sid_file)
    if args.max_sids is not None and args.max_sids > 0:
        sids = sids[: args.max_sids]

    mode = normalize_mode(args.mode)
    alpha = normalize_alpha(args.alpha)
    rows: List[Dict[str, Any]] = []
    counts: Dict[int, int] = {}

    for sid in sids:
        files = discover_spectrum_files(
            base_dir=args.base_dir,
            sid=sid,
            snap=args.snap,
            run_label=args.run_label,
            mode=mode,
            alpha=alpha,
            max_files=args.max_files_per_sid,
        )
        counts[sid] = len(files)
        for spectrum_file in files:
            meta = parse_spectrum_filename(spectrum_file)
            output_csv, _output_txt = per_spectrum_output_paths(
                args.base_dir,
                sid,
                args.snap,
                args.run_label,
                args.output_subdir,
                spectrum_file,
            )
            rows.append(
                {
                    "task_id": len(rows) + 1,
                    "sid": sid,
                    "snap": args.snap,
                    "run_label": args.run_label,
                    "mode": meta.get("mode", "unknown"),
                    "alpha": meta.get("alpha") if meta.get("alpha") is not None else -1,
                    "ray_id": meta.get("ray_id", "unknown"),
                    "spectrum_file": spectrum_file,
                    "output_file": output_csv,
                }
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_dict_rows(args.output, rows, TASK_FIELDNAMES)

    print(f"Wrote task list: {args.output}")
    print(f"SIDs read      : {len(sids)} from {args.sid_file}")
    print(f"Tasks written  : {len(rows)}")
    for sid in sids:
        print(f"  sid{sid}: {counts.get(sid, 0)} spectra")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
