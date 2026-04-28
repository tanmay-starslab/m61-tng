#!/usr/bin/env python3
"""Shared helpers for the saved-spectrum fitting production pipeline."""

from __future__ import annotations

import csv
import glob
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SNAP = 99
DEFAULT_RUN_LABEL = "L4Rvir"
DEFAULT_BASE_DIR = "/scratch/tsingh65/m61-tng/outputs"
DEFAULT_OUTPUT_SUBDIR = "fitted_spectra_snr10_bin3"
DEFAULT_SID_FILE = str(REPO_ROOT / "data" / "sids_from_cutouts_snap99.txt")
DEFAULT_TASK_LIST = str(REPO_ROOT / "batch" / "fit_spectra_tasks_snap99_L4Rvir.tsv")

DEFAULT_Z = 0.0
DEFAULT_SNR = 10.0
DEFAULT_BIN_BEFORE_NOISE = True
DEFAULT_BIN_NPIX = 3
DEFAULT_N_SIGMA = 3.0
DEFAULT_MIN_REGION_WIDTH = 3


LINE_MAP: Dict[str, str] = {
    "Si II 1190": "SiII1190",
    "Si II 1193": "SiII1193",
    "Si III 1206": "SiIII1206",
    "N V 1239": "NV1238",
    "Si II 1260": "SiII1260",
    "O I 1302": "OI1302",
    "C II 1335": "CII1334",
    "Si IV 1403": "SiIV1402",
    "H I 1216": "H1215",
}


LINE_FIT_PARAMS: Dict[str, Dict[str, Any]] = {
    "Si II 1190": {
        "logN_bounds": (12.3, 16.5),
        "b_bounds": (6.0, 60.0),
        "velocity_window": 500.0,
        "upper_limit_window": 50.0,
        "max_lines": 4,
        "chisq_lim": 1.3,
    },
    "Si II 1193": {
        "logN_bounds": (12.2, 16.5),
        "b_bounds": (6.0, 60.0),
        "velocity_window": 500.0,
        "upper_limit_window": 50.0,
        "max_lines": 4,
        "chisq_lim": 1.3,
    },
    "Si III 1206": {
        "logN_bounds": (11.8, 16.5),
        "b_bounds": (6.0, 80.0),
        "velocity_window": 800.0,
        "upper_limit_window": 50.0,
        "max_lines": 5,
        "chisq_lim": 1.3,
    },
    "N V 1239": {
        "logN_bounds": (12.7, 16.8),
        "b_bounds": (6.0, 100.0),
        "velocity_window": 800.0,
        "upper_limit_window": 50.0,
        "max_lines": 5,
        "chisq_lim": 1.3,
    },
    "Si II 1260": {
        "logN_bounds": (11.9, 16.5),
        "b_bounds": (6.0, 60.0),
        "velocity_window": 800.0,
        "upper_limit_window": 50.0,
        "max_lines": 4,
        "chisq_lim": 1.3,
    },
    "O I 1302": {
        "logN_bounds": (13.1, 17.0),
        "b_bounds": (6.0, 50.0),
        "velocity_window": 800.0,
        "upper_limit_window": 50.0,
        "max_lines": 3,
        "chisq_lim": 1.3,
    },
    "C II 1335": {
        "logN_bounds": (12.7, 17.0),
        "b_bounds": (6.0, 70.0),
        "velocity_window": 800.0,
        "upper_limit_window": 50.0,
        "max_lines": 4,
        "chisq_lim": 1.3,
    },
    "Si IV 1403": {
        "logN_bounds": (12.4, 16.8),
        "b_bounds": (6.0, 90.0),
        "velocity_window": 800.0,
        "upper_limit_window": 50.0,
        "max_lines": 5,
        "chisq_lim": 1.3,
    },
    "H I 1216": {
        "logN_bounds": (12.3, 19.5),
        "b_bounds": (8.0, 150.0),
        "velocity_window": 1200.0,
        "upper_limit_window": 50.0,
        "max_lines": 6,
        "chisq_lim": 1.3,
    },
}


TASK_FIELDNAMES = [
    "task_id",
    "sid",
    "snap",
    "run_label",
    "mode",
    "alpha",
    "ray_id",
    "spectrum_file",
    "output_file",
]


RESULT_FIELDNAMES = [
    "SID",
    "snap",
    "run_label",
    "ray_id",
    "mode",
    "alpha",
    "source_spectrum_path",
    "source_filename",
    "spectrum_basename",
    "saved_line_label",
    "pygad_ion_key",
    "EW_mA",
    "dEW_mA",
    "logN",
    "dlogN",
    "b_kms",
    "db_kms",
    "v_kms",
    "dv_kms",
    "lambda_A",
    "dlambda_A",
    "UpLim",
    "Sat",
    "Chisq",
    "Nregions",
    "WaveFrame",
    "Binned",
    "BinNpix",
    "LSFAlreadyIncluded",
    "fit_status",
    "error_message",
    "output_file",
    "fit_z",
    "fit_snr",
    "fit_seed",
    "fit_bin_before_noise",
    "fit_bin_npix",
    "fit_N_sigma",
    "fit_min_region_width",
    "fit_velocity_window_kms",
    "fit_upper_limit_window_kms",
    "fit_max_lines",
    "fit_chisq_lim",
    "fit_logN_min",
    "fit_logN_max",
    "fit_b_min_kms",
    "fit_b_max_kms",
]


STATUS_FIELDNAMES = [
    "task_index",
    "SID",
    "snap",
    "run_label",
    "ray_id",
    "mode",
    "alpha",
    "spectrum_file",
    "output_file",
    "status",
    "n_lines_requested",
    "n_result_rows",
    "n_line_errors",
    "start_time_utc",
    "end_time_utc",
    "elapsed_sec",
    "error_message",
]


def read_sid_file(path: str) -> List[int]:
    sids: List[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not re.fullmatch(r"[0-9]+", line):
                raise ValueError(f"Invalid SID line in {path}: {raw.rstrip()!r}")
            sids.append(int(line))
    return sids


def spectra_run_dir(base_dir: str, sid: int, snap: int, run_label: str) -> str:
    return os.path.join(
        base_dir,
        f"sid{int(sid)}",
        f"rays_and_spectra_sid{int(sid)}_snap{int(snap)}_{run_label}",
    )


def spectra_h5_dir(base_dir: str, sid: int, snap: int, run_label: str) -> str:
    return os.path.join(spectra_run_dir(base_dir, sid, snap, run_label), "spectra_h5")


def output_root(base_dir: str, sid: int, snap: int, run_label: str, output_subdir: str) -> str:
    return os.path.join(spectra_run_dir(base_dir, sid, snap, run_label), output_subdir)


def parse_spectrum_filename(path: str) -> Dict[str, Any]:
    base = os.path.basename(path)

    mode = "unknown"
    if "_noflip_" in base:
        mode = "noflip"
    elif "_flip_" in base:
        mode = "flip"

    alpha: Optional[int] = None
    m_alpha = re.search(r"alpha([0-9]+)", base)
    if m_alpha:
        alpha = int(m_alpha.group(1))

    sid: Optional[int] = None
    m_sid = re.search(r"sid([0-9]+)", base)
    if m_sid:
        sid = int(m_sid.group(1))

    ray_id = "unknown"
    m_ray = re.search(r"sid[0-9]+_(.*?)_(?:no)?flip_alpha", base)
    if m_ray:
        ray_id = m_ray.group(1)

    return {
        "filename": base,
        "basename": os.path.splitext(base)[0],
        "sid": sid,
        "ray_id": ray_id,
        "mode": mode,
        "alpha": alpha,
    }


def normalize_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    value = str(mode).strip().lower()
    if value in ("", "all", "any", "*"):
        return None
    if value not in ("flip", "noflip"):
        raise ValueError("mode must be one of: all, flip, noflip")
    return value


def normalize_alpha(alpha: Any) -> Optional[int]:
    if alpha is None:
        return None
    value = str(alpha).strip().lower()
    if value in ("", "all", "any", "*"):
        return None
    return int(value)


def alpha_dirname(alpha: Optional[int]) -> str:
    if alpha is None:
        return "alpha_unknown"
    return f"alpha{int(alpha):03d}"


def discover_spectrum_files(
    base_dir: str,
    sid: int,
    snap: int = DEFAULT_SNAP,
    run_label: str = DEFAULT_RUN_LABEL,
    mode: Optional[str] = None,
    alpha: Optional[int] = None,
    max_files: Optional[int] = None,
) -> List[str]:
    h5_dir = spectra_h5_dir(base_dir, sid, snap, run_label)
    patterns = [
        os.path.join(h5_dir, "*_spectrum.h5"),
        os.path.join(h5_dir, "*_spectrum.hdf5"),
        os.path.join(h5_dir, "*.h5"),
        os.path.join(h5_dir, "*.hdf5"),
    ]

    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    files = sorted(set(files))
    files = [
        f
        for f in files
        if os.path.isfile(f)
        and os.path.getsize(f) > 0
        and "all_rays" not in os.path.basename(f).lower()
        and "combined" not in f.lower()
        and "summary" not in os.path.basename(f).lower()
    ]

    mode_norm = normalize_mode(mode)
    alpha_norm = normalize_alpha(alpha)

    if mode_norm == "flip":
        files = [
            f
            for f in files
            if "_flip_" in os.path.basename(f) and "_noflip_" not in os.path.basename(f)
        ]
    elif mode_norm == "noflip":
        files = [f for f in files if "_noflip_" in os.path.basename(f)]

    if alpha_norm is not None:
        files = [f for f in files if f"alpha{alpha_norm}_" in os.path.basename(f)]

    files = sorted(files, key=spectrum_sort_key)
    if max_files is not None and int(max_files) > 0:
        files = files[: int(max_files)]
    return files


def spectrum_sort_key(path: str) -> Tuple[Any, ...]:
    meta = parse_spectrum_filename(path)
    mode_rank = {"flip": 0, "noflip": 1}.get(str(meta["mode"]), 9)
    alpha = meta["alpha"] if meta["alpha"] is not None else 10**9
    return (mode_rank, int(alpha), str(meta["ray_id"]), os.path.basename(path))


def per_spectrum_output_paths(
    base_dir: str,
    sid: int,
    snap: int,
    run_label: str,
    output_subdir: str,
    spectrum_file: str,
) -> Tuple[str, str]:
    meta = parse_spectrum_filename(spectrum_file)
    mode = str(meta["mode"])
    alpha = meta["alpha"]
    basename = str(meta["basename"])
    out_dir = os.path.join(
        output_root(base_dir, sid, snap, run_label, output_subdir),
        "per_spectrum",
        mode,
        alpha_dirname(alpha),
    )
    return (
        os.path.join(out_dir, f"{basename}_fits.csv"),
        os.path.join(out_dir, f"{basename}_fits.txt"),
    )


def per_spectrum_status_path(
    base_dir: str,
    sid: int,
    snap: int,
    run_label: str,
    output_subdir: str,
    spectrum_file: str,
) -> str:
    meta = parse_spectrum_filename(spectrum_file)
    out_dir = os.path.join(
        output_root(base_dir, sid, snap, run_label, output_subdir),
        "logs",
        "per_spectrum",
        str(meta["mode"]),
        alpha_dirname(meta["alpha"]),
    )
    return os.path.join(out_dir, f"{meta['basename']}_status.csv")


def plot_output_subdir_for_spectrum(output_subdir: str, spectrum_file: str) -> str:
    meta = parse_spectrum_filename(spectrum_file)
    return os.path.join(
        output_subdir,
        "per_spectrum",
        str(meta["mode"]),
        alpha_dirname(meta["alpha"]),
        "plots",
        str(meta["basename"]),
    )


def stable_seed(text: str, base_seed: int = 42) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return int((int(base_seed) + offset) % (2**31 - 1))


def delimiter_for_path(path: str) -> str:
    return "\t" if str(path).lower().endswith((".tsv", ".tab")) else ","


def write_dict_rows(path: str, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=delimiter_for_path(path),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_dict_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter_for_path(path))
        return list(reader)


def select_task_row(task_list: str, task_index: int) -> Dict[str, str]:
    rows = read_dict_rows(task_list)
    if not rows:
        raise ValueError(f"Task list has no rows: {task_list}")

    wanted = str(int(task_index))
    for row in rows:
        if str(row.get("task_id", "")).strip() == wanted:
            return row

    idx = int(task_index) - 1
    if idx < 0 or idx >= len(rows):
        raise IndexError(
            f"task_index={task_index} is outside task list row range 1..{len(rows)}"
        )
    return rows[idx]
