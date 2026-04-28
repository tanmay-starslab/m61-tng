#!/usr/bin/env python3
"""Production CLI for fitting saved synthetic spectra with pygad."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from astropy.table import Table

from fit_pipeline_common import (
    DEFAULT_BASE_DIR,
    DEFAULT_BIN_BEFORE_NOISE,
    DEFAULT_BIN_NPIX,
    DEFAULT_MIN_REGION_WIDTH,
    DEFAULT_N_SIGMA,
    DEFAULT_OUTPUT_SUBDIR,
    DEFAULT_RUN_LABEL,
    DEFAULT_SNAP,
    DEFAULT_SNR,
    DEFAULT_TASK_LIST,
    DEFAULT_Z,
    LINE_FIT_PARAMS,
    LINE_MAP,
    RESULT_FIELDNAMES,
    STATUS_FIELDNAMES,
    discover_spectrum_files,
    normalize_alpha,
    normalize_mode,
    parse_spectrum_filename,
    per_spectrum_output_paths,
    per_spectrum_status_path,
    plot_output_subdir_for_spectrum,
    select_task_row,
    stable_seed,
    write_dict_rows,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def import_pfit(repo_root: Path):
    notebooks = repo_root / "notebooks"
    if not notebooks.is_dir():
        raise FileNotFoundError(f"Missing notebooks directory: {notebooks}")
    sys.path.insert(0, str(notebooks))
    import pygad_fit_saved_spectra as pfit  # noqa: PLC0415
    import pygad.analysis as pygad_analysis  # noqa: PLC0415

    try:
        import pygad.analysis.absorption_spectra  # noqa: F401, PLC0415
        import pygad.analysis.vpfit  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        pass

    if not hasattr(pygad_analysis, "absorption_spectra") or not hasattr(pygad_analysis, "vpfit"):
        raise RuntimeError(
            "The active pygad import does not expose pygad.analysis.absorption_spectra "
            "and pygad.analysis.vpfit. On SOL, run in the trident environment with "
            "PYTHONNOUSERSITE=1; batch/run_fit_spectra_array.sh does this setup."
        )

    return pfit


def scalar(value: Any) -> Any:
    if value is None:
        return ""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def finite_or_blank(value: Any) -> Any:
    value = scalar(value)
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def bool_or_blank(value: Any) -> Any:
    value = scalar(value)
    if value == "":
        return ""
    return bool(value)


def int_or_default(value: Any, default: int) -> int:
    value = finite_or_blank(value)
    if value == "":
        return int(default)
    return int(value)


def make_base_config(pfit: Any, args: argparse.Namespace, sid: int, seed: int):
    return pfit.FitConfig(
        sid=int(sid),
        snap=int(args.snap),
        run_label=str(args.run_label),
        base_dir=str(args.base_dir),
        z=float(args.z),
        snr=float(args.snr),
        seed=int(seed),
        bin_before_noise=bool(args.bin_before_noise),
        bin_npix=int(args.bin_npix),
        min_region_width=int(args.min_region_width),
        N_sigma=float(args.N_sigma),
        line_labels=list(LINE_MAP.keys()),
        line_map=dict(LINE_MAP),
        output_subdir=str(args.output_subdir),
        make_plots=bool(args.make_plots),
        verbose=bool(args.verbose),
    )


def line_config(pfit: Any, base_cfg: Any, saved_line: str, spectrum_file: str) -> Any:
    cfg = replace(base_cfg)
    cfg.line_labels = [saved_line]
    cfg.line_map = {saved_line: LINE_MAP[saved_line]}

    params = LINE_FIT_PARAMS[saved_line]
    cfg.logN_bounds = tuple(params["logN_bounds"])
    cfg.b_bounds = tuple(params["b_bounds"])
    cfg.velocity_window = float(params["velocity_window"])
    cfg.upper_limit_window = float(params["upper_limit_window"])
    cfg.max_lines = int(params["max_lines"])
    cfg.chisq_lim = float(params["chisq_lim"])

    if cfg.make_plots:
        cfg.output_subdir = plot_output_subdir_for_spectrum(base_cfg.output_subdir, spectrum_file)

    return cfg


def empty_line_error_record(
    cfg: Any,
    spectrum_file: str,
    saved_line: str,
    output_csv: str,
    status: str,
    error_message: str,
) -> Dict[str, Any]:
    meta = parse_spectrum_filename(spectrum_file)
    params = LINE_FIT_PARAMS[saved_line]
    return {
        "SID": int(cfg.sid),
        "snap": int(cfg.snap),
        "run_label": cfg.run_label,
        "ray_id": meta.get("ray_id", "unknown"),
        "mode": meta.get("mode", "unknown"),
        "alpha": meta.get("alpha") if meta.get("alpha") is not None else -1,
        "source_spectrum_path": spectrum_file,
        "source_filename": meta.get("filename", os.path.basename(spectrum_file)),
        "spectrum_basename": meta.get("basename", Path(spectrum_file).stem),
        "saved_line_label": saved_line,
        "pygad_ion_key": LINE_MAP[saved_line],
        "EW_mA": "",
        "dEW_mA": "",
        "logN": "",
        "dlogN": "",
        "b_kms": "",
        "db_kms": "",
        "v_kms": "",
        "dv_kms": "",
        "lambda_A": "",
        "dlambda_A": "",
        "UpLim": "",
        "Sat": "",
        "Chisq": "",
        "Nregions": "",
        "WaveFrame": "",
        "Binned": bool(cfg.bin_before_noise),
        "BinNpix": int(cfg.bin_npix),
        "LSFAlreadyIncluded": True,
        "fit_status": status,
        "error_message": error_message,
        "output_file": output_csv,
        "fit_z": float(cfg.z),
        "fit_snr": float(cfg.snr),
        "fit_seed": int(cfg.seed),
        "fit_bin_before_noise": bool(cfg.bin_before_noise),
        "fit_bin_npix": int(cfg.bin_npix),
        "fit_N_sigma": float(cfg.N_sigma),
        "fit_min_region_width": int(cfg.min_region_width),
        "fit_velocity_window_kms": float(params["velocity_window"]),
        "fit_upper_limit_window_kms": float(params["upper_limit_window"]),
        "fit_max_lines": int(params["max_lines"]),
        "fit_chisq_lim": float(params["chisq_lim"]),
        "fit_logN_min": float(params["logN_bounds"][0]),
        "fit_logN_max": float(params["logN_bounds"][1]),
        "fit_b_min_kms": float(params["b_bounds"][0]),
        "fit_b_max_kms": float(params["b_bounds"][1]),
    }


def normalize_fit_row(
    row: Any,
    cfg: Any,
    spectrum_file: str,
    saved_line: str,
    output_csv: str,
    fit_status: str = "succeeded",
    error_message: str = "",
) -> Dict[str, Any]:
    meta = parse_spectrum_filename(spectrum_file)
    params = LINE_FIT_PARAMS[saved_line]

    def get(name: str, default: Any = "") -> Any:
        try:
            return row[name]
        except Exception:
            return default

    return {
        "SID": int_or_default(get("SID", cfg.sid), int(cfg.sid)),
        "snap": int(cfg.snap),
        "run_label": cfg.run_label,
        "ray_id": finite_or_blank(get("RayID", meta.get("ray_id", "unknown"))),
        "mode": finite_or_blank(get("Mode", meta.get("mode", "unknown"))),
        "alpha": int_or_default(get("Alpha", meta.get("alpha", -1)), -1),
        "source_spectrum_path": finite_or_blank(get("SourceFile", spectrum_file)),
        "source_filename": finite_or_blank(get("Filename", meta.get("filename", os.path.basename(spectrum_file)))),
        "spectrum_basename": meta.get("basename", Path(spectrum_file).stem),
        "saved_line_label": finite_or_blank(get("SavedLine", saved_line)),
        "pygad_ion_key": finite_or_blank(get("PygadIon", LINE_MAP[saved_line])),
        "EW_mA": finite_or_blank(get("EW_mA")),
        "dEW_mA": finite_or_blank(get("dEW_mA")),
        "logN": finite_or_blank(get("logN")),
        "dlogN": finite_or_blank(get("dlogN")),
        "b_kms": finite_or_blank(get("b_kms")),
        "db_kms": finite_or_blank(get("db_kms")),
        "v_kms": finite_or_blank(get("v_kms")),
        "dv_kms": finite_or_blank(get("dv_kms")),
        "lambda_A": finite_or_blank(get("lambda_A")),
        "dlambda_A": finite_or_blank(get("dlambda_A")),
        "UpLim": bool_or_blank(get("UpLim")),
        "Sat": bool_or_blank(get("Sat")),
        "Chisq": finite_or_blank(get("Chisq")),
        "Nregions": finite_or_blank(get("Nregions")),
        "WaveFrame": finite_or_blank(get("WaveFrame")),
        "Binned": bool_or_blank(get("Binned", cfg.bin_before_noise)),
        "BinNpix": int_or_default(get("BinNpix", cfg.bin_npix), int(cfg.bin_npix)),
        "LSFAlreadyIncluded": True,
        "fit_status": fit_status,
        "error_message": error_message,
        "output_file": output_csv,
        "fit_z": float(cfg.z),
        "fit_snr": float(cfg.snr),
        "fit_seed": int(cfg.seed),
        "fit_bin_before_noise": bool(cfg.bin_before_noise),
        "fit_bin_npix": int(cfg.bin_npix),
        "fit_N_sigma": float(cfg.N_sigma),
        "fit_min_region_width": int(cfg.min_region_width),
        "fit_velocity_window_kms": float(params["velocity_window"]),
        "fit_upper_limit_window_kms": float(params["upper_limit_window"]),
        "fit_max_lines": int(params["max_lines"]),
        "fit_chisq_lim": float(params["chisq_lim"]),
        "fit_logN_min": float(params["logN_bounds"][0]),
        "fit_logN_max": float(params["logN_bounds"][1]),
        "fit_b_min_kms": float(params["b_bounds"][0]),
        "fit_b_max_kms": float(params["b_bounds"][1]),
    }


def write_result_files(csv_path: str, txt_path: str, records: List[Dict[str, Any]]) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    write_dict_rows(csv_path, records, RESULT_FIELDNAMES)

    table = Table(rows=[[record.get(name, "") for name in RESULT_FIELDNAMES] for record in records], names=RESULT_FIELDNAMES)
    table.write(txt_path, format="ascii.fixed_width", overwrite=True)


def write_status(
    args: argparse.Namespace,
    cfg: Any,
    spectrum_file: str,
    output_csv: str,
    status: str,
    start_time: float,
    start_time_utc: str,
    n_result_rows: int = 0,
    n_line_errors: int = 0,
    error_message: str = "",
) -> None:
    meta = parse_spectrum_filename(spectrum_file)
    end_time = time.time()
    row = {
        "task_index": args.task_index if args.task_index is not None else "",
        "SID": int(cfg.sid),
        "snap": int(cfg.snap),
        "run_label": cfg.run_label,
        "ray_id": meta.get("ray_id", "unknown"),
        "mode": meta.get("mode", "unknown"),
        "alpha": meta.get("alpha") if meta.get("alpha") is not None else -1,
        "spectrum_file": spectrum_file,
        "output_file": output_csv,
        "status": status,
        "n_lines_requested": len(LINE_MAP),
        "n_result_rows": int(n_result_rows),
        "n_line_errors": int(n_line_errors),
        "start_time_utc": start_time_utc,
        "end_time_utc": utc_now(),
        "elapsed_sec": f"{end_time - start_time:.3f}",
        "error_message": error_message,
    }
    status_path = per_spectrum_status_path(
        args.base_dir,
        int(cfg.sid),
        int(cfg.snap),
        cfg.run_label,
        args.output_subdir,
        spectrum_file,
    )
    write_dict_rows(status_path, [row], STATUS_FIELDNAMES)


def fit_one_spectrum(
    pfit: Any,
    args: argparse.Namespace,
    sid: int,
    spectrum_file: str,
) -> Tuple[str, int]:
    start_time = time.time()
    start_time_utc = utc_now()
    output_csv, output_txt = per_spectrum_output_paths(
        args.base_dir,
        sid,
        args.snap,
        args.run_label,
        args.output_subdir,
        spectrum_file,
    )

    file_seed = stable_seed(spectrum_file, args.seed)
    base_cfg = make_base_config(pfit, args, sid=sid, seed=file_seed)

    if os.path.exists(output_csv) and os.path.exists(output_txt) and not args.overwrite:
        if args.verbose:
            print(f"[SKIP] output exists: {output_csv}")
        write_status(
            args,
            base_cfg,
            spectrum_file,
            output_csv,
            status="skipped",
            start_time=start_time,
            start_time_utc=start_time_utc,
        )
        return "skipped", 0

    try:
        wave, flux_clean, _load_meta = pfit.load_saved_spectrum_h5(
            spectrum_file,
            verbose=bool(args.verbose),
        )
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        write_status(
            args,
            base_cfg,
            spectrum_file,
            output_csv,
            status="failed",
            start_time=start_time,
            start_time_utc=start_time_utc,
            error_message=error,
        )
        print(f"[FAILED-LOAD] {spectrum_file}: {error}", file=sys.stderr)
        return "failed", 0

    records: List[Dict[str, Any]] = []
    line_errors: List[str] = []

    for iline, saved_line in enumerate(LINE_MAP.keys()):
        pg_ion = LINE_MAP[saved_line]
        cfg_line = line_config(pfit, base_cfg, saved_line, spectrum_file)

        try:
            if args.verbose:
                print(
                    f"[FIT] {os.path.basename(spectrum_file)} | {saved_line} -> {pg_ion} "
                    f"window={cfg_line.velocity_window:g} km/s"
                )

            row_table, diag = pfit.fit_line_in_spectrum(
                cfg=cfg_line,
                wave=wave,
                flux_clean=flux_clean,
                source_file=spectrum_file,
                saved_line=saved_line,
                pg_ion=pg_ion,
                file_seed_offset=1000 * iline,
            )

            for row in row_table:
                records.append(
                    normalize_fit_row(
                        row=row,
                        cfg=cfg_line,
                        spectrum_file=spectrum_file,
                        saved_line=saved_line,
                        output_csv=output_csv,
                    )
                )

            if args.make_plots and diag is not None:
                pfit.plot_fit_diagnostic(cfg_line, diag)

        except Exception as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            line_errors.append(f"{saved_line}: {error}")
            records.append(
                empty_line_error_record(
                    cfg=cfg_line,
                    spectrum_file=spectrum_file,
                    saved_line=saved_line,
                    output_csv=output_csv,
                    status="line_failed",
                    error_message=error,
                )
            )
            print(f"[FAILED-LINE] {spectrum_file} | {saved_line}: {error}", file=sys.stderr)

    try:
        write_result_files(output_csv, output_txt, records)
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        write_status(
            args,
            base_cfg,
            spectrum_file,
            output_csv,
            status="failed",
            start_time=start_time,
            start_time_utc=start_time_utc,
            n_result_rows=len(records),
            n_line_errors=len(line_errors),
            error_message=f"write failed: {error}",
        )
        print(f"[FAILED-WRITE] {spectrum_file}: {error}", file=sys.stderr)
        return "failed", len(records)

    status = "succeeded_with_line_errors" if line_errors else "succeeded"
    write_status(
        args,
        base_cfg,
        spectrum_file,
        output_csv,
        status=status,
        start_time=start_time,
        start_time_utc=start_time_utc,
        n_result_rows=len(records),
        n_line_errors=len(line_errors),
        error_message="; ".join(line_errors),
    )

    print(f"[{status.upper()}] {spectrum_file}")
    print(f"  CSV: {output_csv}")
    print(f"  TXT: {output_txt}")
    return status, len(records)


def resolve_work_items(args: argparse.Namespace) -> List[Tuple[int, str]]:
    if args.task_list and args.task_index is not None:
        task = select_task_row(args.task_list, int(args.task_index))
        spectrum_file = task["spectrum_file"]
        sid = int(task["sid"])
        return [(sid, spectrum_file)]

    if args.spectrum_file:
        spectrum_file = str(args.spectrum_file)
        meta = parse_spectrum_filename(spectrum_file)
        sid_value = args.sid if args.sid is not None else meta.get("sid")
        if sid_value is None:
            raise ValueError("--sid is required when --spectrum-file does not encode sid")
        sid = int(sid_value)
        return [(sid, spectrum_file)]

    if args.sid is None:
        raise ValueError("Provide --sid, --spectrum-file, or --task-list with --task-index")

    mode = normalize_mode(args.mode)
    alpha = normalize_alpha(args.alpha)
    files = discover_spectrum_files(
        base_dir=args.base_dir,
        sid=int(args.sid),
        snap=int(args.snap),
        run_label=args.run_label,
        mode=mode,
        alpha=alpha,
        max_files=args.max_files,
    )
    return [(int(args.sid), path) for path in files]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit saved synthetic spectra with line-specific pygad settings."
    )
    parser.add_argument("--sid", type=int, default=None)
    parser.add_argument("--snap", type=int, default=DEFAULT_SNAP)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--mode", choices=["all", "flip", "noflip"], default="all")
    parser.add_argument("--alpha", default="all")
    parser.add_argument("--spectrum-file", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--task-list", default=None)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--z", type=float, default=DEFAULT_Z)
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bin-npix", type=int, default=DEFAULT_BIN_NPIX)
    parser.add_argument("--no-bin-before-noise", dest="bin_before_noise", action="store_false")
    parser.set_defaults(bin_before_noise=DEFAULT_BIN_BEFORE_NOISE)
    parser.add_argument("--N-sigma", dest="N_sigma", type=float, default=DEFAULT_N_SIGMA)
    parser.add_argument("--min-region-width", type=int, default=DEFAULT_MIN_REGION_WIDTH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.task_index is not None and args.task_list is None:
        args.task_list = DEFAULT_TASK_LIST

    repo_root = Path(__file__).resolve().parents[1]
    pfit = import_pfit(repo_root)

    work_items = resolve_work_items(args)
    if not work_items:
        print("[INFO] No spectra matched the request.")
        return 0

    counts: Dict[str, int] = {}
    total_rows = 0

    for i, (sid, spectrum_file) in enumerate(work_items, start=1):
        print(f"[{i}/{len(work_items)}] SID={sid} {spectrum_file}")
        status, n_rows = fit_one_spectrum(pfit, args, sid=sid, spectrum_file=spectrum_file)
        counts[status] = counts.get(status, 0) + 1
        total_rows += n_rows

    print("[SUMMARY]")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"  result rows: {total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
