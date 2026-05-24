#!/usr/bin/env python3
"""Parallel individual-line fitter for one saved spectrum per task.

This is the fast production-oriented variant of the individual-line fitting
workflow.  One job handles one saved spectrum/ray, while multiple worker
processes fit the saved line branches concurrently:

    /spectrum_by_line/<line_group>/raw
    /spectrum_by_line/<line_group>/lsf

The output is a single HDF5 file per spectrum containing all fitted rows,
per-line detection/non-detection status, and enough arrays to regenerate visual
QA plots later without refitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import h5py
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/m61_fit_parallel_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fit_individual_line_pipeline_common import (  # noqa: E402
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
    parse_spectrum_filename,
    select_task_row,
    stable_seed,
    write_dict_rows,
)
from fit_individual_line_spectra_batch import (  # noqa: E402
    import_pfit,
    line_config,
    line_group_name,
    load_individual_line_spectrum_h5,
    make_base_config,
    normalize_fit_row,
)

C_KMS = 299792.458


def read_lines_from_run_spectra() -> List[str]:
    text = (REPO_ROOT / "batch" / "run_spectra_array.sh").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("LINES_CSV="):
            labels = line.split("=", 1)[1].strip().strip('"').split(",")
            labels = [label.strip() for label in labels if label.strip()]
            missing = [label for label in labels if label not in LINE_MAP]
            if missing:
                raise RuntimeError(f"LINES_CSV labels missing from fitter LINE_MAP: {missing}")
            return labels
    raise RuntimeError("Could not find LINES_CSV in batch/run_spectra_array.sh")


def selected_lines(raw: str) -> List[str]:
    labels = read_lines_from_run_spectra()
    raw = str(raw).strip()
    if raw.lower() in {"", "all", "*"}:
        return labels
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    bad = [item for item in requested if item not in labels]
    if bad:
        raise ValueError(f"Requested line(s) not in run_spectra_array.sh LINES_CSV: {bad}")
    return requested


def spectrum_fit_h5_path(base_dir: str, sid: int, snap: int, run_label: str, output_subdir: str, spectrum_file: str) -> Path:
    meta = parse_spectrum_filename(spectrum_file)
    mode = str(meta.get("mode", "unknown"))
    alpha = int(meta.get("alpha", -1))
    outdir = (
        Path(base_dir)
        / f"sid{sid}"
        / f"rays_and_spectra_sid{sid}_snap{snap}_{run_label}"
        / output_subdir
        / "per_spectrum_h5"
        / mode
        / f"alpha{alpha:03d}"
    )
    return outdir / f"{Path(spectrum_file).stem}_individual_line_fits.h5"


def plots_dir_for_h5(path: Path) -> Path:
    return path.parent.parent.parent.parent / "plots_from_h5" / path.parent.parent.name / path.parent.name


def wave_to_vel(wave_rest: np.ndarray, line_rest: float) -> np.ndarray:
    return C_KMS * (np.asarray(wave_rest, dtype=float) / float(line_rest) - 1.0)


def window_arrays(pfit: Any, wave: np.ndarray, flux: np.ndarray, line_rest: float, z: float, window: float) -> Dict[str, np.ndarray]:
    wave_rest, _frame = pfit.convert_wave_to_rest_if_needed(wave, line_rest=line_rest, z=z)
    vel = wave_to_vel(wave_rest, line_rest)
    mask = np.isfinite(vel) & np.isfinite(flux) & (vel >= -window) & (vel <= window)
    return {"velocity_kms": vel[mask], "flux": np.asarray(flux, dtype=float)[mask], "wave_A": wave_rest[mask]}


def row_is_detection(row: Dict[str, Any]) -> bool:
    uplim = str(row.get("UpLim", "")).strip().lower()
    if uplim in {"true", "1"}:
        return False
    value = row.get("logN", "")
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return str(value)


def worker_fit_line(payload: Dict[str, Any]) -> Dict[str, Any]:
    pfit = import_pfit(REPO_ROOT)
    args = argparse.Namespace(**payload["args"])
    spectrum_file = payload["spectrum_file"]
    saved_line = payload["saved_line"]
    sid = int(payload["sid"])
    iline = int(payload["iline"])
    pg_ion = LINE_MAP[saved_line]

    base_cfg = make_base_config(pfit, args, sid=sid, seed=int(payload["file_seed"]))
    cfg_line = replace(line_config(pfit, base_cfg, saved_line, spectrum_file), make_plots=False)
    group_name = line_group_name(saved_line)
    line_rest = float(pfit.get_line_rest_wave(pg_ion))

    out: Dict[str, Any] = {
        "saved_line": saved_line,
        "pg_ion": pg_ion,
        "line_group": group_name,
        "status": "failed",
        "error_message": "",
        "records": [],
        "arrays": {},
        "params": LINE_FIT_PARAMS[saved_line],
    }

    try:
        raw_wave, raw_flux, _raw_meta = load_individual_line_spectrum_h5(spectrum_file, saved_line, "raw")
        lsf_wave, lsf_flux, load_meta = load_individual_line_spectrum_h5(spectrum_file, saved_line, "lsf")

        row_table, diag = pfit.fit_line_in_spectrum(
            cfg=cfg_line,
            wave=lsf_wave,
            flux_clean=lsf_flux,
            source_file=spectrum_file,
            saved_line=saved_line,
            pg_ion=pg_ion,
            file_seed_offset=1000 * iline,
        )

        records = [
            normalize_fit_row(row, cfg_line, spectrum_file, saved_line, "", load_meta=load_meta)
            for row in row_table
        ]
        detection = any(row_is_detection(row) for row in records)
        out["status"] = "detection" if detection else "non_detection"
        out["records"] = records

        arrays: Dict[str, Any] = {
            "raw": window_arrays(pfit, raw_wave, raw_flux, line_rest, cfg_line.z, cfg_line.velocity_window),
            "lsf": window_arrays(pfit, lsf_wave, lsf_flux, line_rest, cfg_line.z, cfg_line.velocity_window),
            "obs": {
                "velocity_kms": np.asarray(diag["velocity"], dtype=float) if diag is not None else np.array([]),
                "wave_A": np.asarray(diag["wave_rest"], dtype=float) if diag is not None else np.array([]),
                "flux": np.asarray(diag["flux_noisy"], dtype=float) if diag is not None else np.array([]),
                "error": np.asarray(diag["error"], dtype=float) if diag is not None else np.array([]),
            },
            "model": {
                "total_flux": np.array([]),
                "component_flux": np.empty((0, 0), dtype=float),
                "component_velocity_kms": np.array([]),
            },
        }

        if diag is not None and diag.get("fit") is not None and len(diag["fit"]["N"]) > 0:
            fit = diag["fit"]
            line_data = pfit.pg.analysis.absorption_spectra.lines[pg_ion]
            params = pfit.generate_params_from_fit(fit)
            tau_total = pfit.pg.analysis.vpfit.model_tau(line_data, params, diag["wave_rest"], mode="Voigt")
            arrays["model"]["total_flux"] = np.exp(-tau_total)

            comps = []
            comp_vel = []
            for j in range(len(fit["N"])):
                one = np.array([float(fit["N"][j]), float(fit["b"][j]), float(fit["l"][j])])
                tau = pfit.pg.analysis.vpfit.model_tau(line_data, one, diag["wave_rest"], mode="Voigt")
                comps.append(np.exp(-tau))
                comp_vel.append(float(pfit.wave_to_vel(float(fit["l"][j]), line_rest)))
            arrays["model"]["component_flux"] = np.asarray(comps, dtype=float)
            arrays["model"]["component_velocity_kms"] = np.asarray(comp_vel, dtype=float)

        out["arrays"] = arrays
        out["n_regions"] = int(diag.get("n_regions", 0)) if diag is not None else 0
        out["n_components"] = int(arrays["model"]["component_flux"].shape[0])

    except Exception as exc:
        out["status"] = "failed"
        out["error_message"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()

    return out


def h5_write_str_array(group: h5py.Group, name: str, values: Sequence[Any]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray([stringify(v) for v in values], dtype=object), dtype=dtype)


def h5_write_rows(group: h5py.Group, name: str, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    data = np.asarray([[stringify(row.get(field, "")) for field in fields] for row in rows], dtype=object)
    group.create_dataset(name, data=data, dtype=dtype)


def h5_write_array(group: h5py.Group, name: str, arr: Any) -> None:
    arr = np.asarray(arr)
    if name in group:
        del group[name]
    group.create_dataset(name, data=arr, compression="gzip", compression_opts=4, shuffle=True)


def write_fit_h5(path: Path, *, spectrum_file: str, args: argparse.Namespace, line_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = parse_spectrum_filename(spectrum_file)
    all_rows: List[Dict[str, Any]] = []

    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "m61_individual_line_parallel_fit_v1"
        h5.attrs["spectrum_file"] = spectrum_file
        h5.attrs["sid"] = int(meta.get("sid") or args.sid)
        h5.attrs["snap"] = int(args.snap)
        h5.attrs["run_label"] = str(args.run_label)
        h5.attrs["mode"] = str(meta.get("mode", "unknown"))
        h5.attrs["alpha"] = int(meta.get("alpha", -1))
        h5.attrs["snr"] = float(args.snr)
        h5.attrs["bin_before_noise"] = bool(args.bin_before_noise)
        h5.attrs["bin_npix"] = int(args.bin_npix)
        h5.attrs["n_workers"] = int(args.n_workers)

        h5_write_str_array(h5, "result_fieldnames", RESULT_FIELDNAMES)

        lines_group = h5.create_group("lines")
        status_rows = []
        for result in line_results:
            line_group = str(result["line_group"])
            group = lines_group.create_group(line_group)
            group.attrs["saved_line_label"] = str(result["saved_line"])
            group.attrs["pygad_ion_key"] = str(result["pg_ion"])
            group.attrs["fit_status"] = str(result["status"])
            group.attrs["non_detection"] = str(result["status"]) == "non_detection"
            group.attrs["error_message"] = str(result.get("error_message", ""))
            group.attrs["n_result_rows"] = int(len(result.get("records", [])))
            group.attrs["n_components"] = int(result.get("n_components", 0))
            group.attrs["n_regions"] = int(result.get("n_regions", 0))
            for key, value in result.get("params", {}).items():
                group.attrs[f"fit_{key}"] = json.dumps(value)

            rows = list(result.get("records", []))
            all_rows.extend(rows)
            h5_write_rows(group, "fit_rows", rows, RESULT_FIELDNAMES)

            arrays = result.get("arrays", {})
            for branch_name in ["raw", "lsf", "obs"]:
                branch = group.create_group(branch_name)
                for key, arr in arrays.get(branch_name, {}).items():
                    h5_write_array(branch, key, arr)
            model = group.create_group("model")
            for key, arr in arrays.get("model", {}).items():
                h5_write_array(model, key, arr)

            status_rows.append(
                {
                    "line_group": line_group,
                    "saved_line_label": result["saved_line"],
                    "pygad_ion_key": result["pg_ion"],
                    "fit_status": result["status"],
                    "non_detection": str(result["status"] == "non_detection"),
                    "n_result_rows": len(rows),
                    "n_components": result.get("n_components", 0),
                    "n_regions": result.get("n_regions", 0),
                    "error_message": result.get("error_message", ""),
                }
            )

        h5_write_rows(h5, "all_fit_rows", all_rows, RESULT_FIELDNAMES)
        status_fields = [
            "line_group",
            "saved_line_label",
            "pygad_ion_key",
            "fit_status",
            "non_detection",
            "n_result_rows",
            "n_components",
            "n_regions",
            "error_message",
        ]
        h5_write_str_array(h5, "line_status_fieldnames", status_fields)
        h5_write_rows(h5, "line_status_rows", status_rows, status_fields)
    return all_rows


def plot_h5_file(path: Path, output_dir: Path | None = None, dpi: int = 180) -> List[str]:
    if output_dir is None:
        output_dir = plots_dir_for_h5(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    with h5py.File(path, "r") as h5:
        sid = h5.attrs.get("sid", "unknown")
        mode = h5.attrs.get("mode", "unknown")
        alpha = h5.attrs.get("alpha", "unknown")
        for line_group, group in h5["lines"].items():
            saved_line = group.attrs["saved_line_label"]
            pg_ion = group.attrs["pygad_ion_key"]
            status = group.attrs["fit_status"]
            params = LINE_FIT_PARAMS[str(saved_line)]
            window = float(params["velocity_window"])

            fig, ax = plt.subplots(figsize=(12.5, 6.4), constrained_layout=True)
            if "velocity_kms" in group["raw"]:
                ax.plot(group["raw/velocity_kms"][:], group["raw/flux"][:], color="0.65", lw=1.1, label="Raw")
            if "velocity_kms" in group["lsf"]:
                ax.plot(group["lsf/velocity_kms"][:], group["lsf/flux"][:], color="black", lw=1.7, label="Clean LSF")
            if "velocity_kms" in group["obs"]:
                v = group["obs/velocity_kms"][:]
                f = group["obs/flux"][:]
                e = group["obs/error"][:]
                ax.step(v, f, where="mid", lw=1.8, color="#1f77b4", label="Binned + noise")
                ax.fill_between(v, f - e, f + e, step="mid", alpha=0.16, color="#1f77b4", linewidth=0)
                if len(group["model/total_flux"][:]):
                    ax.plot(v, group["model/total_flux"][:], color="#d62728", lw=2.7, label="Total Voigt fit")
                    comps = group["model/component_flux"][:]
                    comp_vel = group["model/component_velocity_kms"][:]
                    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(comps))))
                    for idx, comp in enumerate(comps):
                        label = f"component {idx + 1}"
                        if idx < len(comp_vel):
                            label += f": v={comp_vel[idx]:.0f} km/s"
                            ax.axvline(comp_vel[idx], color=colors[idx], lw=1.0, ls=":", alpha=0.85)
                        ax.plot(v, comp, color=colors[idx], lw=1.3, ls="--", alpha=0.9, label=label)
            ax.axhline(1.0, color="0.35", ls=":", lw=1.1)
            ax.axvline(0.0, color="0.25", ls=":", lw=1.1)
            ax.set_xlim(-window, window)
            ax.set_ylim(-0.08, 1.35)
            ax.set_xlabel(r"$v - v_{\rm line}$ [km s$^{-1}$]")
            ax.set_ylabel("Normalized flux")
            ax.set_title(f"SID {sid} | {mode} alpha={alpha} | {saved_line} | {status}")
            ax.legend(fontsize=9, loc="upper right")
            ax.minorticks_on()
            out = output_dir / f"{path.stem}_{line_group}_fit_overlay.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(str(out))
    return written


def fit_one_spectrum(args: argparse.Namespace, spectrum_file: str) -> Path:
    meta = parse_spectrum_filename(spectrum_file)
    sid = int(meta.get("sid") or args.sid)
    file_seed = stable_seed(spectrum_file, int(args.seed))
    lines = selected_lines(args.line_labels)
    out_h5 = Path(args.output_h5) if args.output_h5 else spectrum_fit_h5_path(
        args.base_dir, sid, args.snap, args.run_label, args.output_subdir, spectrum_file
    )

    if out_h5.exists() and not args.overwrite:
        print(f"[SKIP] output exists: {out_h5}")
        return out_h5

    worker_args = {
        "snap": args.snap,
        "run_label": args.run_label,
        "base_dir": args.base_dir,
        "z": args.z,
        "snr": args.snr,
        "seed": args.seed,
        "bin_before_noise": args.bin_before_noise,
        "bin_npix": args.bin_npix,
        "min_region_width": args.min_region_width,
        "N_sigma": args.N_sigma,
        "output_subdir": args.output_subdir,
        "make_plots": False,
        "verbose": args.verbose,
    }
    payloads = [
        {
            "args": worker_args,
            "spectrum_file": spectrum_file,
            "saved_line": line,
            "sid": sid,
            "iline": idx,
            "file_seed": file_seed,
        }
        for idx, line in enumerate(lines)
    ]

    print(f"[FIT] {spectrum_file}")
    print(f"[FIT] lines={len(payloads)} workers={args.n_workers}")
    with ProcessPoolExecutor(max_workers=int(args.n_workers)) as executor:
        futures = {executor.submit(worker_fit_line, payload): payload for payload in payloads}
        results = []
        for future in as_completed(futures):
            result = future.result()
            print(
                f"[LINE] {result['saved_line']} -> {result['status']} "
                f"rows={len(result.get('records', []))} comps={result.get('n_components', 0)}"
            )
            results.append(result)

    order = {line_group_name(line): idx for idx, line in enumerate(lines)}
    results = sorted(results, key=lambda item: order.get(item["line_group"], 10**6))
    write_fit_h5(out_h5, spectrum_file=spectrum_file, args=args, line_results=results)
    if args.make_plots:
        plots = plot_h5_file(out_h5, output_dir=Path(args.plot_dir) if args.plot_dir else None)
        print(f"[PLOTS] wrote {len(plots)}")
    print(f"[OUT] {out_h5}")
    return out_h5


def fit_from_task(args: argparse.Namespace) -> int:
    row = select_task_row(args.task_list, int(args.task_index))
    args.sid = int(row["sid"])
    args.snap = int(row["snap"])
    args.run_label = str(row["run_label"])
    fit_one_spectrum(args, row["spectrum_file"])
    return 0


def fit_sid_index(args: argparse.Namespace) -> int:
    files = discover_spectrum_files(
        base_dir=args.base_dir,
        sid=int(args.sid),
        snap=int(args.snap),
        run_label=str(args.run_label),
        mode=args.mode,
        alpha=args.alpha,
        max_files=None,
    )
    index = int(args.spectrum_index)
    if index < 1:
        raise ValueError("--spectrum-index is 1-based and must be >= 1")
    if index > len(files):
        print(
            f"[SKIP] sid{args.sid} spectrum index {index} exceeds discovered spectra count {len(files)}"
        )
        return 0
    print(f"[DISCOVER] sid{args.sid}: {len(files)} spectra; fitting index {index}")
    fit_one_spectrum(args, files[index - 1])
    return 0


def fit_files(args: argparse.Namespace) -> int:
    files = [item.strip() for item in str(args.spectrum_files).split(",") if item.strip()]
    if not files:
        raise RuntimeError("--spectrum-files is required for fit-files")
    for spectrum_file in files:
        fit_one_spectrum(args, spectrum_file)
    return 0


def plot_h5(args: argparse.Namespace) -> int:
    files = [Path(item.strip()) for item in str(args.fit_h5_files).split(",") if item.strip()]
    if not files:
        raise RuntimeError("--fit-h5-files is required for plot-h5")
    total = 0
    for path in files:
        plots = plot_h5_file(path, output_dir=Path(args.plot_dir) if args.plot_dir else None)
        print(f"[PLOTS] {path}: {len(plots)}")
        total += len(plots)
    print(f"[DONE] total plots={total}")
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sid", type=int, default=348901)
    parser.add_argument("--snap", type=int, default=DEFAULT_SNAP)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-subdir", default="fitted_individual_line_spectra_parallel_snr10_bin3")
    parser.add_argument("--output-h5", default="")
    parser.add_argument("--line-labels", default="all")
    parser.add_argument("--z", type=float, default=DEFAULT_Z)
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bin-before-noise", action=argparse.BooleanOptionalAction, default=DEFAULT_BIN_BEFORE_NOISE)
    parser.add_argument("--bin-npix", type=int, default=DEFAULT_BIN_NPIX)
    parser.add_argument("--N-sigma", dest="N_sigma", type=float, default=DEFAULT_N_SIGMA)
    parser.add_argument("--min-region-width", type=int, default=DEFAULT_MIN_REGION_WIDTH)
    parser.add_argument("--n-workers", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--plot-dir", default="")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_task = sub.add_parser("fit-task", help="Fit one task-list row.")
    add_common(p_task)
    p_task.add_argument("--task-list", default=DEFAULT_TASK_LIST)
    p_task.add_argument("--task-index", type=int, required=True)

    p_sid = sub.add_parser("fit-sid-index", help="Discover one SID's spectra and fit a 1-based spectrum index.")
    add_common(p_sid)
    p_sid.add_argument("--spectrum-index", type=int, required=True)
    p_sid.add_argument("--mode", choices=["all", "flip", "noflip"], default="all")
    p_sid.add_argument("--alpha", default="all")

    p_files = sub.add_parser("fit-files", help="Fit comma-separated spectrum HDF5 files.")
    add_common(p_files)
    p_files.add_argument("--spectrum-files", required=True)

    p_plot = sub.add_parser("plot-h5", help="Regenerate visual QA plots from fit HDF5 files.")
    p_plot.add_argument("--fit-h5-files", required=True)
    p_plot.add_argument("--plot-dir", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "fit-task":
        return fit_from_task(args)
    if args.command == "fit-sid-index":
        return fit_sid_index(args)
    if args.command == "fit-files":
        return fit_files(args)
    if args.command == "plot-h5":
        return plot_h5(args)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
