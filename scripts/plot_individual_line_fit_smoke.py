#!/usr/bin/env python3
"""Visual smoke test for individual-line saved-spectrum fitting.

This script fits the exact line list used by batch/run_spectra_array.sh from
the per-line HDF5 branches:

    /spectrum_by_line/<line_group>/raw
    /spectrum_by_line/<line_group>/lsf

For each spectrum and line it saves a velocity-space diagnostic plot showing
the raw individual-line spectrum, the clean LSF-convolved spectrum, the binned
SNR-noisy observation-like spectrum passed to pygad, the total Voigt model, and
the individual Voigt components.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/m61_fit_visual_smoke_mpl")

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
    DEFAULT_RUN_LABEL,
    DEFAULT_SNAP,
    DEFAULT_SNR,
    DEFAULT_Z,
    LINE_FIT_PARAMS,
    LINE_MAP,
    RESULT_FIELDNAMES,
    discover_spectrum_files,
    parse_spectrum_filename,
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


PLOT_SUMMARY_FIELDS = [
    "spectrum_file",
    "SID",
    "mode",
    "alpha",
    "saved_line_label",
    "pygad_ion_key",
    "source_line_group",
    "fit_status",
    "n_result_rows",
    "n_components",
    "n_regions",
    "plot_png",
    "error_message",
]


def sanitize(text: Any) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
        .replace(":", "")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )


def read_spectra_lines_from_batch(batch_script: Path) -> List[str]:
    text = batch_script.read_text(encoding="utf-8")
    match = re.search(r'^LINES_CSV="([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find LINES_CSV in {batch_script}")
    labels = [item.strip() for item in match.group(1).split(",") if item.strip()]
    missing = [label for label in labels if label not in LINE_MAP]
    if missing:
        raise RuntimeError(
            f"LINES_CSV contains labels missing from individual-line fitter: {missing}"
        )
    return labels


def parse_file_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def wave_to_vel(wave_rest: np.ndarray, line_rest: float) -> np.ndarray:
    return C_KMS * (np.asarray(wave_rest, dtype=float) / float(line_rest) - 1.0)


def rest_and_window(pfit: Any, wave: np.ndarray, flux: np.ndarray, line_rest: float, z: float, window: float):
    wave_rest, _ = pfit.convert_wave_to_rest_if_needed(wave, line_rest=line_rest, z=z)
    vel = wave_to_vel(wave_rest, line_rest)
    mask = np.isfinite(vel) & np.isfinite(flux) & (vel >= -window) & (vel <= window)
    return vel[mask], np.asarray(flux, dtype=float)[mask], wave_rest[mask]


def binned_clean_lsf(pfit: Any, cfg_line: Any, lsf_wave: np.ndarray, lsf_flux: np.ndarray, line_rest: float):
    wave_rest, _ = pfit.convert_wave_to_rest_if_needed(lsf_wave, line_rest=line_rest, z=cfg_line.z)
    if cfg_line.bin_before_noise:
        wave_bin, flux_bin = pfit.bin_spectrum_npix(wave_rest, lsf_flux, npix=cfg_line.bin_npix)
    else:
        wave_bin, flux_bin = wave_rest.copy(), np.asarray(lsf_flux, dtype=float).copy()
    vel_bin = wave_to_vel(wave_bin, line_rest)
    mask = (
        np.isfinite(vel_bin)
        & np.isfinite(flux_bin)
        & (vel_bin >= -cfg_line.velocity_window)
        & (vel_bin <= cfg_line.velocity_window)
    )
    return vel_bin[mask], flux_bin[mask], wave_bin[mask]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.titlesize": 16,
            "legend.fontsize": 10,
            "axes.linewidth": 1.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
        }
    )


def plot_one_line(
    *,
    pfit: Any,
    cfg_line: Any,
    spectrum_file: str,
    saved_line: str,
    pg_ion: str,
    raw_wave: np.ndarray,
    raw_flux: np.ndarray,
    lsf_wave: np.ndarray,
    lsf_flux: np.ndarray,
    diag: Dict[str, Any] | None,
    output_dir: Path,
    fit_rows: Sequence[Dict[str, Any]],
    status: str,
    error_message: str,
) -> Path:
    meta = parse_spectrum_filename(spectrum_file)
    line_rest = float(pfit.get_line_rest_wave(pg_ion))
    window = float(cfg_line.velocity_window)

    v_raw, f_raw, _ = rest_and_window(pfit, raw_wave, raw_flux, line_rest, cfg_line.z, window)
    v_lsf, f_lsf, _ = rest_and_window(pfit, lsf_wave, lsf_flux, line_rest, cfg_line.z, window)

    fig, ax = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True)
    if len(v_raw):
        ax.plot(v_raw, f_raw, color="0.62", lw=1.2, alpha=0.8, label="Raw individual-line spectrum")
    if len(v_lsf):
        ax.plot(v_lsf, f_lsf, color="black", lw=1.8, alpha=0.9, label="Clean LSF-convolved spectrum")

    if diag is not None:
        v_obs = np.asarray(diag["velocity"], dtype=float)
        f_obs = np.asarray(diag["flux_noisy"], dtype=float)
        err = np.asarray(diag["error"], dtype=float)
        ax.step(
            v_obs,
            f_obs,
            where="mid",
            color="#1f77b4",
            lw=1.8,
            alpha=0.95,
            label=rf"Binned + noise observation (SNR={cfg_line.snr:g})",
        )
        ax.fill_between(
            v_obs,
            f_obs - err,
            f_obs + err,
            step="mid",
            color="#1f77b4",
            alpha=0.16,
            linewidth=0,
            label=r"$1\sigma$ flux error",
        )

        fit = diag.get("fit")
        if fit is not None and len(fit["N"]) > 0:
            line_data = pfit.pg.analysis.absorption_spectra.lines[pg_ion]
            params = pfit.generate_params_from_fit(fit)
            tau_total = pfit.pg.analysis.vpfit.model_tau(line_data, params, diag["wave_rest"], mode="Voigt")
            total_flux = np.exp(-tau_total)
            ax.plot(v_obs, total_flux, color="#d62728", lw=2.8, label="Total Voigt fit")

            colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(fit["N"]))))
            for j in range(len(fit["N"])):
                comp_params = np.array([float(fit["N"][j]), float(fit["b"][j]), float(fit["l"][j])])
                tau_comp = pfit.pg.analysis.vpfit.model_tau(
                    line_data,
                    comp_params,
                    diag["wave_rest"],
                    mode="Voigt",
                )
                comp_flux = np.exp(-tau_comp)
                v_j = float(pfit.wave_to_vel(float(fit["l"][j]), line_rest))
                ax.plot(
                    v_obs,
                    comp_flux,
                    color=colors[j],
                    lw=1.4,
                    ls="--",
                    alpha=0.9,
                    label=f"component {j + 1}: v={v_j:.0f} km/s",
                )
                ax.axvline(v_j, color=colors[j], lw=1.1, ls=":", alpha=0.85)

    ax.axhline(1.0, color="0.35", lw=1.2, ls=":")
    ax.axvline(0.0, color="0.25", lw=1.2, ls=":")
    ax.set_xlim(-window, window)
    ax.set_ylim(-0.08, 1.35)
    ax.set_xlabel(r"$v - v_{\rm line}$ [km s$^{-1}$]")
    ax.set_ylabel("Normalized flux")
    ax.minorticks_on()
    title = (
        f"SID {meta.get('sid', cfg_line.sid)} | {meta.get('mode', 'unknown')} "
        f"alpha={meta.get('alpha', 'NA')} | {saved_line} | window=+/-{window:g} km/s"
    )
    ax.set_title(title)

    fit_text = f"{status}"
    if fit_rows:
        detections = [
            row
            for row in fit_rows
            if str(row.get("UpLim", "")).lower() not in {"true", "1"} and row.get("logN", "") != ""
        ]
        fit_text += f"\nrows={len(fit_rows)}, detections={len(detections)}"
    if error_message:
        fit_text += f"\n{error_message[:90]}"
    ax.text(
        0.02,
        0.04,
        fit_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.4", alpha=0.9),
    )
    ax.legend(loc="upper right", ncols=1)

    outdir = output_dir / f"sid{meta.get('sid', cfg_line.sid)}" / str(meta.get("mode", "mode")) / f"alpha{int(meta.get('alpha', -1)):03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / (
        f"{Path(spectrum_file).stem}_{sanitize(saved_line)}_individual_line_fit_overlay.png"
    )
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outpath


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> int:
    configure_matplotlib()
    pfit = import_pfit(REPO_ROOT)

    line_labels = read_spectra_lines_from_batch(REPO_ROOT / "batch" / "run_spectra_array.sh")
    if args.line_labels.strip().lower() not in {"all", "*"}:
        requested = [item.strip() for item in args.line_labels.split(",") if item.strip()]
        bad = [item for item in requested if item not in line_labels]
        if bad:
            raise ValueError(f"Requested labels are not in run_spectra_array.sh LINES_CSV: {bad}")
        line_labels = requested

    if args.spectrum_files:
        spectra = parse_file_list(args.spectrum_files)
    else:
        spectra = discover_spectrum_files(
            str(args.base_dir),
            int(args.sid),
            snap=int(args.snap),
            run_label=str(args.run_label),
            mode=args.mode,
            alpha=args.alpha,
            max_files=int(args.max_spectra),
        )
    spectra = sorted(spectra)[: int(args.max_spectra)]
    if not spectra:
        raise RuntimeError("No spectra selected for visual smoke test.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows: List[Dict[str, Any]] = []
    plot_rows: List[Dict[str, Any]] = []

    for ispec, spectrum_file in enumerate(spectra, start=1):
        meta = parse_spectrum_filename(spectrum_file)
        sid = int(meta.get("sid") or args.sid)
        file_seed = stable_seed(spectrum_file, int(args.seed))
        base_args = argparse.Namespace(
            snap=int(args.snap),
            run_label=str(args.run_label),
            base_dir=str(args.base_dir),
            z=float(args.z),
            snr=float(args.snr),
            seed=int(args.seed),
            bin_before_noise=bool(args.bin_before_noise),
            bin_npix=int(args.bin_npix),
            min_region_width=int(args.min_region_width),
            N_sigma=float(args.N_sigma),
            output_subdir="individual_line_visual_smoke",
            make_plots=False,
            verbose=bool(args.verbose),
        )
        base_cfg = make_base_config(pfit, base_args, sid=sid, seed=file_seed)

        print(f"[{ispec}/{len(spectra)}] {spectrum_file}")
        for iline, saved_line in enumerate(line_labels):
            pg_ion = LINE_MAP[saved_line]
            cfg_line = line_config(pfit, base_cfg, saved_line, spectrum_file)
            cfg_line = replace(cfg_line, make_plots=False)
            status = "succeeded"
            error_message = ""
            line_records: List[Dict[str, Any]] = []
            diag = None
            plot_png = ""
            try:
                raw_wave, raw_flux, _ = load_individual_line_spectrum_h5(
                    spectrum_file, saved_line=saved_line, branch="raw"
                )
                lsf_wave, lsf_flux, load_meta = load_individual_line_spectrum_h5(
                    spectrum_file, saved_line=saved_line, branch="lsf"
                )
                row_table, diag = pfit.fit_line_in_spectrum(
                    cfg=cfg_line,
                    wave=lsf_wave,
                    flux_clean=lsf_flux,
                    source_file=spectrum_file,
                    saved_line=saved_line,
                    pg_ion=pg_ion,
                    file_seed_offset=1000 * iline,
                )
                for row in row_table:
                    record = normalize_fit_row(
                        row=row,
                        cfg=cfg_line,
                        spectrum_file=spectrum_file,
                        saved_line=saved_line,
                        output_csv=str(output_dir / "all_fit_rows.csv"),
                        load_meta=load_meta,
                        fit_status=status,
                        error_message="",
                    )
                    result_rows.append(record)
                    line_records.append(record)
                plot_png = str(
                    plot_one_line(
                        pfit=pfit,
                        cfg_line=cfg_line,
                        spectrum_file=spectrum_file,
                        saved_line=saved_line,
                        pg_ion=pg_ion,
                        raw_wave=raw_wave,
                        raw_flux=raw_flux,
                        lsf_wave=lsf_wave,
                        lsf_flux=lsf_flux,
                        diag=diag,
                        output_dir=output_dir / "plots",
                        fit_rows=line_records,
                        status=status,
                        error_message=error_message,
                    )
                )
            except Exception as exc:
                status = "failed"
                error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                print(f"[FAILED] {Path(spectrum_file).name} | {saved_line}: {error_message}", file=sys.stderr)

            n_components = 0
            n_regions = 0
            if diag is not None and diag.get("fit") is not None:
                n_components = int(len(diag["fit"]["N"]))
                n_regions = int(diag.get("n_regions", 0))
            plot_rows.append(
                {
                    "spectrum_file": spectrum_file,
                    "SID": sid,
                    "mode": meta.get("mode", "unknown"),
                    "alpha": meta.get("alpha", -1),
                    "saved_line_label": saved_line,
                    "pygad_ion_key": pg_ion,
                    "source_line_group": line_group_name(saved_line),
                    "fit_status": status,
                    "n_result_rows": len(line_records),
                    "n_components": n_components,
                    "n_regions": n_regions,
                    "plot_png": plot_png,
                    "error_message": error_message,
                }
            )

    write_dict_rows(str(output_dir / "all_fit_rows.csv"), result_rows, RESULT_FIELDNAMES)
    write_csv(output_dir / "plot_summary.csv", plot_rows, PLOT_SUMMARY_FIELDS)
    write_json(
        output_dir / "visual_smoke_metadata.json",
        {
            "script": str(Path(__file__).resolve()),
            "spectra": spectra,
            "line_labels_from_run_spectra_array": line_labels,
            "output_dir": str(output_dir),
            "snr": float(args.snr),
            "bin_before_noise": bool(args.bin_before_noise),
            "bin_npix": int(args.bin_npix),
            "base_dir": str(args.base_dir),
            "run_label": str(args.run_label),
            "snap": int(args.snap),
            "n_plots_expected": len(spectra) * len(line_labels),
            "n_plots_written": int(sum(1 for row in plot_rows if row.get("plot_png"))),
        },
    )
    print(f"[DONE] wrote {len(result_rows)} fit rows")
    print(f"[DONE] wrote {sum(1 for row in plot_rows if row.get('plot_png'))} plots")
    print(f"[OUT] {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", type=int, default=348901)
    parser.add_argument("--snap", type=int, default=DEFAULT_SNAP)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--mode", choices=["flip", "noflip"], default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--max-spectra", type=int, default=2)
    parser.add_argument("--spectrum-files", default="")
    parser.add_argument(
        "--output-dir",
        default="/scratch/tsingh65/m61-tng/outputs/individual_line_fit_visual_smoke_L2Rvir",
    )
    parser.add_argument("--line-labels", default="all")
    parser.add_argument("--z", type=float, default=DEFAULT_Z)
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bin-before-noise", action=argparse.BooleanOptionalAction, default=DEFAULT_BIN_BEFORE_NOISE)
    parser.add_argument("--bin-npix", type=int, default=DEFAULT_BIN_NPIX)
    parser.add_argument("--N-sigma", dest="N_sigma", type=float, default=DEFAULT_N_SIGMA)
    parser.add_argument("--min-region-width", type=int, default=DEFAULT_MIN_REGION_WIDTH)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
