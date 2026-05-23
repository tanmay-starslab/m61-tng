#!/usr/bin/env python3
"""Quick first-look diagnostics for currently available spectrum-fit outputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ION_ORDER = [
    "Si II 1190",
    "Si II 1193",
    "Si III 1206",
    "N V 1239",
    "Si II 1260",
    "O I 1302",
    "C II 1335",
    "Si IV 1403",
    "H I 1216",
]

LINE_TO_ION = {
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

CANONICAL_COLUMNS = [
    "SID",
    "RayID",
    "Mode",
    "Alpha",
    "Filename",
    "SavedLine",
    "PygadIon",
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
    "SourceFile",
    "result_csv",
]

ALIASES = {
    "SID": ["SID", "sid"],
    "RayID": ["RayID", "ray_id"],
    "Mode": ["Mode", "mode"],
    "Alpha": ["Alpha", "alpha"],
    "Filename": ["Filename", "source_filename"],
    "SavedLine": ["SavedLine", "saved_line_label"],
    "PygadIon": ["PygadIon", "pygad_ion_key"],
    "EW_mA": ["EW_mA"],
    "dEW_mA": ["dEW_mA"],
    "logN": ["logN"],
    "dlogN": ["dlogN"],
    "b_kms": ["b_kms"],
    "db_kms": ["db_kms"],
    "v_kms": ["v_kms"],
    "dv_kms": ["dv_kms"],
    "lambda_A": ["lambda_A"],
    "dlambda_A": ["dlambda_A"],
    "UpLim": ["UpLim"],
    "Sat": ["Sat"],
    "Chisq": ["Chisq"],
    "Nregions": ["Nregions"],
    "WaveFrame": ["WaveFrame"],
    "Binned": ["Binned"],
    "BinNpix": ["BinNpix"],
    "SourceFile": ["SourceFile", "source_spectrum_path"],
}

NUMERIC_COLUMNS = [
    "SID",
    "Alpha",
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
    "Chisq",
    "Nregions",
    "BinNpix",
]

BOOL_COLUMNS = ["UpLim", "Sat", "Binned"]

ABSORBER_KEY_COLUMNS = ["SID", "RayID", "Mode", "Alpha", "SavedLine"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine available per-spectrum fit tables and make quick diagnostics."
    )
    parser.add_argument("--repo", default="/home/tsingh65/m61-tng")
    parser.add_argument("--base-dir", default="/scratch/tsingh65/m61-tng/outputs")
    parser.add_argument("--snap", type=int, default=99)
    parser.add_argument("--run-label", default="L4Rvir")
    parser.add_argument("--output-subdir", default="fitted_spectra_snr10_bin3")
    parser.add_argument(
        "--task-list",
        default="/home/tsingh65/m61-tng/batch/fit_spectra_tasks_snap99_L4Rvir.tsv",
    )
    parser.add_argument(
        "--outdir",
        default="/scratch/tsingh65/m61-tng/outputs/quick_diagnostics_snap99_L4Rvir_snr10_bin3",
    )
    parser.add_argument("--force-recombine", action="store_true")
    parser.add_argument("--make-individual-ion-plots", action="store_true")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        }
    )


def print_progress(message: str) -> None:
    print(f"[quick-fit] {message}", flush=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def parse_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def first_existing_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lower_to_original = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name in df.columns:
            return name
        low = name.lower()
        if low in lower_to_original:
            return lower_to_original[low]
    return None


def find_result_files(base_dir: Path, snap: int, run_label: str, output_subdir: str) -> List[Path]:
    pattern = f"sid*/rays_and_spectra_sid*_snap{snap}_{run_label}/{output_subdir}/**/*.csv"
    files = []
    for path in base_dir.glob(pattern):
        if not path.is_file():
            continue
        low = str(path).lower()
        name = path.name.lower()
        if not name.endswith("_fits.csv"):
            continue
        if any(word in low for word in ("combined", "summary", "_status.csv")):
            continue
        files.append(path)
    return sorted(set(files))


def read_fit_csvs(files: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    n_files = len(files)
    for i, path in enumerate(files, start=1):
        if i == 1 or i % 1000 == 0 or i == n_files:
            print_progress(f"reading fit CSV {i}/{n_files}")
        try:
            df = pd.read_csv(path)
            df["result_csv"] = str(path)
            frames.append(df)
        except Exception as exc:
            print_progress(f"warning: could not read {path}: {exc}")
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    for canonical, names in ALIASES.items():
        src = first_existing_column(raw, names)
        if src is None:
            out[canonical] = np.nan
        else:
            out[canonical] = raw[src]

    if "result_csv" in raw.columns:
        out["result_csv"] = raw["result_csv"]
    else:
        out["result_csv"] = ""

    if out["PygadIon"].isna().all():
        out["PygadIon"] = out["SavedLine"].map(LINE_TO_ION)
    else:
        missing_ion = out["PygadIon"].isna() | (out["PygadIon"].astype(str).str.strip() == "")
        out.loc[missing_ion, "PygadIon"] = out.loc[missing_ion, "SavedLine"].map(LINE_TO_ION)

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in BOOL_COLUMNS:
        out[col] = out[col].map(parse_bool)

    for col in ["RayID", "Mode", "Filename", "SavedLine", "PygadIon", "WaveFrame", "SourceFile"]:
        out[col] = out[col].fillna("").astype(str)

    out["abs_v_kms"] = out["v_kms"].abs()
    out["is_detection"] = ~out["UpLim"]
    out["is_upper_limit"] = out["UpLim"]
    out["line_order"] = out["SavedLine"].map({line: i for i, line in enumerate(ION_ORDER)}).fillna(999)
    return out


def current_combined_is_fresh(combined_csv: Path, result_files: Sequence[Path]) -> bool:
    if not combined_csv.exists() or not result_files:
        return False
    combined_mtime = combined_csv.stat().st_mtime
    newest_result_mtime = max(path.stat().st_mtime for path in result_files)
    return combined_mtime >= newest_result_mtime


def combine_or_load(result_files: Sequence[Path], outdir: Path, force: bool) -> pd.DataFrame:
    combined_csv = outdir / "combined_fit_results_available.csv"
    combined_parquet = outdir / "combined_fit_results_available.parquet"

    if not force and current_combined_is_fresh(combined_csv, result_files):
        print_progress(f"using fresh cached combined table: {combined_csv}")
        df = pd.read_csv(combined_csv)
    else:
        print_progress(f"combining {len(result_files)} currently available per-spectrum fit CSVs")
        raw = read_fit_csvs(result_files)
        df = normalize_columns(raw)
        df.to_csv(combined_csv, index=False)
        print_progress(f"wrote {combined_csv}")

    try:
        df.to_parquet(combined_parquet, index=False)
        print_progress(f"wrote {combined_parquet}")
    except Exception as exc:
        print_progress(f"warning: could not write parquet: {exc}")

    return normalize_columns(df)


def ordered_lines(df: pd.DataFrame) -> List[str]:
    present = [line for line in ION_ORDER if line in set(df["SavedLine"])]
    extras = sorted(set(df["SavedLine"]) - set(ION_ORDER))
    return present + extras


def finite_values(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals[np.isfinite(vals)]


def percentile_or_nan(values: pd.Series, q: float) -> float:
    vals = finite_values(values)
    if len(vals) == 0:
        return np.nan
    return float(np.nanpercentile(vals, q))


def median_or_nan(values: pd.Series) -> float:
    vals = finite_values(values)
    if len(vals) == 0:
        return np.nan
    return float(np.nanmedian(vals))


def summary_stats_for_group(group: pd.DataFrame) -> Dict[str, Any]:
    absorber_keys = [col for col in ABSORBER_KEY_COLUMNS if col in group.columns]
    if absorber_keys:
        absorber_detected = group.groupby(absorber_keys, dropna=False)["is_detection"].any()
        n_absorbers = int(len(absorber_detected))
        n_absorbers_detected = int(absorber_detected.sum())
    else:
        # Fallback for schema edge-cases: treat each row as one absorber.
        n_absorbers = int(len(group))
        n_absorbers_detected = int(group["is_detection"].sum())
    n_absorbers_non_detected = int(n_absorbers - n_absorbers_detected)

    detections = group[group["is_detection"]]
    vdet = detections[np.isfinite(detections["abs_v_kms"])]
    n_rows = int(len(group))
    n_det = int(group["is_detection"].sum())
    n_ul = int(group["is_upper_limit"].sum())
    n_v = len(vdet)
    return {
        "n_rows": n_rows,
        "n_detections": n_det,
        "n_upper_limits": n_ul,
        "component_detection_fraction": float(n_det / n_rows) if n_rows else np.nan,
        "n_absorbers": n_absorbers,
        "n_absorbers_detected": n_absorbers_detected,
        "n_absorbers_non_detected": n_absorbers_non_detected,
        "detection_fraction": float(n_absorbers_detected / n_absorbers) if n_absorbers else np.nan,
        "median_logN_detected": median_or_nan(detections["logN"]),
        "p16_logN_detected": percentile_or_nan(detections["logN"], 16),
        "p84_logN_detected": percentile_or_nan(detections["logN"], 84),
        "median_b_kms_detected": median_or_nan(detections["b_kms"]),
        "p16_b_kms_detected": percentile_or_nan(detections["b_kms"], 16),
        "p84_b_kms_detected": percentile_or_nan(detections["b_kms"], 84),
        "median_abs_v_kms_detected": median_or_nan(detections["abs_v_kms"]),
        "p16_abs_v_kms_detected": percentile_or_nan(detections["abs_v_kms"], 16),
        "p84_abs_v_kms_detected": percentile_or_nan(detections["abs_v_kms"], 84),
        "frac_abs_v_0_50": float(((vdet["abs_v_kms"] >= 0) & (vdet["abs_v_kms"] < 50)).sum() / n_v) if n_v else np.nan,
        "frac_abs_v_50_100": float(((vdet["abs_v_kms"] >= 50) & (vdet["abs_v_kms"] < 100)).sum() / n_v) if n_v else np.nan,
        "frac_abs_v_ge_100": float((vdet["abs_v_kms"] >= 100).sum() / n_v) if n_v else np.nan,
        "n_saturated": int(group["Sat"].sum()),
    }


def make_summary_by_ion(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in ordered_lines(df):
        group = df[df["SavedLine"] == line]
        if len(group) == 0:
            continue
        row = {
            "SavedLine": line,
            "PygadIon": group["PygadIon"].replace("", np.nan).dropna().iloc[0]
            if group["PygadIon"].replace("", np.nan).dropna().size
            else LINE_TO_ION.get(line, ""),
        }
        row.update(summary_stats_for_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def make_summary_by_sid_ion(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["SID", "SavedLine"], dropna=False, sort=False)
    for (sid, line), group in grouped:
        row = {
            "SID": int(sid) if pd.notna(sid) else -1,
            "SavedLine": line,
            "PygadIon": group["PygadIon"].replace("", np.nan).dropna().iloc[0]
            if group["PygadIon"].replace("", np.nan).dropna().size
            else LINE_TO_ION.get(line, ""),
        }
        row.update(summary_stats_for_group(group))
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["line_order"] = out["SavedLine"].map({line: i for i, line in enumerate(ION_ORDER)}).fillna(999)
        out = out.sort_values(["SID", "line_order", "SavedLine"]).drop(columns=["line_order"])
    return out


def read_task_list(path: Path) -> pd.DataFrame:
    if not path.exists():
        print_progress(f"warning: task list not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def write_missing_tasks(task_df: pd.DataFrame, result_files: Sequence[Path], outdir: Path) -> pd.DataFrame:
    found = {str(path) for path in result_files}
    if task_df.empty or "output_file" not in task_df.columns:
        rough = pd.DataFrame(
            [
                {
                    "expected_tasks": len(task_df),
                    "result_files_found": len(found),
                    "rough_missing_count": max(len(task_df) - len(found), 0),
                    "note": "Exact task matching unavailable because task_list/output_file is missing.",
                }
            ]
        )
        rough.to_csv(outdir / "missing_or_unprocessed_tasks.csv", index=False)
        return rough

    task_df = task_df.copy()
    task_df["output_exists"] = task_df["output_file"].astype(str).isin(found)
    missing = task_df[~task_df["output_exists"]].copy()
    missing.to_csv(outdir / "missing_or_unprocessed_tasks.csv", index=False)
    return missing


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def no_data_plot(outdir: Path, stem: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_axis_off()
    save_figure(fig, outdir, stem)


def plot_detection_fraction(summary: pd.DataFrame, outdir: Path) -> None:
    if summary.empty:
        no_data_plot(outdir, "detection_fraction_by_ion", "Detection Fraction by Ion")
        return
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, summary["detection_fraction"], color="tab:blue", edgecolor="black", linewidth=0.8)
    for i, row in summary.iterrows():
        ax.text(
            i,
            min(float(row["detection_fraction"]) + 0.03, 1.03),
            f"{int(row['n_absorbers_detected'])}/{int(row['n_absorbers'])}",
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=90,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["SavedLine"], rotation=45, ha="right")
    ax.set_ylabel("Absorber detection fraction")
    ax.set_ylim(0, 1.12)
    ax.set_title("Absorber Detection Fraction by Ion")
    save_figure(fig, outdir, "detection_fraction_by_ion")


def plot_detection_counts(summary: pd.DataFrame, outdir: Path) -> None:
    if summary.empty:
        no_data_plot(outdir, "detection_counts_by_ion", "Detection Counts by Ion")
        return
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x,
        summary["n_absorbers_detected"],
        label="Detected absorbers",
        color="tab:green",
        edgecolor="black",
        linewidth=0.7,
    )
    ax.bar(
        x,
        summary["n_absorbers_non_detected"],
        bottom=summary["n_absorbers_detected"],
        label="Non-detected absorbers",
        color="tab:gray",
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["SavedLine"], rotation=45, ha="right")
    ax.set_ylabel("Number of absorbers")
    ax.set_title("Detected and Non-Detected Absorbers by Ion")
    ax.legend()
    save_figure(fig, outdir, "detection_counts_by_ion")


def detected_df(df: pd.DataFrame, require: Sequence[str] = ()) -> pd.DataFrame:
    mask = df["is_detection"].copy()
    for col in require:
        mask &= np.isfinite(pd.to_numeric(df[col], errors="coerce"))
    return df[mask]


def common_bins(values: pd.Series, n_bins: int, default: Tuple[float, float]) -> np.ndarray:
    vals = finite_values(values)
    if len(vals) == 0:
        return np.linspace(default[0], default[1], n_bins + 1)
    lo = float(np.nanpercentile(vals, 1))
    hi = float(np.nanpercentile(vals, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = default
    pad = 0.05 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def plot_step_hist_by_ion(
    df: pd.DataFrame,
    outdir: Path,
    column: str,
    stem: str,
    xlabel: str,
    title: str,
    bins: Optional[np.ndarray] = None,
    vertical_lines: Sequence[float] = (),
) -> None:
    data = detected_df(df, require=[column])
    if data.empty:
        no_data_plot(outdir, stem, title)
        return
    if bins is None:
        bins = common_bins(data[column], 28, (0, 1))
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ordered_lines(df)), 1)))
    for color, line in zip(colors, ordered_lines(df)):
        vals = finite_values(data.loc[data["SavedLine"] == line, column])
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, histtype="step", linewidth=2.0, label=line, color=color)
    for x in vertical_lines:
        ax.axvline(x, color="0.25", linestyle="--", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Detected components")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=10)
    save_figure(fig, outdir, stem)


def plot_velocity_bin_fraction(summary: pd.DataFrame, outdir: Path) -> None:
    if summary.empty:
        no_data_plot(outdir, "velocity_bin_fraction_by_ion", "Velocity Bin Fractions by Ion")
        return
    x = np.arange(len(summary))
    y0 = summary["frac_abs_v_0_50"].fillna(0)
    y1 = summary["frac_abs_v_50_100"].fillna(0)
    y2 = summary["frac_abs_v_ge_100"].fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, y0, label="0-50 km/s", color="tab:blue", edgecolor="black", linewidth=0.7)
    ax.bar(x, y1, bottom=y0, label="50-100 km/s", color="tab:orange", edgecolor="black", linewidth=0.7)
    ax.bar(x, y2, bottom=y0 + y1, label=">=100 km/s", color="tab:red", edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["SavedLine"], rotation=45, ha="right")
    ax.set_ylabel("Fraction of detected components")
    ax.set_ylim(0, 1.05)
    ax.set_title("Absolute Velocity Bin Fractions by Ion")
    ax.legend()
    save_figure(fig, outdir, "velocity_bin_fraction_by_ion")


def scatter_by_ion(
    df: pd.DataFrame,
    outdir: Path,
    xcol: str,
    ycol: str,
    stem: str,
    xlabel: str,
    ylabel: str,
    title: str,
    vertical_lines: Sequence[float] = (),
    log_y: bool = False,
) -> None:
    data = detected_df(df, require=[xcol, ycol])
    if data.empty:
        no_data_plot(outdir, stem, title)
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ordered_lines(df)), 1)))
    for color, line in zip(colors, ordered_lines(df)):
        sub = data[data["SavedLine"] == line]
        if sub.empty:
            continue
        ax.scatter(sub[xcol], sub[ycol], s=12, alpha=0.35, label=line, color=color, rasterized=True)
    for x in vertical_lines:
        ax.axvline(x, color="0.25", linestyle="--", linewidth=1.2)
    if log_y:
        positive = data[ycol] > 0
        if positive.any():
            ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=10)
    save_figure(fig, outdir, stem)


def plot_sid_ion_heatmap(summary_sid: pd.DataFrame, outdir: Path) -> None:
    if summary_sid.empty:
        no_data_plot(outdir, "detection_fraction_sid_ion", "Per-SID Detection Fraction")
        return
    pivot = summary_sid.pivot_table(
        index="SID",
        columns="SavedLine",
        values="detection_fraction",
        aggfunc="mean",
    )
    cols = [line for line in ION_ORDER if line in pivot.columns] + [c for c in pivot.columns if c not in ION_ORDER]
    pivot = pivot[cols].sort_index()
    fig_h = max(5, 0.35 * len(pivot.index) + 2)
    fig_w = max(10, 0.75 * len(pivot.columns) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(sid)) for sid in pivot.index])
    ax.set_xlabel("Ion / line")
    ax.set_ylabel("SID")
    ax.set_title("Detection Fraction by SID and Ion")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Detection fraction")
    if pivot.size <= 220:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=8)
    save_figure(fig, outdir, "detection_fraction_sid_ion")


def sid_from_result_path(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.startswith("sid") and part[3:].isdigit():
            return int(part[3:])
    return None


def plot_processing_progress(result_files: Sequence[Path], task_df: pd.DataFrame, outdir: Path) -> None:
    processed: Dict[int, int] = {}
    for path in result_files:
        sid = sid_from_result_path(path)
        if sid is not None:
            processed[sid] = processed.get(sid, 0) + 1

    if not task_df.empty and "sid" in task_df.columns:
        expected_series = task_df.groupby("sid").size()
        sids = sorted(set(expected_series.index.astype(int)) | set(processed))
        expected = np.array([expected_series.get(sid, 0) for sid in sids], dtype=float)
    else:
        sids = sorted(processed)
        expected = np.zeros(len(sids))
    done = np.array([processed.get(int(sid), 0) for sid in sids], dtype=float)

    if len(sids) == 0:
        no_data_plot(outdir, "processing_progress_by_sid", "Processing Progress by SID")
        return
    x = np.arange(len(sids))
    fig, ax = plt.subplots(figsize=(13, 6))
    if expected.sum() > 0:
        ax.bar(x, expected, color="0.85", edgecolor="black", linewidth=0.6, label="Expected spectra")
    ax.bar(x, done, color="tab:blue", edgecolor="black", linewidth=0.6, label="Processed fit CSVs")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(sid)) for sid in sids], rotation=45, ha="right")
    ax.set_ylabel("Number of spectra")
    ax.set_title("Processing Progress by SID")
    ax.legend()
    save_figure(fig, outdir, "processing_progress_by_sid")


def make_all_plots(df: pd.DataFrame, summary: pd.DataFrame, summary_sid: pd.DataFrame, result_files: Sequence[Path], task_df: pd.DataFrame, outdir: Path) -> None:
    print_progress("making diagnostic plots")
    plot_detection_fraction(summary, outdir)
    plot_detection_counts(summary, outdir)
    plot_step_hist_by_ion(
        df,
        outdir,
        "logN",
        "logN_distribution_by_ion",
        r"$\log N$ [cm$^{-2}$]",
        "Detected Column Density Distribution by Ion",
        bins=common_bins(detected_df(df, require=["logN"])["logN"], 30, (11, 19)),
    )
    plot_step_hist_by_ion(
        df,
        outdir,
        "b_kms",
        "b_distribution_by_ion",
        r"$b$ [km s$^{-1}$]",
        "Detected b-Parameter Distribution by Ion",
        bins=common_bins(detected_df(df, require=["b_kms"])["b_kms"], 30, (0, 160)),
    )
    plot_step_hist_by_ion(
        df,
        outdir,
        "abs_v_kms",
        "abs_velocity_distribution_by_ion",
        r"$|v|$ [km s$^{-1}$]",
        "Detected Absolute Velocity Distribution by Ion",
        bins=np.linspace(0, max(1250, np.nanpercentile(finite_values(df["abs_v_kms"]), 99) if len(finite_values(df["abs_v_kms"])) else 1250), 36),
        vertical_lines=[50, 100],
    )
    plot_velocity_bin_fraction(summary, outdir)
    scatter_by_ion(
        df,
        outdir,
        "v_kms",
        "logN",
        "logN_vs_velocity_by_ion",
        r"$v$ [km s$^{-1}$]",
        r"$\log N$ [cm$^{-2}$]",
        r"$\log N$ vs Velocity by Ion",
        vertical_lines=[-100, -50, 0, 50, 100],
    )
    scatter_by_ion(
        df,
        outdir,
        "abs_v_kms",
        "logN",
        "logN_vs_abs_velocity_by_ion",
        r"$|v|$ [km s$^{-1}$]",
        r"$\log N$ [cm$^{-2}$]",
        r"$\log N$ vs Absolute Velocity by Ion",
        vertical_lines=[50, 100],
    )
    scatter_by_ion(
        df,
        outdir,
        "abs_v_kms",
        "EW_mA",
        "ew_vs_abs_velocity_by_ion",
        r"$|v|$ [km s$^{-1}$]",
        "EW [mA]",
        "Equivalent Width vs Absolute Velocity by Ion",
        vertical_lines=[50, 100],
        log_y=True,
    )
    scatter_by_ion(
        df,
        outdir,
        "b_kms",
        "logN",
        "b_vs_logN_by_ion",
        r"$b$ [km s$^{-1}$]",
        r"$\log N$ [cm$^{-2}$]",
        r"$b$ vs $\log N$ by Ion",
    )
    plot_sid_ion_heatmap(summary_sid, outdir)
    plot_processing_progress(result_files, task_df, outdir)


def plot_individual_ions(df: pd.DataFrame, outdir: Path) -> None:
    ind = outdir / "individual_ions"
    print_progress(f"making individual-ion plots under {ind}")
    for line in ordered_lines(df):
        sub = df[df["SavedLine"] == line]
        if sub.empty:
            continue
        line_stem = safe_name(line)
        plot_step_hist_by_ion(
            sub,
            ind,
            "abs_v_kms",
            f"{line_stem}_abs_velocity_histogram",
            r"$|v|$ [km s$^{-1}$]",
            f"{line} Absolute Velocity Distribution",
            bins=np.linspace(0, max(1250, np.nanpercentile(finite_values(sub["abs_v_kms"]), 99) if len(finite_values(sub["abs_v_kms"])) else 1250), 36),
            vertical_lines=[50, 100],
        )
        plot_step_hist_by_ion(
            sub,
            ind,
            "b_kms",
            f"{line_stem}_b_distribution",
            r"$b$ [km s$^{-1}$]",
            f"{line} b-Parameter Distribution",
            bins=common_bins(detected_df(sub, require=["b_kms"])["b_kms"], 28, (0, 160)),
        )
        scatter_by_ion(
            sub,
            ind,
            "v_kms",
            "logN",
            f"{line_stem}_logN_vs_velocity",
            r"$v$ [km s$^{-1}$]",
            r"$\log N$ [cm$^{-2}$]",
            f"{line}: logN vs Velocity",
            vertical_lines=[-100, -50, 0, 50, 100],
        )
        scatter_by_ion(
            sub,
            ind,
            "abs_v_kms",
            "logN",
            f"{line_stem}_logN_vs_abs_velocity",
            r"$|v|$ [km s$^{-1}$]",
            r"$\log N$ [cm$^{-2}$]",
            f"{line}: logN vs Absolute Velocity",
            vertical_lines=[50, 100],
        )


def write_analysis_notes(
    outdir: Path,
    df: pd.DataFrame,
    result_files: Sequence[Path],
    task_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> None:
    n_expected = len(task_df) if not task_df.empty else 0
    n_found = len(result_files)
    n_missing = len(missing_df) if "output_exists" in missing_df.columns or "task_id" in missing_df.columns else max(n_expected - n_found, 0)
    detections = int(df["is_detection"].sum()) if "is_detection" in df else 0
    upper_limits = int(df["is_upper_limit"].sum()) if "is_upper_limit" in df else 0
    absorber_keys = [col for col in ABSORBER_KEY_COLUMNS if col in df.columns]
    if absorber_keys and "is_detection" in df:
        absorber_detected = df.groupby(absorber_keys, dropna=False)["is_detection"].any()
        n_absorbers = int(len(absorber_detected))
        n_absorbers_detected = int(absorber_detected.sum())
        n_absorbers_non_detected = int((~absorber_detected).sum())
    else:
        n_absorbers = 0
        n_absorbers_detected = 0
        n_absorbers_non_detected = 0
    unique_sids = int(df["SID"].dropna().nunique()) if "SID" in df else 0
    unique_ions = int(df["SavedLine"].replace("", np.nan).dropna().nunique()) if "SavedLine" in df else 0

    notes = [
        "Quick fit diagnostics summary",
        "=============================",
        "",
        f"Result files found: {n_found}",
        f"Rows combined: {len(df)}",
        f"Expected tasks from task list: {n_expected}",
        f"Missing or unprocessed tasks: {n_missing}",
        f"Unique SIDs: {unique_sids}",
        f"Unique ions/lines: {unique_ions}",
        f"Detected absorbers (any fitted component): {n_absorbers_detected}",
        f"Non-detected absorbers (no fitted components): {n_absorbers_non_detected}",
        f"Total absorber-line groups: {n_absorbers}",
        f"Detected component rows (UpLim == False): {detections}",
        f"Upper-limit component rows (UpLim == True): {upper_limits}",
        "",
        "Science notes:",
        "- For detection/non-detection counts, an absorber is detected if any fitted component exists for that SID/ray/mode/alpha/line group.",
        "- Non-detected absorbers are those with no detected fitted components (all rows UpLim == True in that group).",
        "- Component-level kinematic plots still use detected component rows (UpLim == False) and additionally require finite kinematics.",
        "- Velocity bins are first-look HVC-style bins: 0 <= |v| < 50, 50 <= |v| < 100, and |v| >= 100 km/s.",
        "- These velocity bins are a kinematic classification only, not a definitive physical interpretation.",
        "- Co-rotating/counter-rotating fractions are not computed here. They require absorber velocity sign relative to the galaxy disk rotation model and line-of-sight geometry.",
        "- Detection fractions in the count/fraction summary plots are absorber-level first-look fractions, not per-sightline covering fractions.",
    ]
    (outdir / "analysis_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def print_run_commands(args: argparse.Namespace) -> None:
    print("\nRun quick diagnostics:")
    print(
        "python /home/tsingh65/m61-tng/scripts/quick_fit_diagnostics.py "
        "\\\n  --repo /home/tsingh65/m61-tng "
        "\\\n  --base-dir /scratch/tsingh65/m61-tng/outputs "
        "\\\n  --snap 99 "
        "\\\n  --run-label L4Rvir "
        "\\\n  --output-subdir fitted_spectra_snr10_bin3 "
        "\\\n  --task-list /home/tsingh65/m61-tng/batch/fit_spectra_tasks_snap99_L4Rvir.tsv "
        "\\\n  --outdir /scratch/tsingh65/m61-tng/outputs/quick_diagnostics_snap99_L4Rvir_snr10_bin3 "
        "\\\n  --make-individual-ion-plots"
    )
    print("\nList output plots:")
    print("ls -lh /scratch/tsingh65/m61-tng/outputs/quick_diagnostics_snap99_L4Rvir_snr10_bin3/*.png")


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    base_dir = Path(args.base_dir)
    outdir = Path(args.outdir)
    task_list = Path(args.task_list)
    outdir.mkdir(parents=True, exist_ok=True)

    print_progress(f"repo: {args.repo}")
    print_progress(f"base_dir: {base_dir}")
    print_progress(f"outdir: {outdir}")

    result_files = find_result_files(base_dir, args.snap, args.run_label, args.output_subdir)
    print_progress(f"found {len(result_files)} per-spectrum fit CSVs")

    df = combine_or_load(result_files, outdir, args.force_recombine)
    task_df = read_task_list(task_list)
    missing_df = write_missing_tasks(task_df, result_files, outdir)

    summary_ion = make_summary_by_ion(df)
    summary_sid = make_summary_by_sid_ion(df)
    summary_ion.to_csv(outdir / "summary_by_ion.csv", index=False)
    summary_sid.to_csv(outdir / "summary_by_sid_ion.csv", index=False)
    print_progress(f"wrote {outdir / 'summary_by_ion.csv'}")
    print_progress(f"wrote {outdir / 'summary_by_sid_ion.csv'}")
    print_progress(f"wrote {outdir / 'missing_or_unprocessed_tasks.csv'}")

    make_all_plots(df, summary_ion, summary_sid, result_files, task_df, outdir)
    if args.make_individual_ion_plots:
        plot_individual_ions(df, outdir)

    write_analysis_notes(outdir, df, result_files, task_df, missing_df)
    print_progress(f"wrote {outdir / 'analysis_notes.txt'}")

    print_progress("done")
    print_run_commands(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
