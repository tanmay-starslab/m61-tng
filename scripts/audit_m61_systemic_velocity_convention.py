#!/usr/bin/env python3
"""
Audit and correct the systemic-velocity convention in M61/TNG50 LOS movies.

The correction uses the IllustrisTNG catalog systemic velocity vector
SubhaloVel, read from local cutout metadata when available:

    v_sys_los = dot(SubhaloVel, los_hat)
    v_los_rest = v_los_raw - v_sys_los

Original products are never overwritten. Corrected products are written under
``systemic_subtracted_SubhaloVel`` inside each workflow output root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-m61-systemic")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SID = 488530
SNAP = 99
ROOT = Path("/scratch/tsingh65/m61-tng/outputs/sid488530")
CUTOUT = Path("/scratch/tsingh65/TNG50-1_snap99/out_sub_488530/cutout_ALLFIELDS_sphere_2p1Rvir_sub488530.hdf5")
WORKFLOW_A = ROOT / "saved_alpha_LOS_alpha000_180_stellar_particle_projection_movies"
WORKFLOW_B = ROOT / "inclination_sweep_alpha000_180_gas_stars_movies"
AUDIT_DIR = ROOT / "systemic_velocity_audit"
INCLINATIONS = [0, 23, 45, 75, 90, 135, 170, 180]
ALPHAS = list(range(181))
REP_ALPHAS = [0, 45, 90, 135, 180]
REP_INCS = [23, 90]
LOGNHI_CUT = float(np.log10(1.25e20))


def jclean(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jclean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jclean(v) for v in obj]
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jclean(payload), indent=2, sort_keys=True))


def normalize_vector(vec: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"LOS vector is invalid: {vec!r}")
    return arr / norm


def project_systemic_velocity(subhalo_vel_kms: Any, los_vector: Any) -> float:
    los_hat = normalize_vector(los_vector)
    vel = np.asarray(subhalo_vel_kms, dtype=float)
    if vel.shape != (3,):
        raise ValueError(f"SubhaloVel must have shape (3,), got {vel.shape}")
    return float(np.dot(vel, los_hat))


def read_subhalo_velocity() -> dict[str, Any]:
    if not CUTOUT.exists():
        raise FileNotFoundError(CUTOUT)
    with h5py.File(CUTOUT, "r") as f:
        header = f["Header"].attrs
        scale_factor = float(header.get("Time", np.nan))
        redshift = float(header.get("Redshift", np.nan))
        hubble = float(header.get("HubbleParam", np.nan))
        group = f.get(str(SID))
        if group is None or "vel" not in group.attrs:
            raise RuntimeError(
                "Local cutout metadata does not contain the subhalo velocity. "
                "Set TNG_API_KEY and add an API fallback before correcting movies."
            )
        pos = np.asarray(group.attrs.get("pos"), dtype=float)
        vel = np.asarray(group.attrs.get("vel"), dtype=float)
    return {
        "simulation": "TNG50-1",
        "snapshot": SNAP,
        "subhalo_id": SID,
        "subhalo_pos_ckpch": pos,
        "subhalo_vel_kms": vel,
        "subhalo_vel_disp_kms": None,
        "source": f"{CUTOUT} group /{SID} attrs['vel']",
        "source_note": "Local cutout metadata; interpreted as TNG SubhaloVel in km/s.",
        "scale_factor": scale_factor,
        "redshift": redshift,
        "hubble_param": hubble,
        "snapshot_velocity_note": "TNG snapshot particle velocities require sqrt(a) to convert stored velocities to peculiar km/s; for snap99 a=1, so this is numerically unchanged.",
    }


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        out: dict[str, Any] = {}
        for key in z.files:
            val = z[key]
            out[key] = val.item() if getattr(val, "shape", None) == () else val
        return out


def save_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 190,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 18,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 15,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )


def robust_vlim(arrays: list[np.ndarray], percentile: float = 98.5, default: float = 200.0) -> float:
    vals = []
    for arr in arrays:
        a = np.asarray(arr)
        finite = np.isfinite(a)
        if finite.any():
            vals.append(np.abs(a[finite]).ravel())
    if not vals:
        return default
    return float(np.clip(np.nanpercentile(np.concatenate(vals), percentile), 50.0, 550.0))


def frame_extent(width_kpc: float = 100.0) -> list[float]:
    half = 0.5 * width_kpc
    return [-half, half, -half, half]


def decorate_axis(ax: plt.Axes, qso: tuple[float, float] | None = None) -> None:
    ax.set_xlabel(r"$x\ [{\rm kpc}]$")
    ax.set_ylabel(r"$y\ [{\rm kpc}]$")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.minorticks_on()
    ax.grid(alpha=0.14, lw=0.5)
    ax.scatter([0], [0], marker="+", s=140, c="cyan", lw=2.0, zorder=5)
    if qso is not None:
        ax.scatter([qso[0]], [qso[1]], marker="*", s=170, c="lime", edgecolors="black", lw=0.8, zorder=5)


def add_cbar(fig: plt.Figure, ax: plt.Axes, im: Any, label: str) -> None:
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=15)


def workflow_a_npz(alpha: int) -> Path:
    return WORKFLOW_A / "data" / f"stellar_projection_alpha{alpha:03d}_saved_alpha_LOS_inner100kpc.npz"


def workflow_b_npz(inc: int, comp: str, alpha: int) -> Path:
    prefix = "gas" if comp == "gas" else "stellar"
    return WORKFLOW_B / f"inc{inc:03d}" / comp / "data" / f"{prefix}_projection_inc{inc:03d}_alpha{alpha:03d}_inner100kpc.npz"


def workflow_b_out_npz(outroot: Path, inc: int, comp: str, alpha: int) -> Path:
    prefix = "gas" if comp == "gas" else "stellar"
    return outroot / f"inc{inc:03d}" / comp / "data" / f"{prefix}_projection_inc{inc:03d}_alpha{alpha:03d}_inner100kpc_restframe.npz"


def workflow_a_out_npz(outroot: Path, alpha: int) -> Path:
    return outroot / "data" / f"stellar_projection_alpha{alpha:03d}_saved_alpha_LOS_inner100kpc_restframe.npz"


def audit_scripts(cat: dict[str, Any]) -> dict[str, Any]:
    script_paths = [
        Path("/scratch/tsingh65/m61-tng/scripts/m61_stellar_saved_alpha_projection_movie.py"),
        Path("/scratch/tsingh65/m61-tng/batch/run_stellar_saved_alpha_projection_array.sh"),
        Path("/scratch/tsingh65/m61-tng/batch/run_stellar_saved_alpha_projection_render.sh"),
        Path("/home/tsingh65/m61-tng/scripts/m61_inclination_sweep_alpha_movies.py"),
        Path("/home/tsingh65/m61-tng/batch/run_inclination_sweep_gas_array.sh"),
        Path("/home/tsingh65/m61-tng/batch/run_inclination_sweep_stars_array.sh"),
        Path("/home/tsingh65/m61-tng/batch/run_inclination_sweep_gas_render_array.sh"),
        Path("/home/tsingh65/m61-tng/batch/run_inclination_sweep_stars_render_array.sh"),
    ]
    inspected = []
    for path in script_paths:
        if not path.exists():
            inspected.append({"path": str(path), "exists": False})
            continue
        text = path.read_text(errors="replace")
        inspected.append(
            {
                "path": str(path),
                "exists": True,
                "mentions_SubhaloVel": "SubhaloVel" in text,
                "mentions_VelDisp": "VelDisp" in text or "veldisp" in text.lower(),
                "mentions_systemic": "systemic" in text.lower(),
                "mentions_subtract_systemic_false": "subtract_systemic: bool = False" in text,
                "mentions_subtract_systemic_true": "subtract_systemic: bool = True" in text,
                "uses_central_hdf5_systemic": "systemic_velocity_from_hdf5" in text,
            }
        )
    return {
        "inspected_scripts": inspected,
        "workflow_A_systemic_status": "incorrect",
        "workflow_A_reason": "Script subtracts gas-density-weighted median velocity within 5 physical kpc, not catalog SubhaloVel.",
        "workflow_B_systemic_status": "incorrect",
        "workflow_B_reason": "Script default subtract_systemic is False; cached maps store zero systemic_velocity_kms.",
        "catalog": cat,
    }


def representative_rows(cat: dict[str, Any]) -> list[dict[str, Any]]:
    vel = np.asarray(cat["subhalo_vel_kms"], dtype=float)
    rows: list[dict[str, Any]] = []
    for alpha in REP_ALPHAS:
        p = workflow_a_npz(alpha)
        if p.exists():
            m = load_npz(p)
            los = normalize_vector(m["los_hat"])
            old_sys = np.asarray(m["systemic_velocity_kms"], dtype=float)
            rows.append(
                {
                    "workflow": "A_saved_alpha_stars",
                    "inclination_deg": "",
                    "alpha_deg": alpha,
                    "los_hat": los.tolist(),
                    "v_sys_los_SubhaloVel_kms": project_systemic_velocity(vel, los),
                    "old_central_sys_los_kms": project_systemic_velocity(old_sys, los),
                }
            )
    for inc in REP_INCS:
        for alpha in REP_ALPHAS:
            p = workflow_b_npz(inc, "gas", alpha)
            if p.exists():
                m = load_npz(p)
                los = normalize_vector(m["los_hat"])
                rows.append(
                    {
                        "workflow": "B_inclination_sweep",
                        "inclination_deg": inc,
                        "alpha_deg": alpha,
                        "los_hat": los.tolist(),
                        "v_sys_los_SubhaloVel_kms": project_systemic_velocity(vel, los),
                        "old_central_sys_los_kms": 0.0,
                    }
                )
    return rows


def write_audit(cat: dict[str, Any]) -> dict[str, Any]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit = audit_scripts(cat)
    rows = representative_rows(cat)
    frame_csv = AUDIT_DIR / "audit_systemic_velocity_frames.csv"
    with frame_csv.open("w", newline="") as f:
        fieldnames = ["workflow", "inclination_deg", "alpha_deg", "los_hat", "v_sys_los_SubhaloVel_kms", "old_central_sys_los_kms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row2 = dict(row)
            row2["los_hat"] = json.dumps(row2["los_hat"])
            writer.writerow(row2)
    summary = {
        "simulation": "TNG50-1",
        "snapshot": SNAP,
        "subhalo_id": SID,
        "subhalo_vel_kms": cat["subhalo_vel_kms"],
        "subhalo_vel_disp_kms": cat["subhalo_vel_disp_kms"],
        "subhalo_velocity_source": cat["source"],
        "workflow_A_systemic_status": "incorrect",
        "workflow_B_systemic_status": "incorrect",
        "workflow_A_requires_regeneration": True,
        "workflow_B_requires_regeneration": True,
        "notes": [
            "Workflow A did subtract a velocity vector, but the vector was a central 5 kpc gas median fallback, not TNG SubhaloVel.",
            "Workflow B cached maps and metadata explicitly use raw simulation-frame LOS velocities with zero systemic vector.",
            "SubhaloVelDisp was not found in local metadata and was not used.",
        ],
        "representative_frames_csv": str(frame_csv),
    }
    write_json(AUDIT_DIR / "audit_systemic_velocity_summary.json", summary)
    write_json(AUDIT_DIR / "audit_systemic_velocity_details.json", audit)
    lines = [
        "M61/TNG50 systemic velocity audit",
        "",
        f"Simulation: TNG50-1",
        f"Snapshot: {SNAP}",
        f"SubhaloID: {SID}",
        f"SubhaloVel source: {cat['source']}",
        f"SubhaloVel = {np.asarray(cat['subhalo_vel_kms']).tolist()} km/s",
        f"SubhaloVelDisp = {cat['subhalo_vel_disp_kms']} km/s (metadata only; not subtracted)",
        "",
        "Formula:",
        "  v_sys_los = dot(SubhaloVel, los_hat)",
        "  v_los_rest = v_los_raw - v_sys_los",
        "",
        "Workflow A: incorrect for the requested convention.",
        "  Existing velocity maps subtract a central 5 kpc gas-density-weighted median velocity.",
        "  They do not subtract catalog SubhaloVel.",
        "",
        "Workflow B: incorrect for the requested convention.",
        "  Existing velocity maps are raw simulation-frame LOS velocities.",
        "  Cached metadata has subtract_systemic=false and systemic_velocity_kms=[0,0,0].",
        "",
        "Representative v_sys_los values are in audit_systemic_velocity_frames.csv.",
    ]
    (AUDIT_DIR / "audit_systemic_velocity_report.txt").write_text("\n".join(lines) + "\n")
    return summary


def correct_workflow_a(outroot: Path, cat: dict[str, Any], overwrite: bool = False) -> list[dict[str, Any]]:
    vel = np.asarray(cat["subhalo_vel_kms"], dtype=float)
    rows = []
    for alpha in ALPHAS:
        src = workflow_a_npz(alpha)
        if not src.exists():
            rows.append({"workflow": "A", "alpha_deg": alpha, "status": "missing_source", "source": str(src)})
            continue
        dst = workflow_a_out_npz(outroot, alpha)
        if dst.exists() and not overwrite:
            rows.append({"workflow": "A", "alpha_deg": alpha, "status": "exists", "output": str(dst)})
            continue
        m = load_npz(src)
        los = normalize_vector(m["los_hat"])
        old_sys = np.asarray(m["systemic_velocity_kms"], dtype=float)
        old_sys_los = project_systemic_velocity(old_sys, los)
        sub_sys_los = project_systemic_velocity(vel, los)
        existing = np.asarray(m["stellar_vlos_unweighted_mean_kms"], dtype=float)
        corrected = existing + old_sys_los - sub_sys_los
        payload = {
            "x_kpc": m["x_kpc"],
            "y_kpc": m["y_kpc"],
            "stellar_vlos_unweighted_mean_restframe_kms": corrected,
            "alpha_deg": float(alpha),
            "los_hat": los,
            "north_hat": m["north_hat"],
            "x_qso_kpc": float(m["x_qso_kpc"]),
            "y_qso_kpc": float(m["y_qso_kpc"]),
            "old_systemic_velocity_kms": old_sys,
            "subhalo_vel_kms": vel,
            "old_systemic_los_kms": old_sys_los,
            "v_sys_los_kms": sub_sys_los,
            "applied_delta_to_existing_kms": old_sys_los - sub_sys_los,
            "velocity_frame": "SubhaloVel systemic-subtracted rest frame",
            "source_npz": str(src),
        }
        save_npz(dst, payload)
        rows.append(
            {
                "workflow": "A",
                "alpha_deg": alpha,
                "status": "corrected",
                "source": str(src),
                "output": str(dst),
                "v_sys_los_kms": sub_sys_los,
                "old_systemic_los_kms": old_sys_los,
                "delta_existing_to_rest_kms": old_sys_los - sub_sys_los,
            }
        )
    return rows


def correct_workflow_b(outroot: Path, cat: dict[str, Any], overwrite: bool = False) -> list[dict[str, Any]]:
    vel = np.asarray(cat["subhalo_vel_kms"], dtype=float)
    rows = []
    for inc in INCLINATIONS:
        for comp in ["gas", "stars"]:
            for alpha in ALPHAS:
                src = workflow_b_npz(inc, comp, alpha)
                if not src.exists():
                    rows.append({"workflow": "B", "component": comp, "inc_deg": inc, "alpha_deg": alpha, "status": "missing_source", "source": str(src)})
                    continue
                dst = workflow_b_out_npz(outroot, inc, comp, alpha)
                if dst.exists() and not overwrite:
                    rows.append({"workflow": "B", "component": comp, "inc_deg": inc, "alpha_deg": alpha, "status": "exists", "output": str(dst)})
                    continue
                m = load_npz(src)
                los = normalize_vector(m["los_hat"])
                sub_sys_los = project_systemic_velocity(vel, los)
                common = {
                    "x_kpc": m["x_kpc"],
                    "y_kpc": m["y_kpc"],
                    "alpha_deg": float(alpha),
                    "inc_deg": float(inc),
                    "pa_deg": float(m.get("pa_deg", 138.0)),
                    "los_hat": los,
                    "north_hat": m["north_hat"],
                    "disk_normal_hat": m["disk_normal_hat"],
                    "x_qso_kpc": float(m["x_qso_kpc"]),
                    "y_qso_kpc": float(m["y_qso_kpc"]),
                    "rho_kpc": float(m["rho_kpc"]),
                    "phi_deg": float(m["phi_deg"]),
                    "subhalo_vel_kms": vel,
                    "v_sys_los_kms": sub_sys_los,
                    "subtract_systemic": True,
                    "velocity_frame": "SubhaloVel systemic-subtracted rest frame",
                    "source_npz": str(src),
                }
                if comp == "gas":
                    payload = {
                        **common,
                        "logNHI_cut": float(m["logNHI_cut"]),
                        "vlos_gasweighted_restframe_kms": np.asarray(m["vlos_gasweighted_kms"], dtype=float) - sub_sys_los,
                        "vlos_HIweighted_restframe_kms": np.asarray(m["vlos_HIweighted_kms"], dtype=float) - sub_sys_los,
                    }
                else:
                    payload = {
                        **common,
                        "stellar_vlos_mass_weighted_restframe_kms": np.asarray(m["stellar_vlos_mass_weighted_kms"], dtype=float) - sub_sys_los,
                    }
                save_npz(dst, payload)
                rows.append(
                    {
                        "workflow": "B",
                        "component": comp,
                        "inc_deg": inc,
                        "alpha_deg": alpha,
                        "status": "corrected",
                        "source": str(src),
                        "output": str(dst),
                        "v_sys_los_kms": sub_sys_los,
                        "delta_raw_to_rest_kms": -sub_sys_los,
                    }
                )
    return rows


def render_a_frames(outroot: Path, overwrite: bool = False) -> dict[str, str]:
    configure_matplotlib()
    frames_combined = outroot / "frames" / "combined"
    frames_sigma = outroot / "frames" / "stellar_surface_density"
    frames_vlos = outroot / "frames" / "stellar_vlos_restframe"
    for p in [frames_combined, frames_sigma, frames_vlos, outroot / "videos"]:
        p.mkdir(parents=True, exist_ok=True)
    maps = {}
    for alpha in ALPHAS:
        src = load_npz(workflow_a_npz(alpha))
        cor = load_npz(workflow_a_out_npz(outroot, alpha))
        maps[alpha] = {**src, **cor}
    log_vals = np.concatenate([m["log_stellar_surface_density"][np.isfinite(m["log_stellar_surface_density"])].ravel() for m in maps.values()])
    vlim = robust_vlim([m["stellar_vlos_unweighted_mean_restframe_kms"] for m in maps.values()])
    scales = {
        "logSigma_vmin": float(np.nanpercentile(log_vals, 1.0)),
        "logSigma_vmax": float(np.nanpercentile(log_vals, 99.5)),
        "vstar_vlim": vlim,
    }
    for alpha, m in maps.items():
        qso = (float(m["x_qso_kpc"]), float(m["y_qso_kpc"]))
        log_sigma = m["log_stellar_surface_density"]
        vlos = m["stellar_vlos_unweighted_mean_restframe_kms"]
        outputs = [
            (frames_sigma / f"frame_{alpha:03d}.png", [(log_sigma, "inferno", scales["logSigma_vmin"], scales["logSigma_vmax"], r"$\log_{10}\Sigma_\star\ [M_\odot\,{\rm kpc}^{-2}]$", r"stellar surface density")], (8.5, 8)),
            (frames_vlos / f"frame_{alpha:03d}.png", [(vlos, "RdBu_r", -vlim, vlim, r"$v_{\rm LOS}-v_{\rm sys,LOS}\ [{\rm km\,s}^{-1}]$", r"unweighted stellar-particle LOS velocity")], (8.5, 8)),
            (frames_combined / f"frame_{alpha:03d}.png", [
                (log_sigma, "inferno", scales["logSigma_vmin"], scales["logSigma_vmax"], r"$\log_{10}\Sigma_\star\ [M_\odot\,{\rm kpc}^{-2}]$", r"stellar surface density"),
                (vlos, "RdBu_r", -vlim, vlim, r"$v_{\rm LOS}-v_{\rm sys,LOS}\ [{\rm km\,s}^{-1}]$", r"unweighted stellar-particle LOS velocity"),
            ], (16, 8)),
        ]
        for out, panels, figsize in outputs:
            if out.exists() and not overwrite:
                continue
            fig, axes = plt.subplots(1, len(panels), figsize=figsize, constrained_layout=True)
            if len(panels) == 1:
                axes = [axes]
            for ax, (arr, cmap, vmin, vmax, label, title) in zip(axes, panels):
                cm = plt.get_cmap(cmap).copy()
                cm.set_bad("white" if "inferno" in cmap else "0.92")
                im = ax.imshow(arr, origin="lower", extent=frame_extent(), cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(title)
                decorate_axis(ax, qso)
                add_cbar(fig, ax, im, label)
            fig.suptitle(rf"M61 / TNG50-1 SID {SID}   $\alpha={alpha:03d}^\circ$   Subhalo-rest-frame", y=1.03, fontsize=24)
            fig.savefig(out)
            plt.close(fig)
    return run_a_ffmpeg(outroot)


def run_ffmpeg(frame_dir: Path, output: Path, fps: int = 18, start_number: int = 0) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not in PATH. Run under `module load ffmpeg-6.0-gcc-12.1.0` or load that module first.")
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        str(frame_dir / "frame_%03d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "15",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return str(output)


def run_a_ffmpeg(outroot: Path) -> dict[str, str]:
    videos = outroot / "videos"
    return {
        "stellar_surface_density": run_ffmpeg(outroot / "frames" / "stellar_surface_density", videos / "m61_saved_alpha_000_180_stellar_surface_density_unchanged.mp4"),
        "stellar_vlos_unweighted_restframe": run_ffmpeg(outroot / "frames" / "stellar_vlos_restframe", videos / "m61_saved_alpha_000_180_stellar_vlos_unweighted_restframe.mp4"),
        "combined_restframe": run_ffmpeg(outroot / "frames" / "combined", videos / "m61_saved_alpha_000_180_stars_sigma_vlos_combined_restframe.mp4"),
    }


def render_b_component(outroot: Path, inc: int, comp: str, overwrite: bool = False) -> str:
    configure_matplotlib()
    base = outroot / f"inc{inc:03d}" / comp
    frames = base / "frames"
    videos = base / "videos"
    frames.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    maps = {}
    for alpha in ALPHAS:
        src = load_npz(workflow_b_npz(inc, comp, alpha))
        cor = load_npz(workflow_b_out_npz(outroot, inc, comp, alpha))
        maps[alpha] = {**src, **cor}
    if comp == "gas":
        vgas_vlim = robust_vlim([m["vlos_gasweighted_restframe_kms"][m["mask_logNHI_cut"].astype(bool)] for m in maps.values()])
        vhi_vlim = robust_vlim([m["vlos_HIweighted_restframe_kms"][m["mask_logNHI_cut"].astype(bool)] for m in maps.values()])
        log_vals = np.concatenate([m["logN_HI"][m["mask_logNHI_cut"].astype(bool) & np.isfinite(m["logN_HI"])].ravel() for m in maps.values()])
        log_vmax = float(max(LOGNHI_CUT + 0.2, np.nanpercentile(log_vals, 99.7)))
        for alpha, m in maps.items():
            out = frames / f"frame_{alpha:03d}.png"
            if out.exists() and not overwrite:
                continue
            mask = m["mask_logNHI_cut"].astype(bool)
            logn = np.where(mask, m["logN_HI"], np.nan)
            vgas = np.where(mask, m["vlos_gasweighted_restframe_kms"], np.nan)
            vhi = np.where(mask, m["vlos_HIweighted_restframe_kms"], np.nan)
            qso = (float(m["x_qso_kpc"]), float(m["y_qso_kpc"]))
            fig, axes = plt.subplots(1, 3, figsize=(25, 8.4), constrained_layout=True)
            panels = [
                (logn, "magma", LOGNHI_CUT, log_vmax, r"$\log_{10}N_{\rm HI}\ [{\rm cm}^{-2}]$", r"H I column, $N_{\rm HI}>1.25\times10^{20}\ {\rm cm}^{-2}$"),
                (vgas, "RdBu_r", -vgas_vlim, vgas_vlim, r"$v_{\rm LOS}-v_{\rm sys,LOS}\ [{\rm km\,s}^{-1}]$", r"gas-density-weighted LOS velocity"),
                (vhi, "RdBu_r", -vhi_vlim, vhi_vlim, r"$v_{\rm LOS}-v_{\rm sys,LOS}\ [{\rm km\,s}^{-1}]$", r"H I-weighted LOS velocity"),
            ]
            for ax, (arr, cmap, vmin, vmax, label, title) in zip(axes, panels):
                cm = plt.get_cmap(cmap).copy()
                cm.set_bad("white" if cmap == "magma" else "0.92")
                im = ax.imshow(arr, origin="lower", extent=frame_extent(), cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(title)
                decorate_axis(ax, qso)
                add_cbar(fig, ax, im, label)
            fig.suptitle(rf"M61 / TNG50-1 SID {SID}   $i={inc}^\circ$   $\alpha={alpha:03d}^\circ$   Subhalo-rest-frame", y=1.03, fontsize=25)
            fig.savefig(out)
            plt.close(fig)
        return run_ffmpeg(frames, videos / f"m61_inc{inc:03d}_alpha000_180_gas_HI_vgas_vHI_restframe.mp4")
    vlim = robust_vlim([m["stellar_vlos_mass_weighted_restframe_kms"] for m in maps.values()])
    mu_vals = np.concatenate([m["stellar_mu_r_mag_arcsec2"][np.isfinite(m["stellar_mu_r_mag_arcsec2"])].ravel() for m in maps.values()])
    mu_vmin = float(np.nanpercentile(mu_vals, 0.5))
    mu_vmax = float(np.nanpercentile(mu_vals, 99.0))
    for alpha, m in maps.items():
        out = frames / f"frame_{alpha:03d}.png"
        if out.exists() and not overwrite:
            continue
        qso = (float(m["x_qso_kpc"]), float(m["y_qso_kpc"]))
        fig, axes = plt.subplots(1, 2, figsize=(18.2, 8.4), constrained_layout=True)
        panels = [
            (m["stellar_mu_r_mag_arcsec2"], "magma_r", mu_vmin, mu_vmax, r"$\mu_r\ [{\rm mag\,arcsec}^{-2}]$", r"stellar $r$-band surface brightness"),
            (m["stellar_vlos_mass_weighted_restframe_kms"], "RdBu_r", -vlim, vlim, r"$v_{\rm LOS}-v_{\rm sys,LOS}\ [{\rm km\,s}^{-1}]$", r"stellar-mass-weighted LOS velocity"),
        ]
        for ax, (arr, cmap, vmin, vmax, label, title) in zip(axes, panels):
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("white" if cmap == "magma_r" else "0.92")
            im = ax.imshow(arr, origin="lower", extent=frame_extent(), cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title)
            decorate_axis(ax, qso)
            add_cbar(fig, ax, im, label)
        fig.suptitle(rf"M61 / TNG50-1 SID {SID}   $i={inc}^\circ$   $\alpha={alpha:03d}^\circ$   Subhalo-rest-frame", y=1.03, fontsize=25)
        fig.savefig(out)
        plt.close(fig)
    return run_ffmpeg(frames, videos / f"m61_inc{inc:03d}_alpha000_180_stars_surface_brightness_vlos_restframe.mp4")


def render_workflow_b(outroot: Path, overwrite: bool = False) -> dict[str, str]:
    videos: dict[str, str] = {}
    for inc in INCLINATIONS:
        videos[f"inc{inc:03d}_gas"] = render_b_component(outroot, inc, "gas", overwrite=overwrite)
        videos[f"inc{inc:03d}_stars"] = render_b_component(outroot, inc, "stars", overwrite=overwrite)
    return videos


def central_mask(m: dict[str, Any], radius_kpc: float = 2.0) -> np.ndarray:
    if "X_kpc" in m and "Y_kpc" in m:
        rr = np.sqrt(np.asarray(m["X_kpc"]) ** 2 + np.asarray(m["Y_kpc"]) ** 2)
    else:
        x = np.asarray(m["x_kpc"])
        y = np.asarray(m["y_kpc"])
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx**2 + yy**2)
    return rr <= radius_kpc


def validate_and_diagnose(a_out: Path, b_out: Path, cat: dict[str, Any]) -> dict[str, Any]:
    configure_matplotlib()
    diag_dir = AUDIT_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    vel = np.asarray(cat["subhalo_vel_kms"], dtype=float)
    rows = []

    def record_diag(label: str, raw: np.ndarray, corrected: np.ndarray, expected_delta: float, meta: dict[str, Any], out: Path) -> None:
        valid = np.isfinite(raw) & np.isfinite(corrected)
        delta = corrected - raw
        med_delta = float(np.nanmedian(delta[valid])) if valid.any() else np.nan
        std_delta = float(np.nanstd(delta[valid])) if valid.any() else np.nan
        maxerr = float(np.nanmax(np.abs(delta[valid] - expected_delta))) if valid.any() else np.nan
        cmask = central_mask(meta)
        cvalid = valid & cmask
        row = {
            **{k: meta.get(k) for k in ["workflow", "component", "inc_deg", "alpha_deg"]},
            "median_raw_valid_pixels": float(np.nanmedian(raw[valid])) if valid.any() else np.nan,
            "median_corrected_valid_pixels": float(np.nanmedian(corrected[valid])) if valid.any() else np.nan,
            "median_delta_valid_pixels": med_delta,
            "std_delta_valid_pixels": std_delta,
            "expected_delta": float(expected_delta),
            "max_abs_delta_minus_expected": maxerr,
            "raw_central_aperture_median_vlos_kms": float(np.nanmedian(raw[cvalid])) if cvalid.any() else np.nan,
            "corrected_central_aperture_median_vlos_kms": float(np.nanmedian(corrected[cvalid])) if cvalid.any() else np.nan,
            "corrected_minus_raw_central_value": float(np.nanmedian(corrected[cvalid] - raw[cvalid])) if cvalid.any() else np.nan,
            "v_sys_los_kms": float(meta["v_sys_los_kms"]),
            "diagnostic_png": str(out),
        }
        rows.append(row)
        if out.exists():
            return
        vlim = robust_vlim([raw, corrected], default=200.0)
        dlim = max(1.0, abs(expected_delta) * 1.15)
        fig, axes = plt.subplots(1, 3, figsize=(22, 7.2), constrained_layout=True)
        panels = [
            (raw, "RdBu_r", -vlim, vlim, "raw / pre-correction"),
            (corrected, "RdBu_r", -vlim, vlim, "SubhaloVel rest frame"),
            (delta, "RdBu_r", -dlim, dlim, r"corrected $-$ raw"),
        ]
        for ax, (arr, cmap, vmin, vmax, title) in zip(axes, panels):
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("0.92")
            im = ax.imshow(arr, origin="lower", extent=frame_extent(), cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title)
            decorate_axis(ax, None)
            add_cbar(fig, ax, im, r"${\rm km\,s}^{-1}$")
        fig.suptitle(f"{label}: expected delta = {expected_delta:.6f} km/s, max error = {maxerr:.3e}", y=1.03, fontsize=22)
        fig.savefig(out)
        plt.close(fig)

    for alpha in REP_ALPHAS:
        src = load_npz(workflow_a_npz(alpha))
        cor = load_npz(workflow_a_out_npz(a_out, alpha))
        los = normalize_vector(src["los_hat"])
        old_los = project_systemic_velocity(src["systemic_velocity_kms"], los)
        sub_los = project_systemic_velocity(vel, los)
        raw = np.asarray(src["stellar_vlos_unweighted_mean_kms"], dtype=float) + old_los
        corrected = np.asarray(cor["stellar_vlos_unweighted_mean_restframe_kms"], dtype=float)
        record_diag(
            f"Workflow A alpha {alpha:03d}",
            raw,
            corrected,
            -sub_los,
            {**src, "workflow": "A_saved_alpha_stars", "component": "stars_unweighted", "inc_deg": np.nan, "alpha_deg": alpha, "v_sys_los_kms": sub_los},
            diag_dir / f"workflowA_alpha{alpha:03d}_raw_vs_restframe.png",
        )
    for inc in REP_INCS:
        for alpha in REP_ALPHAS:
            for comp, raw_key, corr_key in [
                ("gas", "vlos_HIweighted_kms", "vlos_HIweighted_restframe_kms"),
                ("stars", "stellar_vlos_mass_weighted_kms", "stellar_vlos_mass_weighted_restframe_kms"),
            ]:
                src = load_npz(workflow_b_npz(inc, comp, alpha))
                cor = load_npz(workflow_b_out_npz(b_out, inc, comp, alpha))
                sub_los = project_systemic_velocity(vel, src["los_hat"])
                record_diag(
                    f"Workflow B {comp} inc {inc:03d} alpha {alpha:03d}",
                    np.asarray(src[raw_key], dtype=float),
                    np.asarray(cor[corr_key], dtype=float),
                    -sub_los,
                    {**src, "workflow": "B_inclination_sweep", "component": comp, "inc_deg": inc, "alpha_deg": alpha, "v_sys_los_kms": sub_los},
                    diag_dir / f"workflowB_{comp}_inc{inc:03d}_alpha{alpha:03d}_raw_vs_restframe.png",
                )
    csv_path = AUDIT_DIR / "systemic_subtraction_validation_frames.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_validation_rows": len(rows),
        "max_abs_delta_minus_expected": max([r["max_abs_delta_minus_expected"] for r in rows if np.isfinite(r["max_abs_delta_minus_expected"])], default=np.nan),
        "central_aperture_radius_kpc": 2.0,
        "central_aperture_note": "Corrected central median is not forced to zero; it is residual physical projected motion after bulk SubhaloVel subtraction.",
        "validation_csv": str(csv_path),
        "diagnostics_dir": str(diag_dir),
    }
    write_json(AUDIT_DIR / "systemic_subtraction_validation_summary.json", summary)
    return summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(a_out: Path, b_out: Path, a_videos: dict[str, str], b_videos: dict[str, str], cat: dict[str, Any], validation: dict[str, Any]) -> Path:
    lines = [
        "M61/TNG50 SubhaloVel systemic-subtracted movie manifest",
        "",
        f"SubhaloVel source: {cat['source']}",
        f"SubhaloVel: {np.asarray(cat['subhalo_vel_kms']).tolist()} km/s",
        "Formula: v_los_rest = v_los_raw - dot(SubhaloVel, los_hat)",
        "",
        "Workflow A videos:",
    ]
    for key, path in a_videos.items():
        p = Path(path)
        lines.append(f"  {key}: {p} ({p.stat().st_size if p.exists() else 'missing'} bytes)")
    lines.append("")
    lines.append("Workflow B videos:")
    for key, path in b_videos.items():
        p = Path(path)
        lines.append(f"  {key}: {p} ({p.stat().st_size if p.exists() else 'missing'} bytes)")
    lines.extend(
        [
            "",
            f"Validation CSV: {validation.get('validation_csv')}",
            f"Max abs(delta - expected): {validation.get('max_abs_delta_minus_expected')} km/s",
        ]
    )
    path = ROOT / "systemic_subtracted_SubhaloVel" / "final_restframe_video_manifest.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def run_audit_only(args: argparse.Namespace) -> None:
    cat = read_subhalo_velocity()
    summary = write_audit(cat)
    print(json.dumps(jclean(summary), indent=2))


def run_correct(args: argparse.Namespace) -> None:
    cat = read_subhalo_velocity()
    write_audit(cat)
    a_out = WORKFLOW_A / "systemic_subtracted_SubhaloVel"
    b_out = WORKFLOW_B / "systemic_subtracted_SubhaloVel"
    root_jobs = ROOT / "systemic_subtracted_SubhaloVel"
    root_jobs.mkdir(parents=True, exist_ok=True)
    (root_jobs / "slurm_submitted_jobs.txt").write_text(
        "No Slurm jobs submitted by this correction run. Existing cached maps were corrected by the mathematically equivalent scalar SubhaloVel dot los_hat subtraction; no yt projections were recomputed.\n"
    )
    print("Correcting Workflow A caches...")
    a_rows = correct_workflow_a(a_out, cat, overwrite=args.overwrite)
    write_rows_csv(AUDIT_DIR / "workflow_A_correction_manifest.csv", a_rows)
    print("Correcting Workflow B caches...")
    b_rows = correct_workflow_b(b_out, cat, overwrite=args.overwrite)
    write_rows_csv(AUDIT_DIR / "workflow_B_correction_manifest.csv", b_rows)
    videos_a: dict[str, str] = {}
    videos_b: dict[str, str] = {}
    if not args.skip_render:
        print("Rendering Workflow A rest-frame frames/videos...")
        videos_a = render_a_frames(a_out, overwrite=args.overwrite)
        print("Rendering Workflow B rest-frame frames/videos...")
        videos_b = render_workflow_b(b_out, overwrite=args.overwrite)
    validation = validate_and_diagnose(a_out, b_out, cat)
    manifest = build_manifest(a_out, b_out, videos_a, videos_b, cat, validation)
    final_summary = {
        "catalog": cat,
        "workflow_A_output": str(a_out),
        "workflow_B_output": str(b_out),
        "workflow_A_corrected_rows": len([r for r in a_rows if r.get("status") in {"corrected", "exists"}]),
        "workflow_B_corrected_rows": len([r for r in b_rows if r.get("status") in {"corrected", "exists"}]),
        "workflow_A_videos": videos_a,
        "workflow_B_videos": videos_b,
        "validation": validation,
        "final_manifest": str(manifest),
    }
    write_json(ROOT / "systemic_subtracted_SubhaloVel" / "restframe_correction_summary.json", final_summary)
    print(json.dumps(jclean(final_summary), indent=2))


def run_render_a(args: argparse.Namespace) -> None:
    outroot = WORKFLOW_A / "systemic_subtracted_SubhaloVel"
    videos = render_a_frames(outroot, overwrite=args.overwrite)
    write_json(outroot / "data" / "rendered_videos_restframe.json", videos)
    print(json.dumps(jclean(videos), indent=2))


def run_render_b_component(args: argparse.Namespace) -> None:
    outroot = WORKFLOW_B / "systemic_subtracted_SubhaloVel"
    video = render_b_component(outroot, int(args.inc), str(args.component), overwrite=args.overwrite)
    payload = {"inc_deg": int(args.inc), "component": args.component, "video": video}
    write_json(outroot / f"inc{int(args.inc):03d}" / str(args.component) / "data" / "rendered_video_restframe.json", payload)
    print(json.dumps(jclean(payload), indent=2))


def run_validate_manifest(args: argparse.Namespace) -> None:
    cat = read_subhalo_velocity()
    a_out = WORKFLOW_A / "systemic_subtracted_SubhaloVel"
    b_out = WORKFLOW_B / "systemic_subtracted_SubhaloVel"
    validation = validate_and_diagnose(a_out, b_out, cat)
    a_videos = {
        "stellar_surface_density": str(a_out / "videos" / "m61_saved_alpha_000_180_stellar_surface_density_unchanged.mp4"),
        "stellar_vlos_unweighted_restframe": str(a_out / "videos" / "m61_saved_alpha_000_180_stellar_vlos_unweighted_restframe.mp4"),
        "combined_restframe": str(a_out / "videos" / "m61_saved_alpha_000_180_stars_sigma_vlos_combined_restframe.mp4"),
    }
    b_videos = {}
    for inc in INCLINATIONS:
        b_videos[f"inc{inc:03d}_gas"] = str(
            b_out / f"inc{inc:03d}" / "gas" / "videos" / f"m61_inc{inc:03d}_alpha000_180_gas_HI_vgas_vHI_restframe.mp4"
        )
        b_videos[f"inc{inc:03d}_stars"] = str(
            b_out / f"inc{inc:03d}" / "stars" / "videos" / f"m61_inc{inc:03d}_alpha000_180_stars_surface_brightness_vlos_restframe.mp4"
        )
    missing = [p for p in list(a_videos.values()) + list(b_videos.values()) if not Path(p).exists() or Path(p).stat().st_size == 0]
    manifest = build_manifest(a_out, b_out, a_videos, b_videos, cat, validation)
    payload = {"manifest": str(manifest), "missing_or_empty_videos": missing, "validation": validation}
    write_json(ROOT / "systemic_subtracted_SubhaloVel" / "restframe_video_verification.json", payload)
    if missing:
        raise RuntimeError(f"Missing/empty videos: {missing}")
    print(json.dumps(jclean(payload), indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Write non-destructive audit files only.")
    p_corr = sub.add_parser("correct", help="Correct cached maps, render rest-frame frames/videos, and validate.")
    p_corr.add_argument("--overwrite", action="store_true", help="Overwrite existing corrected caches/frames.")
    p_corr.add_argument("--skip-render", action="store_true", help="Only write corrected NPZ caches and diagnostics.")
    p_a = sub.add_parser("render-a", help="Render Workflow A rest-frame frames/videos from corrected caches.")
    p_a.add_argument("--overwrite", action="store_true")
    p_b = sub.add_parser("render-b-component", help="Render one Workflow B inclination/component rest-frame video.")
    p_b.add_argument("--inc", required=True, type=int, choices=INCLINATIONS)
    p_b.add_argument("--component", required=True, choices=["gas", "stars"])
    p_b.add_argument("--overwrite", action="store_true")
    sub.add_parser("validate-manifest", help="Validate representative frames and write final rest-frame video manifest.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "audit":
        run_audit_only(args)
    elif args.command == "correct":
        run_correct(args)
    elif args.command == "render-a":
        run_render_a(args)
    elif args.command == "render-b-component":
        run_render_b_component(args)
    elif args.command == "validate-manifest":
        run_validate_manifest(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
