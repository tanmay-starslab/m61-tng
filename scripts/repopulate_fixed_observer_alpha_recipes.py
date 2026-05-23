#!/usr/bin/env python3
"""
Repopulate ray recipe folders with the corrected fixed-observer alpha convention.

This script uses existing per-SID orientation metadata for centers and Rhalf,
but recomputes the disk normal from the cutout using the current adopted
definition:

  inner stellar PCA = PartType4 positions and masses inside 2*rhalf_star.

Corrected alpha convention:
  - observer/QSO sky-plane coordinates are fixed for all alpha
  - alpha is a galaxy spin about the PCA disk normal
  - native-code ray endpoints use the inverse rotation so spectra sample the
    corresponding region in the unrotated simulation cutout
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import orient_m61 as orient


RUN_SPECS = [("L2Rvir", 1.0), ("L3Rvir", 1.5), ("L4Rvir", 2.0)]
NORMAL_METHOD = "pca_v3_inner_stars_2rhalf"
ALPHA_CONVENTION = "fixed_observer_qso__galaxy_rotated_about_inner_stellar_pca_disk_normal"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def backup_run_dir(run_dir: Path, backup_root: Path, tag: str) -> str:
    if not run_dir.exists():
        return ""
    dest = backup_root / f"{run_dir.name}_{tag}"
    if dest.exists():
        return str(dest)
    shutil.copytree(run_dir, dest)
    return str(dest)


def minimal_delta(pos: np.ndarray, center: np.ndarray, box: float = 35000.0) -> np.ndarray:
    return (pos - center[None, :] + 0.5 * box) % box - 0.5 * box


def cutout_path(cutout_root: Path, sid: int) -> Path:
    sub = cutout_root / f"out_sub_{sid}"
    for pat in (f"cutout_ALLFIELDS_sphere_2p1Rvir_sub{sid}.hdf5", "cutout*sub*.hdf5", "cutout*.hdf5", "*.hdf5"):
        hits = sorted(sub.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No cutout found for sid{sid} under {sub}")


def inner_stellar_pca_normal_from_cutout(cutout: Path, center_ckpch: np.ndarray, rhalf_star_ckpch: float, h_fallback: float) -> tuple[np.ndarray, dict]:
    with h5py.File(cutout, "r") as f:
        h = float(f["Header"].attrs.get("HubbleParam", h_fallback))
        box = float(f["Header"].attrs.get("BoxSize", 35000.0))
        coords = np.asarray(f["PartType4/Coordinates"], dtype=np.float64)
        masses = np.asarray(f["PartType4/Masses"], dtype=np.float64)
    rel = minimal_delta(coords, center_ckpch, box)
    radius = np.linalg.norm(rel, axis=1)
    sel = radius <= 2.0 * float(rhalf_star_ckpch)
    if sel.sum() < 1000:
        raise RuntimeError(f"{cutout} has only {sel.sum()} stars inside 2*rhalf; cannot compute stable inner PCA")
    x_kpc = rel[sel] / h
    w = masses[sel] * 1e10 / h
    evals, evecs, _ = orient.pca3_weighted(x_kpc, w)
    normal = orient.unit(evecs[:, 2])
    meta = {
        "normal_method": NORMAL_METHOD,
        "normal_cutout": str(cutout),
        "normal_selection": "PartType4 stars with radius <= 2*rhalf_star_ckpc_h",
        "normal_selected_particles": int(sel.sum()),
        "normal_eigenvalues_desc": [float(x) for x in evals],
        "normal_axis_ratios": {
            "c_over_a": float(np.sqrt(evals[2] / evals[0])) if evals[0] > 0 else None,
            "b_over_a": float(np.sqrt(evals[1] / evals[0])) if evals[0] > 0 else None,
        },
    }
    return normal, meta


def observed_sightlines_from_radec_pa(obs_csv: Path) -> tuple[list[dict], dict]:
    df = pd.read_csv(obs_csv)
    if len(df) == 0:
        raise RuntimeError(f"Obs CSV is empty: {obs_csv}")
    sightlines = []
    meta_rows = []
    for idx, row in df.iterrows():
        dec0 = np.radians(float(row["dec_deg"]))
        decq = np.radians(float(row["qso_dec_deg"]))
        kpc_per_deg = float(row["distance_mpc"]) * 1000.0 * np.pi / 180.0
        east_kpc = (float(row["qso_ra_deg"]) - float(row["ra_deg"])) * np.cos(0.5 * (dec0 + decq)) * kpc_per_deg
        north_kpc = (float(row["qso_dec_deg"]) - float(row["dec_deg"])) * kpc_per_deg
        pa = np.radians(float(row["PA_deg"]))
        major = np.array([np.sin(pa), np.cos(pa)])
        minor = np.array([-np.cos(pa), np.sin(pa)])
        sky = np.array([east_kpc, north_kpc])
        x_major = float(np.dot(sky, major))
        y_minor = float(np.dot(sky, minor))
        rho = float(np.hypot(x_major, y_minor))
        phi = float(np.degrees(np.arctan2(y_minor, x_major)))
        sid = str(row["qso_name"]) if "qso_name" in row else f"sl_{idx:03d}"
        sl = {
            "sightline_id": sid,
            "rho_kpc": rho,
            "phi_deg": phi,
            "x_qso_major_kpc": x_major,
            "y_qso_minor_kpc": y_minor,
            "rho_table_kpc": float(row["impact_kpc"]),
            "PA_deg_original": float(row["PA_deg"]),
        }
        if "Rvir_kpc" in row and np.isfinite(float(row["Rvir_kpc"])):
            sl["Rvir_kpc"] = float(row["Rvir_kpc"])
        sightlines.append(sl)
        meta_rows.append({
            "sightline_id": sid,
            "east_kpc": float(east_kpc),
            "north_kpc": float(north_kpc),
            **sl,
        })
    meta = {
        "obs_csv": str(obs_csv),
        "qso_geometry_note": "RA/Dec offsets rotated into M61 PA major-axis frame; recipes use PA=0 so image x is projected major axis.",
        "sightline_geometry_rows": meta_rows,
    }
    return sightlines, meta


def build_rows_for_sid(
    sid: int,
    snap: int,
    orient_meta: dict,
    sightlines: list[dict],
    rvir_kpc: float,
    h: float,
    alpha_step: int,
    run_label: str,
    half_R: float,
    normal: np.ndarray,
    normal_meta: dict,
    qso_meta: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    center = np.asarray(orient_meta["center_ckpc_h"], dtype=float)
    inc_deg = float(orient_meta["obs_inc_deg"])
    pa_deg = 0.0
    method = NORMAL_METHOD
    R_base_noflip, R_base_flip, n_hat, n_hat_flip = orient.build_R_bases(normal, inc_deg, pa_deg)
    half_len_ckpch = half_R * float(rvir_kpc) * h
    alphas = list(range(0, 360, int(alpha_step)))

    rays = []
    peralpha = []
    for alpha in alphas:
        for mode, R_base, axis in [
            ("noflip", R_base_noflip, n_hat),
            ("flip", R_base_flip, n_hat_flip),
        ]:
            R_cur = orient.fixed_observer_galaxy_alpha_rotation(R_base, axis, alpha)
            normal_nat = np.array([0.0, 0.0, 1.0]) @ R_cur
            north_nat = np.array([0.0, 1.0, 0.0]) @ R_cur
            peralpha.append({
                "alpha_deg": alpha,
                "mode": mode,
                "orientation_method": method,
                "alpha_convention": ALPHA_CONVENTION,
                "obs_inc_deg": inc_deg,
                "obs_pa_deg_used": pa_deg,
                "obs_pa_deg_original": float(orient_meta.get("obs_pa_deg_used", np.nan)),
                "los_x": float(normal_nat[0]),
                "los_y": float(normal_nat[1]),
                "los_z": float(normal_nat[2]),
                "north_x": float(north_nat[0]),
                "north_y": float(north_nat[1]),
                "north_z": float(north_nat[2]),
            })

            for sl in sightlines:
                rho_kpc = float(sl["rho_kpc"])
                phi_deg = float(sl["phi_deg"])
                p0, p1, anchor, los = orient.sightline_endpoints_codeunits(
                    center_ckpch=center,
                    R_cur=R_cur,
                    rho_ckpch=rho_kpc * h,
                    phi_deg=phi_deg,
                    half_len_ckpch=half_len_ckpch,
                )
                rays.append({
                    "SubhaloID": sid,
                    "sightline_id": sl["sightline_id"],
                    "alpha_deg": alpha,
                    "mode": mode,
                    "orientation_method": method,
                    "alpha_convention": ALPHA_CONVENTION,
                    "obs_inc_deg": inc_deg,
                    "obs_pa_deg_used": pa_deg,
                    "obs_pa_deg_original": float(orient_meta.get("obs_pa_deg_used", np.nan)),
                    "rho_kpc": rho_kpc,
                    "phi_deg": phi_deg,
                    "Rvir_kpc": float(rvir_kpc),
                    "half_len_Rvir": float(half_R),
                    "total_len_Rvir": float(2.0 * half_R),
                    "p0_X_ckpch_abs": float(p0[0]),
                    "p0_Y_ckpch_abs": float(p0[1]),
                    "p0_Z_ckpch_abs": float(p0[2]),
                    "p1_X_ckpch_abs": float(p1[0]),
                    "p1_Y_ckpch_abs": float(p1[1]),
                    "p1_Z_ckpch_abs": float(p1[2]),
                    "anchor_X_ckpch_abs": float(anchor[0]),
                    "anchor_Y_ckpch_abs": float(anchor[1]),
                    "anchor_Z_ckpch_abs": float(anchor[2]),
                    "los_x": float(los[0]),
                    "los_y": float(los[1]),
                    "los_z": float(los[2]),
                })

    header = {
        "SID": sid,
        "SNAP": snap,
        "RUN_LABEL": run_label,
        "half_len_Rvir": float(half_R),
        "total_len_Rvir": float(2.0 * half_R),
        "units_note": "all saved coordinates are ckpc/h (code_length) for TNG50-1 snap99",
        "h": float(h),
        "Rvir_kpc_used": float(rvir_kpc),
        "obs_inc_deg": inc_deg,
        "obs_pa_deg_used": pa_deg,
        "obs_pa_deg_original": float(orient_meta.get("obs_pa_deg_used", np.nan)),
        "PA_FROM_NORTH": bool(orient_meta.get("PA_FROM_NORTH", False)),
        "orientation_method": method,
        "alpha_convention": ALPHA_CONVENTION,
        "alpha_note": (
            "Observer/QSO sky-plane coordinates are fixed for all alpha. "
            "Alpha is a galaxy spin about the PCA disk normal; native ray "
            "endpoints use the inverse rotation to sample the unrotated cutout."
        ),
        "normal_used_hat": orient.unit(normal).tolist(),
        **normal_meta,
        "sightlines": sightlines,
        **qso_meta,
        "alpha_step_deg": int(alpha_step),
    }
    return pd.DataFrame(rays), pd.DataFrame(peralpha), header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-base", default="/scratch/tsingh65/m61-tng/outputs")
    parser.add_argument("--obs-csv", default="/home/tsingh65/m61-tng/data/M61_DIISC_Table1_Table2.csv")
    parser.add_argument("--summary-csv", default="/scratch/tsingh65/m61-tng/outputs/orientation_summary_snap99.csv")
    parser.add_argument("--cutout-root", default="/data/sborthak/m61/cutouts")
    parser.add_argument("--snap", type=int, default=99)
    parser.add_argument("--alpha-step", type=int, default=1)
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--sid-list", default="", help="Optional comma-separated SID filter for smoke tests or targeted repopulation.")
    args = parser.parse_args()

    out_base = Path(args.out_base)
    sightlines, qso_meta = observed_sightlines_from_radec_pa(Path(args.obs_csv))
    summary = pd.read_csv(args.summary_csv)
    summary_by_sid = {int(row.SubhaloID): row for row in summary.itertuples(index=False)}
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = out_base / f"_recipe_backups_pre_fixed_observer_alpha_{tag}"
    report_rows = []
    sid_filter = {int(x) for x in args.sid_list.replace(" ", "").split(",") if x} if args.sid_list else None

    for sid_dir in sorted(out_base.glob("sid*")):
        sid_text = sid_dir.name[3:]
        if not sid_text.isdigit():
            continue
        sid = int(sid_text)
        if sid_filter is not None and sid not in sid_filter:
            continue
        orient_json = sid_dir / "analysis" / f"orientation_sid{sid}_snap{args.snap}.json"
        if not orient_json.exists():
            report_rows.append({"SubhaloID": sid, "status": "missing_orientation_json", "path": str(orient_json)})
            continue
        orient_meta = load_json(orient_json)
        row = summary_by_sid.get(sid)
        obs_rvir = float(sightlines[0].get("Rvir_kpc", 457.0)) if sightlines else 457.0
        rvir = float(getattr(row, "Rvir_kpc_used")) if row is not None and np.isfinite(getattr(row, "Rvir_kpc_used")) else obs_rvir
        h = float(orient_meta.get("h", 0.6774))
        rhalf = float(orient_meta.get("rhalf_star_ckpc_h", np.nan))
        if not np.isfinite(rhalf) or rhalf <= 0:
            report_rows.append({"SubhaloID": sid, "status": "bad_rhalf", "path": str(orient_json)})
            continue
        cutout = cutout_path(Path(args.cutout_root), sid)
        try:
            normal, normal_meta = inner_stellar_pca_normal_from_cutout(
                cutout=cutout,
                center_ckpch=np.asarray(orient_meta["center_ckpc_h"], dtype=float),
                rhalf_star_ckpch=rhalf,
                h_fallback=h,
            )
        except Exception as exc:
            report_rows.append({"SubhaloID": sid, "status": "normal_failed", "error": str(exc), "cutout": str(cutout)})
            continue

        for run_label, half_R in RUN_SPECS:
            run_dir = sid_dir / f"rays_and_recipes_sid{sid}_snap{args.snap}_{run_label}"
            run_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_run_dir(run_dir, backup_root, tag) if args.backup and any(run_dir.iterdir()) else ""
            rays, peralpha, header = build_rows_for_sid(
                sid=sid,
                snap=args.snap,
                orient_meta=orient_meta,
                sightlines=sightlines,
                rvir_kpc=rvir,
                h=h,
                alpha_step=args.alpha_step,
                run_label=run_label,
                half_R=half_R,
                normal=normal,
                normal_meta=normal_meta,
                qso_meta=qso_meta,
            )
            rays.to_csv(run_dir / f"rays_sid{sid}.csv", index=False)
            peralpha.to_csv(run_dir / f"orient_peralpha_sid{sid}.csv", index=False)
            with (run_dir / f"orient_header_sid{sid}.json").open("w") as f:
                json.dump(header, f, indent=2)
            report_rows.append({
                "SubhaloID": sid,
                "RUN_LABEL": run_label,
                "status": "ok",
                "rows_rays": len(rays),
                "rows_orient": len(peralpha),
                "backup_path": backup_path,
                "run_dir": str(run_dir),
            })
            print(f"[OK] sid{sid} {run_label}: rays={len(rays)} orient={len(peralpha)}")

    report = pd.DataFrame(report_rows)
    report_path = out_base / f"fixed_observer_alpha_repopulation_report_snap{args.snap}_{tag}.csv"
    report.to_csv(report_path, index=False)
    print(f"[DONE] report: {report_path}")
    if backup_root.exists():
        print(f"[DONE] backups: {backup_root}")


if __name__ == "__main__":
    main()
