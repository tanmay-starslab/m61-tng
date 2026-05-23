#!/usr/bin/env python3
"""
PCA-v3 orientation sanity diagnostic for one SID.

This does not overwrite existing rays/recipes.  It recomputes the PCA-v3 stellar
disk normal from the data cutout, builds a transparent "major-axis on x-axis"
observer convention, and makes low-cost direct particle/cell histograms for a
few alpha values.
"""

from __future__ import annotations

import json
import math
import sys
import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import orient_m61 as orient


SID = 413372
SNAP = 99
H = 0.6774
INC_DEG = 23.0
RHO_KPC = 26.0
PHI_DEG = 0.0
Rvir_KPC = 457.0
ALPHAS = [0, 45, 90, 135, 180]
WIDTH_KPC = 100.0
NPIX = 500
NORMAL_SOURCE = "pca3_all_stars_10rhalf"
QSO_X_KPC = 26.0
QSO_Y_KPC = 0.0
QSO_NOTE = "synthetic marker at (+rho,0)"
OUT = Path(f"/scratch/tsingh65/m61-tng/outputs/sid{SID}/orientation_rebuild_test_pcav3_majorx")
CUTOUT = Path(f"/data/sborthak/m61/cutouts/out_sub_{SID}/cutout_ALLFIELDS_sphere_2p1Rvir_sub{SID}.hdf5")
OLD_L2 = Path(f"/scratch/tsingh65/m61-tng/outputs/sid{SID}/rays_and_recipes_sid{SID}_snap99_L2Rvir")
ORIENTATION_JSON = Path(f"/scratch/tsingh65/m61-tng/outputs/sid{SID}/analysis/orientation_sid{SID}_snap99.json")
SUBHALO_JSON = Path(f"/scratch/tsingh65/m61-tng/outputs/sid{SID}/analysis/subhalo_catalog_sid{SID}_snap99.json")
OBS_CSV = Path("/home/tsingh65/m61-tng/data/M61_DIISC_Table1_Table2.csv")


def clean_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")


def configure_paths(sid: int, snap: int = 99, normal_source: str = NORMAL_SOURCE, qso_mode: str = "synthetic"):
    global SID, SNAP, OUT, CUTOUT, OLD_L2, ORIENTATION_JSON, SUBHALO_JSON
    SID = int(sid)
    SNAP = int(snap)
    suffix = "" if normal_source == "pca3_all_stars_10rhalf" and qso_mode == "synthetic" else f"_{clean_tag(normal_source)}_{clean_tag(qso_mode)}"
    OUT = Path(f"/scratch/tsingh65/m61-tng/outputs/sid{SID}/orientation_rebuild_test_pcav3_majorx{suffix}")
    CUTOUT = Path(f"/data/sborthak/m61/cutouts/out_sub_{SID}/cutout_ALLFIELDS_sphere_2p1Rvir_sub{SID}.hdf5")
    OLD_L2 = Path(f"/scratch/tsingh65/m61-tng/outputs/sid{SID}/rays_and_recipes_sid{SID}_snap{SNAP}_L2Rvir")
    ORIENTATION_JSON = Path(f"/scratch/tsingh65/m61-tng/outputs/sid{SID}/analysis/orientation_sid{SID}_snap{SNAP}.json")
    SUBHALO_JSON = Path(f"/scratch/tsingh65/m61-tng/outputs/sid{SID}/analysis/subhalo_catalog_sid{SID}_snap{SNAP}.json")


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise ValueError(f"bad vector {v}")
    return v / n


def minimal_delta(pos: np.ndarray, center: np.ndarray, box: float = 35000.0) -> np.ndarray:
    return (pos - center[None, :] + 0.5 * box) % box - 0.5 * box


def hist2d(x, y, w, width=WIDTH_KPC, npix=NPIX):
    half = 0.5 * width
    H2, _, _ = np.histogram2d(
        x,
        y,
        bins=npix,
        range=[[-half, half], [-half, half]],
        weights=w,
    )
    return H2.T


def weighted_mean_map(x, y, value, weight, width=WIDTH_KPC, npix=NPIX):
    den = hist2d(x, y, weight, width, npix)
    num = hist2d(x, y, weight * value, width, npix)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out, den


def robust_vlim(arrays, percentile=98, floor=50, ceiling=300):
    vals = []
    for arr in arrays:
        a = np.asarray(arr)
        vals.append(np.abs(a[np.isfinite(a)]).ravel())
    vals = np.concatenate([v for v in vals if v.size]) if any(v.size for v in vals) else np.array([])
    if vals.size == 0:
        return floor
    return float(np.clip(np.nanpercentile(vals, percentile), floor, ceiling))


def load_catalog():
    sub = json.loads(SUBHALO_JSON.read_text())
    orient_meta = json.loads(ORIENTATION_JSON.read_text())
    center = np.asarray(orient_meta["center_ckpc_h"], dtype=float)
    if "subhalo_pos_ckpch" in sub:
        api_center = np.asarray(sub["subhalo_pos_ckpch"], dtype=float)
        if np.linalg.norm(api_center - center) > 0.2:
            print(f"WARNING center mismatch orientation={center} api={api_center}")
    vsub = np.asarray(sub["subhalo_vel_kms"], dtype=float)
    return orient_meta, center, vsub


def load_particles(center, load_gas: bool = True):
    with h5py.File(CUTOUT, "r") as f:
        h = float(f["Header"].attrs.get("HubbleParam", H))
        box = float(f["Header"].attrs.get("BoxSize", 35000.0))
        st_pos = np.asarray(f["PartType4/Coordinates"], dtype=np.float32)
        st_mass = np.asarray(f["PartType4/Masses"], dtype=np.float32)
        st_vel = np.asarray(f["PartType4/Velocities"], dtype=np.float32)
        st_sft = np.asarray(f["PartType4/GFM_StellarFormationTime"], dtype=np.float32) if "PartType4/GFM_StellarFormationTime" in f else None
        if load_gas:
            gas_pos = np.asarray(f["PartType0/Coordinates"], dtype=np.float32)
            gas_mass = np.asarray(f["PartType0/Masses"], dtype=np.float32)
            gas_hi = np.asarray(f["PartType0/NeutralHydrogenAbundance"], dtype=np.float32)
            gas_vel = np.asarray(f["PartType0/Velocities"], dtype=np.float32)
        else:
            gas_pos = gas_mass = gas_hi = gas_vel = None
    st_rel = minimal_delta(st_pos.astype(np.float64), center, box).astype(np.float32)
    gas_rel = minimal_delta(gas_pos.astype(np.float64), center, box).astype(np.float32) if gas_pos is not None else None
    return h, st_rel, st_mass, st_vel, st_sft, gas_rel, gas_mass, gas_hi, gas_vel


def recompute_pcav3_normal(st_rel_ckpch, st_mass, h, rhalf_ckpch):
    r = np.linalg.norm(st_rel_ckpch, axis=1)
    sel = r <= 10.0 * float(rhalf_ckpch)
    x_kpc = st_rel_ckpch[sel].astype(np.float64) / h
    w = st_mass[sel].astype(np.float64) * 1e10 / h
    evals, evecs, _ = orient.pca3_weighted(x_kpc, w)
    return norm(evecs[:, 2]), evals.tolist(), int(sel.sum())


def pca_normal_for_selection(rel_ckpch, weight, h, sel):
    x_kpc = rel_ckpch[sel].astype(np.float64) / h
    w = weight[sel].astype(np.float64)
    evals, evecs, _ = orient.pca3_weighted(x_kpc, w)
    return norm(evecs[:, 2]), evals.tolist(), int(sel.sum())


def spin_normal_for_selection(rel_ckpch, vel, weight, h, vsub, sel):
    x_kpc = rel_ckpch[sel].astype(np.float64) / h
    vrel = vel[sel].astype(np.float64) - np.asarray(vsub, dtype=float)[None, :]
    w = weight[sel].astype(np.float64)
    J = np.sum(np.cross(x_kpc, vrel) * w[:, None], axis=0)
    return norm(J), [np.nan, np.nan, np.nan], int(sel.sum())


def choose_normal(source, saved_normal, st_rel, st_mass, st_vel, st_sft, gas_rel, gas_mass, gas_hi, gas_vel, h, rhalf_ckpch, vsub):
    rstar = np.linalg.norm(st_rel, axis=1)
    stellar_weight = st_mass * 1e10 / h
    if source == "saved":
        n, evals, count = norm(saved_normal), [np.nan, np.nan, np.nan], 0
    elif source == "pca3_all_stars_10rhalf":
        n, evals, count = recompute_pcav3_normal(st_rel, st_mass, h, rhalf_ckpch)
    elif source == "star_pca_r2":
        n, evals, count = pca_normal_for_selection(st_rel, stellar_weight, h, rstar <= 2.0 * rhalf_ckpch)
    elif source == "star_pca_r3":
        n, evals, count = pca_normal_for_selection(st_rel, stellar_weight, h, rstar <= 3.0 * rhalf_ckpch)
    elif source == "young_star_pca_a08":
        if st_sft is None:
            raise RuntimeError("PartType4/GFM_StellarFormationTime is required for young_star_pca_a08")
        sel = (rstar <= 10.0 * rhalf_ckpch) & (st_sft > 0.8)
        n, evals, count = pca_normal_for_selection(st_rel, stellar_weight, h, sel)
    elif source == "star_spin_r5":
        n, evals, count = spin_normal_for_selection(st_rel, st_vel, stellar_weight, h, vsub, rstar <= 5.0 * rhalf_ckpch)
    elif source == "hi_spin_r50":
        if gas_rel is None:
            raise RuntimeError("Gas arrays are required for hi_spin_r50")
        rgas_kpc = np.linalg.norm(gas_rel, axis=1) / h
        hi_weight = gas_mass * np.clip(gas_hi, 0, None)
        sel = (rgas_kpc <= 50.0) & (hi_weight > 0)
        n, evals, count = spin_normal_for_selection(gas_rel, gas_vel, hi_weight, h, vsub, sel)
    else:
        raise ValueError(f"Unknown normal source: {source}")
    if np.dot(n, saved_normal) < 0:
        n = -n
    return n, {
        "normal_source": source,
        "normal_selected_count": int(count),
        "normal_eigenvalues_desc": [float(x) if np.isfinite(x) else None for x in evals],
        "normal_dot_saved": float(np.dot(n, saved_normal)),
    }


def observed_qso_major_axis_coords(obs_csv: Path = OBS_CSV) -> dict:
    row = pd.read_csv(obs_csv).iloc[0]
    dec0 = math.radians(float(row["dec_deg"]))
    decq = math.radians(float(row["qso_dec_deg"]))
    kpc_per_deg = float(row["distance_mpc"]) * 1000.0 * math.pi / 180.0
    east_kpc = (float(row["qso_ra_deg"]) - float(row["ra_deg"])) * math.cos(0.5 * (dec0 + decq)) * kpc_per_deg
    north_kpc = (float(row["qso_dec_deg"]) - float(row["dec_deg"])) * kpc_per_deg
    pa = math.radians(float(row["PA_deg"]))
    major = np.array([math.sin(pa), math.cos(pa)])
    minor = np.array([-math.cos(pa), math.sin(pa)])
    sky = np.array([east_kpc, north_kpc])
    x_major = float(np.dot(sky, major))
    y_minor = float(np.dot(sky, minor))
    return {
        "obs_csv": str(obs_csv),
        "east_kpc": float(east_kpc),
        "north_kpc": float(north_kpc),
        "x_major_kpc": x_major,
        "y_minor_kpc": y_minor,
        "rho_from_radec_kpc": float(np.hypot(east_kpc, north_kpc)),
        "rho_table_kpc": float(row["impact_kpc"]),
        "PA_deg": float(row["PA_deg"]),
        "theta_table_deg": float(row["theta_deg"]),
    }


def basis_from_R(R_cur):
    xhat = norm(np.array([1.0, 0.0, 0.0]) @ R_cur)
    yhat = norm(np.array([0.0, 1.0, 0.0]) @ R_cur)
    zhat = norm(np.array([0.0, 0.0, 1.0]) @ R_cur)
    return xhat, yhat, zhat


def build_candidate_geometry(normal_hat, vsub):
    # PA=0 puts the projected disk major axis on the image x-axis.
    R_base, R_base_flip, n_hat, n_hat_flip = orient.build_R_bases(normal_hat, INC_DEG, 0.0)
    rho_kpc = float(np.hypot(QSO_X_KPC, QSO_Y_KPC))
    phi_deg = float(np.degrees(np.arctan2(QSO_Y_KPC, QSO_X_KPC)))
    rows = []
    orient_rows = []
    for alpha in ALPHAS:
        R_cur = orient.fixed_observer_galaxy_alpha_rotation(R_base, n_hat, alpha)
        p0, p1, anchor, los = orient.sightline_endpoints_codeunits(
            center_ckpch=CENTER_CKPCH,
            R_cur=R_cur,
            rho_ckpch=rho_kpc * H,
            phi_deg=phi_deg,
            half_len_ckpch=Rvir_KPC * H,
        )
        xhat, yhat, los_hat = basis_from_R(R_cur)
        major = norm(np.cross(n_hat, los_hat))
        major_angle = float(np.degrees(np.arctan2(np.dot(major, yhat), np.dot(major, xhat))) % 180.0)
        inc = float(np.degrees(np.arccos(np.clip(np.dot(n_hat, los_hat), -1, 1))))
        v_sys_los = float(np.dot(vsub, los_hat))
        rows.append(
            {
                "SubhaloID": SID,
                "sightline_id": "J122138+043026",
                "alpha_deg": alpha,
                "mode": "noflip",
                "orientation_method": "pca_v3_recomputed",
                "alpha_convention": f"major_axis_x_alpha_samples_intrinsic_azimuth__{QSO_NOTE}",
                "obs_inc_deg": INC_DEG,
                "obs_pa_deg_used": 0.0,
                "rho_kpc": rho_kpc,
                "phi_deg": phi_deg,
                "Rvir_kpc": Rvir_KPC,
                "half_len_Rvir": 1.0,
                "total_len_Rvir": 2.0,
                "p0_X_ckpch_abs": p0[0],
                "p0_Y_ckpch_abs": p0[1],
                "p0_Z_ckpch_abs": p0[2],
                "p1_X_ckpch_abs": p1[0],
                "p1_Y_ckpch_abs": p1[1],
                "p1_Z_ckpch_abs": p1[2],
                "anchor_X_ckpch_abs": anchor[0],
                "anchor_Y_ckpch_abs": anchor[1],
                "anchor_Z_ckpch_abs": anchor[2],
                "los_x": los[0],
                "los_y": los[1],
                "los_z": los[2],
                "v_sys_los_kms": v_sys_los,
                "inc_check_deg": inc,
                "projected_major_axis_angle_deg": major_angle,
                "x_qso_kpc": QSO_X_KPC,
                "y_qso_kpc": QSO_Y_KPC,
            }
        )
        orient_rows.append(
            {
                "alpha_deg": alpha,
                "mode": "noflip",
                "los_x": los_hat[0],
                "los_y": los_hat[1],
                "los_z": los_hat[2],
                "v_sys_los_kms": v_sys_los,
                "north_x": yhat[0],
                "north_y": yhat[1],
                "north_z": yhat[2],
                "x_axis_x": xhat[0],
                "x_axis_y": xhat[1],
                "x_axis_z": xhat[2],
                "inc_check_deg": inc,
                "projected_major_axis_angle_deg": major_angle,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(orient_rows)


def make_normal_validation(st_rel, st_mass, saved_normal, selected_normal):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    setups = [
        ("saved PCA-v3 face-on", saved_normal, orient.unit(np.cross(saved_normal, [0, 0, 1]))),
        (f"{NORMAL_SOURCE} face-on", selected_normal, orient.unit(np.cross(selected_normal, [0, 0, 1]))),
    ]
    if not np.all(np.isfinite(setups[0][2])) or np.linalg.norm(setups[0][2]) == 0:
        setups[0] = (setups[0][0], setups[0][1], np.array([1.0, 0.0, 0.0]))
    if not np.all(np.isfinite(setups[1][2])) or np.linalg.norm(setups[1][2]) == 0:
        setups[1] = (setups[1][0], setups[1][1], np.array([1.0, 0.0, 0.0]))
    for col, (label, los, xhat) in enumerate(setups):
        los = norm(los)
        xhat = norm(xhat - np.dot(xhat, los) * los)
        yhat = norm(np.cross(los, xhat))
        X = st_rel @ xhat / H
        Y = st_rel @ yhat / H
        Z = st_rel @ los / H
        m1 = hist2d(X, Y, st_mass)
        m2 = hist2d(X, Z, st_mass)
        for ax, img, ttl, ylabel in [
            (axes[0, col], m1, label, "minor-like [kpc]"),
            (axes[1, col], m2, label.replace("face-on", "edge-on"), "LOS/depth [kpc]"),
        ]:
            with np.errstate(divide="ignore"):
                im = ax.imshow(np.log10(np.where(img > 0, img, np.nan)), origin="lower", extent=[-50, 50, -50, 50], cmap="magma")
            ax.set_title(ttl)
            ax.set_xlabel("major-like [kpc]")
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"$\log_{10}$ stellar mass [arb.]")
    fig.savefig(OUT / "figures" / f"sid{SID}_pcav3_normal_validation.png", dpi=180)
    plt.close(fig)


def make_alpha_diagnostic(st_rel, st_mass, st_vel, gas_rel, gas_mass, gas_hi, gas_vel, vsub, geom):
    maps = []
    for _, row in geom.iterrows():
        alpha = int(row["alpha_deg"])
        xhat = np.array([row["x_axis_x"], row["x_axis_y"], row["x_axis_z"]])
        yhat = np.array([row["north_x"], row["north_y"], row["north_z"]])
        los = np.array([row["los_x"], row["los_y"], row["los_z"]])
        sx = st_rel @ xhat / H
        sy = st_rel @ yhat / H
        sv = (st_vel.astype(np.float32) - vsub.astype(np.float32)) @ los.astype(np.float32)
        smap = hist2d(sx, sy, st_mass)
        svmap, _ = weighted_mean_map(sx, sy, sv, st_mass)
        if gas_rel is not None:
            gx = gas_rel @ xhat / H
            gy = gas_rel @ yhat / H
            gv = (gas_vel.astype(np.float32) - vsub.astype(np.float32)) @ los.astype(np.float32)
            hi_weight = gas_mass * np.clip(gas_hi, 0, None)
            himap = hist2d(gx, gy, hi_weight)
            hivmap, _ = weighted_mean_map(gx, gy, gv, hi_weight)
        else:
            himap = hivmap = None
        maps.append((alpha, smap, svmap, himap, hivmap))

    star_vlim = robust_vlim([m[2] for m in maps])
    has_gas = maps[0][3] is not None
    gas_vlim = robust_vlim([m[4] for m in maps]) if has_gas else None
    extent = [-50, 50, -50, 50]
    ncols = 4 if has_gas else 2
    fig, axes = plt.subplots(len(maps), ncols, figsize=(4.5 * ncols, 4.1 * len(maps)), constrained_layout=True)
    for i, (alpha, smap, svmap, himap, hivmap) in enumerate(maps):
        panels = [
            (np.log10(np.where(smap > 0, smap, np.nan)), "magma", None, None, r"$\log_{10} M_\star$ [arb.]"),
            (svmap, "RdBu_r", -star_vlim, star_vlim, r"$v_{\rm LOS,\star}-v_{\rm sys,LOS}$ [km/s]"),
        ]
        if has_gas:
            panels.extend([
                (np.log10(np.where(himap > 0, himap, np.nan)), "magma", None, None, r"$\log_{10} M_{\rm HI}$ [arb.]"),
                (hivmap, "RdBu_r", -gas_vlim, gas_vlim, r"$v_{\rm LOS,HI}-v_{\rm sys,LOS}$ [km/s]"),
            ])
        for j, (arr, cmap, vmin, vmax, label) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axhline(0, color="w", lw=0.5, alpha=0.45)
            ax.axvline(0, color="w", lw=0.5, alpha=0.45)
            ax.scatter([0], [0], marker="+", s=55, c="cyan", lw=1.3)
            ax.scatter([QSO_X_KPC], [QSO_Y_KPC], marker="*", s=90, c="lime", edgecolors="black", lw=0.5)
            ax.add_patch(
                Ellipse(
                    (0, 0),
                    width=60.0,
                    height=60.0 * math.cos(math.radians(INC_DEG)),
                    angle=0.0,
                    fill=False,
                    edgecolor="white",
                    lw=0.9,
                    ls="--",
                    alpha=0.7,
                )
            )
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.set_aspect("equal")
            ax.set_xlabel("projected major-axis x [kpc]")
            ax.set_ylabel("projected minor-axis y [kpc]")
            ax.set_title(f"alpha={alpha} deg" if j == 0 else "")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=label)
    fig.suptitle(
        f"SID{SID} diagnostic: normal={NORMAL_SOURCE}; major axis fixed on x; {QSO_NOTE}; noflip; inc=23 deg",
        fontsize=15,
    )
    fig.savefig(OUT / "figures" / f"sid{SID}_majorx_discrete_alpha_diagnostic.png", dpi=180)
    plt.close(fig)


def old_geometry_summary():
    old = pd.read_csv(OLD_L2 / f"orient_peralpha_sid{SID}.csv")
    header = json.loads((OLD_L2 / f"orient_header_sid{SID}.json").read_text())
    n = np.asarray(header["normal_used_hat"], dtype=float)
    rows = []
    for alpha in ALPHAS:
        row = old[(old.alpha_deg == alpha) & (old["mode"] == "noflip")].iloc[0]
        los = np.array([row.los_x, row.los_y, row.los_z], dtype=float)
        north = np.array([row.north_x, row.north_y, row.north_z], dtype=float)
        north = norm(north - np.dot(north, los) * los)
        xhat = norm(np.cross(north, los))
        major = norm(np.cross(n, los))
        rows.append(
            {
                "alpha_deg": alpha,
                "old_inc_check_deg": float(np.degrees(np.arccos(np.clip(np.dot(n, los), -1, 1)))),
                "old_major_axis_angle_deg": float(np.degrees(np.arctan2(np.dot(major, north), np.dot(major, xhat))) % 180),
                "old_los_x": los[0],
                "old_los_y": los[1],
                "old_los_z": los[2],
            }
        )
    return pd.DataFrame(rows)


def main():
    global CENTER_CKPCH, VSUB, NORMAL_SOURCE, QSO_X_KPC, QSO_Y_KPC, QSO_NOTE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sid", type=int, default=413372)
    ap.add_argument("--snap", type=int, default=99)
    ap.add_argument(
        "--normal-source",
        default="pca3_all_stars_10rhalf",
        choices=["saved", "pca3_all_stars_10rhalf", "star_pca_r2", "star_pca_r3", "young_star_pca_a08", "star_spin_r5", "hi_spin_r50"],
    )
    ap.add_argument("--qso-mode", default="synthetic", choices=["synthetic", "observed_radec_pa"])
    ap.add_argument("--stars-only", action="store_true", help="Skip gas arrays/maps for memory-heavy diagnostic runs.")
    args = ap.parse_args()
    NORMAL_SOURCE = args.normal_source
    configure_paths(args.sid, args.snap, args.normal_source, args.qso_mode)

    for sub in ("figures", "data", "logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    orient_meta, CENTER_CKPCH, VSUB = load_catalog()
    h, st_rel, st_mass, st_vel, st_sft, gas_rel, gas_mass, gas_hi, gas_vel = load_particles(CENTER_CKPCH, load_gas=not args.stars_only)
    saved_normal = norm(np.asarray(orient_meta["normal_used_hat"], dtype=float))
    if args.qso_mode == "observed_radec_pa":
        qso_meta = observed_qso_major_axis_coords()
        QSO_X_KPC = qso_meta["x_major_kpc"]
        QSO_Y_KPC = qso_meta["y_minor_kpc"]
        QSO_NOTE = "observed QSO RA/Dec rotated into M61 PA major-axis frame"
    else:
        qso_meta = {"x_major_kpc": QSO_X_KPC, "y_minor_kpc": QSO_Y_KPC, "rho_table_kpc": RHO_KPC}
        QSO_NOTE = "synthetic marker at (+rho,0)"
    selected_normal, normal_meta = choose_normal(
        args.normal_source,
        saved_normal,
        st_rel,
        st_mass,
        st_vel,
        st_sft,
        gas_rel,
        gas_mass,
        gas_hi,
        gas_vel,
        h,
        orient_meta["rhalf_star_ckpc_h"],
        VSUB,
    )

    rays_preview, orient_preview = build_candidate_geometry(selected_normal, VSUB)
    old = old_geometry_summary()
    audit = orient_preview.merge(old, on="alpha_deg", how="left")

    make_normal_validation(st_rel, st_mass, saved_normal, selected_normal)
    make_alpha_diagnostic(st_rel, st_mass, st_vel, gas_rel, gas_mass, gas_hi, gas_vel, VSUB, orient_preview)

    rays_preview.to_csv(OUT / "data" / f"rays_preview_sid{SID}_L2Rvir_majorx_discrete_alphas.csv", index=False)
    orient_preview.to_csv(OUT / "data" / f"orient_preview_sid{SID}_majorx_discrete_alphas.csv", index=False)
    audit.to_csv(OUT / "data" / f"sid{SID}_old_vs_majorx_geometry_audit.csv", index=False)
    metadata = {
        "sid": SID,
        "snap": SNAP,
        "cutout": str(CUTOUT),
        "output_dir": str(OUT),
        "normal_saved_hat": saved_normal.tolist(),
        "normal_selected_hat": selected_normal.tolist(),
        **normal_meta,
        "inc_deg": INC_DEG,
        "diagnostic_PA_deg_used": 0.0,
        "diagnostic_note": f"Projected major axis fixed on image x-axis; {QSO_NOTE}.",
        "qso_marker": qso_meta,
        "rho_kpc": float(np.hypot(QSO_X_KPC, QSO_Y_KPC)),
        "phi_deg": float(np.degrees(np.arctan2(QSO_Y_KPC, QSO_X_KPC))),
        "vsub_kms": VSUB.tolist(),
        "stars_only": bool(args.stars_only),
    }
    (OUT / "data" / f"sid{SID}_pcav3_majorx_diagnostic_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"FIG {OUT / 'figures' / f'sid{SID}_pcav3_normal_validation.png'}")
    print(f"FIG {OUT / 'figures' / f'sid{SID}_majorx_discrete_alpha_diagnostic.png'}")


if __name__ == "__main__":
    main()
