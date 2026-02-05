#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
orient_m61.py

Batch: orientations + LOS endpoints for M61 matched TNG subhalos on HPC.

You provide:
  (1) An "observation geometry" CSV for the M61 QSO pair
      (e.g., M61_DIISC_Table1_Table2.csv)
  (2) A matches CSV listing simulated SubhaloIDs ranked by closeness
      (e.g., m61_closest_matches_3d.csv)

This script:
  - takes the top-N matched SubhaloIDs
  - loads each subhalo cutout from:   --cutout-root /scratch/.../out_sub_<SID>/
  - reads group catalog from:         --groupcat-base /scratch/.../groups_099/
  - computes orientation bases using either:
        * PCA v3 (default)
        * stellar spin (optional override list)
  - applies the OBS inc + PA to set the on-sky disk orientation
  - for each sightline in the obs CSV (each row can be a sightline):
        uses its rho,phi to place the impact point in the observer plane
  - loops over alpha rotations about the (post-inc) disk normal
  - writes rays CSVs and small JSON headers per SID and per run length

Important convention notes:
  - Observer frame axes: x,y in sky plane; +z is LOS direction.
  - rho,phi define a position in sky plane:
        x = rho cos(phi), y = rho sin(phi)     (degrees, phi measured from +x)
  - inc tilts the galaxy about observer x-axis (rot_x(inc)).
  - PA is implemented as a rotation about observer z-axis (rot_z(PA)) AFTER the tilt.
  - If your PA is measured "east of north" (astronomy convention),
    pass --pa-from-north to convert to this script’s +x-based convention.

Outputs (under --out-base):
  <out-base>/sid<SID>/
      analysis/
          debug_sid<SID>_snap99.png
          orientation_sid<SID>_snap99.json
      rays_and_recipes_sid<SID>_snap99_L3Rvir/
          rays_sid<SID>.csv
          orient_peralpha_sid<SID>.csv
          orient_header_sid<SID>.json
      rays_and_recipes_sid<SID>_snap99_L4Rvir/
          (same)
  <out-base>/orientation_summary_snap99.csv
"""

import os
import re
import glob
import json
import math
import argparse
import traceback
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import h5py


# ──────────────────────────────────────────────────────────────────────
# Runtime helper: reduce Matplotlib ABI headaches on HPC
# (forces Agg backend; optionally prefers conda libstdc++ if present)
# ──────────────────────────────────────────────────────────────────────

def _force_noninteractive_matplotlib():
    os.environ.setdefault("MPLBACKEND", "Agg")

def _prefer_conda_libstdcxx():
    """
    If Matplotlib import fails due to GLIBCXX mismatch, it usually means the
    system libstdc++ is being picked instead of conda's.

    This tries to prefer conda's lib directory before importing matplotlib.
    It is safe to run even if you already fixed your environment.
    """
    try:
        import sys
        conda_prefix = sys.prefix
        conda_lib = os.path.join(conda_prefix, "lib")
        if os.path.isdir(conda_lib):
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            if conda_lib not in ld.split(":"):
                os.environ["LD_LIBRARY_PATH"] = conda_lib + (":" + ld if ld else "")
            # Optional: LD_PRELOAD conda libstdc++ if it exists
            cand = os.path.join(conda_lib, "libstdc++.so.6")
            if os.path.isfile(cand):
                pre = os.environ.get("LD_PRELOAD", "")
                if cand not in pre.split(":"):
                    os.environ["LD_PRELOAD"] = cand + (":" + pre if pre else "")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────

def prefer(df: pd.DataFrame, base: str) -> str:
    """Pick a column name for `base` among common variants."""
    for cand in (base, f"{base}_obs", f"{base}_sum", f"{base}_x", f"{base}_y",
                 base.lower(), base.upper()):
        if cand in df.columns:
            return cand
    raise KeyError(f"Column not found: {base} (tried common variants)")

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def minimal_image_delta(dpos, box):
    return (dpos + 0.5 * box) % box - 0.5 * box

def recenter_positions(x, center, box):
    return minimal_image_delta(x - center[None, :], box)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def rot_x(theta_deg: float) -> np.ndarray:
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], float)

def rot_z(theta_deg: float) -> np.ndarray:
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], float)

def rodrigues_axis_angle(axis, theta_rad: float) -> np.ndarray:
    a = unit(axis)
    ax, ay, az = a
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    C = 1 - c
    return np.array([
        [c + ax*ax*C,     ax*ay*C - az*s, ax*az*C + ay*s],
        [ay*ax*C + az*s,  c + ay*ay*C,    ay*az*C - ax*s],
        [az*ax*C - ay*s,  az*ay*C + ax*s, c + az*az*C]
    ], float)

def R_from_u_to_v(u, v) -> np.ndarray:
    """Active rotation taking u -> v."""
    u = unit(u)
    v = unit(v)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        a = unit(np.cross(u, [1, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1, 0]))
        return rodrigues_axis_angle(a, math.pi)
    axis = unit(np.cross(u, v))
    return rodrigues_axis_angle(axis, math.acos(c))

def pca3_weighted(X, w):
    """Mass-weighted 3D PCA. Return evals(desc), evecs(cols), and COM."""
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    wsum = np.sum(w)
    xc = np.sum(X * w[:, None], axis=0) / max(wsum, 1e-30)
    X0 = X - xc
    C = (X0 * w[:, None]).T @ X0 / max(wsum, 1e-30)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx], xc

def pca2_weighted(XY, w):
    XY = np.asarray(XY, float)
    w = np.asarray(w, float)
    wsum = np.sum(w)
    xc = np.sum(XY * w[:, None], axis=0) / max(wsum, 1e-30)
    X0 = XY - xc
    C = (X0 * w[:, None]).T @ X0 / max(wsum, 1e-30)
    evals, evecs = np.linalg.eigh(C)
    maj = unit(evecs[:, np.argmax(evals)])
    ang = (np.degrees(np.arctan2(maj[1], maj[0])) % 180.0)
    return maj, ang

def stellar_spin_vector(X_rel_ckpch, V_kms, M_1e10, remove_bulk_vel: bool = False):
    """Mass-weighted stellar angular momentum vector (code units)."""
    if remove_bulk_vel:
        vcm = np.sum(V_kms * M_1e10[:, None], axis=0) / max(np.sum(M_1e10), 1e-30)
        V_rel = V_kms - vcm[None, :]
    else:
        V_rel = V_kms
    return np.sum(np.cross(X_rel_ckpch, V_rel) * M_1e10[:, None], axis=0)

def inc_PA_from_vector(J):
    """
    inc (deg): inclination wrt +z (0 face-on, 90 edge-on), uses |cos|.
    PA  (deg): position angle of line of nodes (z × J) in [0,180), from +x.
    """
    J = np.asarray(J, float)
    Jn = np.linalg.norm(J)
    if Jn == 0:
        return np.nan, np.nan
    Jhat = J / Jn
    inc = np.degrees(np.arccos(np.clip(abs(Jhat[2]), 0, 1)))
    n = np.cross([0, 0, 1.0], Jhat)
    PA = 0.0 if np.hypot(n[0], n[1]) < 1e-14 else (np.degrees(np.arctan2(n[1], n[0])) % 180.0)
    return inc, PA

def sightline_endpoints_codeunits(center_ckpch, R_cur, rho_ckpch, phi_deg, half_len_ckpch):
    """
    Build endpoints for a LOS in code units (ckpc/h).
      - rho,phi specify the anchor in observer plane z=0
      - LOS direction is +z in observer frame, mapped into native by R_cur
    """
    x = rho_ckpch * math.cos(math.radians(phi_deg))
    y = rho_ckpch * math.sin(math.radians(phi_deg))
    r_obs = np.array([x, y, 0.0], float)

    ez_obs = np.array([0.0, 0.0, 1.0], float)

    # Map observer vectors into native by right-multiplication:
    # r_nat = r_obs @ R_cur
    r_nat = r_obs @ R_cur
    L_nat = unit(ez_obs @ R_cur)

    p0 = center_ckpch + r_nat - half_len_ckpch * L_nat
    p1 = center_ckpch + r_nat + half_len_ckpch * L_nat
    anchor = center_ckpch + r_nat
    return p0, p1, anchor, L_nat


# ──────────────────────────────────────────────────────────────────────
# Group catalog readers (no illustris_python dependency)
# ──────────────────────────────────────────────────────────────────────

def _chunk_path(groupcat_base: str, snap: int, filenum: int) -> str:
    return os.path.join(groupcat_base, f"fof_subhalo_tab_{snap:03d}.{filenum}.hdf5")

def _find_chunks(groupcat_base: str, snap: int) -> List[str]:
    files = sorted(
        glob.glob(os.path.join(groupcat_base, f"fof_subhalo_tab_{snap:03d}.*.hdf5")),
        key=lambda p: int(os.path.splitext(p)[0].split(".")[-1]) if re.search(r"\.\d+\.hdf5$", p) else -1
    )
    alt = os.path.join(groupcat_base, f"groups_{snap:03d}.hdf5")
    if (not files) and os.path.exists(alt):
        files = [alt]
    return files

def _read_header_any(groupcat_base: str, snap: int):
    """Return (h, box_ckpch, offs_sub, offs_grp) if offsets exist, else offs_* = None."""
    candidates = []
    p0 = _chunk_path(groupcat_base, snap, 0)
    if os.path.exists(p0):
        candidates = [p0]
    else:
        candidates = _find_chunks(groupcat_base, snap)

    if not candidates:
        raise FileNotFoundError(f"No group catalog files found under {groupcat_base} for snap {snap}.")

    for ff in candidates:
        try:
            with h5py.File(ff, "r") as f:
                h = float(f["Header"].attrs["HubbleParam"])
                bs = float(f["Header"].attrs["BoxSize"])
                offs_sub = None
                offs_grp = None
                for key in ("FileOffsets_Subhalo", "FileOffsets_SubFindSubhalo", "FileOffsets_SubFind"):
                    if key in f["Header"].attrs:
                        offs_sub = np.array(f["Header"].attrs[key], dtype=np.int64)
                        break
                for key in ("FileOffsets_Group", "FileOffsets_SubfindGroup", "FileOffsets_FoFGroup"):
                    if key in f["Header"].attrs:
                        offs_grp = np.array(f["Header"].attrs[key], dtype=np.int64)
                        break
                return h, bs, offs_sub, offs_grp
        except Exception:
            continue

    ff = candidates[0]
    with h5py.File(ff, "r") as f:
        h = float(f["Header"].attrs["HubbleParam"])
        bs = float(f["Header"].attrs["BoxSize"])
    return h, bs, None, None

def _is_valid_chunk_with(dataset_name: str, path: str) -> bool:
    try:
        with h5py.File(path, "r") as f:
            if dataset_name not in f:
                return False
            obj = f[dataset_name]
            try:
                return len(obj) > 0
            except TypeError:
                return True
    except Exception:
        return False

def read_single_subhalo(groupcat_base: str, snap: int, subhalo_id: int) -> Dict[str, Any]:
    """
    Return dict: {h, BoxSize, SubhaloPos, SubhaloHalfmassRadType, SubhaloSpin, SubhaloGrNr}
    """
    subhalo_id = int(subhalo_id)
    h, boxsize, offs_sub, _ = _read_header_any(groupcat_base, snap)

    if offs_sub is not None:
        rel = subhalo_id - offs_sub
        fileNum = int(np.max(np.where(rel >= 0)))
        local_index = int(rel[fileNum])
        candidate = _chunk_path(groupcat_base, snap, fileNum)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Missing expected chunk {fileNum}: {candidate}")
        if not _is_valid_chunk_with("Subhalo", candidate):
            raise RuntimeError(f"{os.path.basename(candidate)} lacks Subhalo datasets.")
        with h5py.File(candidate, "r") as f:
            sub = f["Subhalo"]
            pos = np.asarray(sub["SubhaloPos"][local_index], dtype=np.float64)
            hmr = np.asarray(sub["SubhaloHalfmassRadType"][local_index], dtype=np.float64)
            spin = np.asarray(sub["SubhaloSpin"][local_index], dtype=np.float64)
            grnr = int(sub["SubhaloGrNr"][local_index]) if "SubhaloGrNr" in sub else -1
        return {"h": h, "BoxSize": boxsize, "SubhaloPos": pos,
                "SubhaloHalfmassRadType": hmr, "SubhaloSpin": spin, "SubhaloGrNr": grnr}

    files = _find_chunks(groupcat_base, snap)
    remaining = subhalo_id
    for ff in files:
        if not _is_valid_chunk_with("Subhalo", ff):
            continue
        with h5py.File(ff, "r") as f:
            n_here = f["Subhalo"]["SubhaloPos"].shape[0]
            if remaining >= n_here:
                remaining -= n_here
                continue
            sub = f["Subhalo"]
            pos = np.asarray(sub["SubhaloPos"][remaining], dtype=np.float64)
            hmr = np.asarray(sub["SubhaloHalfmassRadType"][remaining], dtype=np.float64)
            spin = np.asarray(sub["SubhaloSpin"][remaining], dtype=np.float64)
            grnr = int(sub["SubhaloGrNr"][remaining]) if "SubhaloGrNr" in sub else -1
        return {"h": h, "BoxSize": boxsize, "SubhaloPos": pos,
                "SubhaloHalfmassRadType": hmr, "SubhaloSpin": spin, "SubhaloGrNr": grnr}

    raise IndexError(f"SubhaloID {subhalo_id} not found.")

def read_group_field(groupcat_base: str, snap: int, group_index: int, field: str = "Group_R_Crit200"):
    """
    Read a scalar group field for a given FoF group index (global).
    Returns (val, h).
    """
    group_index = int(group_index)
    h, _, _, offs_grp = _read_header_any(groupcat_base, snap)

    if offs_grp is not None:
        rel = group_index - offs_grp
        fileNum = int(np.max(np.where(rel >= 0)))
        local_index = int(rel[fileNum])
        candidate = _chunk_path(groupcat_base, snap, fileNum)
        if (not os.path.exists(candidate)) or (not _is_valid_chunk_with("Group", candidate)):
            return np.nan, h
        with h5py.File(candidate, "r") as f:
            grp = f["Group"]
            if field not in grp:
                return np.nan, h
            val = grp[field][local_index]
            try:
                return float(val), h
            except Exception:
                return np.nan, h

    for ff in _find_chunks(groupcat_base, snap):
        if not _is_valid_chunk_with("Group", ff):
            continue
        with h5py.File(ff, "r") as f:
            grp = f["Group"]
            if field not in grp:
                continue
            n_here = grp[field].shape[0]
            if group_index < n_here:
                try:
                    return float(grp[field][group_index]), h
                except Exception:
                    return np.nan, h
            group_index -= n_here

    return np.nan, h


# ──────────────────────────────────────────────────────────────────────
# Cutout I/O (HPC: out_sub_<SID>/...)
# ──────────────────────────────────────────────────────────────────────

def find_cutout_h5(sid: int, cutout_root: str) -> str:
    """For HPC layout: cutout_root/out_sub_<SID>/*.hdf5 (pick first match)."""
    sid = int(sid)
    sub_dir = os.path.join(cutout_root, f"out_sub_{sid}")
    pats = [
        os.path.join(sub_dir, "cutout*sub*.hdf5"),
        os.path.join(sub_dir, "cutout*.hdf5"),
        os.path.join(sub_dir, "*.hdf5"),
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return ""

def read_cutout_particles(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        stars = {
            "Coordinates": np.asarray(f["/PartType4/Coordinates"][...], dtype=np.float64),
            "Masses":      np.asarray(f["/PartType4/Masses"][...],      dtype=np.float64),
            "Velocities":  np.asarray(f["/PartType4/Velocities"][...],  dtype=np.float64),
        }
    return stars


# ──────────────────────────────────────────────────────────────────────
# Orientation builders
# ──────────────────────────────────────────────────────────────────────

def build_face_rotation_from_normal(normal_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R_face_noflip, R_face_flip) that map +/-normal -> +z."""
    n_hat = unit(np.asarray(normal_vec, float))
    if np.linalg.norm(n_hat) == 0:
        return np.eye(3), np.eye(3)
    R_face_noflip = R_from_u_to_v(n_hat,  np.array([0, 0, 1.0]))
    R_face_flip   = R_from_u_to_v(-n_hat, np.array([0, 0, 1.0]))
    return R_face_noflip, R_face_flip

def build_R_bases(normal_vec: np.ndarray, inc_deg: float, pa_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build two bases (noflip, flip) and two corresponding normal hats (n_hat, -n_hat).

    Convention:
      - first rotate so that normal -> +z (face-on)
      - then apply tilt by inc about x: rot_x(inc)
      - then rotate in sky plane by PA about z: rot_z(PA)

    R_base = rot_z(PA) @ rot_x(inc) @ R_face
    """
    n_hat = unit(np.asarray(normal_vec, float))
    if np.linalg.norm(n_hat) == 0:
        n_hat = np.array([0, 0, 1.0], float)

    R_face_noflip, R_face_flip = build_face_rotation_from_normal(n_hat)

    R_base_noflip = rot_z(pa_deg) @ rot_x(inc_deg) @ R_face_noflip
    R_base_flip   = rot_z(pa_deg) @ rot_x(inc_deg) @ R_face_flip

    return R_base_noflip, R_base_flip, n_hat, -n_hat

def build_normal_from_pca_v3(Xs_rel_ckpch: np.ndarray, Ms_1e10: np.ndarray, h: float) -> np.ndarray:
    """PCA v3 from stellar positions (in kpc), weighted by stellar masses."""
    Xs_kpc  = Xs_rel_ckpch / h
    Ms_msun = Ms_1e10 * 1e10 / h
    _, evecs3, _ = pca3_weighted(Xs_kpc, Ms_msun)
    v3_hat = unit(evecs3[:, 2])
    return v3_hat


# ──────────────────────────────────────────────────────────────────────
# Debug figure (optional)
# ──────────────────────────────────────────────────────────────────────

def _mpl_import():
    """
    Import matplotlib lazily.
    If this fails with GLIBCXX mismatch, the environment is still loading system libstdc++.
    """
    _force_noninteractive_matplotlib()
    _prefer_conda_libstdcxx()
    import matplotlib as mpl  # noqa
    import matplotlib.pyplot as plt  # noqa
    return mpl, plt

def setup_plot_style(mpl):
    bg_color = "#0f111a"
    fg_color = "#eaeaea"
    mpl.rcParams.update({
        "figure.facecolor": bg_color,
        "axes.facecolor": bg_color,
        "savefig.facecolor": bg_color,
        "axes.edgecolor": fg_color,
        "axes.labelcolor": fg_color,
        "xtick.color": fg_color,
        "ytick.color": fg_color,
        "text.color": fg_color,
        "font.family": "DejaVu Sans",
        "font.size": 12,
    })

def mass_map_arbitrary(M_weights, XY_kpc, half_width_kpc, nbin):
    L = half_width_kpc
    x = np.clip(XY_kpc[:, 0], -L, L)
    y = np.clip(XY_kpc[:, 1], -L, L)
    H, xe, ye = np.histogram2d(x, y, bins=nbin, range=[[-L, L], [-L, L]], weights=M_weights)
    return H.T, xe, ye, (2 * L) / nbin

def rms_z(arr_xyz):
    z = arr_xyz[:, 2]
    return float(np.sqrt(np.mean((z - np.mean(z))**2)))

def make_debug_figure(
    sid: int,
    snap: int,
    out_dir: str,
    Xs_rel_ckpch: np.ndarray,
    Ms_1e10: np.ndarray,
    Vs_kms: np.ndarray,
    center_ckpch: np.ndarray,
    rhalf_star_ckpch: float,
    J_tot: np.ndarray,
    h: float,
    extent_kpc_h: float = 80.0,
    nbin: int = 1200,
    cmap_name: str = "jet",
    vmin: float = 4.0,
    vmax: float = 7.0,
    remove_bulk_vel: bool = False,
):
    mpl, plt = _mpl_import()
    setup_plot_style(mpl)

    Xs_kpc  = Xs_rel_ckpch / h
    Ms_msun = Ms_1e10 * 1e10 / h

    J_star = stellar_spin_vector(Xs_rel_ckpch, Vs_kms, Ms_1e10, remove_bulk_vel=remove_bulk_vel)

    evals3, evecs3, _ = pca3_weighted(Xs_kpc, Ms_msun)
    v3_hat = unit(evecs3[:, 2])

    inc_t, PA_t = inc_PA_from_vector(J_tot)
    inc_s, PA_s = inc_PA_from_vector(J_star)
    _, ang2_xy  = pca2_weighted(Xs_kpc[:, [0, 1]], Ms_msun)

    L_kpc = (extent_kpc_h / 2.0) / h

    def mass_map(MW, XY):
        H, _, _, _ = mass_map_arbitrary(MW, XY, L_kpc, nbin)
        return H

    R_face_tot = R_from_u_to_v(unit(J_tot),  np.array([0, 0, 1.0]))
    R_face_str = R_from_u_to_v(unit(J_star), np.array([0, 0, 1.0]))
    R_face_pca = R_from_u_to_v(v3_hat,       np.array([0, 0, 1.0]))

    Ximg_kpc  = Xs_kpc
    Wimg_msun = Ms_msun

    H_native   = mass_map(Wimg_msun, Ximg_kpc[:, [0, 1]])
    X_face_tot = (Ximg_kpc @ R_face_tot.T)
    X_face_str = (Ximg_kpc @ R_face_str.T)
    X_face_pca = (Ximg_kpc @ R_face_pca.T)

    H_face_t = mass_map(Wimg_msun, X_face_tot[:, [0, 1]])
    H_edge_t = mass_map(Wimg_msun, X_face_tot[:, [0, 2]])
    H_face_s = mass_map(Wimg_msun, X_face_str[:, [0, 1]])
    H_edge_s = mass_map(Wimg_msun, X_face_str[:, [0, 2]])
    H_face_p = mass_map(Wimg_msun, X_face_pca[:, [0, 1]])
    H_edge_p = mass_map(Wimg_msun, X_face_pca[:, [0, 2]])

    fig, axs = plt.subplots(2, 4, figsize=(22, 11), constrained_layout=True)

    def show_map(ax, H, title, xlabel="x [kpc]", ylabel="y [kpc]"):
        im = ax.imshow(np.log10(H + 1e-12), origin="lower",
                       extent=[-L_kpc, L_kpc, -L_kpc, L_kpc],
                       interpolation="nearest", cmap=cmap_name,
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(r"$\log_{10}$ projected mass [arb.]")
        return im

    title0 = f"Native XY | inc_tot={inc_t:.1f}°, PA_tot={PA_t:.1f}° | inc_*={inc_s:.1f}°, PA_*={PA_s:.1f}° | sid={sid}"
    show_map(axs[0, 0], H_native, title0)
    show_map(axs[0, 1], H_face_t, "Face-on (SubhaloSpin → +z)")
    show_map(axs[0, 2], H_edge_t, "Edge-on XZ (SubhaloSpin)", xlabel="x [kpc]", ylabel="z [kpc]")
    show_map(axs[0, 3], H_face_s, "Face-on (stellar spin → +z)")
    show_map(axs[1, 0], H_edge_s, "Edge-on XZ (stellar spin)", xlabel="x [kpc]", ylabel="z [kpc]")
    show_map(axs[1, 1], H_face_p, "Face-on (PCA v3 → +z)")
    show_map(axs[1, 2], H_edge_p, "Edge-on XZ (PCA v3)", xlabel="x [kpc]", ylabel="z [kpc]")

    ax_txt = axs[1, 3]
    ax_txt.axis("off")
    ax_txt.text(
        0.02, 0.98,
        "\n".join([
            f"sid = {sid}",
            f"N*  = {len(Ms_msun)}",
            "",
            f"inc_tot = {inc_t:.2f}°, PA_tot = {PA_t:.2f}°",
            f"inc_*   = {inc_s:.2f}°, PA_*   = {PA_s:.2f}°",
            f"PCA 2D major angle (XY) = {ang2_xy:.2f}°",
            "",
            f"RMS z (edge SubhaloSpin) = {rms_z(X_face_tot):.3f} kpc",
            f"RMS z (edge stellar)     = {rms_z(X_face_str):.3f} kpc",
            f"RMS z (edge PCA)         = {rms_z(X_face_pca):.3f} kpc",
        ]),
        transform=ax_txt.transAxes,
        va="top", ha="left"
    )

    ensure_dir(out_dir)
    outfile = os.path.join(out_dir, f"debug_sid{sid}_snap{snap}.png")
    plt.savefig(outfile, dpi=120)
    plt.close(fig)

    extra = {
        "inc_tot_deg": float(inc_t), "PA_tot_deg": float(PA_t),
        "inc_star_deg": float(inc_s), "PA_star_deg": float(PA_s),
        "PCA_major2D_XY_deg": float(ang2_xy),
        "J_star": J_star.tolist(),
        "v3_hat": v3_hat.tolist(),
    }
    return outfile, extra


# ──────────────────────────────────────────────────────────────────────
# Loaders: obs CSV and matches CSV
# ──────────────────────────────────────────────────────────────────────

def load_obs_geometry(obs_csv: str) -> Dict[str, Any]:
    """
    Load observational geometry from a CSV.

    FIXED BEHAVIOR:
      - Always returns obs["sightlines"] as a list (one per row in CSV).
      - Returns inc_deg and pa_deg as scalars taken from the first row,
        or an M61 row if a galaxy/name column contains "M61".

    Required columns (with many accepted aliases):
      - rho (impact parameter) in kpc
      - phi in degrees
      - inc in degrees
      - pa in degrees

    Output dict:
      obs = {
        "inc_deg": float,
        "pa_deg": float,
        "Rvir_kpc": optional float,
        "sightlines": [
            {"sightline_id": str, "rho_kpc": float, "phi_deg": float, "Rvir_kpc": optional float},
            ...
        ],
        ... optional metadata ...
      }
    """
    df = pd.read_csv(obs_csv)
    if df.shape[0] == 0:
        raise RuntimeError(f"Obs CSV is empty: {obs_csv}")

    colmap = {c.strip().lower(): c for c in df.columns}

    def pick_col(cands):
        for cand in cands:
            key = cand.strip().lower()
            if key in colmap:
                return colmap[key]
        return None

    rho_col = pick_col(["rho", "rho_kpc", "impact_kpc", "impact_parameter_kpc", "impact", "b_kpc", "b"])
    phi_col = pick_col(["phi", "phi_deg", "theta_deg", "theta", "azimuth_deg", "azimuth", "phi(deg)"])
    inc_col = pick_col(["inc", "inc_deg", "inclination_deg", "inclination", "i_deg", "i"])
    pa_col  = pick_col(["pa", "pa_deg", "pa(deg)", "position_angle_deg", "positionangle_deg", "pa_gal_deg", "pa_gal", "PA_deg"])
    rvir_col= pick_col(["rvir_kpc", "r_vir", "rvir", "rvir(kpc)", "r200_kpc", "r200"])

    missing = []
    if rho_col is None: missing.append("rho (e.g. rho/impact_kpc)")
    if phi_col is None: missing.append("phi (e.g. phi/theta_deg)")
    if inc_col is None: missing.append("inc (e.g. inc/inclination_deg)")
    if pa_col  is None: missing.append("pa (e.g. pa_deg/PA_deg)")
    if missing:
        raise RuntimeError(
            "Obs CSV missing required geometry columns: "
            + ", ".join(missing)
            + f". Found columns: {list(df.columns)}"
        )

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    # Pick a representative row for galaxy-wide inc/PA (prefer row matching M61 if possible)
    row0 = df.iloc[0]
    for name_col in ["galaxy_name", "galaxy", "name"]:
        if name_col.lower() in colmap:
            actual = colmap[name_col.lower()]
            m = df[actual].astype(str).str.contains("M61", case=False, na=False)
            if m.any():
                row0 = df.loc[m].iloc[0]
                break

    inc_deg = to_float(row0[inc_col])
    pa_deg  = to_float(row0[pa_col])
    if not np.isfinite(inc_deg):
        raise RuntimeError(f"inc is not finite (from column '{inc_col}'): {row0[inc_col]}")
    if not np.isfinite(pa_deg):
        raise RuntimeError(f"pa is not finite (from column '{pa_col}'): {row0[pa_col]}")

    # Build per-row sightlines
    sightlines: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        sid = None
        for cand in ["sightline_id", "sightline", "qso", "qso_name", "target", "id"]:
            if cand.lower() in colmap:
                sid = str(row[colmap[cand.lower()]])
                break
        if sid is None:
            sid = f"sl_{idx:03d}"

        rho_kpc = to_float(row[rho_col])
        phi_deg = to_float(row[phi_col])

        if not (np.isfinite(rho_kpc) and np.isfinite(phi_deg)):
            continue

        sl = {"sightline_id": sid, "rho_kpc": rho_kpc, "phi_deg": phi_deg}
        if rvir_col is not None:
            rv = to_float(row[rvir_col])
            if np.isfinite(rv):
                sl["Rvir_kpc"] = rv
        sightlines.append(sl)

    if len(sightlines) == 0:
        raise RuntimeError(
            f"No valid sightlines found (finite rho/phi) using rho_col={rho_col}, phi_col={phi_col}."
        )

    obs: Dict[str, Any] = {
        "inc_deg": float(inc_deg),
        "pa_deg": float(pa_deg),
        "sightlines": sightlines,
    }

    # Optional global Rvir: prefer row0 if present and finite
    if rvir_col is not None:
        rv0 = to_float(row0[rvir_col])
        if np.isfinite(rv0):
            obs["Rvir_kpc"] = float(rv0)

    # Optional metadata passthrough
    for k in ["galaxy_name", "galaxy_id", "ra_deg", "dec_deg", "distance_mpc"]:
        if k.lower() in colmap:
            obs[k] = row0[colmap[k.lower()]]

    return obs

def load_top_matches(matches_csv: str, topn: int) -> pd.DataFrame:
    """
    Load matches CSV and return top-N rows.

    Requirements:
      - must contain a SubhaloID-like column

    If a distance/rank column exists, we sort by it; otherwise we keep file order.
    """
    df = pd.read_csv(matches_csv)

    sid_col = None
    for c in ["SubhaloID", "subhalo_id", "sid", "SubfindID"]:
        if c in df.columns:
            sid_col = c
            break
    if sid_col is None:
        sid_col = prefer(df, "SubhaloID")

    df[sid_col] = pd.to_numeric(df[sid_col], errors="coerce").astype("Int64")
    df = df[pd.notna(df[sid_col])].copy()
    df[sid_col] = df[sid_col].astype(int)

    sort_col = None
    for c in ["rank", "Rank", "distance", "dist", "d3d", "D3D", "delta", "score"]:
        if c in df.columns:
            sort_col = c
            break
    if sort_col is not None:
        df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce")
        df = df.sort_values(sort_col, ascending=True)

    df = df.head(int(topn)).reset_index(drop=True)
    df = df.rename(columns={sid_col: "SubhaloID"})
    return df


# ──────────────────────────────────────────────────────────────────────
# Core processing per subhalo
# ──────────────────────────────────────────────────────────────────────

def process_subhalo(
    sid: int,
    matches_meta: Dict[str, Any],
    obs: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process one SubhaloID:
      - load cutout particles
      - read groupcat record (pos, spin, etc.)
      - compute normal (PCA v3 or stellar spin)
      - build R bases with obs inc/PA
      - loop over alpha and (noflip/flip)
      - compute endpoints for each obs sightline
      - write outputs
    """
    sid = int(sid)
    snap = cfg["SNAP"]

    cutout = find_cutout_h5(sid, cfg["CUTOUT_ROOT"])
    if not cutout or (not os.path.isfile(cutout)):
        return {"SubhaloID": sid, "error": f"cutout missing under out_sub_{sid}", "cutout": cutout}

    sh = read_single_subhalo(cfg["GROUPCAT_BASE"], snap, sid)
    h = float(sh["h"])
    box_ckpch = float(sh["BoxSize"])
    center_ckpch = np.array(sh["SubhaloPos"], float)
    rhalf_star_ckpch_cat = float(sh["SubhaloHalfmassRadType"][4]) if len(sh["SubhaloHalfmassRadType"]) > 4 else np.nan
    J_tot = np.array(sh["SubhaloSpin"], float)
    grnr = int(sh.get("SubhaloGrNr", -1))

    stars = read_cutout_particles(cutout)
    Xs_ckpch = stars["Coordinates"]
    Ms_1e10 = stars["Masses"]
    Vs_kms = stars["Velocities"]

    Xs_rel_ckpch = recenter_positions(Xs_ckpch, center_ckpch, box_ckpch)

    if cfg["USE_RHALF_APERTURE"] and np.isfinite(rhalf_star_ckpch_cat) and rhalf_star_ckpch_cat > 0:
        R = np.linalg.norm(Xs_rel_ckpch, axis=1)
        sel = (R <= cfg["RHALF_MULTIPLIER"] * rhalf_star_ckpch_cat)
        if np.any(sel):
            Xs_rel_ckpch = Xs_rel_ckpch[sel]
            Ms_1e10 = Ms_1e10[sel]
            Vs_kms = Vs_kms[sel]

    # Determine Rvir fallback if not in obs
    Rvir_kpc_used = obs.get("Rvir_kpc", np.nan)
    if (not np.isfinite(Rvir_kpc_used)) and (grnr >= 0):
        R200_ckpch, _ = read_group_field(cfg["GROUPCAT_BASE"], snap, grnr, field="Group_R_Crit200")
        if np.isfinite(R200_ckpch):
            Rvir_kpc_used = float(R200_ckpch / h)  # ckpc/h -> kpc

    if not np.isfinite(Rvir_kpc_used) or (Rvir_kpc_used <= 0):
        Rvir_kpc_used = cfg["DEFAULT_RVIR_KPC"]

    inc_deg = float(obs["inc_deg"])
    pa_deg = float(obs["pa_deg"])
    if cfg["PA_FROM_NORTH"]:
        # astronomy PA: east of north; convert to angle from +x (east):
        # angle_from_x = 90 - PA_north
        pa_deg = (90.0 - pa_deg) % 360.0

    method = cfg["ORIENTATION_OVERRIDE"].get(sid, cfg["DEFAULT_METHOD"])

    if method == "stellar_spin":
        normal_vec = stellar_spin_vector(
            Xs_rel_ckpch, Vs_kms, Ms_1e10,
            remove_bulk_vel=cfg["REMOVE_BULK_VEL"]
        )
    else:
        normal_vec = build_normal_from_pca_v3(Xs_rel_ckpch, Ms_1e10, h)

    R_base_noflip, R_base_flip, n_hat, n_hat_flip = build_R_bases(normal_vec, inc_deg, pa_deg)

    debug_png = ""
    extra = {}
    if cfg["MAKE_FIGURES"]:
        sub_out_dir = os.path.join(cfg["OUT_BASE"], f"sid{sid}")
        an_dir = os.path.join(sub_out_dir, "analysis")
        debug_png, extra = make_debug_figure(
            sid=sid, snap=snap, out_dir=an_dir,
            Xs_rel_ckpch=Xs_rel_ckpch, Ms_1e10=Ms_1e10, Vs_kms=Vs_kms,
            center_ckpch=center_ckpch, rhalf_star_ckpch=rhalf_star_ckpch_cat,
            J_tot=J_tot, h=h,
            extent_kpc_h=cfg["FIG_EXTENT_KPC_H"],
            nbin=cfg["FIG_NBIN"],
            cmap_name=cfg["FIG_CMAP"],
            vmin=cfg["FIG_VMIN"],
            vmax=cfg["FIG_VMAX"],
            remove_bulk_vel=cfg["REMOVE_BULK_VEL"],
        )
    else:
        J_star = stellar_spin_vector(Xs_rel_ckpch, Vs_kms, Ms_1e10, remove_bulk_vel=cfg["REMOVE_BULK_VEL"])
        inc_t, PA_t = inc_PA_from_vector(J_tot)
        inc_s, PA_s = inc_PA_from_vector(J_star)
        _, ang2_xy = pca2_weighted((Xs_rel_ckpch / h)[:, [0, 1]], (Ms_1e10 * 1e10 / h))
        extra = {
            "inc_tot_deg": float(inc_t), "PA_tot_deg": float(PA_t),
            "inc_star_deg": float(inc_s), "PA_star_deg": float(PA_s),
            "PCA_major2D_XY_deg": float(ang2_xy),
        }

    sub_out_dir = os.path.join(cfg["OUT_BASE"], f"sid{sid}")
    an_dir = os.path.join(sub_out_dir, "analysis")
    ensure_dir(an_dir)

    orientation_json = os.path.join(an_dir, f"orientation_sid{sid}_snap{snap}.json")
    with open(orientation_json, "w") as f:
        json.dump({
            "sid": sid,
            "snap": snap,
            "h": h,
            "box_ckpc_h": box_ckpch,
            "center_ckpc_h": center_ckpch.tolist(),
            "rhalf_star_ckpc_h": float(rhalf_star_ckpch_cat) if np.isfinite(rhalf_star_ckpch_cat) else None,
            "J_tot": J_tot.tolist(),
            "normal_used_hat": unit(normal_vec).tolist(),
            "orientation_method": method,
            "obs_inc_deg": inc_deg,
            "obs_pa_deg_used": pa_deg,
            "PA_FROM_NORTH": bool(cfg["PA_FROM_NORTH"]),
            "inc_tot_deg": extra.get("inc_tot_deg"),
            "PA_tot_deg": extra.get("PA_tot_deg"),
            "inc_star_deg": extra.get("inc_star_deg"),
            "PA_star_deg": extra.get("PA_star_deg"),
            "PCA_major2D_XY_deg": extra.get("PCA_major2D_XY_deg"),
        }, f, indent=2)

    alpha_step = int(cfg["ALPHA_STEP"])
    alphas = list(range(0, 360, alpha_step))

    run_specs = [("L3Rvir", 1.5), ("L4Rvir", 2.0)]

    summary_row = {
        "SubhaloID": sid,
        "cutout": cutout,
        "orientation_method": method,
        "h": h,
        "center_x_ckpch": center_ckpch[0],
        "center_y_ckpch": center_ckpch[1],
        "center_z_ckpch": center_ckpch[2],
        "rhalf_star_ckpch": rhalf_star_ckpch_cat,
        "obs_inc_deg": inc_deg,
        "obs_pa_deg_raw": float(obs["pa_deg"]),
        "obs_pa_deg_used": pa_deg,
        "Rvir_kpc_used": float(Rvir_kpc_used),
        "debug_image": debug_png,
        "orientation_json": orientation_json,
        "inc_tot_deg": extra.get("inc_tot_deg", np.nan),
        "PA_tot_deg": extra.get("PA_tot_deg", np.nan),
        "inc_star_deg": extra.get("inc_star_deg", np.nan),
        "PA_star_deg": extra.get("PA_star_deg", np.nan),
        "PCA_major2D_XY_deg": extra.get("PCA_major2D_XY_deg", np.nan),
        "error": "",
    }

    for run_label, half_R in run_specs:
        out_run = os.path.join(sub_out_dir, f"rays_and_recipes_sid{sid}_snap{snap}_{run_label}")
        ensure_dir(out_run)

        rows_rays = []
        rows_orient = []

        half_len_ckpch = (half_R * Rvir_kpc_used) * h

        for alpha in alphas:
            for mode, R_base, axis in [
                ("noflip", R_base_noflip, n_hat),
                ("flip",   R_base_flip,   n_hat_flip),
            ]:
                S_alpha = rodrigues_axis_angle(axis, math.radians(alpha))
                R_cur = R_base @ S_alpha

                ey_obs = np.array([0.0, 1.0, 0.0])
                ez_obs = np.array([0.0, 0.0, 1.0])
                normal_nat = ez_obs @ R_cur
                north_nat = ey_obs @ R_cur

                rows_orient.append({
                    "alpha_deg": alpha,
                    "mode": mode,
                    "orientation_method": method,
                    "obs_inc_deg": inc_deg,
                    "obs_pa_deg_used": pa_deg,
                    "los_x": float(normal_nat[0]),
                    "los_y": float(normal_nat[1]),
                    "los_z": float(normal_nat[2]),
                    "north_x": float(north_nat[0]),
                    "north_y": float(north_nat[1]),
                    "north_z": float(north_nat[2]),
                })

                for sl in obs["sightlines"]:
                    rho_kpc = float(sl["rho_kpc"])
                    phi_deg = float(sl["phi_deg"])
                    rho_ckpch = rho_kpc * h

                    p0, p1, anchor, L_nat = sightline_endpoints_codeunits(
                        center_ckpch=center_ckpch,
                        R_cur=R_cur,
                        rho_ckpch=rho_ckpch,
                        phi_deg=phi_deg,
                        half_len_ckpch=half_len_ckpch
                    )

                    rows_rays.append({
                        "SubhaloID": sid,
                        "sightline_id": sl["sightline_id"],
                        "alpha_deg": alpha,
                        "mode": mode,
                        "orientation_method": method,
                        "obs_inc_deg": inc_deg,
                        "obs_pa_deg_used": pa_deg,
                        "rho_kpc": rho_kpc,
                        "phi_deg": phi_deg,
                        "Rvir_kpc": float(Rvir_kpc_used),
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

                        "los_x": float(L_nat[0]),
                        "los_y": float(L_nat[1]),
                        "los_z": float(L_nat[2]),
                    })

        rays_csv = os.path.join(out_run, f"rays_sid{sid}.csv")
        orient_csv = os.path.join(out_run, f"orient_peralpha_sid{sid}.csv")
        header_json = os.path.join(out_run, f"orient_header_sid{sid}.json")

        pd.DataFrame(rows_rays).to_csv(rays_csv, index=False)
        pd.DataFrame(rows_orient).to_csv(orient_csv, index=False)

        with open(header_json, "w") as f:
            json.dump({
                "SID": sid,
                "SNAP": snap,
                "RUN_LABEL": run_label,
                "half_len_Rvir": float(half_R),
                "total_len_Rvir": float(2.0 * half_R),
                "units_note": "all saved coordinates are ckpc/h (code_length) for TNG50-1 snap99",
                "h": float(h),
                "Rvir_kpc_used": float(Rvir_kpc_used),
                "obs_inc_deg": float(inc_deg),
                "obs_pa_deg_used": float(pa_deg),
                "PA_FROM_NORTH": bool(cfg["PA_FROM_NORTH"]),
                "orientation_method": method,
                "normal_used_hat": unit(normal_vec).tolist(),
                "sightlines": obs["sightlines"],
                "alpha_step_deg": alpha_step,
            }, f, indent=2)

    return summary_row


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="M61: orientations + LOS endpoints for top-N TNG matches on HPC."
    )
    p.add_argument("--topn", type=int, default=20)
    p.add_argument("--alpha-step", type=int, default=1)
    p.add_argument("--snap", type=int, default=99)

    p.add_argument("--out-base", required=True, help="Output base directory (scratch).")
    p.add_argument("--cutout-root", required=True, help="Cutout root containing out_sub_<SID>/")
    p.add_argument("--groupcat-base", required=True, help="Group catalog directory groups_099/ or chunk files directory.")

    p.add_argument("--obs-csv", required=True, help="Obs CSV with inc/PA and rho/phi per sightline (one row per sightline).")
    p.add_argument("--matches-csv", required=True, help="Matches CSV with SubhaloIDs ranked; topN used.")

    p.add_argument("--no-fig", action="store_true", help="Disable debug figures to reduce runtime.")
    p.add_argument("--remove-bulk-vel", action="store_true", help="Remove stellar bulk velocity for stellar spin.")
    p.add_argument("--no-rhalf-aperture", action="store_true", help="Disable rhalf aperture cut.")
    p.add_argument("--rhalf-multiplier", type=float, default=10.0)

    p.add_argument("--default-rvir-kpc", type=float, default=300.0)

    p.add_argument("--pa-from-north", action="store_true",
                   help="If PA in obs CSV is east-of-north, convert to angle from +x.")

    p.add_argument("--stellar-spin-override", default="",
                   help="Optional text file with one SubhaloID per line to use stellar spin.")
    return p.parse_args()

def load_override_list(path: str) -> Dict[int, str]:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Override file not found: {path}")
    out = {}
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                sid = int(s)
                out[sid] = "stellar_spin"
            except Exception:
                continue
    return out

def main():
    args = parse_args()

    obs = load_obs_geometry(args.obs_csv)
    matches = load_top_matches(args.matches_csv, args.topn)

    override = load_override_list(args.stellar_spin_override)

    cfg = {
        "SNAP": int(args.snap),
        "OUT_BASE": os.path.abspath(args.out_base),
        "CUTOUT_ROOT": os.path.abspath(args.cutout_root),
        "GROUPCAT_BASE": os.path.abspath(args.groupcat_base),

        "TOPN": int(args.topn),
        "ALPHA_STEP": int(args.alpha_step),

        "DEFAULT_METHOD": "pca_v3",
        "ORIENTATION_OVERRIDE": override,

        "MAKE_FIGURES": (not args.no_fig),
        "REMOVE_BULK_VEL": bool(args.remove_bulk_vel),

        "USE_RHALF_APERTURE": (not args.no_rhalf_aperture),
        "RHALF_MULTIPLIER": float(args.rhalf_multiplier),

        "DEFAULT_RVIR_KPC": float(args.default_rvir_kpc),

        "PA_FROM_NORTH": bool(args.pa_from_north),

        "FIG_EXTENT_KPC_H": 80.0,
        "FIG_NBIN": 1200,
        "FIG_CMAP": "jet",
        "FIG_VMIN": 4.0,
        "FIG_VMAX": 7.0,
    }

    ensure_dir(cfg["OUT_BASE"])

    print(f"[INFO] obs inc_deg={obs['inc_deg']:.3f}, pa_deg(raw)={obs['pa_deg']:.3f}, PA_FROM_NORTH={cfg['PA_FROM_NORTH']}")
    print(f"[INFO] sightlines parsed: {len(obs['sightlines'])}: {[s['sightline_id'] for s in obs['sightlines']]}")
    print(f"[INFO] matches loaded: {len(matches)} (topn={cfg['TOPN']})")
    print(f"[INFO] outputs: {cfg['OUT_BASE']}")

    summary_rows = []
    for i in range(len(matches)):
        sid = int(matches.loc[i, "SubhaloID"])
        meta = matches.loc[i].to_dict()

        try:
            method = cfg["ORIENTATION_OVERRIDE"].get(sid, cfg["DEFAULT_METHOD"])
            print(f"[{i+1}/{len(matches)}] sid={sid} method={method}")
            row = process_subhalo(sid=sid, matches_meta=meta, obs=obs, cfg=cfg)
            if row.get("error"):
                print(f"   ↳ WARN: {row['error']}")
            summary_rows.append({**meta, **row})
        except Exception as e:
            print(f"[ERROR] sid={sid}: {e}")
            traceback.print_exc()
            summary_rows.append({**meta, "SubhaloID": sid, "error": str(e)})

    summary_csv = os.path.join(cfg["OUT_BASE"], f"orientation_summary_snap{cfg['SNAP']}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print(f"[OK] wrote summary: {summary_csv}")
    print(f"[OK] per-SID outputs under: {cfg['OUT_BASE']}/sid<SID>/")

if __name__ == "__main__":
    main()