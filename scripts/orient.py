#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M61 → TNG50-1 snap99
Batch: orientations + LOS endpoints for TOP-N matched subhalos (HPC paths).

What this does (high-level)
---------------------------
1) Reads your M61 observation geometry from:
     data/M61_DIISC_Table1_Table2.csv
   Uses:
     - inclination_deg   (tilt)
     - theta_deg         (azimuth of QSO wrt major axis on sky)
     - impact_kpc        (rho)
     - Rvir_kpc          (ray length scaling; fallback to Group_R_Crit200 if missing)

2) Reads your matched simulated galaxies from:
     data/m61_closest_matches_3d.csv
   Takes top N rows (default 20) and extracts SubhaloID.

3) For each SubhaloID:
   - Loads groupcat info (center, SubhaloSpin, halfmass radius, h, box)
   - Loads the cutout HDF5 (stars: Coordinates/Masses/Velocities)
   - Builds an orientation:
       a) choose normal (default PCA-v3 of stellar distribution; optional override to stellar-spin)
       b) rotate to face-on
       c) apply the observed inclination
       d) rotate about LOS so that the *projected* stellar major axis lies along +x
          → after this, theta_deg can be used directly as phi_deg in the observer plane.
   - Computes LOS endpoints for the observed (rho, phi) and writes per-subhalo outputs.

Directory layout (reasonable for SOL scratch)
--------------------------------------------
Base output:
  /scratch/tsingh65/m61-tng/outputs/

Per subhalo:
  outputs/sub_<SID>/
    analysis/
      debug_sid<SID>_snap99.png
      orientation_sid<SID>_snap99.json
    rays_m61_sid<SID>_snap99_L3Rvir/
      rays_sid<SID>.csv
      orient_header_sid<SID>.json
    rays_m61_sid<SID>_snap99_L4Rvir/
      rays_sid<SID>.csv
      orient_header_sid<SID>.json

Global:
  outputs/m61_orientation_summary_snap99_top<N>.csv

HPC inputs assumed
-----------------
Cutouts:
  /scratch/tsingh65/TNG50-1_snap99/out_sub_<SID>/
    (any *.hdf5 containing /PartType4/{Coordinates,Masses,Velocities})

Group catalog:
  /scratch/tsingh65/TNG50-1_snap99/groups_099/
    fof_subhalo_tab_099.*.hdf5  (or groups_099.hdf5)

Run
---
From: /home/tsingh65/m61-tng/scripts
  python orient_m61.py --topn 20
"""

import os, re, glob, json, math, argparse, traceback
import numpy as np
import pandas as pd
import h5py

# matplotlib only for debug figure
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Defaults for SOL/HPC
# ----------------------------
SNAP = 99

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

M61_OBS_CSV   = os.path.join(DATA_DIR, "M61_DIISC_Table1_Table2.csv")
MATCHES_CSV   = os.path.join(DATA_DIR, "m61_closest_matches_3d.csv")

CUTOUT_ROOT   = "/scratch/tsingh65/TNG50-1_snap99"
GROUPCAT_BASE = "/scratch/tsingh65/TNG50-1_snap99/groups_099"

OUT_BASE      = "/scratch/tsingh65/m61-tng/outputs"

# Orientation override (optional): force some SIDs to use stellar spin as the normal
ORIENTATION_OVERRIDE = {
    # 143885: "stellar_spin",
}
DEFAULT_METHOD = "pca_v3"

# Debug/figure knobs (keep lightweight for HPC)
extent_kpc_h       = 80.0
nbin               = 600
dpi_out            = 120
cmap_name          = "jet"
VMIN, VMAX         = 4.0, 7.0

REMOVE_BULK_VEL    = False
use_rhalf_aperture = True
rhalf_multiplier   = 10.0


# ----------------------------
# Math helpers
# ----------------------------
def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def minimal_image_delta(dpos, box):
    return (dpos + 0.5 * box) % box - 0.5 * box

def recenter_positions(x, center, box):
    return minimal_image_delta(x - center[None, :], box)

def rot_x(theta_deg):
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], float)

def rot_z(theta_deg):
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], float)

def rodrigues_axis_angle(axis, theta_rad):
    a = unit(axis)
    ax, ay, az = a
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    C = 1.0 - c
    return np.array([
        [c + ax*ax*C,     ax*ay*C - az*s, ax*az*C + ay*s],
        [ay*ax*C + az*s,  c + ay*ay*C,    ay*az*C - ax*s],
        [az*ax*C - ay*s,  az*ay*C + ax*s, c + az*az*C]
    ], float)

def R_from_u_to_v(u, v):
    u, v = unit(u), unit(v)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        a = unit(np.cross(u, [1, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1, 0]))
        return rodrigues_axis_angle(a, math.pi)
    axis = unit(np.cross(u, v))
    return rodrigues_axis_angle(axis, math.acos(c))

def pca3_weighted(X, w):
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
    w  = np.asarray(w, float)
    wsum = np.sum(w)
    xc = np.sum(XY * w[:, None], axis=0) / max(wsum, 1e-30)
    X0 = XY - xc
    C = (X0 * w[:, None]).T @ X0 / max(wsum, 1e-30)
    evals, evecs = np.linalg.eigh(C)
    maj = unit(evecs[:, np.argmax(evals)])
    ang = (np.degrees(np.arctan2(maj[1], maj[0])) % 180.0)
    return maj, ang

def stellar_spin_vector(X_rel_ckpch, V_kms, M_1e10):
    if REMOVE_BULK_VEL:
        vcm = np.sum(V_kms * M_1e10[:, None], axis=0) / np.sum(M_1e10)
        V_rel = V_kms - vcm[None, :]
    else:
        V_rel = V_kms
    return np.sum(np.cross(X_rel_ckpch, V_rel) * M_1e10[:, None], axis=0)

def build_R_base_from_normal(normal_vec, inc_deg):
    """
    Returns R_base mapping observer -> native:
      - face-on: normal -> +z_obs
      - tilt by inc about +x_obs
    """
    n_hat = unit(np.asarray(normal_vec, float))
    if np.linalg.norm(n_hat) == 0:
        n_hat = np.array([0, 0, 1.0])
    R_face = R_from_u_to_v(n_hat, np.array([0, 0, 1.0]))
    R_base = rot_x(inc_deg) @ R_face
    return R_base, n_hat

def sightline_endpoints_codeunits(center_ckpch, R_cur, rho_ckpch, phi_deg, half_len_ckpch):
    """
    Observer-plane location: (rho,phi) in XY_obs, LOS along +Z_obs.
    R_cur maps (row-vector) observer -> native:
      r_nat = r_obs @ R_cur
      L_nat = ez_obs @ R_cur
    """
    x = rho_ckpch * math.cos(math.radians(phi_deg))
    y = rho_ckpch * math.sin(math.radians(phi_deg))
    r_obs = np.array([x, y, 0.0], float)
    ez_obs = np.array([0.0, 0.0, 1.0], float)

    r_nat = r_obs @ R_cur
    L_nat = unit(ez_obs @ R_cur)

    p0 = center_ckpch + r_nat - half_len_ckpch * L_nat
    p1 = center_ckpch + r_nat + half_len_ckpch * L_nat
    anchor = center_ckpch + r_nat
    return p0, p1, anchor, L_nat


# ----------------------------
# Groupcat readers (no illustris_python)
# ----------------------------
def _groupcat_dir(base, snap):
    d = os.path.join(base, f"groups_{snap:03d}")
    return d if os.path.isdir(d) else base

def _chunk_path(base, snap, filenum):
    d = _groupcat_dir(base, snap)
    return os.path.join(d, f"fof_subhalo_tab_{snap:03d}.{filenum}.hdf5")

def _find_chunks(base, snap):
    d = _groupcat_dir(base, snap)
    files = sorted(
        glob.glob(os.path.join(d, f"fof_subhalo_tab_{snap:03d}.*.hdf5")),
        key=lambda p: int(os.path.splitext(p)[0].split(".")[-1]) if re.search(r"\.\d+\.hdf5$", p) else -1
    )
    alt = os.path.join(base, f"groups_{snap:03d}.hdf5")
    if not files and os.path.exists(alt):
        files = [alt]
    return files

def _read_header_any(base, snap):
    p0 = _chunk_path(base, snap, 0)
    candidates = [p0] if os.path.exists(p0) else _find_chunks(base, snap)
    if not candidates:
        raise FileNotFoundError(f"No group catalog files found under {base} for snap {snap}.")
    for ff in candidates:
        try:
            with h5py.File(ff, "r") as f:
                h  = float(f["Header"].attrs["HubbleParam"])
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
    ff = _find_chunks(base, snap)[0]
    with h5py.File(ff, "r") as f:
        h  = float(f["Header"].attrs["HubbleParam"])
        bs = float(f["Header"].attrs["BoxSize"])
    return h, bs, None, None

def _is_valid_chunk_with(dataset_name, path):
    try:
        with h5py.File(path, "r") as f:
            if dataset_name not in f:
                return False
            obj = f[dataset_name]
            try:
                _ = len(obj)
                return True
            except TypeError:
                return True
    except Exception:
        return False

def read_single_subhalo(base, snap, subhalo_id):
    subhalo_id = int(subhalo_id)
    h, boxsize, offsets, _ = _read_header_any(base, snap)

    if offsets is not None:
        rel = subhalo_id - offsets
        fileNum = int(np.max(np.where(rel >= 0)))
        local_index = int(rel[fileNum])
        candidate = _chunk_path(base, snap, fileNum)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Missing expected chunk {fileNum}: {candidate}")
        if not _is_valid_chunk_with("Subhalo", candidate):
            raise RuntimeError(f"{os.path.basename(candidate)} lacks Subhalo datasets.")
        with h5py.File(candidate, "r") as f:
            sub = f["Subhalo"]
            pos  = np.asarray(sub["SubhaloPos"][local_index], dtype=np.float64)
            hmr  = np.asarray(sub["SubhaloHalfmassRadType"][local_index], dtype=np.float64)
            spin = np.asarray(sub["SubhaloSpin"][local_index], dtype=np.float64)
            grnr = int(sub["SubhaloGrNr"][local_index]) if "SubhaloGrNr" in sub else -1
        return {"h": h, "BoxSize": boxsize,
                "SubhaloPos": pos, "SubhaloHalfmassRadType": hmr,
                "SubhaloSpin": spin, "SubhaloGrNr": grnr}

    files = _find_chunks(base, snap)
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
            pos  = np.asarray(sub["SubhaloPos"][remaining], dtype=np.float64)
            hmr  = np.asarray(sub["SubhaloHalfmassRadType"][remaining], dtype=np.float64)
            spin = np.asarray(sub["SubhaloSpin"][remaining], dtype=np.float64)
            grnr = int(sub["SubhaloGrNr"][remaining]) if "SubhaloGrNr" in sub else -1
        return {"h": h, "BoxSize": boxsize,
                "SubhaloPos": pos, "SubhaloHalfmassRadType": hmr,
                "SubhaloSpin": spin, "SubhaloGrNr": grnr}
    raise IndexError(f"SubhaloID {subhalo_id} not found.")

def read_group_field(base, snap, group_index, field="Group_R_Crit200"):
    group_index = int(group_index)
    h, _, _, offs_grp = _read_header_any(base, snap)
    if offs_grp is not None:
        rel = group_index - offs_grp
        fileNum = int(np.max(np.where(rel >= 0)))
        local_index = int(rel[fileNum])
        candidate = _chunk_path(base, snap, fileNum)
        if not os.path.exists(candidate) or not _is_valid_chunk_with("Group", candidate):
            return np.nan, h
        with h5py.File(candidate, "r") as f:
            grp = f["Group"]
            if field not in grp:
                return np.nan, h
            try:
                return float(grp[field][local_index]), h
            except Exception:
                return np.nan, h

    for ff in _find_chunks(base, snap):
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


# ----------------------------
# Cutout I/O
# ----------------------------
def find_cutout_h5(sid):
    """
    Your scratch layout:
      /scratch/tsingh65/TNG50-1_snap99/out_sub_<SID>/
    We accept any *.hdf5 that contains /PartType4/{Coordinates,Masses,Velocities}.
    """
    sub_dir = os.path.join(CUTOUT_ROOT, f"out_sub_{int(sid)}")
    if not os.path.isdir(sub_dir):
        return None

    candidates = sorted(glob.glob(os.path.join(sub_dir, "*.hdf5")))
    for h5 in candidates:
        try:
            with h5py.File(h5, "r") as f:
                ok = ("/PartType4/Coordinates" in f and "/PartType4/Masses" in f and "/PartType4/Velocities" in f)
            if ok:
                return h5
        except Exception:
            continue
    return None

def read_cutout_stars(h5_path):
    with h5py.File(h5_path, "r") as f:
        X = np.asarray(f["/PartType4/Coordinates"][...], dtype=np.float64)
        M = np.asarray(f["/PartType4/Masses"][...],      dtype=np.float64)
        V = np.asarray(f["/PartType4/Velocities"][...],  dtype=np.float64)
    return X, M, V


# ----------------------------
# Debug figure
# ----------------------------
def mass_map_arbitrary(M_weights, XY_kpc, half_width_kpc, nbin):
    L = half_width_kpc
    x = np.clip(XY_kpc[:, 0], -L, L)
    y = np.clip(XY_kpc[:, 1], -L, L)
    H, xe, ye = np.histogram2d(x, y, bins=nbin, range=[[-L, L], [-L, L]], weights=M_weights)
    return H.T, xe, ye

def make_debug_figure(sid, out_png, X_obs_kpc, M_msun):
    L = (extent_kpc_h / 2.0)
    H, _, _ = mass_map_arbitrary(M_msun, X_obs_kpc[:, [0, 1]], L, nbin)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(np.log10(H + 1e-12), origin="lower",
                   extent=[-L, L, -L, L], cmap=cmap_name, vmin=VMIN, vmax=VMAX)
    ax.set_title(f"sid={sid} | observer XY (major axis aligned to +x)")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"$\log_{10}$ proj mass [arb]")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi_out)
    plt.close(fig)


# ----------------------------
# Loaders for M61 + matches
# ----------------------------
def load_m61_obs(path):
    df = pd.read_csv(path)

    # expected columns in your attached file:
    # inclination_deg, theta_deg, impact_kpc, Rvir_kpc (plus PA_deg present but not required here)
    need = ["inclination_deg", "theta_deg", "impact_kpc"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"M61 obs file missing column: {c}")

    # enforce numeric
    for c in ["inclination_deg", "theta_deg", "impact_kpc", "Rvir_kpc", "PA_deg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def load_topn_matches(path, topn=20, sort_col="dist3d"):
    df = pd.read_csv(path)
    if "SubhaloID" not in df.columns:
        raise KeyError("matches file must have SubhaloID column")
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=True)
    return df.head(int(topn)).reset_index(drop=True)


# ----------------------------
# Core processing
# ----------------------------
def process_one_sid_m61(sid, m61_rows, topmatch_row, alpha_list):
    sid = int(sid)

    cutout = find_cutout_h5(sid)
    if not cutout:
        return {"SubhaloID": sid, "error": f"cutout missing under {CUTOUT_ROOT}/out_sub_{sid}/"}

    sh = read_single_subhalo(GROUPCAT_BASE, SNAP, sid)
    h            = float(sh["h"])
    box_ckpch    = float(sh["BoxSize"])
    center_ckpch = np.array(sh["SubhaloPos"], float)
    rhalf_star_ckpch = float(sh["SubhaloHalfmassRadType"][4]) if len(sh["SubhaloHalfmassRadType"]) > 4 else np.nan
    J_tot        = np.array(sh["SubhaloSpin"], float)
    grnr         = int(sh.get("SubhaloGrNr", -1))

    X_ckpch, M_1e10, V_kms = read_cutout_stars(cutout)
    X_rel_ckpch = recenter_positions(X_ckpch, center_ckpch, box_ckpch)

    if use_rhalf_aperture and np.isfinite(rhalf_star_ckpch) and rhalf_star_ckpch > 0:
        R = np.linalg.norm(X_rel_ckpch, axis=1)
        sel = (R <= rhalf_multiplier * rhalf_star_ckpch)
        if np.any(sel):
            X_rel_ckpch = X_rel_ckpch[sel]
            M_1e10      = M_1e10[sel]
            V_kms       = V_kms[sel]

    # Choose normal method
    method = ORIENTATION_OVERRIDE.get(sid, DEFAULT_METHOD)
    if method == "stellar_spin":
        nvec = stellar_spin_vector(X_rel_ckpch, V_kms, M_1e10)
    else:
        # PCA-v3 in *physical* coordinates for stability
        X_kpc  = X_rel_ckpch / h
        M_msun = M_1e10 * 1e10 / h
        _, evecs3, _ = pca3_weighted(X_kpc, M_msun)
        nvec = evecs3[:, 2]  # v3

    # Use M61 obs geometry (if multiple rows, we will compute rays for each row)
    rows_written = 0
    per_obs_summaries = []

    # Per-subhalo base output directory
    sub_out = os.path.join(OUT_BASE, f"sub_{sid}")
    an_dir  = os.path.join(sub_out, "analysis")
    os.makedirs(an_dir, exist_ok=True)

    for obs_i, obs in m61_rows.iterrows():
        inc_deg = float(obs["inclination_deg"]) if np.isfinite(obs["inclination_deg"]) else np.nan
        theta_deg = float(obs["theta_deg"]) if np.isfinite(obs["theta_deg"]) else 0.0
        rho_kpc = float(obs["impact_kpc"]) if np.isfinite(obs["impact_kpc"]) else 0.0
        Rvir_kpc = float(obs["Rvir_kpc"]) if ("Rvir_kpc" in obs and np.isfinite(obs["Rvir_kpc"])) else np.nan

        # fallback Rvir from groupcat if needed
        if (not np.isfinite(Rvir_kpc)) and (grnr >= 0):
            R200_ckpch, _ = read_group_field(GROUPCAT_BASE, SNAP, grnr, field="Group_R_Crit200")
            if np.isfinite(R200_ckpch):
                Rvir_kpc = float(R200_ckpch / h)
        if not np.isfinite(Rvir_kpc) or Rvir_kpc <= 0:
            Rvir_kpc = 300.0

        # Base rotation: normal -> +z, then tilt by inc
        R_base, n_hat = build_R_base_from_normal(nvec, inc_deg)

        # Align projected stellar major axis with +x:
        #   X_obs = X_native @ R_base.T   (since R_base maps obs->native)
        X_kpc  = X_rel_ckpch / h
        M_msun = M_1e10 * 1e10 / h
        X_obs0 = X_kpc @ R_base.T

        _, ang_major = pca2_weighted(X_obs0[:, [0, 1]], M_msun)
        # rotate observer frame by -ang_major so major axis lies along +x
        R_align = rot_z(-ang_major)
        R_cur0  = R_align @ R_base

        # Debug image (once per obs row; cheap)
        debug_png = os.path.join(an_dir, f"debug_sid{sid}_snap{SNAP}_obs{obs_i:02d}.png")
        X_obs = X_kpc @ R_cur0.T
        make_debug_figure(sid, debug_png, X_obs, M_msun)

        # Now use theta_deg directly as phi_deg in the aligned observer frame.
        phi_deg = theta_deg
        rho_ckpch = rho_kpc * h

        # Write a small orientation json for this obs row
        orient_json = os.path.join(an_dir, f"orientation_sid{sid}_snap{SNAP}_obs{obs_i:02d}.json")
        with open(orient_json, "w") as f:
            json.dump({
                "sid": sid, "snap": SNAP, "h": h,
                "cutout": cutout,
                "center_ckpc_h": center_ckpch.tolist(),
                "orientation_method": method,
                "normal_used_hat": unit(n_hat).tolist(),
                "inc_deg_used": inc_deg,
                "major_axis_angle_before_align_deg": float(ang_major),
                "align_applied_deg": float(-ang_major),
                "m61_obs": {
                    "row_index": int(obs_i),
                    "rho_kpc": rho_kpc,
                    "theta_deg": theta_deg,
                    "phi_deg_used": phi_deg,
                    "Rvir_kpc_used": Rvir_kpc,
                },
                "topmatch": {k: (None if pd.isna(topmatch_row.get(k)) else float(topmatch_row.get(k)) if isinstance(topmatch_row.get(k), (int, float, np.number)) else str(topmatch_row.get(k)))
                             for k in topmatch_row.index if k != "SubhaloID"},
                "debug_png": debug_png
            }, f, indent=2)

        # Rays for L3/L4 with chosen alpha_list (default: [0])
        run_specs = [("L3Rvir", 1.5), ("L4Rvir", 2.0)]

        for RUN_LABEL, HALF_R in run_specs:
            out_run = os.path.join(sub_out, f"rays_m61_sid{sid}_snap{SNAP}_{RUN_LABEL}_obs{obs_i:02d}")
            os.makedirs(out_run, exist_ok=True)

            rows = []
            half_len_ckpch = (HALF_R * Rvir_kpc) * h

            # We rotate about the *native* normal axis to generate alpha variants
            axis_native = unit(n_hat)

            for alpha in alpha_list:
                S_alpha = rodrigues_axis_angle(axis_native, math.radians(float(alpha)))
                R_cur = R_cur0 @ S_alpha  # keep major-axis alignment at alpha=0, then spin about normal

                p0, p1, anchor, L_nat = sightline_endpoints_codeunits(
                    center_ckpch, R_cur, rho_ckpch, phi_deg, half_len_ckpch
                )

                rows.append({
                    "SubhaloID": sid,
                    "obs_row": int(obs_i),
                    "alpha_deg": float(alpha),
                    "orientation_method": method,
                    "inc_deg": float(inc_deg),
                    "rho_kpc": float(rho_kpc),
                    "theta_deg": float(theta_deg),
                    "phi_deg_used": float(phi_deg),
                    "Rvir_kpc": float(Rvir_kpc),
                    "half_len_Rvir": float(HALF_R),
                    "total_len_Rvir": float(2.0 * HALF_R),
                    "p0_X_ckpch_abs": float(p0[0]), "p0_Y_ckpch_abs": float(p0[1]), "p0_Z_ckpch_abs": float(p0[2]),
                    "p1_X_ckpch_abs": float(p1[0]), "p1_Y_ckpch_abs": float(p1[1]), "p1_Z_ckpch_abs": float(p1[2]),
                    "anchor_X_ckpch_abs": float(anchor[0]), "anchor_Y_ckpch_abs": float(anchor[1]), "anchor_Z_ckpch_abs": float(anchor[2]),
                    "los_x": float(L_nat[0]), "los_y": float(L_nat[1]), "los_z": float(L_nat[2]),
                    "cutout": cutout,
                    "debug_png": debug_png,
                    "orientation_json": orient_json,
                })

            rays_csv = os.path.join(out_run, f"rays_sid{sid}.csv")
            pd.DataFrame(rows).to_csv(rays_csv, index=False)

            header_json = os.path.join(out_run, f"orient_header_sid{sid}.json")
            with open(header_json, "w") as f:
                json.dump({
                    "SID": sid, "SNAP": SNAP, "RUN_LABEL": RUN_LABEL,
                    "obs_row": int(obs_i),
                    "hints": "all coordinates are ckpc/h (code_length); multiply by 1/h for kpc at z~0",
                    "orientation_method": method,
                    "normal_used_hat": unit(n_hat).tolist(),
                    "inc_deg": float(inc_deg),
                    "major_axis_align_deg": float(-ang_major),
                    "rho_kpc": float(rho_kpc),
                    "theta_deg": float(theta_deg),
                    "phi_deg_used": float(phi_deg),
                    "Rvir_kpc": float(Rvir_kpc),
                    "alpha_list": [float(a) for a in alpha_list],
                }, f, indent=2)

        rows_written += 1
        per_obs_summaries.append({
            "obs_row": int(obs_i),
            "inc_deg": float(inc_deg),
            "rho_kpc": float(rho_kpc),
            "theta_deg": float(theta_deg),
            "phi_deg_used": float(phi_deg),
            "Rvir_kpc": float(Rvir_kpc),
            "debug_png": debug_png,
            "orientation_json": orient_json,
        })

    return {
        "SubhaloID": sid,
        "cutout": cutout,
        "method": method,
        "n_obs_rows": int(rows_written),
        "per_obs": json.dumps(per_obs_summaries),
        "error": ""
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--alpha-step", type=int, default=1,
                    help="alpha grid step in degrees. Use 1 for 0..359, use 360 for only alpha=0.")
    ap.add_argument("--out-base", type=str, default=OUT_BASE)
    ap.add_argument("--cutout-root", type=str, default=CUTOUT_ROOT)
    ap.add_argument("--groupcat-base", type=str, default=GROUPCAT_BASE)
    ap.add_argument("--obs-csv", type=str, default=M61_OBS_CSV)
    ap.add_argument("--matches-csv", type=str, default=MATCHES_CSV)
    args = ap.parse_args()

    global OUT_BASE, CUTOUT_ROOT, GROUPCAT_BASE
    OUT_BASE      = args.out_base
    CUTOUT_ROOT   = args.cutout_root
    GROUPCAT_BASE = args.groupcat_base

    os.makedirs(OUT_BASE, exist_ok=True)

    if not os.path.isfile(args.obs_csv):
        raise FileNotFoundError(f"M61 obs CSV not found: {args.obs_csv}")
    if not os.path.isfile(args.matches_csv):
        raise FileNotFoundError(f"matches CSV not found: {args.matches_csv}")

    m61_obs = load_m61_obs(args.obs_csv)
    matches = load_topn_matches(args.matches_csv, topn=args.topn)

    alpha_step = int(args.alpha_step)
    if alpha_step <= 0:
        alpha_step = 1
    alpha_list = list(range(0, 360, alpha_step))
    if alpha_step >= 360:
        alpha_list = [0]

    summary_rows = []
    for i, r in matches.iterrows():
        sid = int(r["SubhaloID"])
        try:
            print(f"[{i+1}/{len(matches)}] sid={sid}")
            row = process_one_sid_m61(sid, m61_obs, r, alpha_list)
            if row.get("error"):
                print(f"   WARN: {row['error']}")
            summary_rows.append(row)
        except Exception as e:
            print(f"[ERROR] sid={sid}: {e}")
            traceback.print_exc()
            summary_rows.append({"SubhaloID": sid, "error": str(e)})

    summary = pd.DataFrame(summary_rows)
    out_summary = os.path.join(OUT_BASE, f"m61_orientation_summary_snap{SNAP}_top{args.topn}.csv")
    summary.to_csv(out_summary, index=False)
    print(f"[OK] wrote summary: {out_summary}")


if __name__ == "__main__":
    main()