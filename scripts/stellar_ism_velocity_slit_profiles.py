#!/usr/bin/env python3
"""
Measure the stellar disk/ISM proxy velocity along the QSO radial sightline.

For each saved alpha/mode recipe row, this script:
  - reads the saved LOS and north vectors from the LxRvir recipe folder;
  - projects PartType4 stellar particles into that exact sky frame;
  - subtracts the catalog SubhaloVel vector before taking v_LOS;
  - samples a finite-width signed radial strip through the galaxy center along
    the fixed QSO sky-plane direction; and
  - writes chunked HDF5 per-bin sufficient statistics, profiles, and the value
    in the bin containing the sightline impact parameter.

The script intentionally uses the saved recipe/orientation products rather than
recomputing alpha geometry on the fly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py

os.environ.setdefault("MPLCONFIGDIR", "/tmp/m61_matplotlib_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("/scratch/tsingh65/m61-tng/outputs")
DEFAULT_CUTOUT_ROOT = Path("/data/sborthak/m61/cutouts")
DEFAULT_SIGHTLINE_ID = "J122138+043026"
H5_STRING = h5py.string_dtype(encoding="utf-8")


@dataclass(frozen=True)
class CatalogContext:
    center_ckpch: np.ndarray
    subhalo_vel_kms: np.ndarray
    h: float
    scale_factor: float
    redshift: float
    box_ckpch: float
    center_source: str
    velocity_source: str


def unit(vec: Iterable[float], name: str = "vector") -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Cannot normalize {name}: {arr}")
    return arr / norm


def minimal_delta(pos: np.ndarray, center: np.ndarray, box: float) -> np.ndarray:
    return (pos - center[None, :] + 0.5 * box) % box - 0.5 * box


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def parse_int_list(text: str | None) -> list[int] | None:
    if text is None or str(text).strip().lower() == "all":
        return None
    values: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = [int(x) for x in part.split("-", 1)]
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
        else:
            values.append(int(part))
    return sorted(set(values))


def parse_modes(text: str) -> list[str]:
    modes = [m.strip() for m in text.split(",") if m.strip()]
    bad = sorted(set(modes) - {"noflip", "flip"})
    if bad:
        raise ValueError(f"Unsupported mode(s): {bad}; expected noflip and/or flip")
    return modes


def sid_recipe_dir(output_root: Path, sid: int, snap: int, run_label: str) -> Path:
    return output_root / f"sid{sid}" / f"rays_and_recipes_sid{sid}_snap{snap}_{run_label}"


def default_cutout_path(cutout_root: Path, sid: int) -> Path:
    return cutout_root / f"out_sub_{sid}" / f"cutout_ALLFIELDS_sphere_2p1Rvir_sub{sid}.hdf5"


def load_catalog_context(output_root: Path, cutout_path: Path, sid: int, snap: int) -> CatalogContext:
    analysis_dir = output_root / f"sid{sid}" / "analysis"
    orient_path = analysis_dir / f"orientation_sid{sid}_snap{snap}.json"
    subhalo_path = analysis_dir / f"subhalo_catalog_sid{sid}_snap{snap}.json"

    center = None
    center_source = ""
    if orient_path.exists():
        orient = read_json(orient_path)
        if "center_ckpc_h" in orient:
            center = np.asarray(orient["center_ckpc_h"], dtype=float)
            center_source = str(orient_path)

    velocity = None
    velocity_source = ""
    redshift = 0.0
    scale_factor = 1.0
    if subhalo_path.exists():
        subhalo = read_json(subhalo_path)
        if "subhalo_vel_kms" in subhalo:
            velocity = np.asarray(subhalo["subhalo_vel_kms"], dtype=float)
            velocity_source = str(subhalo_path)
        if center is None and "subhalo_pos_ckpch" in subhalo:
            center = np.asarray(subhalo["subhalo_pos_ckpch"], dtype=float)
            center_source = str(subhalo_path)
        redshift = float(subhalo.get("redshift", 0.0))
        scale_factor = float(subhalo.get("scale_factor", 1.0))

    with h5py.File(cutout_path, "r") as f:
        header = f["Header"].attrs
        h = float(header.get("HubbleParam", 0.6774))
        box = float(header.get("BoxSize", 35000.0))
        scale_factor = float(header.get("Time", scale_factor))
        redshift = float(header.get("Redshift", redshift))
        group = f.get(str(sid))
        if group is not None:
            if center is None and "pos" in group.attrs:
                center = np.asarray(group.attrs["pos"], dtype=float)
                center_source = f"{cutout_path}/{sid}.attrs['pos']"
            if velocity is None and "vel" in group.attrs:
                velocity = np.asarray(group.attrs["vel"], dtype=float)
                velocity_source = f"{cutout_path}/{sid}.attrs['vel']"
        if center is None and "CutoutInfo" in f and "Selection" in f["CutoutInfo"].attrs:
            selection = json.loads(f["CutoutInfo"].attrs["Selection"])
            if "center_ckpc_h" in selection:
                center = np.asarray(selection["center_ckpc_h"], dtype=float)
                center_source = f"{cutout_path}/CutoutInfo.attrs['Selection']['center_ckpc_h']"

    if center is None:
        raise RuntimeError(
            f"Could not determine true center for SID {sid}. Expected {orient_path}, "
            f"{subhalo_path}, cutout /{sid}.attrs['pos'], or CutoutInfo Selection center."
        )
    if velocity is None:
        raise RuntimeError(
            f"Could not determine SubhaloVel for SID {sid}. Expected {subhalo_path} "
            f"or cutout /{sid}.attrs['vel']."
        )
    if center.shape != (3,) or velocity.shape != (3,):
        raise RuntimeError(f"Bad catalog vector shape for SID {sid}: center={center.shape}, vel={velocity.shape}")

    return CatalogContext(
        center_ckpch=center,
        subhalo_vel_kms=velocity,
        h=h,
        scale_factor=scale_factor,
        redshift=redshift,
        box_ckpch=box,
        center_source=center_source,
        velocity_source=velocity_source,
    )


def load_recipe_tables(recipe_dir: Path, sid: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rays_path = recipe_dir / f"rays_sid{sid}.csv"
    orient_path = recipe_dir / f"orient_peralpha_sid{sid}.csv"
    header_path = recipe_dir / f"orient_header_sid{sid}.json"
    missing = [str(p) for p in (rays_path, orient_path, header_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing recipe product(s):\n" + "\n".join(missing))
    return pd.read_csv(rays_path), pd.read_csv(orient_path), read_json(header_path)


def qso_xy_from_header(header: dict, sightline_id: str, fallback_row: pd.Series) -> tuple[float, float, str]:
    for item in header.get("sightlines", []):
        if str(item.get("sightline_id")) == str(sightline_id):
            if "x_qso_major_kpc" in item and "y_qso_minor_kpc" in item:
                return float(item["x_qso_major_kpc"]), float(item["y_qso_minor_kpc"]), "orient_header sightlines x_qso_major/y_qso_minor"
    rho = float(fallback_row["rho_kpc"])
    phi = math.radians(float(fallback_row["phi_deg"]))
    return rho * math.cos(phi), rho * math.sin(phi), "ray row rho_kpc/phi_deg"


def saved_basis(row: pd.Series, orient_row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = np.array([row["p0_X_ckpch_abs"], row["p0_Y_ckpch_abs"], row["p0_Z_ckpch_abs"]], dtype=float)
    p1 = np.array([row["p1_X_ckpch_abs"], row["p1_Y_ckpch_abs"], row["p1_Z_ckpch_abs"]], dtype=float)
    los_from_endpoints = unit(p1 - p0, "p1-p0")
    los_saved = unit([row["los_x"], row["los_y"], row["los_z"]], "saved los")
    if np.dot(los_from_endpoints, los_saved) < 0.999999:
        raise RuntimeError(f"Saved LOS does not match p1-p0: dot={np.dot(los_from_endpoints, los_saved):.8f}")
    yhat = unit([orient_row["north_x"], orient_row["north_y"], orient_row["north_z"]], "saved north")
    yhat = unit(yhat - np.dot(yhat, los_saved) * los_saved, "north perpendicular to LOS")
    xhat = unit(np.cross(yhat, los_saved), "x = north cross LOS")
    return xhat, yhat, los_saved


def select_recipe_row(
    rays: pd.DataFrame,
    orient: pd.DataFrame,
    sid: int,
    sightline_id: str,
    alpha: int,
    mode: str,
) -> tuple[pd.Series, pd.Series]:
    mask = (
        (rays["SubhaloID"].astype(int) == int(sid))
        & (rays["sightline_id"].astype(str) == str(sightline_id))
        & (rays["mode"].astype(str) == str(mode))
        & (rays["alpha_deg"].astype(float).round().astype(int) == int(alpha))
    )
    rows = rays.loc[mask]
    if rows.empty:
        raise RuntimeError(f"No ray row found for SID={sid}, sightline={sightline_id}, alpha={alpha}, mode={mode}")
    omask = (
        (orient["mode"].astype(str) == str(mode))
        & (orient["alpha_deg"].astype(float).round().astype(int) == int(alpha))
    )
    orows = orient.loc[omask]
    if orows.empty:
        raise RuntimeError(f"No orient_peralpha row found for SID={sid}, alpha={alpha}, mode={mode}")
    return rows.iloc[0], orows.iloc[0]


def load_stellar_particles(cutout_path: Path, context: CatalogContext) -> dict[str, np.ndarray]:
    with h5py.File(cutout_path, "r") as f:
        pos = np.asarray(f["PartType4/Coordinates"], dtype=np.float64)
        vel = np.asarray(f["PartType4/Velocities"], dtype=np.float64)
        mass = np.asarray(f["PartType4/Masses"], dtype=np.float64)
        if "PartType4/GFM_StellarFormationTime" in f:
            sft = np.asarray(f["PartType4/GFM_StellarFormationTime"], dtype=np.float64)
            keep = sft > 0.0
        else:
            keep = np.ones(pos.shape[0], dtype=bool)

    rel_ckpch = minimal_delta(pos[keep], context.center_ckpch, context.box_ckpch)
    mass_msun = mass[keep] * 1.0e10 / context.h
    # TNG snapshot velocities have units km*sqrt(a)/s. Multiplying by sqrt(a)
    # gives peculiar velocity in km/s. At snap99 a=1, but applying this keeps
    # the script portable.
    velocity_peculiar_kms = vel[keep] * math.sqrt(float(context.scale_factor))
    vrel_kms = velocity_peculiar_kms - context.subhalo_vel_kms[None, :]
    return {
        "rel_ckpch": rel_ckpch,
        "mass_msun": mass_msun,
        "vrel_kms": vrel_kms,
        "kept_count": np.array([keep.sum()]),
        "velocity_scale_factor_sqrt_a": np.array([math.sqrt(float(context.scale_factor))], dtype=float),
    }


def radial_strip_profile(
    x_kpc: np.ndarray,
    y_kpc: np.ndarray,
    vlos_kms: np.ndarray,
    mass_msun: np.ndarray,
    qso_x_kpc: float,
    qso_y_kpc: float,
    s_min_kpc: float,
    s_max_kpc: float,
    n_bins: int,
    slit_width_kpc: float,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    rho = float(np.hypot(qso_x_kpc, qso_y_kpc))
    if rho <= 0:
        raise ValueError("QSO projected radius is zero; cannot define radial sightline strip")
    u = np.array([qso_x_kpc, qso_y_kpc], dtype=float) / rho
    q = np.array([-u[1], u[0]], dtype=float)
    s = x_kpc * u[0] + y_kpc * u[1]
    t = x_kpc * q[0] + y_kpc * q[1]
    if not s_min_kpc < s_max_kpc:
        raise ValueError(f"Bad signed slit bounds: s_min={s_min_kpc}, s_max={s_max_kpc}")
    strip_mask = (
        np.isfinite(s)
        & np.isfinite(t)
        & np.isfinite(vlos_kms)
        & np.isfinite(mass_msun)
        & (mass_msun > 0)
        & (s >= s_min_kpc)
        & (s <= s_max_kpc)
        & (np.abs(t) <= 0.5 * slit_width_kpc)
    )
    edges = np.linspace(s_min_kpc, s_max_kpc, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    m = mass_msun[strip_mask]
    v = vlos_kms[strip_mask]
    ss = s[strip_mask]
    den, _ = np.histogram(ss, bins=edges, weights=m)
    mass_vlos_sum, _ = np.histogram(ss, bins=edges, weights=m * v)
    mass_vlos2_sum, _ = np.histogram(ss, bins=edges, weights=m * v * v)
    count, _ = np.histogram(ss, bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = mass_vlos_sum / den
        var = mass_vlos2_sum / den - mean * mean
    sigma = np.sqrt(np.clip(var, 0.0, None))
    mean[~np.isfinite(mean)] = np.nan
    sigma[~np.isfinite(sigma)] = np.nan
    profile = pd.DataFrame(
        {
            "bin_index": np.arange(n_bins, dtype=int),
            "s_inner_kpc": edges[:-1],
            "s_outer_kpc": edges[1:],
            "s_center_kpc": centers,
            "stellar_mass_sum_msun": den,
            "stellar_mass_vlos_sum_msun_kms": mass_vlos_sum,
            "stellar_mass_vlos2_sum_msun_kms2": mass_vlos2_sum,
            "stellar_particle_count": count,
            "stellar_mass_weighted_vlos_rest_kms": mean,
            "stellar_mass_weighted_sigma_vlos_kms": sigma,
        }
    )
    sightline_bin = int(np.searchsorted(edges, rho, side="right") - 1)
    sightline_bin = max(0, min(n_bins - 1, sightline_bin))
    opposite_bin = int(np.searchsorted(edges, -rho, side="right") - 1)
    opposite_bin = max(0, min(n_bins - 1, opposite_bin))
    finite = np.isfinite(mean)
    interp = np.nan
    opposite_interp = np.nan
    if finite.sum() >= 2:
        interp = float(np.interp(rho, centers[finite], mean[finite]))
        opposite_interp = float(np.interp(-rho, centers[finite], mean[finite]))
    summary = {
        "rho_kpc": rho,
        "qso_x_kpc": float(qso_x_kpc),
        "qso_y_kpc": float(qso_y_kpc),
        "radial_unit_x": float(u[0]),
        "radial_unit_y": float(u[1]),
        "slit_perp_unit_x": float(q[0]),
        "slit_perp_unit_y": float(q[1]),
        "s_min_kpc": float(s_min_kpc),
        "s_max_kpc": float(s_max_kpc),
        "n_bins": int(n_bins),
        "bin_width_kpc": float(edges[1] - edges[0]),
        "slit_width_kpc": float(slit_width_kpc),
        "strip_particle_count": int(strip_mask.sum()),
        "sightline_bin_index": int(sightline_bin),
        "sightline_bin_center_kpc": float(centers[sightline_bin]),
        "sightline_bin_s_inner_kpc": float(edges[sightline_bin]),
        "sightline_bin_s_outer_kpc": float(edges[sightline_bin + 1]),
        "vlos_rest_at_sightline_bin_kms": float(mean[sightline_bin]) if np.isfinite(mean[sightline_bin]) else np.nan,
        "sigma_vlos_at_sightline_bin_kms": float(sigma[sightline_bin]) if np.isfinite(sigma[sightline_bin]) else np.nan,
        "mass_in_sightline_bin_msun": float(den[sightline_bin]),
        "particle_count_in_sightline_bin": int(count[sightline_bin]),
        "vlos_rest_at_sightline_interp_kms": interp,
        "opposite_rho_bin_index": int(opposite_bin),
        "opposite_rho_bin_center_kpc": float(centers[opposite_bin]),
        "opposite_rho_bin_s_inner_kpc": float(edges[opposite_bin]),
        "opposite_rho_bin_s_outer_kpc": float(edges[opposite_bin + 1]),
        "vlos_rest_at_opposite_rho_bin_kms": float(mean[opposite_bin]) if np.isfinite(mean[opposite_bin]) else np.nan,
        "sigma_vlos_at_opposite_rho_bin_kms": float(sigma[opposite_bin]) if np.isfinite(sigma[opposite_bin]) else np.nan,
        "mass_in_opposite_rho_bin_msun": float(den[opposite_bin]),
        "particle_count_in_opposite_rho_bin": int(count[opposite_bin]),
        "vlos_rest_at_opposite_rho_interp_kms": opposite_interp,
    }
    return profile, summary, strip_mask


def make_mass_map(x_kpc: np.ndarray, y_kpc: np.ndarray, mass: np.ndarray, width_kpc: float, npix: int) -> np.ndarray:
    half = 0.5 * width_kpc
    hist, _, _ = np.histogram2d(
        x_kpc,
        y_kpc,
        bins=npix,
        range=[[-half, half], [-half, half]],
        weights=mass,
    )
    return hist.T


def strip_polygon(summary: dict, s0: float, s1: float, width: float) -> np.ndarray:
    u = np.array([summary["radial_unit_x"], summary["radial_unit_y"]], dtype=float)
    q = np.array([summary["slit_perp_unit_x"], summary["slit_perp_unit_y"]], dtype=float)
    hw = 0.5 * width
    points = [
        s0 * u - hw * q,
        s1 * u - hw * q,
        s1 * u + hw * q,
        s0 * u + hw * q,
    ]
    return np.asarray(points)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )


def make_diagnostic_plots(
    out_dir: Path,
    sid: int,
    alpha: int,
    mode: str,
    profile: pd.DataFrame,
    summary: dict,
    x_kpc: np.ndarray,
    y_kpc: np.ndarray,
    mass_msun: np.ndarray,
    map_width_kpc: float,
    map_npix: int,
) -> tuple[str, str]:
    configure_matplotlib()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag = f"sid{sid}_alpha{alpha:03d}_{mode}"
    profile_png = fig_dir / f"stellar_vlos_rest_slit_profile_{tag}.png"
    patch_png = fig_dir / f"stellar_slit_patch_diagnostic_{tag}.png"

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        profile["s_center_kpc"],
        profile["stellar_mass_weighted_vlos_rest_kms"],
        color="black",
        lw=1.5,
    )
    ax.axhline(0, color="0.65", lw=0.9)
    ax.axvline(summary["rho_kpc"], color="#1f78b4", ls="--", lw=1.2, label=r"$\rho_{\rm QSO}$")
    ax.axvline(-summary["rho_kpc"], color="#777777", ls=":", lw=1.0, label=r"$-\rho_{\rm QSO}$")
    y = summary["vlos_rest_at_sightline_bin_kms"]
    if np.isfinite(y):
        ax.scatter([summary["sightline_bin_center_kpc"]], [y], s=70, color="#d62728", zorder=5, label="sightline bin")
    ax.set_xlabel(r"Signed distance along center--QSO strip [kpc]")
    ax.set_ylabel(r"Stellar mass-weighted $v_{\rm LOS}-v_{\rm sys,LOS}$ [km s$^{-1}$]")
    ax.set_title(f"SID {sid}; alpha={alpha} deg; {mode}; slit width={summary['slit_width_kpc']:.1f} kpc")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2, lw=0.5)
    fig.savefig(profile_png)
    plt.close(fig)

    img = make_mass_map(x_kpc, y_kpc, mass_msun, map_width_kpc, map_npix)
    log_img = np.full_like(img, np.nan, dtype=float)
    good = img > 0
    log_img[good] = np.log10(img[good])
    finite = np.isfinite(log_img)
    vmin = float(np.nanpercentile(log_img[finite], 3)) if finite.any() else 0.0
    vmax = float(np.nanpercentile(log_img[finite], 99.5)) if finite.any() else 1.0

    half = 0.5 * map_width_kpc
    fig, ax = plt.subplots(figsize=(7.4, 6.7))
    im = ax.imshow(
        log_img,
        origin="lower",
        extent=[-half, half, -half, half],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    full_poly = Polygon(
        strip_polygon(summary, summary["s_min_kpc"], summary["s_max_kpc"], summary["slit_width_kpc"]),
        closed=True,
        facecolor="none",
        edgecolor="#00c8ff",
        lw=1.2,
        ls="--",
        label="sampled strip",
    )
    bin_poly = Polygon(
        strip_polygon(
            summary,
            summary["sightline_bin_s_inner_kpc"],
            summary["sightline_bin_s_outer_kpc"],
            summary["slit_width_kpc"],
        ),
        closed=True,
        facecolor="#00c8ff",
        edgecolor="#00324a",
        alpha=0.35,
        lw=1.0,
        label="sightline bin patch",
    )
    ax.add_patch(full_poly)
    ax.add_patch(bin_poly)
    ax.scatter([0.0], [0.0], marker="+", color="cyan", s=80, lw=1.5, label="center")
    ax.scatter([summary["qso_x_kpc"]], [summary["qso_y_kpc"]], marker="*", color="#25d366", edgecolor="black", s=100, label="QSO")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("projected major-axis x [kpc]")
    ax.set_ylabel("projected minor-axis y [kpc]")
    ax.set_title(f"SID {sid}; alpha={alpha} deg; {mode}; stellar mass map")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cb.set_label(r"$\log_{10}\,M_\star$ per pixel [$M_\odot$]")
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    fig.savefig(patch_png)
    plt.close(fig)

    return str(profile_png), str(patch_png)


def h5_write_array(group: h5py.Group, name: str, values) -> None:
    if name in group:
        del group[name]
    arr = np.asarray(values)
    if arr.dtype.kind in {"U", "O"}:
        arr = arr.astype(H5_STRING)
        group.create_dataset(name, data=arr)
    elif arr.ndim == 0:
        group.create_dataset(name, data=arr)
    else:
        group.create_dataset(
            name,
            data=arr,
            chunks=True,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )


def set_h5_attrs(obj: h5py.Group | h5py.Dataset, payload: dict) -> None:
    for key, value in json_ready(payload).items():
        if value is None:
            obj.attrs[key] = "null"
        elif isinstance(value, (list, dict)):
            obj.attrs[key] = json.dumps(value)
        elif isinstance(value, str):
            obj.attrs[key] = value
        else:
            obj.attrs[key] = value


def combined_hdf5_path(output_root: Path, sid: int, run_label: str) -> Path:
    return (
        output_root
        / f"sid{sid}"
        / f"ism_velocity_slit_profiles_{run_label}"
        / "data"
        / f"stellar_vlos_rest_slit_profiles_combined_sid{sid}_{run_label}.hdf5"
    )


def write_profile_to_hdf5(path: Path, profile: pd.DataFrame, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = str(summary["mode"])
    alpha = int(summary["alpha_deg"])
    group_name = f"profiles/{mode}/alpha_{alpha:03d}"
    with h5py.File(path, "a") as h5:
        h5.attrs["file_format"] = "stellar_ism_velocity_slit_profiles_v2"
        h5.attrs["chunked_datasets"] = True
        h5.attrs["velocity_unit_treatment"] = (
            "PartType4/Velocities have units km*sqrt(a)/s; values are multiplied by sqrt(a) "
            "to obtain peculiar km/s before subtracting TNG catalog SubhaloVel."
        )
        if group_name in h5:
            del h5[group_name]
        group = h5.create_group(group_name)
        set_h5_attrs(group, summary)
        for column in profile.columns:
            h5_write_array(group, column, profile[column].to_numpy())


def write_summary_to_hdf5(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).copy()
    drop_cols = [
        "profile_plot_png",
        "patch_diagnostic_png",
        "profile_csv",
        "summary_json",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    with h5py.File(path, "a") as h5:
        if "summary" in h5:
            del h5["summary"]
        group = h5.create_group("summary")
        group.attrs["description"] = "One row per SID/alpha/mode profile; detailed per-bin arrays live under /profiles/<mode>/alpha_XXX."
        for column in df.columns:
            values = df[column].to_numpy()
            if values.dtype.kind == "O":
                values = np.array([json.dumps(v) if isinstance(v, (list, dict)) else str(v) for v in values], dtype=H5_STRING)
            h5_write_array(group, column, values)


def process_one(
    sid: int,
    snap: int,
    run_label: str,
    sightline_id: str,
    mode: str,
    alpha: int,
    output_root: Path,
    cutout_root: Path,
    s_min_kpc: float,
    s_max_kpc: float,
    n_bins: int,
    slit_width_kpc: float,
    make_plots: bool,
    map_width_kpc: float,
    map_npix: int,
    hdf5_path: Path | None = None,
    save_csv: bool = False,
    save_json: bool = False,
    stellar_cache: dict[str, np.ndarray] | None = None,
    context_cache: CatalogContext | None = None,
    tables_cache: tuple[pd.DataFrame, pd.DataFrame, dict] | None = None,
) -> dict:
    out_dir = output_root / f"sid{sid}" / f"ism_velocity_slit_profiles_{run_label}"
    data_dir = out_dir / "data"
    log_dir = out_dir / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cutout = default_cutout_path(cutout_root, sid)
    context = context_cache or load_catalog_context(output_root, cutout, sid, snap)
    tables = tables_cache or load_recipe_tables(sid_recipe_dir(output_root, sid, snap, run_label), sid)
    rays, orient, header = tables
    row, orient_row = select_recipe_row(rays, orient, sid, sightline_id, alpha, mode)
    xhat, yhat, los_hat = saved_basis(row, orient_row)
    qso_x, qso_y, qso_source = qso_xy_from_header(header, sightline_id, row)

    if stellar_cache is None:
        stellar_cache = load_stellar_particles(cutout, context)

    rel_kpc = stellar_cache["rel_ckpch"] / context.h
    x_kpc = rel_kpc @ xhat
    y_kpc = rel_kpc @ yhat
    vlos_rest = stellar_cache["vrel_kms"] @ los_hat
    mass_msun = stellar_cache["mass_msun"]

    profile, summary, _ = radial_strip_profile(
        x_kpc=x_kpc,
        y_kpc=y_kpc,
        vlos_kms=vlos_rest,
        mass_msun=mass_msun,
        qso_x_kpc=qso_x,
        qso_y_kpc=qso_y,
        s_min_kpc=s_min_kpc,
        s_max_kpc=s_max_kpc,
        n_bins=n_bins,
        slit_width_kpc=slit_width_kpc,
    )
    v_sys_los = float(np.dot(context.subhalo_vel_kms, los_hat))
    anchor = np.array([row["anchor_X_ckpch_abs"], row["anchor_Y_ckpch_abs"], row["anchor_Z_ckpch_abs"]], dtype=float)
    anchor_rel_kpc = minimal_delta(anchor[None, :], context.center_ckpch, context.box_ckpch)[0] / context.h
    anchor_x = float(np.dot(anchor_rel_kpc, xhat))
    anchor_y = float(np.dot(anchor_rel_kpc, yhat))
    summary.update(
        {
            "sid": sid,
            "snap": snap,
            "run_label": run_label,
            "sightline_id": sightline_id,
            "alpha_deg": int(alpha),
            "mode": mode,
            "quantity": "stellar_mass_weighted_los_velocity_minus_subhalo_systemic_los",
            "subhalo_vel_kms": context.subhalo_vel_kms.tolist(),
            "los_hat": los_hat.tolist(),
            "north_hat": yhat.tolist(),
            "xhat_major_axis": xhat.tolist(),
            "v_sys_los_kms": v_sys_los,
            "center_ckpch": context.center_ckpch.tolist(),
            "center_source": context.center_source,
            "subhalo_velocity_source": context.velocity_source,
            "cutout_path": str(cutout),
            "recipe_csv": str(sid_recipe_dir(output_root, sid, snap, run_label) / f"rays_sid{sid}.csv"),
            "orient_peralpha_csv": str(sid_recipe_dir(output_root, sid, snap, run_label) / f"orient_peralpha_sid{sid}.csv"),
            "qso_xy_source": qso_source,
            "anchor_projected_x_kpc_check": anchor_x,
            "anchor_projected_y_kpc_check": anchor_y,
            "anchor_minus_qso_projected_error_kpc": float(np.hypot(anchor_x - qso_x, anchor_y - qso_y)),
            "scale_factor": context.scale_factor,
            "sqrt_scale_factor_applied_to_parttype4_velocities": float(stellar_cache["velocity_scale_factor_sqrt_a"][0]),
            "redshift": context.redshift,
            "snapshot_velocity_note": "PartType4/Velocities are km*sqrt(a)/s and are multiplied by sqrt(a) before subtracting SubhaloVel; at snap99 a=1.",
            "stellar_particles_used": int(stellar_cache["kept_count"][0]),
        }
    )

    tag = f"sid{sid}_alpha{alpha:03d}_{mode}"
    if hdf5_path is not None:
        write_profile_to_hdf5(hdf5_path, profile, summary)
        summary["combined_hdf5"] = str(hdf5_path)
        summary["hdf5_group"] = f"/profiles/{mode}/alpha_{alpha:03d}"
    if save_csv:
        profile_csv = data_dir / f"stellar_vlos_rest_slit_profile_{tag}.csv"
        profile.to_csv(profile_csv, index=False)
        summary["profile_csv"] = str(profile_csv)
    if make_plots:
        profile_png, patch_png = make_diagnostic_plots(
            out_dir=out_dir,
            sid=sid,
            alpha=alpha,
            mode=mode,
            profile=profile,
            summary=summary,
            x_kpc=x_kpc,
            y_kpc=y_kpc,
            mass_msun=mass_msun,
            map_width_kpc=map_width_kpc,
            map_npix=map_npix,
        )
        summary["profile_plot_png"] = profile_png
        summary["patch_diagnostic_png"] = patch_png
    if save_json:
        summary_json = data_dir / f"stellar_vlos_rest_at_sightline_{tag}.json"
        write_json(summary_json, summary)
        summary["summary_json"] = str(summary_json)
    return summary


def existing_sids_from_outputs(output_root: Path) -> list[int]:
    sids: list[int] = []
    for path in sorted(output_root.glob("sid*")):
        if path.is_dir() and path.name[3:].isdigit():
            sids.append(int(path.name[3:]))
    return sids


def read_sid_list(path: Path) -> list[int]:
    sids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            sids.append(int(line.split()[0]))
    return sids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", type=int, action="append", help="SubhaloID. May be repeated.")
    parser.add_argument("--sid-list", type=Path, help="Text file with one SID per line.")
    parser.add_argument("--all-sids", action="store_true", help="Process all sid* directories under output-root.")
    parser.add_argument("--snap", type=int, default=99)
    parser.add_argument("--run-label", default="L2Rvir")
    parser.add_argument("--sightline-id", default=DEFAULT_SIGHTLINE_ID)
    parser.add_argument("--alphas", default="0,45", help="Comma list, ranges like 0-180, or all.")
    parser.add_argument("--modes", default="noflip,flip")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cutout-root", type=Path, default=DEFAULT_CUTOUT_ROOT)
    parser.add_argument("--s-min-kpc", type=float, default=-60.0, help="Signed lower strip coordinate in kpc.")
    parser.add_argument("--s-max-kpc", type=float, default=60.0, help="Signed upper strip coordinate in kpc.")
    parser.add_argument("--r-max-kpc", type=float, default=None, help="Legacy alias: if supplied without signed bounds, uses 0..r_max.")
    parser.add_argument("--n-bins", type=int, default=1000)
    parser.add_argument("--slit-width-kpc", type=float, default=2.0)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--save-csv", action="store_true", help="Also write per-alpha profile CSV files.")
    parser.add_argument("--save-json", action="store_true", help="Also write per-alpha summary JSON files.")
    parser.add_argument("--save-summary-csv", action="store_true", help="Also write a small per-SID summary CSV.")
    parser.add_argument("--no-fresh-hdf5", action="store_true", help="Append/overwrite groups in an existing combined HDF5 instead of starting fresh per SID.")
    parser.add_argument("--map-width-kpc", type=float, default=120.0)
    parser.add_argument("--map-npix", type=int, default=500)
    args = parser.parse_args()
    if args.r_max_kpc is not None and args.s_min_kpc == -60.0 and args.s_max_kpc == 60.0:
        args.s_min_kpc = 0.0
        args.s_max_kpc = float(args.r_max_kpc)

    sids: list[int] = []
    if args.sid:
        sids.extend(args.sid)
    if args.sid_list:
        sids.extend(read_sid_list(args.sid_list))
    if args.all_sids:
        sids.extend(existing_sids_from_outputs(args.output_root))
    sids = sorted(set(sids))
    if not sids:
        raise SystemExit("No SIDs requested. Use --sid, --sid-list, or --all-sids.")

    modes = parse_modes(args.modes)
    rows: list[dict] = []
    for sid in sids:
        recipe_dir = sid_recipe_dir(args.output_root, sid, args.snap, args.run_label)
        rays, orient, header = load_recipe_tables(recipe_dir, sid)
        available_alphas = sorted(rays["alpha_deg"].astype(float).round().astype(int).unique())
        alphas = parse_int_list(args.alphas) or available_alphas
        cutout = default_cutout_path(args.cutout_root, sid)
        context = load_catalog_context(args.output_root, cutout, sid, args.snap)
        stellar = load_stellar_particles(cutout, context)
        hdf5_path = combined_hdf5_path(args.output_root, sid, args.run_label)
        if hdf5_path.exists() and not args.no_fresh_hdf5:
            hdf5_path.unlink()
        for mode in modes:
            for alpha in alphas:
                summary = process_one(
                    sid=sid,
                    snap=args.snap,
                    run_label=args.run_label,
                    sightline_id=args.sightline_id,
                    mode=mode,
                    alpha=int(alpha),
                    output_root=args.output_root,
                    cutout_root=args.cutout_root,
                    s_min_kpc=args.s_min_kpc,
                    s_max_kpc=args.s_max_kpc,
                    n_bins=args.n_bins,
                    slit_width_kpc=args.slit_width_kpc,
                    make_plots=args.make_plots,
                    map_width_kpc=args.map_width_kpc,
                    map_npix=args.map_npix,
                    hdf5_path=hdf5_path,
                    save_csv=args.save_csv,
                    save_json=args.save_json,
                    stellar_cache=stellar,
                    context_cache=context,
                    tables_cache=(rays, orient, header),
                )
                rows.append(summary)
                print(
                    f"SID {sid} alpha={alpha:03d} {mode}: "
                    f"v_bin={summary['vlos_rest_at_sightline_bin_kms']:.3f} km/s, "
                    f"v_interp={summary['vlos_rest_at_sightline_interp_kms']:.3f} km/s, "
                    f"v_opp={summary['vlos_rest_at_opposite_rho_bin_kms']:.3f} km/s, "
                    f"Nbin={summary['particle_count_in_sightline_bin']}, "
                    f"stripN={summary['strip_particle_count']}"
                )
        out_dir = args.output_root / f"sid{sid}" / f"ism_velocity_slit_profiles_{args.run_label}"
        sid_rows = [r for r in rows if int(r["sid"]) == int(sid)]
        write_summary_to_hdf5(hdf5_path, sid_rows)
        size_gb = hdf5_path.stat().st_size / 1024**3
        print(f"Wrote combined chunked HDF5 {hdf5_path} ({size_gb:.3f} GiB)")
        if args.save_summary_csv:
            combined = out_dir / "data" / f"stellar_vlos_rest_at_sightline_summary_sid{sid}_{args.run_label}.csv"
            pd.DataFrame(sid_rows).to_csv(combined, index=False)
            print(f"Wrote {combined}")

    if len(sids) > 1 and args.save_summary_csv:
        combined_all = args.output_root / f"stellar_vlos_rest_at_sightline_summary_all_sids_{args.run_label}.csv"
        pd.DataFrame(rows).to_csv(combined_all, index=False)
        print(f"Wrote {combined_all}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
