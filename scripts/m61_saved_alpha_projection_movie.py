#!/usr/bin/env python3
"""
Saved-alpha LOS projection movie for M61/TNG50 SID 488530.

This is a projection-only diagnostic for alpha = 0..180 in 1 degree steps.
It uses the corrected saved-alpha ray recipes, keeps the observer/QSO sky-plane
coordinates fixed, and animates how the unrotated cutout is sampled as the
galaxy is spun about its disk normal.

Outputs are PNG frames only plus MP4 videos made with ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yt

import m61_oriented_HI_vlos_rotation_alpha5 as base


@dataclass
class MovieConfig:
    output_dir: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "saved_alpha_LOS_alpha000_180_projection_movies"
    )
    recipe_csv: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "rays_and_recipes_sid488530_snap99_L4Rvir/rays_sid488530.csv"
    )
    orient_csv: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "rays_and_recipes_sid488530_snap99_L4Rvir/orient_peralpha_sid488530.csv"
    )
    orient_header: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "rays_and_recipes_sid488530_snap99_L4Rvir/orient_header_sid488530.json"
    )
    sid: int = 488530
    snap: int = 99
    sightline_id: str = "J122138+043026"
    mode: str = "noflip"
    alpha_min: int = 0
    alpha_max: int = 180
    width_kpc: float = 100.0
    npix: int = 1024
    logNHI_threshold: float = 19.5
    fps: int = 18
    recompute: bool = False
    rerender: bool = False
    maps_only: bool = False
    render_only: bool = False

    @property
    def alphas(self) -> list[int]:
        return list(range(int(self.alpha_min), int(self.alpha_max) + 1))

    @property
    def pixel_scale_kpc(self) -> float:
        return self.width_kpc / self.npix


def setup_paths(config: MovieConfig) -> dict[str, Path]:
    out = Path(config.output_dir)
    paths = {
        "out": out,
        "data": out / "data",
        "logs": out / "logs",
        "frames_combined": out / "frames" / "combined",
        "frames_hi": out / "frames" / "HI_logNHI",
        "frames_vgas": out / "frames" / "vlos_gas_weighted",
        "frames_vhi": out / "frames" / "vlos_HI_weighted",
        "videos": out / "videos",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(paths: dict[str, Path], config: MovieConfig) -> None:
    suffix = f"alpha{config.alpha_min:03d}_{config.alpha_max:03d}"
    err = paths["logs"] / f"errors_{suffix}.txt"
    if err.exists():
        err.unlink()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(paths["logs"] / f"run_{suffix}.log", mode="w")
    file_handler.setFormatter(fmt)
    root.addHandler(stream)
    root.addHandler(file_handler)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(base.json_sanitize(payload), f, indent=2, sort_keys=True)


def normalize(vec, name="vector") -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Cannot normalize {name}: {arr}")
    return arr / norm


def row_vec(row, prefix: str) -> np.ndarray:
    if prefix == "p0":
        keys = ["p0_X_ckpch_abs", "p0_Y_ckpch_abs", "p0_Z_ckpch_abs"]
    elif prefix == "p1":
        keys = ["p1_X_ckpch_abs", "p1_Y_ckpch_abs", "p1_Z_ckpch_abs"]
    elif prefix == "anchor":
        keys = ["anchor_X_ckpch_abs", "anchor_Y_ckpch_abs", "anchor_Z_ckpch_abs"]
    elif prefix == "los":
        keys = ["los_x", "los_y", "los_z"]
    elif prefix == "north":
        keys = ["north_x", "north_y", "north_z"]
    else:
        raise ValueError(prefix)
    return np.asarray([row[k] for k in keys], dtype=float)


def load_rows(config: MovieConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rays = pd.read_csv(config.recipe_csv)
    orient = pd.read_csv(config.orient_csv)
    header = json.load(open(config.orient_header))
    wanted = set(config.alphas)
    rmask = (
        (rays["SubhaloID"].astype(int) == config.sid)
        & (rays["sightline_id"].astype(str) == config.sightline_id)
        & (rays["mode"].astype(str) == config.mode)
        & rays["alpha_deg"].astype(int).isin(wanted)
    )
    omask = (
        (orient["mode"].astype(str) == config.mode)
        & orient["alpha_deg"].astype(int).isin(wanted)
    )
    rays = rays.loc[rmask].copy().sort_values("alpha_deg")
    orient = orient.loc[omask].copy().sort_values("alpha_deg")
    if len(rays) != len(config.alphas) or len(orient) != len(config.alphas):
        raise RuntimeError(f"Expected {len(config.alphas)} rows, got rays={len(rays)} orient={len(orient)}")
    return rays, orient, header


def npz_path(paths: dict[str, Path], alpha: int) -> Path:
    return paths["data"] / f"projection_alpha{alpha:03d}_saved_alpha_LOS_inner100kpc.npz"


def grid(config: MovieConfig):
    half = 0.5 * config.width_kpc
    dx = config.pixel_scale_kpc
    x = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    y = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y


def velocity_component_fields(ds):
    candidates = [
        (("gas", "velocity_x"), ("gas", "velocity_y"), ("gas", "velocity_z")),
        (("PartType0", "velocity_x"), ("PartType0", "velocity_y"), ("PartType0", "velocity_z")),
        (("PartType0", "particle_velocity_x"), ("PartType0", "particle_velocity_y"), ("PartType0", "particle_velocity_z")),
    ]
    for cand in candidates:
        if all(base.yt_field_exists(ds, field) for field in cand):
            return cand
    return None


def add_dynamic_los_fields(ds, los_holder: dict[str, np.ndarray], systemic_kms: np.ndarray):
    component_fields = velocity_component_fields(ds)
    raw_vector_field = ("PartType0", "Velocities")
    if component_fields is None and not base.yt_field_exists(ds, raw_vector_field):
        raise RuntimeError("Could not find gas velocity fields.")

    def velocity_component(data, index):
        if component_fields is not None:
            return data[component_fields[index]].to("km/s")
        return data[raw_vector_field][:, index].to("km/s")

    def _vlos(field, data):
        los = los_holder["los"]
        vx = velocity_component(data, 0) - data.ds.quan(systemic_kms[0], "km/s")
        vy = velocity_component(data, 1) - data.ds.quan(systemic_kms[1], "km/s")
        vz = velocity_component(data, 2) - data.ds.quan(systemic_kms[2], "km/s")
        return (los[0] * vx + los[1] * vy + los[2] * vz).to("km/s")

    def _hi_vlos(field, data):
        return data[("gas", "H_p0_number_density")] * data[("gas", "velocity_los_movie")]

    def _gas_vlos(field, data):
        return data[("gas", "density")] * data[("gas", "velocity_los_movie")]

    if not base.yt_field_exists(ds, ("gas", "velocity_los_movie")):
        ds.add_field(("gas", "velocity_los_movie"), function=_vlos, sampling_type="particle", units="km/s")
    if not base.yt_field_exists(ds, ("gas", "HI_vlos_integrand_movie")):
        ds.add_field(("gas", "HI_vlos_integrand_movie"), function=_hi_vlos, sampling_type="particle", units="km/(s*cm**3)")
    if not base.yt_field_exists(ds, ("gas", "gas_density_vlos_integrand_movie")):
        ds.add_field(("gas", "gas_density_vlos_integrand_movie"), function=_gas_vlos, sampling_type="particle", units="g*km/(s*cm**3)")


def compute_maps(config: MovieConfig, paths, ds, geometry, rows_by_alpha, orient_by_alpha, los_holder):
    x, y, X, Y = grid(config)
    center = ds.arr(geometry["galaxy_center_ckpch"], "code_length")
    width = ds.arr([config.width_kpc * base.H_TNG] * 2, "code_length")
    outputs = {}
    for alpha in config.alphas:
        out = npz_path(paths, alpha)
        if out.exists() and not config.recompute:
            try:
                z = np.load(out, allow_pickle=True)
                outputs[alpha] = {key: z[key] for key in z.files}
                logging.info("alpha %03d: loaded cached maps", alpha)
                continue
            except (zipfile.BadZipFile, EOFError, ValueError, OSError) as exc:
                logging.warning("alpha %03d: corrupt cached NPZ %s (%r); deleting and recomputing", alpha, out, exc)
                out.unlink(missing_ok=True)

        row = rows_by_alpha[alpha]
        orow = orient_by_alpha[alpha]
        los = normalize(row_vec(row, "los"), f"los alpha {alpha}")
        north = normalize(row_vec(orow, "north"), f"north alpha {alpha}")
        los_holder["los"] = los
        hi_den = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "H_p0_number_density"), north)
        hi_num = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "HI_vlos_integrand_movie"), north)
        gas_den = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "density"), north)
        gas_num = base.off_axis_integral(ds, center, los, width, config.npix, ("gas", "gas_density_vlos_integrand_movie"), north)

        N_HI = base.yt_array_to_numpy(hi_den, "cm**-2")
        HI_num = base.yt_array_to_numpy(hi_num, "km/(s*cm**2)")
        gas_sigma = base.yt_array_to_numpy(gas_den, "g/cm**2")
        gas_num_np = base.yt_array_to_numpy(gas_num, "g*km/(s*cm**2)")
        with np.errstate(divide="ignore", invalid="ignore"):
            logN = np.log10(np.where(N_HI > 0, N_HI, np.nan))
            vhi = HI_num / N_HI
            vgas = gas_num_np / gas_sigma
        vhi[~np.isfinite(vhi)] = np.nan
        vgas[~np.isfinite(vgas)] = np.nan
        mask = np.isfinite(logN) & (logN >= config.logNHI_threshold)
        anchor = row_vec(row, "anchor")
        delta = anchor - geometry["galaxy_center_ckpch"]
        rho_check = float(np.linalg.norm(delta - np.dot(delta, los) * los) / base.H_TNG)

        np.savez_compressed(
            out,
            x_kpc=x,
            y_kpc=y,
            X_kpc=X,
            Y_kpc=Y,
            N_HI_cm2=N_HI,
            logN_HI=logN,
            vlos_gasweighted_kms=vgas,
            vlos_HIweighted_kms=vhi,
            mask_logNHI_gt19p5=mask,
            alpha_deg=float(alpha),
            los_hat=los,
            north_hat=north,
            x_qso_kpc=float(row["rho_kpc"]) * np.cos(np.deg2rad(float(row["phi_deg"]))),
            y_qso_kpc=float(row["rho_kpc"]) * np.sin(np.deg2rad(float(row["phi_deg"]))),
            rho_check_kpc=rho_check,
        )
        outputs[alpha] = {
            "x_kpc": x,
            "y_kpc": y,
            "X_kpc": X,
            "Y_kpc": Y,
            "N_HI_cm2": N_HI,
            "logN_HI": logN,
            "vlos_gasweighted_kms": vgas,
            "vlos_HIweighted_kms": vhi,
            "mask_logNHI_gt19p5": mask,
            "alpha_deg": np.asarray(float(alpha)),
            "los_hat": los,
            "north_hat": north,
            "x_qso_kpc": np.asarray(float(row["rho_kpc"]) * np.cos(np.deg2rad(float(row["phi_deg"])))),
            "y_qso_kpc": np.asarray(float(row["rho_kpc"]) * np.sin(np.deg2rad(float(row["phi_deg"])))),
            "rho_check_kpc": np.asarray(rho_check),
        }
        good_v = vhi[mask & np.isfinite(vhi)]
        logging.info(
            "alpha %03d/%03d: rho_check=%.4f logN[p50,p99]=%s vHI[p2,p98]=%s",
            alpha, config.alpha_max, rho_check,
            np.nanpercentile(logN[np.isfinite(logN)], [50, 99]).tolist(),
            np.nanpercentile(good_v, [2, 98]).tolist() if good_v.size else [],
        )
    return outputs


def scales(config: MovieConfig, maps):
    log_vals = []
    vgas_vals = []
    vhi_vals = []
    for m in maps.values():
        logN = m["logN_HI"]
        mask = m["mask_logNHI_gt19p5"].astype(bool)
        log_vals.append(logN[np.isfinite(logN) & (logN >= config.logNHI_threshold)])
        vg = m["vlos_gasweighted_kms"]
        vh = m["vlos_HIweighted_kms"]
        vgas_vals.append(np.abs(vg[mask & np.isfinite(vg)]))
        vhi_vals.append(np.abs(vh[mask & np.isfinite(vh)]))
    log_vals = np.concatenate([v for v in log_vals if v.size])
    vgas_vals = np.concatenate([v for v in vgas_vals if v.size])
    vhi_vals = np.concatenate([v for v in vhi_vals if v.size])
    return {
        "logN_vmin": config.logNHI_threshold,
        "logN_vmax": float(max(config.logNHI_threshold + 0.2, np.nanpercentile(log_vals, 99))),
        "vgas_vlim": float(min(max(np.nanpercentile(vgas_vals, 98), 50.0), 300.0)),
        "vhi_vlim": float(min(max(np.nanpercentile(vhi_vals, 98), 50.0), 300.0)),
    }


def draw_map(ax, data, kind, vmin, vmax, alpha, qso_xy):
    extent = [-50, 50, -50, 50]
    if kind == "HI":
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("white")
        label = r"$\log_{10}N_{\rm HI}$"
    else:
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("white")
        label = r"$v_{\rm los}$ [km/s]"
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.plot(0, 0, marker="+", ms=10, mew=1.6, color="cyan")
    ax.scatter([qso_xy[0]], [qso_xy[1]], s=40, facecolors="none", edgecolors="lime", lw=1.4)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(f"{label}  |  alpha={alpha:03d} deg")
    ax.minorticks_on()
    return im


def render_frames(config: MovieConfig, paths, maps, scale):
    for alpha, m in maps.items():
        frame_combined = paths["frames_combined"] / f"frame_{alpha:03d}.png"
        frame_hi = paths["frames_hi"] / f"frame_{alpha:03d}.png"
        frame_vgas = paths["frames_vgas"] / f"frame_{alpha:03d}.png"
        frame_vhi = paths["frames_vhi"] / f"frame_{alpha:03d}.png"
        if all(p.exists() for p in [frame_combined, frame_hi, frame_vgas, frame_vhi]) and not config.rerender:
            continue
        mask = m["mask_logNHI_gt19p5"].astype(bool)
        logN = np.where(mask, m["logN_HI"], np.nan)
        vgas = np.where(mask, m["vlos_gasweighted_kms"], np.nan)
        vhi = np.where(mask, m["vlos_HIweighted_kms"], np.nan)
        qso = (float(m["x_qso_kpc"]), float(m["y_qso_kpc"]))

        fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        ims = [
            draw_map(axes[0], logN, "HI", scale["logN_vmin"], scale["logN_vmax"], alpha, qso),
            draw_map(axes[1], vgas, "v", -scale["vgas_vlim"], scale["vgas_vlim"], alpha, qso),
            draw_map(axes[2], vhi, "v", -scale["vhi_vlim"], scale["vhi_vlim"], alpha, qso),
        ]
        axes[0].set_title(f"H I column, alpha={alpha:03d} deg")
        axes[1].set_title("gas-density-weighted LOS velocity")
        axes[2].set_title("H I-weighted LOS velocity")
        labels = [
            r"$\log_{10}N_{\rm HI}\ [{\rm cm}^{-2}]$",
            r"$v_{\rm los,gas-weighted}$ [km/s]",
            r"$v_{\rm los,HI-weighted}$ [km/s]",
        ]
        for ax, im, label in zip(axes, ims, labels):
            plt.colorbar(im, ax=ax, shrink=0.86, label=label)
        fig.savefig(frame_combined, dpi=160)
        plt.close(fig)

        for frame, arr, kind, vmin, vmax, cbar_label in [
            (frame_hi, logN, "HI", scale["logN_vmin"], scale["logN_vmax"], r"$\log_{10}N_{\rm HI}\ [{\rm cm}^{-2}]$"),
            (frame_vgas, vgas, "v", -scale["vgas_vlim"], scale["vgas_vlim"], r"$v_{\rm los,gas-weighted}$ [km/s]"),
            (frame_vhi, vhi, "v", -scale["vhi_vlim"], scale["vhi_vlim"], r"$v_{\rm los,HI-weighted}$ [km/s]"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
            im = draw_map(ax, arr, kind, vmin, vmax, alpha, qso)
            plt.colorbar(im, ax=ax, shrink=0.86, label=cbar_label)
            fig.savefig(frame, dpi=200)
            plt.close(fig)
        if alpha % 10 == 0:
            logging.info("Rendered PNG frames through alpha %03d", alpha)


def run_ffmpeg(config: MovieConfig, paths):
    videos = {
        "combined": (paths["frames_combined"], paths["videos"] / "m61_saved_alpha_000_180_combined_HI_vgas_vHI.mp4"),
        "HI": (paths["frames_hi"], paths["videos"] / "m61_saved_alpha_000_180_HI_logNHI.mp4"),
        "vlos_gas": (paths["frames_vgas"], paths["videos"] / "m61_saved_alpha_000_180_vlos_gas_weighted.mp4"),
        "vlos_HI": (paths["frames_vhi"], paths["videos"] / "m61_saved_alpha_000_180_vlos_HI_weighted.mp4"),
    }
    made = {}
    for name, (frame_dir, out) in videos.items():
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(config.fps),
            "-start_number", f"{config.alpha_min}",
            "-i", str(frame_dir / "frame_%03d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "15",
            "-pix_fmt", "yuv420p",
            str(out),
        ]
        logging.info("Running ffmpeg for %s -> %s", name, out)
        subprocess.run(cmd, check=True)
        made[name] = str(out)
    return made


def load_context(config: MovieConfig, paths):
    base_config = base.AnalysisConfig(output_dir=str(paths["out"]))
    center, center_meta = base.load_true_galaxy_center(base_config)
    cutout = base.find_cutout_h5(base_config)
    ds = yt.load(str(cutout))
    field_info = base.ensure_gas_alias_fields(ds)
    systemic = base.systemic_velocity_from_hdf5(cutout, center)
    field_info["systemic_velocity"] = systemic
    los_holder = {"los": np.array([0.0, 0.0, 1.0])}
    add_dynamic_los_fields(ds, los_holder, np.asarray(systemic["velocity_kms"], dtype=float))
    return ds, {"galaxy_center_ckpch": center, "galaxy_center_source": center_meta, "cutout_h5": str(cutout)}, field_info, los_holder


def run(config: MovieConfig):
    paths = setup_paths(config)
    setup_logging(paths, config)
    try:
        rays, orient, header = load_rows(config)
        rows_by_alpha = {int(row["alpha_deg"]): row for _, row in rays.iterrows()}
        orient_by_alpha = {int(row["alpha_deg"]): row for _, row in orient.iterrows()}
        ds, geometry, field_info, los_holder = load_context(config, paths)
        logging.info("Output: %s", paths["out"])
        logging.info("Alpha convention: %s", header.get("alpha_convention"))
        logging.info("Systemic velocity: %s", field_info["systemic_velocity"])
        maps = compute_maps(config, paths, ds, geometry, rows_by_alpha, orient_by_alpha, los_holder)
        if config.maps_only:
            logging.info("Maps-only mode complete for alpha %03d..%03d", config.alpha_min, config.alpha_max)
            return {"paths": paths, "maps": maps}
        scale = scales(config, maps)
        logging.info("Common plot scales: %s", scale)
        render_frames(config, paths, maps, scale)
        videos = run_ffmpeg(config, paths)
        metadata = {
            "SID": config.sid,
            "SNAP": config.snap,
            "alpha_min": config.alpha_min,
            "alpha_max": config.alpha_max,
            "mode": config.mode,
            "recipe_csv": config.recipe_csv,
            "orient_csv": config.orient_csv,
            "orient_header": header,
            "projection_width_kpc": config.width_kpc,
            "Npix": config.npix,
            "pixel_scale_kpc": config.pixel_scale_kpc,
            "logNHI_threshold": config.logNHI_threshold,
            "systemic_velocity": field_info["systemic_velocity"],
            "galaxy_center": geometry,
            "plot_scales": scale,
            "videos": videos,
            "output_paths": {k: str(v) for k, v in paths.items()},
        }
        meta_path = paths["data"] / "saved_alpha_projection_movie_metadata.json"
        write_json(meta_path, metadata)
        print("\nSaved-alpha projection movie complete")
        print(f"output_dir = {paths['out']}")
        for name, path in videos.items():
            print(f"{name}: {path}")
        print(f"metadata = {meta_path}")
        return {"paths": paths, "videos": videos, "metadata": metadata}
    except Exception:
        err = paths["logs"] / f"errors_alpha{config.alpha_min:03d}_{config.alpha_max:03d}.txt"
        err.write_text(traceback.format_exc())
        logging.exception("Movie workflow failed; wrote %s", err)
        raise


def parse_args(argv=None) -> MovieConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npix", type=int, default=1024)
    parser.add_argument("--alpha-min", type=int, default=0)
    parser.add_argument("--alpha-max", type=int, default=180)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--rerender", action="store_true")
    parser.add_argument("--maps-only", action="store_true", help="Only compute/cache NPZ projection maps; skip frames and videos.")
    parser.add_argument("--render-only", action="store_true", help="Require cached maps and only render frames/videos.")
    parser.add_argument("--output-dir", default=MovieConfig.output_dir)
    args = parser.parse_args(argv)
    return MovieConfig(
        output_dir=args.output_dir,
        npix=args.npix,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        fps=args.fps,
        recompute=args.recompute and not args.render_only,
        rerender=args.rerender,
        maps_only=args.maps_only,
        render_only=args.render_only,
    )


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
