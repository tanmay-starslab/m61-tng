#!/usr/bin/env python3
"""
Saved-alpha stellar off-axis particle projection movie for M61/TNG50 SID 488530.

This is the stellar-particle analogue of the gas saved-alpha movie:
- alpha = 0..180, mode = noflip, saved alpha LOS/north vectors from the recipes
- true galaxy center as projection center
- yt OffAxisParticleProjectionPlot for stellar surface density and stellar v_los
- Slurm-friendly one-alpha worker mode plus a render/video combine mode
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
from yt.visualization.particle_plots import OffAxisParticleProjectionPlot

import m61_oriented_HI_vlos_rotation_alpha5 as base


@dataclass
class Config:
    output_dir: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "saved_alpha_LOS_alpha000_180_stellar_particle_projection_movies"
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
    fps: int = 18
    recompute: bool = False
    rerender: bool = False
    subtract_systemic: bool = True

    @property
    def alphas(self) -> list[int]:
        return list(range(int(self.alpha_min), int(self.alpha_max) + 1))

    @property
    def half_width_kpc(self) -> float:
        return 0.5 * self.width_kpc

    @property
    def pixel_scale_kpc(self) -> float:
        return self.width_kpc / self.npix


def setup_paths(config: Config) -> dict[str, Path]:
    out = Path(config.output_dir)
    paths = {
        "out": out,
        "data": out / "data",
        "figures": out / "figures",
        "logs": out / "logs",
        "frames_combined": out / "frames" / "combined",
        "frames_sigma": out / "frames" / "stellar_surface_density",
        "frames_vlos": out / "frames" / "stellar_vlos",
        "videos": out / "videos",
        "sbatch_logs": out / "logs_sbatch",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(paths: dict[str, Path], label: str) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(paths["logs"] / f"{label}.log", mode="w")
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
    keys = {
        "los": ["los_x", "los_y", "los_z"],
        "north": ["north_x", "north_y", "north_z"],
    }[prefix]
    return np.asarray([row[k] for k in keys], dtype=float)


def npz_path(paths: dict[str, Path], alpha: int) -> Path:
    return paths["data"] / f"stellar_projection_alpha{alpha:03d}_saved_alpha_LOS_inner100kpc.npz"


def grid(config: Config):
    half = config.half_width_kpc
    dx = config.pixel_scale_kpc
    x = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    y = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, config.npix)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y


def load_rows(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rays = pd.read_csv(config.recipe_csv)
    orient = pd.read_csv(config.orient_csv)
    with open(config.orient_header) as f:
        header = json.load(f)
    wanted = set(config.alphas)
    rays = rays[
        (rays["SubhaloID"].astype(int) == config.sid)
        & (rays["sightline_id"].astype(str) == config.sightline_id)
        & (rays["mode"].astype(str) == config.mode)
        & rays["alpha_deg"].astype(int).isin(wanted)
    ].copy()
    orient = orient[
        (orient["mode"].astype(str) == config.mode)
        & orient["alpha_deg"].astype(int).isin(wanted)
    ].copy()
    if len(rays) != len(config.alphas) or len(orient) != len(config.alphas):
        raise RuntimeError(f"Expected {len(config.alphas)} rows, got rays={len(rays)} orient={len(orient)}")
    return rays.sort_values("alpha_deg"), orient.sort_values("alpha_deg"), header


def add_dynamic_stellar_los_field(ds, los_holder: dict[str, np.ndarray], systemic_kms: np.ndarray) -> None:
    if base.yt_field_exists(ds, ("PartType4", "stellar_vlos_saved_alpha")):
        return

    def _stellar_vlos(field, data):
        los = los_holder["los"]
        vx = data[("PartType4", "particle_velocity_x")].to("km/s") - data.ds.quan(systemic_kms[0], "km/s")
        vy = data[("PartType4", "particle_velocity_y")].to("km/s") - data.ds.quan(systemic_kms[1], "km/s")
        vz = data[("PartType4", "particle_velocity_z")].to("km/s") - data.ds.quan(systemic_kms[2], "km/s")
        return (los[0] * vx + los[1] * vy + los[2] * vz).to("km/s")

    ds.add_field(
        ("PartType4", "stellar_vlos_saved_alpha"),
        function=_stellar_vlos,
        sampling_type="particle",
        units="km/s",
    )


def particle_projection_frb(config, ds, center_ckpch, los, north, field, *, weight_field=None, density=False):
    plot = OffAxisParticleProjectionPlot(
        ds,
        normal=normalize(los, "projection LOS"),
        fields=field,
        center=ds.arr(center_ckpch, "code_length"),
        width=ds.quan(config.width_kpc * base.H_TNG, "code_length"),
        depth=ds.quan(config.width_kpc * base.H_TNG, "code_length"),
        weight_field=weight_field,
        deposition="cic",
        density=density,
        north_vector=normalize(north, "projection north"),
    )
    plot.set_buff_size((int(config.npix), int(config.npix)))
    return plot.frb[field]


def load_context(config: Config, paths: dict[str, Path]):
    base_config = base.AnalysisConfig(output_dir=str(paths["out"]), width_kpc=config.width_kpc, npix=config.npix)
    center_ckpch, center_meta = base.load_true_galaxy_center(base_config)
    cutout = base.find_cutout_h5(base_config)
    ds = yt.load(str(cutout))
    if config.subtract_systemic:
        systemic = base.systemic_velocity_from_hdf5(cutout, center_ckpch)
    else:
        systemic = {
            "velocity_kms": np.array([0.0, 0.0, 0.0]),
            "method": "systemic velocity subtraction disabled; raw stellar velocities",
            "center_subtracted": False,
        }
    los_holder = {"los": np.array([0.0, 0.0, 1.0])}
    add_dynamic_stellar_los_field(ds, los_holder, np.asarray(systemic["velocity_kms"], dtype=float))
    return ds, center_ckpch, center_meta, cutout, systemic, los_holder


def project_alpha(config: Config, alpha: int) -> Path:
    paths = setup_paths(config)
    setup_logging(paths, f"project_alpha{alpha:03d}")
    out = npz_path(paths, alpha)
    if out.exists() and not config.recompute:
        try:
            with np.load(out, allow_pickle=True) as z:
                _ = z.files
            logging.info("alpha %03d cached: %s", alpha, out)
            return out
        except (zipfile.BadZipFile, EOFError, ValueError, OSError) as exc:
            logging.warning("alpha %03d corrupt cache %s (%r); recomputing", alpha, out, exc)
            out.unlink(missing_ok=True)

    one = Config(**{**config.__dict__, "alpha_min": alpha, "alpha_max": alpha})
    rays, orient, header = load_rows(one)
    row = rays.iloc[0]
    orow = orient.iloc[0]
    los = normalize(row_vec(row, "los"), f"saved LOS alpha {alpha}")
    north = normalize(row_vec(orow, "north"), f"saved north alpha {alpha}")
    ds, center_ckpch, center_meta, cutout, systemic, los_holder = load_context(config, paths)
    los_holder["los"] = los
    x, y, X, Y = grid(config)

    sigma_arr = particle_projection_frb(
        config,
        ds,
        center_ckpch,
        los,
        north,
        ("PartType4", "particle_mass"),
        density=True,
    )
    vlos_arr = particle_projection_frb(
        config,
        ds,
        center_ckpch,
        los,
        north,
        ("PartType4", "stellar_vlos_saved_alpha"),
        weight_field=("PartType4", "particle_ones"),
        density=False,
    )
    sigma = base.yt_array_to_numpy(sigma_arr, "Msun/kpc**2")
    vlos = base.yt_array_to_numpy(vlos_arr, "km/s")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sigma = np.log10(np.where(sigma > 0, sigma, np.nan))
    vlos[~np.isfinite(vlos)] = np.nan
    x_qso = float(row["rho_kpc"]) * np.cos(np.deg2rad(float(row["phi_deg"])))
    y_qso = float(row["rho_kpc"]) * np.sin(np.deg2rad(float(row["phi_deg"])))
    np.savez_compressed(
        out,
        x_kpc=x,
        y_kpc=y,
        X_kpc=X,
        Y_kpc=Y,
        stellar_surface_density_msun_kpc2=sigma,
        log_stellar_surface_density=log_sigma,
        stellar_vlos_unweighted_mean_kms=vlos,
        alpha_deg=float(alpha),
        los_hat=los,
        north_hat=north,
        x_qso_kpc=x_qso,
        y_qso_kpc=y_qso,
        systemic_velocity_kms=np.asarray(systemic["velocity_kms"], dtype=float),
        projection_method="yt.OffAxisParticleProjectionPlot CIC; vlos weight_field=PartType4/particle_ones",
    )
    meta = {
        "alpha_deg": alpha,
        "output_npz": str(out),
        "cutout_h5": str(cutout),
        "galaxy_center_ckpch": center_ckpch,
        "galaxy_center_source": center_meta,
        "systemic_velocity": systemic,
        "recipe_row": row.to_dict(),
        "orient_row": orow.to_dict(),
        "orient_header": header,
        "npix": config.npix,
        "width_kpc": config.width_kpc,
    }
    write_json(paths["logs"] / f"project_alpha{alpha:03d}_metadata.json", meta)
    logging.info(
        "alpha %03d done: logSigma[p1,p99]=%s vlos[p2,p98]=%s -> %s",
        alpha,
        np.nanpercentile(log_sigma[np.isfinite(log_sigma)], [1, 99]).tolist(),
        np.nanpercentile(vlos[np.isfinite(vlos)], [2, 98]).tolist(),
        out,
    )
    return out


def load_cached_map(paths: dict[str, Path], alpha: int) -> dict[str, Any]:
    path = npz_path(paths, alpha)
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def robust_vlim(arrays, floor=30.0, ceiling=300.0) -> float:
    vals = []
    for arr in arrays:
        good = np.asarray(arr)[np.isfinite(arr)]
        if good.size:
            vals.append(np.abs(good))
    if not vals:
        return floor
    vals = np.concatenate(vals)
    return float(np.clip(np.nanpercentile(vals, 98), floor, ceiling))


def draw_map(ax, arr, cmap, vmin, vmax, alpha, qso, cbar_label, title):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white")
    im = ax.imshow(arr, origin="lower", extent=[-50, 50, -50, 50], cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.plot(0, 0, marker="+", ms=10, mew=1.5, color="cyan")
    ax.scatter([qso[0]], [qso[1]], s=40, facecolors="none", edgecolors="lime", lw=1.4)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(title)
    ax.minorticks_on()
    plt.colorbar(im, ax=ax, shrink=0.86, label=cbar_label)


def render_frames(config: Config, paths: dict[str, Path], maps: dict[int, dict[str, Any]], scales: dict[str, float]):
    for alpha in config.alphas:
        m = maps[alpha]
        qso = (float(m["x_qso_kpc"]), float(m["y_qso_kpc"]))
        log_sigma = m["log_stellar_surface_density"]
        vlos = m["stellar_vlos_unweighted_mean_kms"]
        combined = paths["frames_combined"] / f"frame_{alpha:03d}.png"
        sigma_frame = paths["frames_sigma"] / f"frame_{alpha:03d}.png"
        vlos_frame = paths["frames_vlos"] / f"frame_{alpha:03d}.png"
        if combined.exists() and sigma_frame.exists() and vlos_frame.exists() and not config.rerender:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        draw_map(
            axes[0],
            log_sigma,
            "inferno",
            scales["logSigma_vmin"],
            scales["logSigma_vmax"],
            alpha,
            qso,
            r"$\log_{10}\Sigma_\star\ [M_\odot\,{\rm kpc}^{-2}]$",
            f"stellar surface density, alpha={alpha:03d}",
        )
        draw_map(
            axes[1],
            vlos,
            "RdBu_r",
            -scales["vstar_vlim"],
            scales["vstar_vlim"],
            alpha,
            qso,
            r"mean stellar $v_{\rm los}$ [km/s]",
            "unweighted stellar-particle LOS velocity",
        )
        fig.savefig(combined, dpi=200)
        plt.close(fig)
        for frame, arr, cmap, vmin, vmax, label, title in [
            (
                sigma_frame,
                log_sigma,
                "inferno",
                scales["logSigma_vmin"],
                scales["logSigma_vmax"],
                r"$\log_{10}\Sigma_\star\ [M_\odot\,{\rm kpc}^{-2}]$",
                f"stellar surface density, alpha={alpha:03d}",
            ),
            (
                vlos_frame,
                vlos,
                "RdBu_r",
                -scales["vstar_vlim"],
                scales["vstar_vlim"],
                r"mean stellar $v_{\rm los}$ [km/s]",
                f"stellar vlos, alpha={alpha:03d}",
            ),
        ]:
            fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
            draw_map(ax, arr, cmap, vmin, vmax, alpha, qso, label, title)
            fig.savefig(frame, dpi=200)
            plt.close(fig)
        if alpha % 10 == 0:
            logging.info("Rendered stellar frames through alpha %03d", alpha)


def run_ffmpeg(config: Config, paths: dict[str, Path]) -> dict[str, str]:
    outputs = {
        "combined": (paths["frames_combined"], paths["videos"] / "m61_saved_alpha_000_180_stars_sigma_vlos_combined.mp4"),
        "stellar_surface_density": (paths["frames_sigma"], paths["videos"] / "m61_saved_alpha_000_180_stellar_surface_density.mp4"),
        "stellar_vlos": (paths["frames_vlos"], paths["videos"] / "m61_saved_alpha_000_180_stellar_vlos_unweighted.mp4"),
    }
    made = {}
    for name, (frame_dir, out) in outputs.items():
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(config.fps),
            "-start_number",
            str(config.alpha_min),
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
            str(out),
        ]
        logging.info("Running ffmpeg for %s -> %s", name, out)
        subprocess.run(cmd, check=True)
        made[name] = str(out)
    return made


def render_video(config: Config) -> dict[str, Any]:
    paths = setup_paths(config)
    setup_logging(paths, "render_video")
    maps = {}
    missing = []
    for alpha in config.alphas:
        path = npz_path(paths, alpha)
        if not path.exists():
            missing.append(alpha)
            continue
        try:
            maps[alpha] = load_cached_map(paths, alpha)
        except (zipfile.BadZipFile, EOFError, ValueError, OSError) as exc:
            raise RuntimeError(f"Corrupt NPZ for alpha {alpha}: {path} ({exc!r})") from exc
    if missing:
        raise RuntimeError(f"Missing stellar projection NPZ files for alphas: {missing}")
    log_vals = np.concatenate(
        [m["log_stellar_surface_density"][np.isfinite(m["log_stellar_surface_density"])].ravel() for m in maps.values()]
    )
    scales = {
        "logSigma_vmin": float(np.nanpercentile(log_vals, 1)),
        "logSigma_vmax": float(np.nanpercentile(log_vals, 99.5)),
        "vstar_vlim": robust_vlim([m["stellar_vlos_unweighted_mean_kms"] for m in maps.values()]),
    }
    logging.info("Common stellar plot scales: %s", scales)
    render_frames(config, paths, maps, scales)
    videos = run_ffmpeg(config, paths)
    metadata = {
        "SID": config.sid,
        "SNAP": config.snap,
        "mode": config.mode,
        "alpha_min": config.alpha_min,
        "alpha_max": config.alpha_max,
        "Npix": config.npix,
        "width_kpc": config.width_kpc,
        "pixel_scale_kpc": config.pixel_scale_kpc,
        "recipe_csv": config.recipe_csv,
        "orient_csv": config.orient_csv,
        "projection_method": "yt.OffAxisParticleProjectionPlot, CIC deposition; stellar vlos weighted by particle_ones",
        "subtract_systemic": config.subtract_systemic,
        "plot_scales": scales,
        "videos": videos,
        "output_paths": {k: str(v) for k, v in paths.items()},
    }
    write_json(paths["data"] / "stellar_saved_alpha_projection_movie_metadata.json", metadata)
    print("Stellar saved-alpha projection movie complete")
    print(f"output_dir = {paths['out']}")
    for key, value in videos.items():
        print(f"{key}: {value}")
    return {"paths": paths, "videos": videos, "metadata": metadata}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", default=Config.output_dir)
    common.add_argument("--npix", type=int, default=1024)
    common.add_argument("--alpha-min", type=int, default=0)
    common.add_argument("--alpha-max", type=int, default=180)
    common.add_argument("--fps", type=int, default=18)
    common.add_argument("--recompute", action="store_true")
    common.add_argument("--rerender", action="store_true")
    common.add_argument("--no-systemic", action="store_true")
    p_proj = sub.add_parser("project-alpha", parents=[common])
    p_proj.add_argument("--alpha", type=int, required=True)
    sub.add_parser("render-video", parents=[common])
    args = parser.parse_args(argv)
    config = Config(
        output_dir=args.output_dir,
        npix=args.npix,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        fps=args.fps,
        recompute=args.recompute,
        rerender=args.rerender,
        subtract_systemic=not args.no_systemic,
    )
    return args, config


def main(argv=None):
    args, config = parse_args(argv)
    try:
        if args.command == "project-alpha":
            return project_alpha(config, int(args.alpha))
        if args.command == "render-video":
            return render_video(config)
        raise RuntimeError(args.command)
    except Exception:
        paths = setup_paths(config)
        err = paths["logs"] / f"errors_{args.command}.txt"
        err.write_text(traceback.format_exc())
        logging.exception("Workflow failed; wrote %s", err)
        raise


if __name__ == "__main__":
    main()
