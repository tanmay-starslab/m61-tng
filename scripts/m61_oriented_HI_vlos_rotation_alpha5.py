#!/usr/bin/env python
"""
Oriented H I column, LOS velocity, and rotation-curve analysis for M61/TNG50.

This module is intentionally notebook-friendly: each major stage is exposed as a
function, while ``main()`` runs the full workflow from the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yt
from matplotlib.patches import Ellipse


H_TNG = 0.6774
M_P_G = 1.67262192369e-24
X_H = 0.76


@dataclass
class AnalysisConfig:
    subhalo_id: int = 488530
    sightline_id: str = "J122138+043026"
    alpha_deg: float = 5.0
    mode: str = "noflip"
    run_label: str = "L4Rvir"
    rho_kpc: float = 26.0
    rvir_kpc: float = 457.0
    inc_deg: float = 23.0
    width_kpc: float = 400.0
    npix: int = 2000
    logN_HI_min: float = 17.0
    annulus_dr_kpc: float = 5.0
    annulus_rmax_kpc: float = 200.0
    bootstrap_samples: int = 200
    bootstrap_max_pixels: int = 5000
    random_seed: int = 488530
    subtract_systemic_velocity: bool = True
    preferred_cutout_path: str = (
        "/scratch/tsingh65/TNG50-1_snap99/out_sub_488530/"
        "cutout_ALLFIELDS_sphere_2p1Rvir_sub488530.hdf5"
    )
    recipe_csv: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "rays_and_recipes_sid488530_snap99_L4Rvir/rays_sid488530.csv"
    )
    output_dir: str = (
        "/scratch/tsingh65/m61-tng/outputs/sid488530/"
        "oriented_HI_vlos_projection_alpha5_noflip"
    )


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def setup_output_dirs(config: AnalysisConfig) -> dict[str, Path]:
    out = Path(config.output_dir)
    paths = {
        "out": out,
        "figures": out / "figures",
        "data": out / "data",
        "logs": out / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(paths: dict[str, Path]) -> None:
    log_path = paths["logs"] / "run.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(json_sanitize(payload), f, indent=2, sort_keys=True)


def normalize(vec: Iterable[float], name: str = "vector") -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Cannot normalize {name}: {arr}")
    return arr / norm


def deterministic_plane_basis(los_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z_axis, los_hat))) > 0.95:
        z_axis = np.array([0.0, 1.0, 0.0])
    e1 = normalize(np.cross(z_axis, los_hat), "image-plane e1")
    e2 = normalize(np.cross(los_hat, e1), "image-plane e2")
    return e1, e2


def groupcat_chunk_number(path: Path) -> int:
    try:
        return int(path.name.split(".")[-2])
    except Exception:
        return -1


def find_group_catalog_files() -> list[Path]:
    roots = [
        Path("/scratch/tsingh65/TNG50-1_snap99"),
        Path("/scratch/tsingh65/TNG50-1"),
        Path("/scratch/tsingh65"),
        Path("/scratch/tsingh65/m61-tng"),
        Path("/scratch/tsingh65/m61-tng/outputs/sid488530"),
    ]
    patterns = [
        "groups_099/fof_subhalo_tab_099.*.hdf5",
        "groups_099/fof_subhalo_tab_099.0.hdf5",
        "fof_subhalo_tab_099.*.hdf5",
    ]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            files.extend(root.glob(pattern))
            if root.name not in {"tsingh65"}:
                files.extend(root.rglob(pattern))
        if files:
            break
    files = sorted(set(files), key=groupcat_chunk_number)
    return [f for f in files if f.is_file()]


def load_subhalo_pos_from_groupcat(subhalo_id: int) -> tuple[np.ndarray, dict[str, Any]]:
    files = find_group_catalog_files()
    if not files:
        raise FileNotFoundError("No TNG snap99 group catalog files found.")
    offset = 0
    for path in files:
        with h5py.File(path, "r") as h5:
            header = h5["Header"].attrs
            n_this = int(header.get("Nsubgroups_ThisFile", 0))
            if offset <= subhalo_id < offset + n_this:
                local = subhalo_id - offset
                if "Subhalo" not in h5 or "SubhaloPos" not in h5["Subhalo"]:
                    raise RuntimeError(f"{path} contains target chunk but no Subhalo/SubhaloPos field.")
                pos = np.asarray(h5["Subhalo/SubhaloPos"][local], dtype=float)
                meta = {
                    "source": "group_catalog",
                    "groupcat_file": str(path),
                    "chunk_number": groupcat_chunk_number(path),
                    "chunk_offset": offset,
                    "local_index": int(local),
                    "Nsubgroups_ThisFile": n_this,
                }
                return pos, meta
            offset += n_this
    raise IndexError(
        f"SubhaloID {subhalo_id} not found after accumulating {offset} subhalos "
        f"across {len(files)} group catalog chunks."
    )


def find_center_in_saved_metadata(subhalo_id: int) -> tuple[np.ndarray, dict[str, Any]] | None:
    root = Path("/scratch/tsingh65/m61-tng/outputs/sid488530")
    if not root.exists():
        return None
    keys = [
        "SubhaloPos",
        "subhalo_pos",
        "galaxy_center_ckpch",
        "center_ckpch",
        "center",
    ]
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.load(path.open())
        except Exception:
            continue
        stack = [payload]
        while stack:
            obj = stack.pop()
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in keys and isinstance(value, (list, tuple)) and len(value) == 3:
                        arr = np.asarray(value, dtype=float)
                        if np.all(np.isfinite(arr)):
                            return arr, {"source": "saved_metadata", "metadata_file": str(path), "key": key}
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(obj, list):
                stack.extend(v for v in obj if isinstance(v, (dict, list)))
    return None


def find_center_in_cutout_header(cutout_h5: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    candidate_attrs = ["SubhaloPos", "subhalo_pos", "Center", "center", "cutout_center", "CutoutCenter"]
    try:
        with h5py.File(cutout_h5, "r") as h5:
            groups = [h5]
            for group_name in ["Header", "CutoutInfo", "488530"]:
                if group_name in h5:
                    groups.append(h5[group_name])
            for group in groups:
                for key in candidate_attrs:
                    if key in group.attrs:
                        arr = np.asarray(group.attrs[key], dtype=float)
                        if arr.shape == (3,) and np.all(np.isfinite(arr)):
                            return arr, {"source": "cutout_header", "cutout_h5": str(cutout_h5), "attr": key}
    except Exception:
        return None
    return None


def load_true_galaxy_center(config: AnalysisConfig) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        return load_subhalo_pos_from_groupcat(config.subhalo_id)
    except Exception as exc:
        logging.warning("Group catalog SubhaloPos lookup failed: %r", exc)
    saved = find_center_in_saved_metadata(config.subhalo_id)
    if saved is not None:
        return saved
    cutout = Path(config.preferred_cutout_path)
    header = find_center_in_cutout_header(cutout)
    if header is not None:
        return header
    raise RuntimeError(
        "True galaxy center is required for projections and rotation curves. "
        "Could not load SubhaloPos from the group catalog, saved metadata, or cutout header. "
        "The ray anchor is intentionally not used as a fallback."
    )


def basis_from_los_and_pa(los_hat: np.ndarray, pa_deg: float) -> tuple[np.ndarray, np.ndarray, str]:
    east_like, north_like = deterministic_plane_basis(los_hat)
    pa_rad = np.deg2rad(pa_deg)
    e_major = normalize(
        np.sin(pa_rad) * east_like + np.cos(pa_rad) * north_like,
        "PA-rotated projected major axis",
    )
    e_minor = normalize(np.cross(los_hat, e_major), "projected minor axis")
    convention = (
        "e_major = sin(PA)*deterministic_east_like + cos(PA)*deterministic_north_like; "
        "this treats obs_pa_deg_used as east-of-north in the ray image plane because "
        "the recipe does not store an absolute sky north vector."
    )
    return e_major, e_minor, convention


def find_cutout_h5(config: AnalysisConfig) -> Path:
    preferred = Path(config.preferred_cutout_path)
    logging.info("Preferred cutout path: %s", preferred)
    if preferred.exists():
        logging.info("Selected cutout path: %s", preferred)
        return preferred

    roots = [
        Path("/scratch/tsingh65"),
        Path("/scratch/tsingh65/m61-tng"),
        Path("/scratch/tsingh65/TNG50-1_snap99"),
        Path("/scratch/tsingh65/m61-tng/cutouts"),
        Path("/scratch/tsingh65/m61-tng/outputs/sid488530"),
    ]
    patterns = [
        f"out_sub_{config.subhalo_id}/*.hdf5",
        "cutout*.hdf5",
        "*.hdf5",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.glob(pattern))
            candidates.extend(root.rglob(pattern))

    candidates = sorted(set(candidates), key=lambda p: (f"{config.subhalo_id}" not in str(p), len(str(p)), str(p)))
    candidates = [p for p in candidates if p.is_file() and p.suffix == ".hdf5"]
    if not candidates:
        searched = "\n".join(str(r) for r in roots)
        raise FileNotFoundError(f"No cutout HDF5 found for SID={config.subhalo_id}. Searched:\n{searched}")
    logging.info("Selected cutout path: %s", candidates[0])
    return candidates[0]


def load_recipe_and_geometry(config: AnalysisConfig, paths: dict[str, Path]) -> dict[str, Any]:
    recipe_csv = Path(config.recipe_csv)
    if not recipe_csv.exists():
        raise FileNotFoundError(f"Recipe CSV not found: {recipe_csv}")

    df = pd.read_csv(recipe_csv)
    mask = (
        (df["SubhaloID"].astype(int) == config.subhalo_id)
        & (df["sightline_id"].astype(str) == config.sightline_id)
        & np.isclose(df["alpha_deg"].astype(float), config.alpha_deg)
        & (df["mode"].astype(str) == config.mode)
    )
    if "Rvir_kpc" in df:
        mask &= np.isclose(df["Rvir_kpc"].astype(float), config.rvir_kpc)
    if "rho_kpc" in df:
        mask &= np.isclose(df["rho_kpc"].astype(float), config.rho_kpc)

    selected = df.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one matching recipe row, found {len(selected)} in {recipe_csv}")
    row = selected.iloc[0].to_dict()

    p0 = np.array([row["p0_X_ckpch_abs"], row["p0_Y_ckpch_abs"], row["p0_Z_ckpch_abs"]], dtype=float)
    p1 = np.array([row["p1_X_ckpch_abs"], row["p1_Y_ckpch_abs"], row["p1_Z_ckpch_abs"]], dtype=float)
    anchor = np.array(
        [row["anchor_X_ckpch_abs"], row["anchor_Y_ckpch_abs"], row["anchor_Z_ckpch_abs"]], dtype=float
    )
    saved_los_hat = normalize([row["los_x"], row["los_y"], row["los_z"]], "saved LOS")
    los_hat = normalize(p1 - p0, "p1-p0 LOS")
    dot = float(np.dot(saved_los_hat, los_hat))
    inc_deg = float(row["obs_inc_deg"])
    pa_deg = float(row["obs_pa_deg_used"])
    e1, e2, pa_convention = basis_from_los_and_pa(los_hat, pa_deg)
    galaxy_center, center_meta = load_true_galaxy_center(config)
    delta_anchor = anchor - galaxy_center
    delta_perp = delta_anchor - np.dot(delta_anchor, los_hat) * los_hat
    rho_check_kpc = float(np.linalg.norm(delta_perp) / H_TNG)
    x_qso_kpc = float(np.dot(delta_anchor, e1) / H_TNG)
    y_qso_kpc = float(np.dot(delta_anchor, e2) / H_TNG)

    geometry = {
        "recipe_csv": str(recipe_csv),
        "selected_recipe_row": row,
        "p0_ckpch": p0,
        "p1_ckpch": p1,
        "anchor_ckpch": anchor,
        "anchor_physical_kpc": anchor / H_TNG,
        "galaxy_center_ckpch": galaxy_center,
        "galaxy_center_physical_kpc": galaxy_center / H_TNG,
        "galaxy_center_source": center_meta,
        "saved_los_hat": saved_los_hat,
        "los_hat": los_hat,
        "los_from_p1_minus_p0_hat": los_hat,
        "dot_saved_los_with_p1_minus_p0": dot,
        "e1_hat": e1,
        "e2_hat": e2,
        "e1_e2_pa_convention": pa_convention,
        "normal_vector": los_hat,
        "rho_kpc": float(row["rho_kpc"]),
        "rho_check_kpc": rho_check_kpc,
        "x_qso_kpc": x_qso_kpc,
        "y_qso_kpc": y_qso_kpc,
        "Rvir_kpc": float(row["Rvir_kpc"]),
        "inc_deg": inc_deg,
        "pa_deg": pa_deg,
        "axis_ratio_q": float(np.cos(np.deg2rad(inc_deg))),
    }
    logging.info("Selected recipe row:\n%s", selected.to_string(index=False))
    logging.info("True galaxy center ckpc/h: %s", galaxy_center)
    logging.info("Galaxy center source: %s", center_meta)
    logging.info("LOS vector from p1-p0: %s", los_hat)
    logging.info("dot(saved_los, normalize(p1-p0)) = %.8f", dot)
    logging.info("Anchor ckpc/h: %s", anchor)
    logging.info("rho_check_kpc = %.6f", rho_check_kpc)
    logging.info("QSO marker x,y [kpc] = %.6f, %.6f", x_qso_kpc, y_qso_kpc)
    logging.info("inc=%.3f deg, PA=%.3f deg, q=cos(i)=%.6f", inc_deg, pa_deg, geometry["axis_ratio_q"])
    write_json(paths["data"] / "geometry_and_projection_metadata_alpha5_noflip_true_center.json", geometry)
    return geometry


def yt_field_exists(ds: yt.data_objects.static_output.Dataset, field: tuple[str, str]) -> bool:
    return field in ds.field_list or field in ds.derived_field_list


def add_field_if_missing(ds: yt.data_objects.static_output.Dataset, field: tuple[str, str], **kwargs: Any) -> None:
    if yt_field_exists(ds, field):
        return
    ds.add_field(field, **kwargs)


def ensure_gas_alias_fields(ds: yt.data_objects.static_output.Dataset) -> dict[str, Any]:
    info: dict[str, Any] = {"added_fields": [], "notes": []}

    if not yt_field_exists(ds, ("gas", "metallicity")) and yt_field_exists(ds, ("PartType0", "GFM_Metallicity")):
        def _metallicity(field, data):
            return data[("PartType0", "GFM_Metallicity")]

        ds.add_field(("gas", "metallicity"), function=_metallicity, sampling_type="particle", units="dimensionless")
        info["added_fields"].append(("gas", "metallicity"))

    if not yt_field_exists(ds, ("gas", "neutral_hydrogen_fraction")) and yt_field_exists(
        ds, ("PartType0", "NeutralHydrogenAbundance")
    ):
        def _nh_frac(field, data):
            return data[("PartType0", "NeutralHydrogenAbundance")]

        ds.add_field(
            ("gas", "neutral_hydrogen_fraction"),
            function=_nh_frac,
            sampling_type="particle",
            units="dimensionless",
        )
        info["added_fields"].append(("gas", "neutral_hydrogen_fraction"))

    if not yt_field_exists(ds, ("gas", "H_number_density")):
        def _h_number_density(field, data):
            return (X_H * data[("gas", "density")] / data.ds.quan(M_P_G, "g")).to("cm**-3")

        ds.add_field(
            ("gas", "H_number_density"),
            function=_h_number_density,
            sampling_type="particle",
            units="cm**-3",
        )
        info["added_fields"].append(("gas", "H_number_density"))

    hi_fields = [
        ("gas", "H_p0_number_density"),
        ("gas", "H_p0_number_density"),
        ("gas", "H_I_number_density"),
    ]
    if not any(yt_field_exists(ds, f) for f in hi_fields):
        try:
            import trident

            trident.add_ion_fields(ds, ions=["H I"])
            info["notes"].append("Added H I fields with trident.add_ion_fields.")
        except Exception as exc:
            info["notes"].append(f"trident.add_ion_fields failed; falling back to TNG neutral fraction: {exc!r}")

    if not yt_field_exists(ds, ("gas", "H_p0_number_density")):
        if not yt_field_exists(ds, ("gas", "neutral_hydrogen_fraction")):
            raise RuntimeError("Cannot derive H_p0_number_density: neutral hydrogen fraction field is missing.")

        def _hi_number_density(field, data):
            return (data[("gas", "neutral_hydrogen_fraction")] * data[("gas", "H_number_density")]).to("cm**-3")

        ds.add_field(
            ("gas", "H_p0_number_density"),
            function=_hi_number_density,
            sampling_type="particle",
            units="cm**-3",
        )
        info["added_fields"].append(("gas", "H_p0_number_density"))

    def _hi_mass_density(field, data):
        return (data[("gas", "H_p0_number_density")] * data.ds.quan(M_P_G, "g")).to("g/cm**3")

    add_field_if_missing(
        ds,
        ("gas", "HI_mass_density"),
        function=_hi_mass_density,
        sampling_type="particle",
        units="g/cm**3",
    )
    if ("gas", "HI_mass_density") not in info["added_fields"]:
        info["added_fields"].append(("gas", "HI_mass_density"))

    return info


def compute_weighted_median(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        good &= np.isfinite(weights) & (weights > 0)
    values = values[good]
    if values.size == 0:
        return np.nan
    if weights is None:
        return float(np.median(values))
    weights = weights[good]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cutoff = 0.5 * cdf[-1]
    return float(values[np.searchsorted(cdf, cutoff)])


def systemic_velocity_from_hdf5(cutout_h5: Path, center_ckpch: np.ndarray, radius_kpc: float = 5.0) -> dict[str, Any]:
    radius_ckpch = radius_kpc * H_TNG
    radius2 = radius_ckpch**2
    chunk = 1_000_000
    gas_vels: list[np.ndarray] = []
    gas_weights: list[np.ndarray] = []
    star_vels: list[np.ndarray] = []

    with h5py.File(cutout_h5, "r") as h5:
        if "PartType0" in h5 and "Coordinates" in h5["PartType0"] and "Velocities" in h5["PartType0"]:
            coords = h5["PartType0/Coordinates"]
            vels = h5["PartType0/Velocities"]
            weights_ds = h5["PartType0/Density"] if "Density" in h5["PartType0"] else None
            for i0 in range(0, coords.shape[0], chunk):
                i1 = min(i0 + chunk, coords.shape[0])
                delta = coords[i0:i1] - center_ckpch[None, :]
                mask = np.sum(delta * delta, axis=1) <= radius2
                if np.any(mask):
                    gas_vels.append(np.asarray(vels[i0:i1][mask], dtype=float))
                    if weights_ds is not None:
                        gas_weights.append(np.asarray(weights_ds[i0:i1][mask], dtype=float))

        if "PartType4" in h5 and "Coordinates" in h5["PartType4"] and "Velocities" in h5["PartType4"]:
            coords = h5["PartType4/Coordinates"]
            vels = h5["PartType4/Velocities"]
            for i0 in range(0, coords.shape[0], chunk):
                i1 = min(i0 + chunk, coords.shape[0])
                delta = coords[i0:i1] - center_ckpch[None, :]
                mask = np.sum(delta * delta, axis=1) <= radius2
                if np.any(mask):
                    star_vels.append(np.asarray(vels[i0:i1][mask], dtype=float))

    if gas_vels:
        gas = np.vstack(gas_vels)
        weights = np.concatenate(gas_weights) if gas_weights else None
        if weights is not None and np.any(np.isfinite(weights) & (weights > 0)):
            vec = np.array([compute_weighted_median(gas[:, j], weights) for j in range(3)])
            method = f"gas-density-weighted median velocity within {radius_kpc:g} physical kpc of true galaxy center"
        else:
            vec = np.nanmedian(gas, axis=0)
            method = f"median gas velocity within {radius_kpc:g} physical kpc of true galaxy center"
        return {"velocity_kms": vec, "method": method, "n_gas": int(gas.shape[0]), "n_stars": 0, "center_subtracted": True}

    if star_vels:
        stars = np.vstack(star_vels)
        vec = np.nanmedian(stars, axis=0)
        return {
            "velocity_kms": vec,
            "method": f"median stellar velocity within {radius_kpc:g} physical kpc of true galaxy center",
            "n_gas": 0,
            "n_stars": int(stars.shape[0]),
            "center_subtracted": True,
        }

    return {
        "velocity_kms": np.array([0.0, 0.0, 0.0]),
        "method": "no gas or stellar particles found within central aperture; vlos is not center-subtracted",
        "n_gas": 0,
        "n_stars": 0,
        "center_subtracted": False,
    }


def add_velocity_fields(
    ds: yt.data_objects.static_output.Dataset,
    los_hat: np.ndarray,
    systemic_velocity_kms: np.ndarray,
) -> dict[str, Any]:
    raw_vector_field = ("PartType0", "Velocities")
    component_candidates = [
        (("gas", "velocity_x"), ("gas", "velocity_y"), ("gas", "velocity_z")),
        (("PartType0", "velocity_x"), ("PartType0", "velocity_y"), ("PartType0", "velocity_z")),
        (("PartType0", "particle_velocity_x"), ("PartType0", "particle_velocity_y"), ("PartType0", "particle_velocity_z")),
    ]
    component_fields = None
    for cand in component_candidates:
        if all(yt_field_exists(ds, f) for f in cand):
            component_fields = cand
            break
    if component_fields is None and not yt_field_exists(ds, raw_vector_field):
        raise RuntimeError("Could not find gas velocity component fields or PartType0/Velocities.")

    def velocity_component(data, index: int):
        if component_fields is not None:
            return data[component_fields[index]].to("km/s")
        return data[raw_vector_field][:, index].to("km/s")

    def _velocity_los(field, data):
        vx = velocity_component(data, 0) - data.ds.quan(systemic_velocity_kms[0], "km/s")
        vy = velocity_component(data, 1) - data.ds.quan(systemic_velocity_kms[1], "km/s")
        vz = velocity_component(data, 2) - data.ds.quan(systemic_velocity_kms[2], "km/s")
        return (los_hat[0] * vx + los_hat[1] * vy + los_hat[2] * vz).to("km/s")

    if yt_field_exists(ds, ("gas", "velocity_los_oriented")):
        logging.warning("Field ('gas','velocity_los_oriented') already exists; using existing field.")
    else:
        ds.add_field(
            ("gas", "velocity_los_oriented"),
            function=_velocity_los,
            sampling_type="particle",
            units="km/s",
        )

    def _hi_vlos_integrand(field, data):
        return data[("gas", "H_p0_number_density")] * data[("gas", "velocity_los_oriented")]

    add_field_if_missing(
        ds,
        ("gas", "HI_vlos_integrand"),
        function=_hi_vlos_integrand,
        sampling_type="particle",
        units="km/(s*cm**3)",
    )

    def _gas_density_vlos_integrand(field, data):
        return data[("gas", "density")] * data[("gas", "velocity_los_oriented")]

    add_field_if_missing(
        ds,
        ("gas", "gas_density_vlos_integrand"),
        function=_gas_density_vlos_integrand,
        sampling_type="particle",
        units="g*km/(s*cm**3)",
    )

    return {
        "velocity_source": component_fields if component_fields is not None else raw_vector_field,
        "systemic_velocity_kms": systemic_velocity_kms,
    }


def load_dataset_and_define_fields(
    config: AnalysisConfig,
    geometry: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[yt.data_objects.static_output.Dataset, dict[str, Any]]:
    cutout_h5 = find_cutout_h5(config)
    logging.info("Loading yt dataset: %s", cutout_h5)
    ds = yt.load(str(cutout_h5))
    logging.info("Dataset type: %s", type(ds).__name__)

    field_info = ensure_gas_alias_fields(ds)
    if config.subtract_systemic_velocity:
        systemic = systemic_velocity_from_hdf5(cutout_h5, geometry["galaxy_center_ckpch"])
    else:
        systemic = {
            "velocity_kms": np.array([0.0, 0.0, 0.0]),
            "method": "systemic velocity subtraction disabled by config; using raw simulation velocities",
            "n_gas": 0,
            "n_stars": 0,
            "center_subtracted": False,
        }
    field_info["systemic_velocity"] = systemic
    vel_info = add_velocity_fields(ds, geometry["los_hat"], np.asarray(systemic["velocity_kms"], dtype=float))
    field_info.update(vel_info)
    field_info["cutout_h5"] = str(cutout_h5)

    required = [
        ("gas", "density"),
        ("gas", "metallicity"),
        ("gas", "H_number_density"),
        ("gas", "H_p0_number_density"),
        ("gas", "HI_mass_density"),
        ("gas", "velocity_los_oriented"),
        ("gas", "HI_vlos_integrand"),
        ("gas", "gas_density_vlos_integrand"),
    ]
    missing = [f for f in required if not yt_field_exists(ds, f)]
    if missing:
        raise RuntimeError(f"Missing required fields after setup: {missing}")

    relevant = [
        f
        for f in sorted(set(ds.field_list + ds.derived_field_list))
        if f[0] in {"gas", "PartType0"} and any(
            token in f[1].lower()
            for token in ["density", "temperature", "metal", "hydrogen", "velocity", "neutral", "h_p0"]
        )
    ]
    field_info["relevant_fields"] = relevant[:200]
    logging.info("Relevant H I / velocity fields: %s", relevant[:80])
    logging.info("Systemic velocity used: %s", systemic)

    write_json(paths["data"] / "field_and_systemic_metadata_alpha5_noflip.json", field_info)
    return ds, field_info


def off_axis_integral(
    ds: yt.data_objects.static_output.Dataset,
    center,
    normal_vector: np.ndarray,
    width,
    npix: int,
    field: tuple[str, str],
    north_vector: np.ndarray,
):
    kwargs = dict(north_vector=north_vector, no_ghost=False, method="integrate")
    resolution = (int(npix), int(npix))
    try:
        return yt.off_axis_projection(ds, center, normal_vector, width, resolution, field, **kwargs)
    except TypeError:
        kwargs.pop("method", None)
        try:
            return yt.off_axis_projection(ds, center, normal_vector, width, resolution, field, **kwargs)
        except TypeError:
            return yt.off_axis_projection(ds, center, normal_vector, width, resolution, field)


def yt_array_to_numpy(arr, units: str | None = None) -> np.ndarray:
    if units is not None and hasattr(arr, "to"):
        arr = arr.to(units)
    if hasattr(arr, "d"):
        return np.asarray(arr.d, dtype=float)
    return np.asarray(arr, dtype=float)


def finite_percentiles(values: np.ndarray, percentiles: Iterable[float]) -> list[float | None]:
    good = np.asarray(values)[np.isfinite(values)]
    if good.size == 0:
        return [None for _ in percentiles]
    return [float(x) for x in np.nanpercentile(good, list(percentiles))]


def save_map_npz(path: Path, **arrays: Any) -> None:
    np.savez_compressed(path, **arrays)
    logging.info("Saved %s", path)


def plot_scalar_map(
    data: np.ndarray,
    extent: list[float],
    path_base: Path,
    cmap: str,
    label: str,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    qso_xy_kpc: tuple[float, float] | None = None,
    rho_check_kpc: float | None = None,
    ellipses: bool = False,
    inc_deg: float | None = None,
) -> None:
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12})
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.94)
    cbar.set_label(label)
    ax.axhline(0, color="white", lw=0.7, alpha=0.45)
    ax.axvline(0, color="white", lw=0.7, alpha=0.45)
    ax.plot(0, 0, marker="+", ms=11, mew=1.8, color="cyan", label="galaxy center")
    if qso_xy_kpc is not None:
        ax.scatter(
            [qso_xy_kpc[0]],
            [qso_xy_kpc[1]],
            s=46,
            facecolors="none",
            edgecolors="lime",
            lw=1.5,
            label="QSO sightline",
        )
        if rho_check_kpc is not None:
            ax.text(
                0.03,
                0.04,
                fr"$\rho_{{check}}={rho_check_kpc:.1f}$ kpc",
                transform=ax.transAxes,
                color="white",
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=3),
            )
    if ellipses and inc_deg is not None:
        add_ellipse_annuli(ax, inc_deg)
    ax.set_xlabel("Projected x [kpc]")
    ax.set_ylabel("Projected y [kpc]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    for suffix in ("png", "pdf"):
        out = path_base.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=300)
        logging.info("Saved %s", out)
    plt.close(fig)


def add_ellipse_annuli(
    ax,
    inc_deg: float,
    levels: tuple[float, ...] = (10, 20, 30, 50, 75, 100, 150, 200),
    color: str = "white",
    alpha: float = 0.78,
) -> None:
    q = float(np.cos(np.deg2rad(inc_deg)))
    for radius in levels:
        ellipse = Ellipse(
            (0.0, 0.0),
            width=2.0 * radius,
            height=2.0 * radius * q,
            angle=0.0,
            fill=False,
            color="cyan" if np.isclose(radius, 26.0) else color,
            lw=1.4 if radius in (20, 30) else 0.9,
            alpha=alpha,
        )
        ax.add_patch(ellipse)


def make_projections(
    config: AnalysisConfig,
    ds: yt.data_objects.static_output.Dataset,
    geometry: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    center = ds.arr(geometry["galaxy_center_ckpch"], "code_length")
    # This local yt/unyt stack can fail converting cosmological kpc to
    # code_length with an unsupported float128 dtype. At z~0 for TNG, physical
    # kpc = (ckpc/h) / h, so 400 physical kpc is 400*h code_length.
    width = ds.arr([config.width_kpc * H_TNG, config.width_kpc * H_TNG], "code_length")
    normal = geometry["normal_vector"]
    north = geometry["e2_hat"]
    npix = int(config.npix)
    extent = [-config.width_kpc / 2, config.width_kpc / 2, -config.width_kpc / 2, config.width_kpc / 2]
    qso_xy = (float(geometry["x_qso_kpc"]), float(geometry["y_qso_kpc"]))

    logging.info("Projection center passed to yt as code_length: %s", center)
    logging.info("Projection width: %s; npix=%d", width, npix)

    hi_den = off_axis_integral(ds, center, normal, width, npix, ("gas", "H_p0_number_density"), north)
    hi_num = off_axis_integral(ds, center, normal, width, npix, ("gas", "HI_vlos_integrand"), north)
    gas_den = off_axis_integral(ds, center, normal, width, npix, ("gas", "density"), north)
    gas_num = off_axis_integral(ds, center, normal, width, npix, ("gas", "gas_density_vlos_integrand"), north)

    N_HI = yt_array_to_numpy(hi_den, "cm**-2")
    HI_vlos_num = yt_array_to_numpy(hi_num, "km/(s*cm**2)")
    gas_sigma = yt_array_to_numpy(gas_den, "g/cm**2")
    gas_vlos_num = yt_array_to_numpy(gas_num, "g*km/(s*cm**2)")

    with np.errstate(divide="ignore", invalid="ignore"):
        vlos_hi = HI_vlos_num / N_HI
        vlos_gas = gas_vlos_num / gas_sigma
        logN_HI = np.log10(np.where(N_HI > 0, N_HI, np.nan))

    vlos_hi[~np.isfinite(vlos_hi)] = np.nan
    vlos_gas[~np.isfinite(vlos_gas)] = np.nan

    stats = {
        "N_HI_cm2_shape": list(N_HI.shape),
        "logN_HI_percentiles_0_1_2_50_98_99_100": finite_percentiles(logN_HI, [0, 1, 2, 50, 98, 99, 100]),
        "vlos_HI_percentiles_0_1_2_50_98_99_100": finite_percentiles(vlos_hi, [0, 1, 2, 50, 98, 99, 100]),
        "vlos_gas_percentiles_0_1_2_50_98_99_100": finite_percentiles(vlos_gas, [0, 1, 2, 50, 98, 99, 100]),
    }
    logging.info("Projection array shapes: N_HI=%s, vlos_HI=%s, vlos_gas=%s", N_HI.shape, vlos_hi.shape, vlos_gas.shape)
    logging.info("Projection stats: %s", stats)

    title = (
        "M61 / TNG50-1 snap99 / SID 488530 / "
        f"alpha={config.alpha_deg:g} / {config.mode} / inc={geometry['inc_deg']:.0f} deg"
    )
    save_map_npz(
        paths["data"] / "HI_column_density_projection_alpha5_noflip.npz",
        N_HI_cm2=N_HI,
        logN_HI=logN_HI,
        extent_kpc=np.asarray(extent),
        los_hat=geometry["los_hat"],
        saved_los_hat=geometry["saved_los_hat"],
        e1_hat=geometry["e1_hat"],
        e2_hat=geometry["e2_hat"],
        galaxy_center_ckpch=geometry["galaxy_center_ckpch"],
        anchor_ckpch=geometry["anchor_ckpch"],
        x_qso_kpc=qso_xy[0],
        y_qso_kpc=qso_xy[1],
        rho_check_kpc=geometry["rho_check_kpc"],
    )
    save_map_npz(
        paths["data"] / "vlos_HI_weighted_projection_alpha5_noflip.npz",
        vlos_HIweighted_kms=vlos_hi,
        N_HI_cm2=N_HI,
        extent_kpc=np.asarray(extent),
    )
    save_map_npz(
        paths["data"] / "vlos_gas_density_weighted_projection_alpha5_noflip.npz",
        vlos_gas_density_weighted_kms=vlos_gas,
        gas_surface_density_g_cm2=gas_sigma,
        extent_kpc=np.asarray(extent),
    )

    hi_good = logN_HI[np.isfinite(logN_HI)]
    hi_vmin = max(12.0, float(np.nanpercentile(hi_good, 1))) if hi_good.size else 12.0
    hi_vmax = float(np.nanpercentile(hi_good, 99)) if hi_good.size else 22.0
    plot_scalar_map(
        logN_HI,
        extent,
        paths["figures"] / "HI_column_density_projection_alpha5_noflip",
        "magma",
        r"$\log_{10}\,N_{\rm HI}\ [{\rm cm}^{-2}]$",
        title,
        vmin=hi_vmin,
        vmax=hi_vmax,
        qso_xy_kpc=qso_xy,
        rho_check_kpc=geometry["rho_check_kpc"],
    )
    plot_scalar_map(
        logN_HI,
        extent,
        paths["figures"] / "HI_projection_true_center_with_qso_marker",
        "magma",
        r"$\log_{10}\,N_{\rm HI}\ [{\rm cm}^{-2}]$",
        title,
        vmin=hi_vmin,
        vmax=hi_vmax,
        qso_xy_kpc=qso_xy,
        rho_check_kpc=geometry["rho_check_kpc"],
    )

    for data, name, label in [
        (vlos_hi, "vlos_HI_weighted_projection_alpha5_noflip", r"$v_{\rm los,HI-weighted}$ [km s$^{-1}$]"),
        (
            vlos_gas,
            "vlos_gas_density_weighted_projection_alpha5_noflip",
            r"$v_{\rm los,gas-density-weighted}$ [km s$^{-1}$]",
        ),
    ]:
        good = data[np.isfinite(data)]
        if good.size:
            lo, hi = np.nanpercentile(good, [2, 98])
            vmax = min(300.0, max(abs(float(lo)), abs(float(hi))))
            if vmax < 20:
                vmax = max(abs(float(np.nanpercentile(good, 1))), abs(float(np.nanpercentile(good, 99))))
            vmax = float(max(vmax, 1.0))
        else:
            vmax = 300.0
        plot_scalar_map(
            data,
            extent,
            paths["figures"] / name,
            "RdBu_r",
            label,
            title,
            vmin=-vmax,
            vmax=vmax,
            qso_xy_kpc=qso_xy,
            rho_check_kpc=geometry["rho_check_kpc"],
        )

    plot_scalar_map(
        logN_HI,
        extent,
        paths["figures"] / "ellipse_annuli_overlay_alpha5_noflip_true_center",
        "magma",
        r"$\log_{10}\,N_{\rm HI}\ [{\rm cm}^{-2}]$",
        title,
        vmin=hi_vmin,
        vmax=hi_vmax,
        qso_xy_kpc=qso_xy,
        rho_check_kpc=geometry["rho_check_kpc"],
        ellipses=True,
        inc_deg=geometry["inc_deg"],
    )
    hi_vmax_vel = float(max(abs(x) for x in finite_percentiles(vlos_hi, [2, 98]) if x is not None))
    hi_vmax_vel = min(max(hi_vmax_vel, 1.0), 300.0)
    plot_scalar_map(
        vlos_hi,
        extent,
        paths["figures"] / "vlos_HI_weighted_with_ellipses_alpha5_noflip_true_center",
        "RdBu_r",
        r"$v_{\rm los,HI-weighted}$ [km s$^{-1}$]",
        title,
        vmin=-hi_vmax_vel,
        vmax=hi_vmax_vel,
        qso_xy_kpc=qso_xy,
        rho_check_kpc=geometry["rho_check_kpc"],
        ellipses=True,
        inc_deg=geometry["inc_deg"],
    )

    maps = {
        "N_HI_cm2": N_HI,
        "logN_HI": logN_HI,
        "vlos_HIweighted_kms": vlos_hi,
        "vlos_gas_density_weighted_kms": vlos_gas,
        "gas_surface_density_g_cm2": gas_sigma,
        "extent_kpc": np.asarray(extent),
        "stats": stats,
        "x_qso_kpc": qso_xy[0],
        "y_qso_kpc": qso_xy[1],
        "rho_check_kpc": geometry["rho_check_kpc"],
    }
    write_json(
        paths["data"] / "projection_metadata_alpha5_noflip_true_center.json",
        {
            "stats": stats,
            "config": asdict(config),
            "galaxy_center_ckpch": geometry["galaxy_center_ckpch"],
            "anchor_ckpch": geometry["anchor_ckpch"],
            "x_qso_kpc": geometry["x_qso_kpc"],
            "y_qso_kpc": geometry["y_qso_kpc"],
            "rho_check_kpc": geometry["rho_check_kpc"],
        },
    )
    return maps


def image_plane_grid(npix: int, width_kpc: float) -> tuple[np.ndarray, np.ndarray]:
    dx = width_kpc / npix
    centers = np.linspace(-width_kpc / 2 + dx / 2, width_kpc / 2 - dx / 2, npix)
    return np.meshgrid(centers, centers, indexing="xy")


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan, np.nan
    values = values[good]
    weights = weights[good]
    mean = float(np.average(values, weights=weights))
    var = float(np.average((values - mean) ** 2, weights=weights))
    return mean, math.sqrt(max(var, 0.0))


def tilted_ring_fit(v: np.ndarray, cos_theta: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    good = np.isfinite(v) & np.isfinite(cos_theta) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(good) < 5:
        return np.nan, np.nan, np.nan
    v = v[good]
    c = cos_theta[good]
    w = weights[good]
    x = np.vstack([c, np.ones_like(c)]).T
    sw = np.sqrt(w / np.nanmedian(w))
    xw = x * sw[:, None]
    vw = v * sw
    try:
        beta, *_ = np.linalg.lstsq(xw, vw, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    a, b = float(beta[0]), float(beta[1])
    resid = v - (a * c + b)
    _, scatter = weighted_mean_std(resid, w)
    return a, b, scatter


def bootstrap_tilted_vrot(
    v: np.ndarray,
    cos_theta: np.ndarray,
    weights: np.ndarray,
    sin_inc: float,
    rng: np.random.Generator,
    n_boot: int,
    max_pixels: int,
) -> float:
    good = np.isfinite(v) & np.isfinite(cos_theta) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(good) < 20 or n_boot <= 0:
        return np.nan
    v = v[good]
    c = cos_theta[good]
    w = weights[good]
    n = v.size
    p = w / np.sum(w)
    sample_n = min(n, max_pixels)
    estimates = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=sample_n, replace=True, p=p)
        a, _, _ = tilted_ring_fit(v[idx], c[idx], w[idx])
        if np.isfinite(a):
            estimates.append(abs(a) / sin_inc)
    if len(estimates) < max(10, n_boot // 10):
        return np.nan
    return float(np.nanstd(estimates, ddof=1))


def run_rotation_curve_analysis(
    config: AnalysisConfig,
    maps: dict[str, Any],
    geometry: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return run_rotation_curve_for_threshold(config, maps, geometry, paths, config.logN_HI_min, save_outputs=True)


def run_rotation_curve_for_threshold(
    config: AnalysisConfig,
    maps: dict[str, Any],
    geometry: dict[str, Any],
    paths: dict[str, Path],
    logN_HI_min: float,
    save_outputs: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    vmap = maps["vlos_HIweighted_kms"]
    logN = maps["logN_HI"]
    N_HI = maps["N_HI_cm2"]
    npix = int(vmap.shape[0])
    x_proj, y_proj = image_plane_grid(npix, config.width_kpc)
    inc_rad = np.deg2rad(float(geometry["inc_deg"]))
    sin_inc = float(np.sin(inc_rad))
    cos_inc = float(np.cos(inc_rad))
    if sin_inc <= 0:
        raise ValueError(f"Invalid inclination for deprojection: {geometry['inc_deg']}")

    x_major = x_proj
    y_minor = y_proj
    y_disk = y_minor / cos_inc
    r_disk = np.sqrt(x_major**2 + y_disk**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(r_disk > 0, x_major / r_disk, np.nan)

    bins = np.arange(0, config.annulus_rmax_kpc + config.annulus_dr_kpc, config.annulus_dr_kpc)
    rng = np.random.default_rng(config.random_seed)
    rows = []
    for r0, r1 in zip(bins[:-1], bins[1:]):
        mask = (
            (r_disk >= r0)
            & (r_disk < r1)
            & np.isfinite(vmap)
            & np.isfinite(logN)
            & (logN >= logN_HI_min)
            & np.isfinite(N_HI)
            & (N_HI > 0)
        )
        n = int(np.count_nonzero(mask))
        logging.info(
            "Annulus %.1f-%.1f kpc, logN_HI_min=%.1f: N_valid=%d",
            r0,
            r1,
            logN_HI_min,
            n,
        )
        r_mid = 0.5 * (r0 + r1)
        if n < 5:
            rows.append(
                {
                    "R_inner_kpc": r0,
                    "R_outer_kpc": r1,
                    "R_mid_kpc": r_mid,
                    "N_pixels": n,
                    "v_sys_kms": np.nan,
                    "A_kms": np.nan,
                    "vrot_tilted_kms": np.nan,
                    "abs_vrot_tilted_kms": np.nan,
                    "weighted_residual_sigma_kms": np.nan,
                    "sigma_vrot_kms": np.nan,
                    "simple_abs_vrot_kms": np.nan,
                    "median_logN_HI": np.nan,
                    "mean_logN_HI": np.nan,
                }
            )
            continue
        v = vmap[mask]
        w = N_HI[mask]
        c = cos_theta[mask]
        mean_v, std_v = weighted_mean_std(v, w)
        med_v = compute_weighted_median(v, w)
        mean_abs = float(np.average(np.abs(v), weights=w))
        simple_vrot = mean_abs / sin_inc
        a, b, resid_scatter = tilted_ring_fit(v, c, w)
        tilted_vrot = a / sin_inc if np.isfinite(a) else np.nan
        abs_tilted_vrot = abs(tilted_vrot) if np.isfinite(tilted_vrot) else np.nan
        boot_sigma = bootstrap_tilted_vrot(
            v,
            c,
            w,
            sin_inc,
            rng,
            config.bootstrap_samples,
            config.bootstrap_max_pixels,
        )
        sigma_vrot = boot_sigma if np.isfinite(boot_sigma) else resid_scatter / sin_inc
        rows.append(
            {
                "R_inner_kpc": r0,
                "R_outer_kpc": r1,
                "R_mid_kpc": r_mid,
                "N_pixels": n,
                "v_sys_kms": b,
                "A_kms": a,
                "vrot_tilted_kms": tilted_vrot,
                "abs_vrot_tilted_kms": abs_tilted_vrot,
                "weighted_residual_sigma_kms": resid_scatter,
                "sigma_vrot_kms": sigma_vrot,
                "simple_abs_vrot_kms": simple_vrot,
                "median_logN_HI": float(np.nanmedian(logN[mask])),
                "mean_logN_HI": float(np.nanmean(logN[mask])),
            }
        )

    table = pd.DataFrame(rows)

    rho = float(geometry["rho_kpc"])
    good_tilt = np.isfinite(table["abs_vrot_tilted_kms"].values)
    good_simple = np.isfinite(table["simple_abs_vrot_kms"].values)
    good_sigma = np.isfinite(table["sigma_vrot_kms"].values)
    v_tilt_rho = float(np.interp(rho, table.loc[good_tilt, "R_mid_kpc"], table.loc[good_tilt, "abs_vrot_tilted_kms"])) if np.any(good_tilt) else np.nan
    v_simple_rho = float(np.interp(rho, table.loc[good_simple, "R_mid_kpc"], table.loc[good_simple, "simple_abs_vrot_kms"])) if np.any(good_simple) else np.nan
    sigma_rho = float(np.interp(rho, table.loc[good_sigma, "R_mid_kpc"], table.loc[good_sigma, "sigma_vrot_kms"])) if np.any(good_sigma) else np.nan

    ism = {
        "rho_kpc": rho,
        "vrot_tilted_at_rho_kms": v_tilt_rho,
        "simple_abs_vrot_at_rho_kms": v_simple_rho,
        "sigma_at_rho_kms": sigma_rho,
        "inc_deg": float(geometry["inc_deg"]),
        "sin_inc": sin_inc,
        "logN_HI_min": logN_HI_min,
        "rho_check_kpc": geometry["rho_check_kpc"],
        "x_qso_kpc": geometry["x_qso_kpc"],
        "y_qso_kpc": geometry["y_qso_kpc"],
        "method_notes": (
            "True galaxy center is the origin. Projected x is the PA-rotated major-axis direction. "
            "Tilted-ring fit uses v_los = B + A cos(theta) in each annulus and reports abs(A/sin(i)). "
            "Simple comparison uses weighted mean(|v_los|)/sin(i). Weights are H I column density."
        ),
    }
    if save_outputs:
        csv_path = paths["data"] / "rotation_curve_alpha5_noflip_true_center_HIweighted.csv"
        table.to_csv(csv_path, index=False)
        logging.info("Saved %s", csv_path)
        write_json(paths["data"] / "ism_velocity_at_rho_alpha5_noflip_true_center.json", ism)
        plot_rotation_curve(table, ism, paths)
        plot_tilted_ring_diagnostics(config, maps, geometry, paths, r_disk)
        plot_threshold_comparison(config, maps, geometry, paths)
        logging.info("Estimated v_ISM at rho=%.2f kpc: %s", rho, ism)
    return table, ism


def plot_rotation_curve(table: pd.DataFrame, ism: dict[str, Any], paths: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    r = table["R_mid_kpc"].to_numpy()
    vt = table["abs_vrot_tilted_kms"].to_numpy()
    vs = table["simple_abs_vrot_kms"].to_numpy()
    sig = table["sigma_vrot_kms"].to_numpy()
    good = np.isfinite(vt)
    ax.plot(r[good], vt[good], color="black", lw=1.6, marker="o", ms=3.5, label="tilted-ring |A|/sin(i)")
    band = good & np.isfinite(sig)
    ax.fill_between(r[band], vt[band] - sig[band], vt[band] + sig[band], color="0.2", alpha=0.18, lw=0, label="1 sigma")
    good_s = np.isfinite(vs)
    ax.plot(r[good_s], vs[good_s], color="tab:blue", lw=1.4, ls="--", label=r"mean |$v_{\rm los}$|/sin(i)")
    rho = ism["rho_kpc"]
    ax.axvline(rho, color="tab:red", lw=1.2, ls=":", label=f"rho={rho:g} kpc")
    vt_rho = ism["vrot_tilted_at_rho_kms"]
    sig_rho = ism["sigma_at_rho_kms"]
    if np.isfinite(vt_rho):
        ax.scatter([rho], [vt_rho], color="tab:red", zorder=5)
        ax.annotate(
            f"{vt_rho:.0f} +/- {sig_rho:.0f} km/s" if np.isfinite(sig_rho) else f"{vt_rho:.0f} km/s",
            xy=(rho, vt_rho),
            xytext=(rho + 8, vt_rho + 15),
            arrowprops=dict(arrowstyle="-", lw=0.8, color="tab:red"),
            fontsize=9,
        )
    ax.set_xlabel("Deprojected radius [kpc]")
    ax.set_ylabel("Rotation velocity [km/s]")
    ax.set_title("M61/TNG50 SID 488530 true-centered H I rotation estimate")
    ax.set_xlim(0, np.nanmax(r) + 2)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    for suffix in ("png", "pdf"):
        out = paths["figures"] / f"rotation_curve_alpha5_noflip_true_center_HIweighted.{suffix}"
        fig.savefig(out, dpi=300)
        logging.info("Saved %s", out)
    plt.close(fig)


def plot_tilted_ring_diagnostics(
    config: AnalysisConfig,
    maps: dict[str, Any],
    geometry: dict[str, Any],
    paths: dict[str, Path],
    r_disk: np.ndarray,
) -> None:
    vmap = maps["vlos_HIweighted_kms"]
    logN = maps["logN_HI"]
    N_HI = maps["N_HI_cm2"]
    inc_rad = np.deg2rad(float(geometry["inc_deg"]))
    sin_inc = float(np.sin(inc_rad))
    cos_theta = np.asarray(r_disk) * np.nan
    x, y = image_plane_grid(vmap.shape[0], config.width_kpc)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(r_disk > 0, x / r_disk, np.nan)
    selected = [10, 20, 26, 40, 60]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
    axes = axes.ravel()
    for ax, r0 in zip(axes, selected):
        r_inner = r0 - config.annulus_dr_kpc / 2
        r_outer = r0 + config.annulus_dr_kpc / 2
        mask = (
            (r_disk >= r_inner)
            & (r_disk < r_outer)
            & np.isfinite(vmap)
            & np.isfinite(cos_theta)
            & np.isfinite(logN)
            & (logN >= config.logN_HI_min)
            & (N_HI > 0)
        )
        if np.count_nonzero(mask) == 0:
            ax.set_title(f"R~{r0} kpc: no pixels")
            continue
        c = cos_theta[mask]
        v = vmap[mask]
        w = N_HI[mask]
        if c.size > 6000:
            rng = np.random.default_rng(config.random_seed + int(r0))
            idx = rng.choice(c.size, size=6000, replace=False, p=w / np.sum(w))
            c_plot, v_plot = c[idx], v[idx]
        else:
            c_plot, v_plot = c, v
        a, b, _ = tilted_ring_fit(v, c, w)
        ax.scatter(c_plot, v_plot, s=3, alpha=0.18, color="black", rasterized=True)
        xline = np.linspace(-1, 1, 100)
        ax.plot(xline, b + a * xline, color="tab:red", lw=1.6)
        vrot = a / sin_inc if np.isfinite(a) else np.nan
        ax.set_title(f"R~{r0} kpc, vrot={vrot:.1f} km/s")
        ax.set_xlabel(r"$\cos\theta$")
        ax.set_ylabel(r"$v_{\rm los}$ [km/s]")
        ax.grid(alpha=0.2)
    axes[-1].axis("off")
    for suffix in ("png", "pdf"):
        out = paths["figures"] / f"tilted_ring_fit_diagnostics_alpha5_noflip_true_center.{suffix}"
        fig.savefig(out, dpi=300)
        logging.info("Saved %s", out)
    plt.close(fig)


def plot_threshold_comparison(
    config: AnalysisConfig,
    maps: dict[str, Any],
    geometry: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for threshold in [17.0, 18.0, 19.0]:
        cfg = AnalysisConfig(**asdict(config))
        cfg.logN_HI_min = threshold
        cfg.bootstrap_samples = 0
        table, _ = run_rotation_curve_for_threshold(cfg, maps, geometry, paths, threshold, save_outputs=False)
        r = table["R_mid_kpc"].to_numpy()
        v = table["abs_vrot_tilted_kms"].to_numpy()
        good = np.isfinite(v)
        ax.plot(r[good], v[good], marker="o", ms=2.8, lw=1.2, label=fr"$\log N_{{HI}}\geq {threshold:.0f}$")
    ax.axvline(geometry["rho_kpc"], color="tab:red", ls=":", lw=1)
    ax.set_xlabel("Deprojected radius [kpc]")
    ax.set_ylabel("abs tilted-ring vrot [km/s]")
    ax.set_title("H I threshold comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        out = paths["figures"] / f"rotation_curve_threshold_comparison_alpha5_noflip_true_center.{suffix}"
        fig.savefig(out, dpi=300)
        logging.info("Saved %s", out)
    plt.close(fig)


def run_workflow(config: AnalysisConfig | None = None) -> dict[str, Any]:
    if config is None:
        config = AnalysisConfig()
    paths = setup_output_dirs(config)
    stale_error_log = paths["logs"] / "errors.txt"
    if stale_error_log.exists():
        stale_error_log.unlink()
    setup_logging(paths)
    try:
        logging.info("Analysis config: %s", asdict(config))
        geometry = load_recipe_and_geometry(config, paths)
        ds, field_info = load_dataset_and_define_fields(config, geometry, paths)
        maps = make_projections(config, ds, geometry, paths)
        rotation_table, ism = run_rotation_curve_analysis(config, maps, geometry, paths)
        metadata = {
            "config": asdict(config),
            "paths": {key: str(value) for key, value in paths.items()},
            "geometry": geometry,
            "field_info": field_info,
            "projection_stats": maps["stats"],
            "projection_settings": {
                "center_ckpch_code_length": geometry["galaxy_center_ckpch"],
                "normal_vector": geometry["normal_vector"],
                "north_vector": geometry["e2_hat"],
                "width_kpc": config.width_kpc,
                "width_code_length_ckpch": config.width_kpc * H_TNG,
                "npix": config.npix,
            },
            "ism_velocity_at_rho": ism,
        }
        write_json(paths["data"] / "analysis_metadata_alpha5_noflip.json", metadata)
        write_json(paths["data"] / "geometry_and_projection_metadata_alpha5_noflip_true_center.json", metadata)
        return {
            "config": config,
            "paths": paths,
            "geometry": geometry,
            "field_info": field_info,
            "maps": maps,
            "rotation_table": rotation_table,
            "ism_velocity_at_rho": ism,
            "metadata": metadata,
        }
    except Exception:
        err_path = paths["logs"] / "errors.txt"
        err_path.write_text(traceback.format_exc())
        logging.exception("Analysis failed; wrote %s", err_path)
        raise


def parse_args(argv: list[str] | None = None) -> AnalysisConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npix", type=int, default=AnalysisConfig.npix)
    parser.add_argument("--logN-HI-min", type=float, default=AnalysisConfig.logN_HI_min)
    parser.add_argument("--bootstrap-samples", type=int, default=AnalysisConfig.bootstrap_samples)
    parser.add_argument("--output-dir", type=str, default=AnalysisConfig.output_dir)
    parser.add_argument("--cutout-path", type=str, default=AnalysisConfig.preferred_cutout_path)
    parser.add_argument("--recipe-csv", type=str, default=AnalysisConfig.recipe_csv)
    args = parser.parse_args(argv)
    config = AnalysisConfig(
        npix=args.npix,
        logN_HI_min=args.logN_HI_min,
        bootstrap_samples=args.bootstrap_samples,
        output_dir=args.output_dir,
        preferred_cutout_path=args.cutout_path,
        recipe_csv=args.recipe_csv,
    )
    return config


def main(argv: list[str] | None = None) -> dict[str, Any]:
    config = parse_args(argv)
    return run_workflow(config)


if __name__ == "__main__":
    main()
