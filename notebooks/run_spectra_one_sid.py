#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_spectra_one_sid.py

Thin wrapper to run spectra generation for a single SubhaloID (SID) using
orient_m61 outputs + a cutout root.

Typical layout:
  cutout_root/out_sub_<SID>/*.hdf5
  orient_out_base/sid<SID>/rays_and_recipes_sid<SID>_snap<SNAP>_<RUN>/rays_sid<SID>.csv

Sanity-check knobs:
  --alpha-keep 0
  --max-rays 4
  --filter-mode noflip
  --no-plots
"""

import os
import glob
import argparse

from spectra_batch_module import JobPaths, JobParams, SpectraConfig, run_all_runs_for_sid


def find_cutout_h5(sid: int, cutout_root: str) -> str:
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Run spectra for one SID using orient_m61 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sid", type=int, required=True)
    p.add_argument("--snap", type=int, default=99)

    p.add_argument("--cutout-root", required=True, help="Root containing out_sub_<SID>/")
    p.add_argument("--orient-out-base", required=True, help="Same --out-base used in orient_m61.py")
    p.add_argument("--spectra-out-base", default="", help="If empty, writes under orient-out-base/sid<SID>/")

    p.add_argument("--run-labels", default="L3Rvir,L4Rvir")
    p.add_argument("--filter-mode", choices=["noflip", "flip"], default=None)

    p.add_argument("--alpha-keep", default="", help="Comma-separated alpha list, e.g. '0,90'")
    p.add_argument("--max-rays", type=int, default=0)
    p.add_argument("--sightline-ids", default="", help="Comma-separated IDs, e.g. 'QSO-A,QSO-B'")

    p.add_argument("--lines", default="H I 1216,C II 1335,Si III 1206")
    p.add_argument("--instrument", default="COS-G130M")
    p.add_argument("--zoom-half-A", type=float, default=3.0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main():
    a = parse_args()

    sid = int(a.sid)
    snap = int(a.snap)

    cutout_h5 = find_cutout_h5(sid, a.cutout_root)
    if not cutout_h5:
        raise FileNotFoundError(f"No cutout .hdf5 found for SID={sid} under {a.cutout_root}/out_sub_{sid}/")

    rays_base = os.path.join(a.orient_out_base, f"sid{sid}")

    spectra_base = a.spectra_out_base.strip()
    if spectra_base:
        output_base = os.path.join(spectra_base, f"sid{sid}")
    else:
        output_base = os.path.join(a.orient_out_base, f"sid{sid}")

    alpha_keep = None
    if a.alpha_keep.strip():
        alpha_keep = [int(x) for x in a.alpha_keep.split(",") if x.strip()]

    sightline_ids = None
    if a.sightline_ids.strip():
        sightline_ids = [x.strip() for x in a.sightline_ids.split(",") if x.strip()]

    paths = JobPaths(
        cutout_h5=cutout_h5,
        rays_base=rays_base,
        output_base=output_base,
    )

    params = JobParams(
        sid=sid,
        snap=snap,
        run_labels=[s.strip() for s in a.run_labels.split(",") if s.strip()],
        filter_mode=a.filter_mode,
        alpha_keep=alpha_keep,
        sightline_ids=sightline_ids,
        max_rays=(int(a.max_rays) if a.max_rays and int(a.max_rays) > 0 else None),
        verbose=bool(a.verbose),
    )

    cfg = SpectraConfig(
        lines=[s.strip() for s in a.lines.split(",") if s.strip()],
        instrument=a.instrument,
        zoom_half_A=float(a.zoom_half_A),
        make_plots=(not a.no_plots),
    )

    run_all_runs_for_sid(paths, params, cfg)


if __name__ == "__main__":
    main()