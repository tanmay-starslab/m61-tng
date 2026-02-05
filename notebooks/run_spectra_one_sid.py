#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_spectra_one_sid.py

Wrapper around spectra_batch_module.py for the M61-TNG workflow.

Expected directory layout from orient_m61.py:
  OUT_BASE/sid<SID>/rays_and_recipes_sid<SID>_snap<SNAP>_<RUN_LABEL>/rays_sid<SID>.csv

Cutout layout:
  CUTOUT_ROOT/out_sub_<SID>/*.hdf5

This script:
  - finds the cutout file for SID
  - points spectra_batch_module at rays_base=OUT_BASE/sid<SID>
  - writes spectra under OUT_BASE/sid<SID>/rays_and_spectra_sid<SID>_snap<SNAP>_<RUN_LABEL>/
"""

import os
import glob
import argparse

from spectra_batch_module import JobPaths, RunConfig, run_all_runs_for_sid


def find_cutout_h5(cutout_root: str, sid: int) -> str:
    sid = int(sid)
    d = os.path.join(os.path.abspath(cutout_root), f"out_sub_{sid}")
    pats = [
        os.path.join(d, "cutout*sub*.hdf5"),
        os.path.join(d, "cutout*.hdf5"),
        os.path.join(d, "*.hdf5"),
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sid", type=int, required=True)
    p.add_argument("--snap", type=int, default=99)

    p.add_argument("--out-base", required=True,
                   help="Orientation outputs base (contains sid<SID>/).")
    p.add_argument("--cutout-root", required=True,
                   help="Cutout root containing out_sub_<SID>/.")

    p.add_argument("--run-label", default="",
                   help="If set, run only this label (e.g. L3Rvir or L4Rvir). If empty, run all available.")
    p.add_argument("--ion", default="H I",
                   help="Ion species in Trident (e.g. 'H I', 'Mg II', 'O VI').")
    p.add_argument("--line-list", default="lines.txt",
                   help="Trident line list filename (must be on PYTHONPATH or local).")
    p.add_argument("--n-processes", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    sid = int(args.sid)
    snap = int(args.snap)
    out_base = os.path.abspath(args.out_base)
    cutout_root = os.path.abspath(args.cutout_root)

    sid_dir = os.path.join(out_base, f"sid{sid}")
    if not os.path.isdir(sid_dir):
        raise FileNotFoundError(f"Missing orientation output directory: {sid_dir}")

    cutout = find_cutout_h5(cutout_root, sid)
    if not cutout or (not os.path.isfile(cutout)):
        raise FileNotFoundError(f"Missing cutout for sid={sid} under {cutout_root}/out_sub_{sid}/")

    # spectra_batch_module expects rays_base to contain rays_and_recipes_sid{sid}_snap{snap}_{run}/
    paths = JobPaths(cutout_h5=cutout, rays_base=sid_dir, output_base=sid_dir)

    cfg = RunConfig(
        sid=sid,
        snap=snap,
        ion=args.ion,
        line_list=args.line_list,
        n_processes=int(args.n_processes),
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
    )

    if args.run_label:
        run_all_runs_for_sid(paths, cfg, allowed_run_labels=[args.run_label])
    else:
        run_all_runs_for_sid(paths, cfg, allowed_run_labels=None)


if __name__ == "__main__":
    main()
