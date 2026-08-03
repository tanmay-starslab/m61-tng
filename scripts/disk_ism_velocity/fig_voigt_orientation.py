#!/usr/bin/env python3
r"""Orientation / geometry figures from the pyGad Voigt component catalog (fig18-fig21).

The 720 orientations per galaxy are a CONTROLLED EXPERIMENT in viewing geometry: one QSO at
a fixed impact parameter rho = 25.56 kpc, a fixed inclination, and 360 rotation angles alpha
in two modes.  These four figures exploit that:

  fig18_covfrac_vs_orientation   HVC covering fraction / column / blue-red split vs the DISK
                                 AZIMUTH probed by the sightline, stacked over 20 galaxies,
                                 with a circular-shift null test of the azimuthal modulation.
  fig19_covfrac_vs_galaxy        per-galaxy HVC covering fraction and its correlation with
                                 Mstar, SFR, M200c and whether rho lies inside the cold disk.
  fig20_edge_fallback_systematics  are the HVC results driven by the ~40-50 per cent of
                                 sightlines whose v_ISM uses the disk-edge fallback?
  fig21_vism_variant_sensitivity how much does the choice of v_ISM (v3a / v3b / v1) move the
                                 observables?

DETECTION is the pipeline's own definition (m61_voigt.py): a fitted pyGad Voigt component
with UpLim == False.  Covering fraction = (# sightlines with >= 1 such component satisfying
the velocity condition) / (# sightlines where that line was fit).  Pure sightline counting;
never weighted by N or EW.  No oscillator strengths, no curve-of-growth thresholds.

DEFAULT SCIENCE SAMPLE (`--raw` disables it): components with
    ~b_at_ceiling  &  ~beyond_common_window
i.e. the Doppler parameter is not pinned at the fitter's upper bound and |dv| < 500 km/s (the
velocity range common to every transition).  H I 1216 was fit over +-1200 km/s with
b_max = 150 km/s while the metals used +-800 km/s with b_max = 50-100, so 59 per cent of H I
components sit at the b ceiling and 61 per cent lie beyond 500 km/s -- multi-component fits to
the damped, saturated Lyman-alpha profile, not physical high-velocity clouds.  The metals are
essentially unaffected (1-1.6 per cent beyond 500 km/s).  Every panel is tagged with the
sample used, and fig18a shows the raw H I curve alongside the cleaned one.

Usage:  python fig_voigt_orientation.py [v3a|v3b] [--raw]      (default v3b)
Output: outputs/disk_ism_velocity/paper_figures_<variant>/fig18..fig21 (pdf + png)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, "/home/tsingh65/m61-tng/scripts/disk_ism_velocity")
sys.path.insert(0, "/scratch/tsingh65/m61-tng/scripts")
import m61_style as S           # noqa: E402
import m61_voigt as V           # noqa: E402
from pm_general import TNG_H        # noqa: E402

OUTBASE = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity")
GALPROP = OUTBASE / "galaxy_properties/galaxy_properties_tng.csv"
DISKEXT = OUTBASE / "diagnostics_v3/disk_extent.csv"
RCDIR = OUTBASE / "rotation_curves_v2"
RAYS = "/scratch/tsingh65/m61-tng/outputs/sid{sid}/rays_and_recipes_sid{sid}_snap99_L2Rvir/rays_sid{sid}.csv"

SLK = ["sid", "mode", "alpha"]
NBIN = 24                      # azimuth bins -> 15 deg = 6.7 kpc of arc at rho = 25.6 kpc
BW = 360.0 / NBIN
NSIM = 4000                    # circular-shift null realisations
RNG = np.random.default_rng(20260803)

PRIMARY = "Si_II_1260"                      # strongest Si II transition: best statistics
REP = ["H_I_1216", "C_II_1335", "Si_II_1260", "Si_III_1206", "Si_IV_1403"]
REP3 = ["C_II_1335", "Si_II_1260", "Si_III_1206"]

SAMPLE_TAG = {True: r"clean sample: $b<b_{\max}$, $|\Delta v|<500$",
              False: r"raw sample (no $b$/window cuts)"}

# Filename suffix so that a `--raw` run cannot silently overwrite the default (clean)
# science figures: they differ by more than a factor two in f_c for H I.
FILE_SUFFIX = ""


def save_fig(fig, name):
    return S.save(fig, name + FILE_SUFFIX)


# ═══════════════════════════════════════════════════════════════════════════════════════
# AZIMUTH -- how (mode, alpha) maps to the disk azimuth probed by the sightline
# ═══════════════════════════════════════════════════════════════════════════════════════
# The orientation sweep is  "fixed_observer_qso__galaxy_rotated_about_inner_stellar_pca_disk
# _normal"  (the `alpha_convention` column of rays_sid<SID>.csv).  In the stored simulation
# frame the galaxy is held still and the OBSERVER is rotated instead, which is the same
# experiment:  for each (sid, mode)
#     n_disk                    fixed  (young-star PCA v3 normal, rotation_curves_v2/rc_sid*.npz)
#     los(alpha)                = los(0) rotated about n_disk by alpha
#     anchor(alpha)             rotated likewise, so the impact direction
#     rhat = unit(anchor - center - [(anchor-center).los] los)   rotates with alpha too.
#
# VERIFIED here for all 14,400 sightlines (printed by --report):
#   * rho = 25.5583 kpc and |i| = angle(n_disk, los) = 23.000 deg are EXACTLY constant;
#   * the angle between rhat and the disk's projected kinematic major axis is EXACTLY
#     constant WITHIN each (sid, mode) block and takes only two values over the whole
#     sample: 21.034 deg (7200 sightlines) and 158.966 deg = 180 - 21.034 (the other
#     7200) -- i.e. the QSO sits on the approaching side of the projected major axis for
#     half of the (galaxy, mode) blocks and on the receding side for the other half.
#     Within a block the SKY-PROJECTED geometry (inclination, sky azimuthal angle, which
#     side of the rotation axis the QSO sits on) does NOT change with alpha at all.
#   ==> the only thing alpha changes is WHICH AZIMUTH OF THE DISK MATERIAL is pierced.
#       Projection is fixed, material varies: that is what makes the sweep controlled.
#
# The azimuth is therefore defined in the galaxy's OWN disk plane:
#     n_L  = n_disk oriented along the cold-gas angular momentum.  The sign is taken from the
#            median v_phi at R = 5-20 kpc in rc_sid<SID>.npz (that rotation curve is measured
#            in the (e1, e2, n_disk) frame), so v_phi > 0 about n_L for every galaxy.
#     f1   = e1;   f2 = sign * e2, which makes (f1, f2, n_L) right-handed, so phi INCREASES
#            in the direction the disk rotates.  (e1 x e2 = n_disk is verified numerically.)
#     r_ip = rhat - (rhat . n_L) n_L            (in-plane part of the impact direction)
#     phi  = atan2(r_ip . f2, r_ip . f1)  mod 360 deg
#
# !! WHAT phi = 0 ACTUALLY IS -- READ BEFORE INTERPRETING fig18a. !!
# e1/e2 in rc_sid<SID>.npz come from disk_velocity_v2/dv_core.disk_basis(n_disk), which is
#     e1 = unit(n_disk x yhat),   e2 = unit(n_disk x e1)
# with yhat the FIXED SIMULATION-BOX y axis (xhat if n_disk is nearly parallel to yhat).
# VERIFIED numerically for all 20 galaxies: e1 == unit(n_disk x yhat) to 1e-6, e1.y == 0.
# n_disk itself is the inner young-star PCA normal, but e1 is NOT a stellar major axis, a
# bar axis, or any other galaxy-intrinsic direction -- it is an arbitrary orthonormal
# completion whose orientation in the disk plane depends only on how that galaxy's disk
# normal happens to sit relative to the box axes.  CONSEQUENCES:
#   * WITHIN a galaxy phi is a perfectly good azimuth: it is linear in alpha, covers the
#     full 360 deg, and any modulation of f_c with phi is a real, reference-free statement
#     about that galaxy.  fig18d (per-galaxy modulation amplitude) is therefore meaningful.
#   * ACROSS galaxies the zero point phi_0 is RANDOM.  Stacking f(phi) over the 20 galaxies
#     (fig18a) co-adds 20 profiles with independent, meaningless phase offsets, so any
#     galaxy-common azimuthal preference would be washed out BY CONSTRUCTION.  The null
#     result of the circular-shift test (p = 0.31-0.63) is therefore EXPECTED and is NOT
#     evidence that HVC incidence is azimuth-independent.  The stacked curve is shown only
#     as the mean level plus its galaxy-to-galaxy scatter; it is labelled as such.
#   Fixing this would need a galaxy-intrinsic in-plane reference (bar/spiral phase or a
#   photometric major axis), which the rotation-curve products do not carry.
#
# VERIFIED: phi = phi_0(sid, mode) +- alpha EXACTLY -- |dphi/dalpha| = 1.000000 and each
# (sid, mode) block covers 360 distinct values -- so alpha sweeps the full 360 deg of disk
# azimuth at constant R_anchor, exactly as VISM_V3.md states.
#
# INDEPENDENT CHECK OF THE WHOLE CONSTRUCTION.  For a rotating disk the line-of-sight ISM
# velocity must be  v_ISM ~ v_c sin(i) cos(phi - phi_major).  Because phi - phi_major is
# constant, this predicts a nearly constant v_ISM within each (sid, mode) block, of a
# specific magnitude and sign.  The measured per-block mean of v_ism_v3b tracks that
# prediction with r = 0.85 and a median ratio of 0.89 -- confirming the handedness, the
# angular-momentum sign calibration and the velocity sign convention all at once.  (This
# check does NOT depend on the e1 zero point: phi - phi_major is frame-independent.)
#
# THE 720 SIGHTLINES OF ONE GALAXY ARE NOT INDEPENDENT.  Neighbouring alphas are 0.45 kpc
# apart at rho = 25.6 kpc, and the binary HVC-detection sequence along alpha has a measured
# lag-1 autocorrelation of 0.23 (H I) to 0.63 (C II), decaying over ~10-15 deg.  The
# variance-inflation factor for the mean of the 15 consecutive alphas that make up one
# 15 deg bin (per mode) is VIF = 2-7, measured per (sid, mode) block by `vif_per_galaxy`.
# Every binomial/Wilson quantity here (Wilson bands in fig18a/c, the "shot" reference in
# fig18d, the naive chi2/dof) therefore UNDERSTATES the uncertainty by sqrt(VIF) ~ 1.5-2.6
# and OVERSTATES chi2/dof by VIF.  fig18d plots the VIF-corrected reference and the report
# prints both the naive and the corrected chi2/dof.  The galaxy-to-galaxy scatter band in
# fig18a is free of this problem and is the honest error on the stacked curve.
# ═══════════════════════════════════════════════════════════════════════════════════════
def build_geometry(sids):
    """Per-sightline geometry table: phi (disk azimuth), phi_major, inclination, v_c."""
    rows = []
    for sid in sids:
        rec = pd.read_csv(RAYS.format(sid=sid))
        rec = rec[rec.sightline_id == V.RAY_ID]
        rc = np.load(RCDIR / f"rc_sid{sid}.npz")
        center, nd, e1, e2 = rc["center_kpc"], rc["n_disk"], rc["e1"], rc["e2"]
        R, vf = rc["R_center"], rc["v_fid_median"]
        m = (R > 5) & (R < 20) & np.isfinite(vf)
        vmed = float(np.nanmedian(vf[m]))
        sgn = float(np.sign(vmed))
        n_L, f1, f2 = sgn * nd, e1, sgn * e2          # (f1, f2, n_L) right-handed
        vcirc = abs(vmed)
        # sanity: the PCA frame must be right-handed before the sign flip
        assert np.allclose(np.cross(e1, e2), nd, atol=1e-6), f"sid {sid}: (e1,e2,n) not RH"
        los = np.stack([rec.los_x, rec.los_y, rec.los_z], 1)
        los = los / np.linalg.norm(los, axis=1)[:, None]
        anc = np.stack([rec.anchor_X_ckpch_abs, rec.anchor_Y_ckpch_abs,
                        rec.anchor_Z_ckpch_abs], 1) / TNG_H
        delta = anc - center[None, :]
        rhat = delta - (delta * los).sum(1)[:, None] * los
        rhat = rhat / np.linalg.norm(rhat, axis=1)[:, None]
        rip = rhat - (rhat @ n_L)[:, None] * n_L[None, :]
        rip = rip / np.linalg.norm(rip, axis=1)[:, None]
        phi = np.degrees(np.arctan2(rip @ f2, rip @ f1)) % 360.0
        # projected kinematic major axis in the disk plane: mhat = unit(los x n_L)
        mh = np.cross(los, n_L[None, :])
        sini = np.linalg.norm(mh, axis=1)
        mh = mh / sini[:, None]
        lh = np.cross(np.broadcast_to(n_L, mh.shape), mh)          # in-plane part of los
        phim = np.degrees(np.arctan2((rip * lh).sum(1), (rip * mh).sum(1)))
        rows.append(pd.DataFrame(dict(
            sid=sid, mode=rec["mode"].to_numpy(), alpha=rec["alpha_deg"].to_numpy().astype(int),
            phi_deg=phi, phi_major_deg=phim, sini=sini,
            inc_deg=np.degrees(np.arccos(np.clip(los @ n_L, -1, 1))),
            vcirc=vcirc, rho_kpc=rec.rho_kpc.to_numpy())))
    g = pd.concat(rows, ignore_index=True)
    g["abin"] = np.floor(g.phi_deg / BW).astype(int) % NBIN
    return g


def check_geometry(SL, variant="v3b", verbose=True):
    """Print the verification numbers quoted in the header comment."""
    vcol = f"v_ism_{variant}"
    out = {}
    dphi_bad = 0
    for _, g in SL.groupby(["sid", "mode"]):
        g = g.sort_values("alpha")
        d = np.degrees(np.diff(np.unwrap(np.radians(g.phi_deg.to_numpy()))))
        if not np.allclose(np.abs(d), 1.0, atol=1e-6) or g.phi_deg.round(4).nunique() != 360:
            dphi_bad += 1
    out["nonlinear_blocks"] = dphi_bad
    out["rho"] = (SL.rho_kpc.min(), SL.rho_kpc.max())
    inc = np.minimum(SL.inc_deg, 180 - SL.inc_deg)
    out["inc"] = (inc.min(), inc.max())
    out["phi_major"] = np.sort(np.unique(np.abs(SL.phi_major_deg).round(4)))
    pred = SL.vcirc * SL.sini * np.cos(np.radians(SL.phi_major_deg))
    ok = SL[vcol].notna()
    out["r_pred"] = float(np.corrcoef(pred[ok], SL[vcol][ok])[0, 1])
    out["ratio"] = float(np.nanmedian(SL[vcol][ok] / pred[ok]))
    if verbose:
        print("\n── azimuth geometry verification " + "─" * 50)
        print(f"  (sid,mode) blocks where phi is NOT phi_0 +- alpha : {dphi_bad}/40")
        print(f"  rho          : {out['rho'][0]:.4f} - {out['rho'][1]:.4f} kpc")
        print(f"  inclination  : {out['inc'][0]:.3f} - {out['inc'][1]:.3f} deg (constant)")
        print(f"  |phi - phi_major| takes only the values {out['phi_major']} deg "
              f"(constant within every (sid,mode) block)")
        print("               -> sky projection is fixed as alpha varies; only the disk")
        print("                  material rotates.  The two values are 21.034 and "
              "180-21.034: the")
        print("                  QSO sits on the approaching side of the projected major "
              "axis for")
        print("                  half of the (galaxy, mode) blocks and the receding side "
              "for the rest.")
        print(f"  v_ISM({variant}) vs v_c sin(i) cos(phi-phi_major): r = {out['r_pred']:.3f}, "
              f"median ratio = {out['ratio']:.3f}")
        print("  phi ZERO POINT: e1 = unit(n_disk x y_box) (dv_core.disk_basis) -- an "
              "ARBITRARY")
        print("                  per-galaxy in-plane direction, NOT a stellar major axis. "
              "Galaxy-")
        print("                  stacked f(phi) therefore carries no azimuthal information; "
              "only the")
        print("                  WITHIN-galaxy modulation (fig18d) is meaningful.")
    return out


# ═══════════════════════════════════════════════════════════════════════════════════════
# counting  (the pipeline's covering-fraction rule, sliced by any sightline property)
# ═══════════════════════════════════════════════════════════════════════════════════════
def clean_sample(comp, clean=True):
    if not clean:
        return comp
    return comp[~comp.b_at_ceiling & ~comp.beyond_common_window].reset_index(drop=True)


def hit_sightlines(comp, line, dvcol, vmin=V.HVC, vmax=np.inf, signed=None):
    """Sightlines with >= 1 DETECTED component of `line` in the velocity window."""
    c = comp[comp.line == line]
    dv = c[dvcol]
    m = (dv.abs() >= vmin) & (dv.abs() < vmax)
    if signed == "blue":
        m = m & (dv < 0)
    elif signed == "red":
        m = m & (dv > 0)
    return c.loc[m, SLK].drop_duplicates()


def denom_sightlines(stat, line):
    s = stat[(stat.line == line) & (stat.fit_status.isin(V.FIT_RAN))]
    return s[SLK].drop_duplicates()


def counts_by(comp, stat, SL, line, dvcol, by, signed=None, vmin=V.HVC, vmax=np.inf,
              sl_sub=None):
    """(k, n) as pandas Series indexed by the `by` columns of SL.

    k = # sightlines with >= 1 detected component in the window; n = # sightlines where the
    line was fit.  `sl_sub` optionally restricts the sightline set (SLK frame).
    """
    d = denom_sightlines(stat, line)
    h = hit_sightlines(comp, line, dvcol, vmin, vmax, signed)
    if sl_sub is not None:
        d = d.merge(sl_sub, on=SLK)
        h = h.merge(sl_sub, on=SLK)
    d = d.merge(SL, on=SLK)
    h = h.merge(SL, on=SLK)
    n = d.groupby(by, observed=True).size()
    k = h.groupby(by, observed=True).size().reindex(n.index, fill_value=0)
    return k, n


def kn_grid(comp, stat, SL, line, dvcol, sids, signed=None, sl_sub=None):
    """(K, N) arrays of shape (n_gal, NBIN) -- galaxy x azimuth bin."""
    k, n = counts_by(comp, stat, SL, line, dvcol, ["sid", "abin"], signed=signed,
                     sl_sub=sl_sub)
    K = np.zeros((len(sids), NBIN)); N = np.zeros((len(sids), NBIN))
    si = {s: i for i, s in enumerate(sids)}
    for (s, b), v in n.items():
        N[si[s], b] = v
    for (s, b), v in k.items():
        K[si[s], b] = v
    return K, N


def frac(k, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, k / np.where(n > 0, n, 1), np.nan)


def tag_at(ax, text, x, y, ha="left", va="top", fs=7.4):
    """S.tag with a free position (used where the panel label occupies the corner)."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.92), zorder=8)


def leg_ul(ax, **kw):
    """Upper-left legend, nudged right so it clears the bold panel label."""
    return ax.legend(loc="upper left", bbox_to_anchor=(0.075, 0.995), **kw)


# ═══════════════════════════════════════════════════════════════════════════════════════
# azimuthal-modulation statistics
# ═══════════════════════════════════════════════════════════════════════════════════════
def shift_null(K, N, nsim=NSIM):
    """RMS of the galaxy-STACKED f(phi) under the null 'no common azimuthal pattern'.

    Each galaxy's (k, n) profile is circularly shifted by a random amount before stacking.
    A circular shift preserves each galaxy's own azimuthal structure AND the correlation
    between neighbouring alphas exactly; it only destroys the alignment BETWEEN galaxies.
    So this is the correct null for 'is the stacked modulation galaxy-coherent?'.
    """
    ng, nb = K.shape
    sh = RNG.integers(0, nb, size=(nsim, ng))
    idx = (np.arange(nb)[None, None, :] + sh[:, :, None]) % nb
    Ks = np.take_along_axis(np.broadcast_to(K, (nsim, ng, nb)), idx, axis=2)
    Ns = np.take_along_axis(np.broadcast_to(N, (nsim, ng, nb)), idx, axis=2)
    f = Ks.sum(1) / np.maximum(Ns.sum(1), 1)
    return f.std(axis=1)


def stacked_modulation(K, N):
    """(f(phi), rms, p-value, null 95th pct) for the galaxy-stacked azimuth profile."""
    f = frac(K.sum(0), N.sum(0))
    rms = float(np.nanstd(f))
    null = shift_null(K, N)
    return f, rms, float((null >= rms).mean()), float(np.percentile(null, 95))


NPB = 360 // NBIN          # consecutive alphas contributing to one azimuth bin, per mode


def vif_per_galaxy(comp, stat, SL, line, dvcol, sids, signed=None, npb=NPB):
    """Variance-inflation factor of a `npb`-sightline bin mean, measured per galaxy.

    Adjacent alphas are 0.45 kpc apart at rho = 25.6 kpc, so the binary detection sequence
    along alpha is autocorrelated and the 30 sightlines of one (galaxy, azimuth bin) are
    NOT 30 independent Bernoulli trials.  For a stationary sequence with autocorrelation
    rho_k the variance of the mean of npb consecutive samples is inflated by
        VIF = 1 + 2 sum_{k=1}^{npb-1} (1 - k/npb) rho_k
    which is estimated here from the circular ACF of each (sid, mode) block and averaged
    over the two modes.  A bin holds two such runs (one per mode), so this VIF is also the
    inflation factor for the 30-sightline bin.  Deliberately conservative: rho_k is measured
    on the raw sequence, so any genuine azimuthal trend contributes to it too (removing a
    61-deg running mean first gives VIF smaller by ~2x -- see the module header).
    """
    hits = set(map(tuple, hit_sightlines(comp, line, dvcol, signed=signed).to_numpy()))
    lag = np.arange(1, npb)
    taper = 1.0 - lag / npb
    out = np.ones(len(sids))
    for i, s in enumerate(sids):
        vs = []
        for _, blk in SL[SL.sid == s].groupby("mode"):
            blk = blk.sort_values("alpha")
            y = np.fromiter(((s, m_, a) in hits for m_, a in
                             zip(blk["mode"].to_numpy(), blk.alpha.to_numpy())),
                            float, len(blk))
            y = y - y.mean()
            v = float((y * y).mean())
            if v <= 0:
                continue
            r = np.array([float((y * np.roll(y, -int(k))).mean()) / v for k in lag])
            vs.append(1.0 + 2.0 * float(np.sum(taper * r)))
        out[i] = max(float(np.mean(vs)), 1.0) if vs else 1.0
    return out


def within_galaxy_excess(K, N, vif=None):
    """Per galaxy: observed RMS of f(phi), shot-noise RMS, excess, chi2/dof.

    `vif` (one value per galaxy, from `vif_per_galaxy`) inflates the binomial variance to
    account for the correlation between neighbouring sightlines.  With vif=None the naive
    (independent-Bernoulli) numbers are returned -- those OVERSTATE the significance and are
    kept only for comparison in the report.
    """
    w = np.ones(K.shape[0]) if vif is None else np.asarray(vif, float)
    fg = frac(K.sum(1), N.sum(1))
    f = frac(K, N)
    var_obs = np.nanvar(f, axis=1)
    var_shot = w * np.nanmean(fg[:, None] * (1 - fg[:, None]) / np.maximum(N, 1), axis=1)
    chi2 = np.nansum((K - N * fg[:, None]) ** 2 /
                     np.maximum(N * fg[:, None] * (1 - fg[:, None]), 1e-9), axis=1)
    return (np.sqrt(var_obs), np.sqrt(var_shot),
            np.sqrt(np.clip(var_obs - var_shot, 0, None)), chi2 / (w * (NBIN - 1)))


# ═══════════════════════════════════════════════════════════════════════════════════════
# data
# ═══════════════════════════════════════════════════════════════════════════════════════
def load_all(clean=True):
    comp_raw = V.load_components()                     # detections only (UpLim == False)
    for c in ("b_at_ceiling", "beyond_common_window"):
        if c not in comp_raw.columns:
            raise KeyError(f"voigt_components.parquet is missing `{c}` -- rebuild the catalog")
    comp = clean_sample(comp_raw, clean)
    stat = V.load_line_status()
    master = V.load_master_v3().rename(columns={"alpha_deg": "alpha"})
    sids = sorted(master.sid.unique())
    geo = build_geometry(sids)
    SL = geo.merge(master[SLK + ["v_ism_v3a", "v_ism_v3b", "v_ism_direct_cool", "v_sys",
                                 "v3a_edge_fallback", "v3b_edge_fallback", "in_disk_v1"]],
                   on=SLK, how="left")
    SL["v1_defined"] = SL.v_ism_direct_cool.notna()
    gal = pd.read_csv(GALPROP)
    dext = pd.read_csv(DISKEXT)
    gal = gal.merge(dext[["sid", "R_disk", "rho_in_disk"]], on="sid")
    print(f"[load] {len(comp_raw)} detected components -> {len(comp)} in sample "
          f"({'clean' if clean else 'raw'}); {len(SL)} sightlines, {len(sids)} galaxies")
    return comp, comp_raw, stat, SL, gal, sids


# ═══════════════════════════════════════════════════════════════════════════════════════
# fig18 -- covering fraction vs disk azimuth
# ═══════════════════════════════════════════════════════════════════════════════════════
def fig18(D):
    comp, comp_raw, stat, SL, sids, dvc = D["comp"], D["comp_raw"], D["stat"], D["SL"], D["sids"], D["dvcol"]
    xc = BW * (np.arange(NBIN) + 0.5)
    fig, ax = plt.subplots(2, 2, figsize=(14.0, 10.4))
    rep = {}

    # ── (a) HVC covering fraction vs azimuth, per line ────────────────────────────────
    a = ax[0, 0]
    for L in V.LINES:
        K, N = kn_grid(comp, stat, SL, L.key, dvc, sids)
        f, rms, p, n95 = stacked_modulation(K, N)
        rep[L.key] = dict(f=f, K=K, N=N, rms=rms, p=p, n95=n95, mean=np.nansum(K) / np.nansum(N))
        lo, hi = V.wilson(K.sum(0), N.sum(0))
        a.plot(xc, f, color=L.color, ls=L.ls, lw=L.lw, label=L.label)
        a.fill_between(xc, lo, hi, color=L.color, alpha=0.16, lw=0)
    # raw H I for comparison (the b-ceiling / >500 km/s artefact)
    Kr, Nr = kn_grid(comp_raw, stat, SL, "H_I_1216", dvc, sids)
    a.plot(xc, frac(Kr.sum(0), Nr.sum(0)), color="0.45", ls=(0, (1, 1.6)), lw=1.6,
           label=r"$\mathrm{H\,I}$ raw (b-ceiling artefact)")
    # galaxy-to-galaxy scatter for the primary line.  Drawn in black dash-dot, NOT in the
    # Si II colour: a green dashed envelope is indistinguishable from the Si II 1193 curve.
    fp = frac(rep[PRIMARY]["K"], rep[PRIMARY]["N"])
    for q in (16, 84):
        a.plot(xc, np.nanpercentile(fp, q, axis=0), color="0.12", ls=(0, (5, 1.6, 1, 1.6)),
               lw=1.3, alpha=0.9,
               label=(r"galaxy-to-galaxy $16/84$th pct "
                      r"($\mathrm{Si\,II}\ \lambda1260$)") if q == 16 else None)
    a.set_xlim(0, 360); a.set_ylim(0, 1.13)
    a.set_xticks(np.arange(0, 361, 60))
    a.set_xlabel(r"disk azimuth of the impact point $\phi$ [deg]"
                 "\n" r"($\phi=0$ is an arbitrary, galaxy-dependent in-plane direction)")
    a.set_ylabel(r"HVC covering fraction $f_{\rm c}(|\Delta v|\geq100)$ [fraction]")
    hh = [h for h, l in zip(*a.get_legend_handles_labels()) if l][-2:]
    ll = [l for l in a.get_legend_handles_labels()[1] if l][-2:]
    a.legend(handles=hh, labels=ll, ncol=1, fontsize=7.6, loc="upper left",
             bbox_to_anchor=(0.075, 0.995), framealpha=0.9)
    S.grid(a); S.panel_label(a, "(a)")
    S.tag(a, "line colours / styles as in (b)\n" + D["stag"] + "\n"
          + r"stacked $f(\phi)$ is consistent with NO common" + "\n"
          + r"azimuthal pattern (see (d)); $\phi_0$ is arbitrary" + "\n"
          + r"per galaxy, so a common pattern could not survive", "lr", fs=6.8)

    # ── (b) mean fitted HVC logN vs azimuth ───────────────────────────────────────────
    # H I sits ~3 dex above the metals (damped Lyman-alpha), so it gets its own right axis.
    b = ax[0, 1]
    b2 = b.twinx()
    b2.tick_params(axis="y", colors=V.LINE_BY_KEY["H_I_1216"].color)
    for L in V.LINES:
        c = comp[(comp.line == L.key) & (comp[dvc].abs() >= V.HVC)]
        c = c.merge(SL[SLK + ["abin"]], on=SLK)
        # equal weight per galaxy: mean of the per-galaxy means
        pg = c.groupby(["sid", "abin"], observed=True).logN.mean().unstack("abin")
        pg = pg.reindex(columns=range(NBIN))
        mu = pg.mean(axis=0).to_numpy()
        se = (pg.std(axis=0) / np.sqrt(pg.notna().sum(axis=0))).to_numpy()
        tgt = b2 if L.key == "H_I_1216" else b
        tgt.plot(xc, mu, color=L.color, ls=L.ls, lw=L.lw, label=L.label)
        tgt.fill_between(xc, mu - se, mu + se, color=L.color, alpha=0.16, lw=0)
        rep[L.key]["logN"] = mu
    b.set_xlim(0, 360); b.set_xticks(np.arange(0, 361, 60))
    b.set_xlabel(r"disk azimuth of the impact point $\phi$ [deg]")
    b.set_ylabel(r"$\langle \log_{10} N_{\rm HVC}\rangle$ (fitted, $\mathrm{cm^{-2}}$)")
    b2.set_ylabel(r"$\langle \log_{10} N_{\rm HVC}\rangle$: $\mathrm{H\,I}$ only",
                  color=V.LINE_BY_KEY["H_I_1216"].color)
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    b.legend(handles=h2 + h1, labels=l2 + l1, ncol=3, fontsize=7.4, loc="upper center",
             framealpha=0.92)
    yl = b.get_ylim(); b.set_ylim(yl[0], yl[0] + (yl[1] - yl[0]) * 1.42)
    yl = b2.get_ylim(); b2.set_ylim(yl[0], yl[0] + (yl[1] - yl[0]) * 1.42)
    S.grid(b); S.panel_label(b, "(b)")
    S.tag(b, r"error $=$ s.e.m. over 20 galaxies", "lr", fs=7.6)

    # ── (c) blue- vs red-shifted HVC covering fraction vs azimuth ────────────────────
    # ION is encoded by LINESTYLE, blue/red by COLOUR.  The three REP3 transitions all
    # carry ls="-" in the line registry, so the registry style cannot be used here: three
    # identical solid curves in two colours are unassignable to an ion.
    c = ax[1, 0]
    C3_LS = {"C_II_1335": (0, (1, 1.5)), "Si_II_1260": "-", "Si_III_1206": (0, (5, 2))}
    C3_LW = {"C_II_1335": 2.1, "Si_II_1260": 2.0, "Si_III_1206": 1.7}
    asym = {}
    for L in [V.LINE_BY_KEY[k] for k in REP3]:
        for sgn, col in (("blue", S.INFLOW), ("red", S.OUTFLOW)):
            K, N = kn_grid(comp, stat, SL, L.key, dvc, sids, signed=sgn)
            f = frac(K.sum(0), N.sum(0))
            c.plot(xc, f, color=col, ls=C3_LS[L.key], lw=C3_LW[L.key], alpha=0.95)
            if L.key == PRIMARY:
                lo, hi = V.wilson(K.sum(0), N.sum(0))
                c.fill_between(xc, lo, hi, color=col, alpha=0.20, lw=0)
            asym.setdefault(L.key, {})[sgn] = (K, N, f)
    hl = [plt.Line2D([], [], color="0.25", ls=C3_LS[k], lw=C3_LW[k],
                     label=V.LINE_BY_KEY[k].label) for k in REP3]
    hl += [plt.Line2D([], [], color=S.INFLOW, lw=2.4, label=r"blueshifted ($\Delta v<0$)"),
           plt.Line2D([], [], color=S.OUTFLOW, lw=2.4, label=r"redshifted ($\Delta v>0$)")]
    c.legend(handles=hl, ncol=2, fontsize=7.8, loc="upper center", framealpha=0.92,
             title=r"transition $=$ linestyle, sign $=$ colour", title_fontsize=7.2)
    c.set_xlim(0, 360); c.set_xticks(np.arange(0, 361, 60))
    c.set_ylim(0, 1.62 * np.nanmax([v[2] for d_ in asym.values() for v in d_.values()]))
    c.set_xlabel(r"disk azimuth of the impact point $\phi$ [deg]"
                 "\n" r"($\phi=0$ is an arbitrary, galaxy-dependent in-plane direction)")
    c.set_ylabel(r"signed HVC covering fraction [fraction]")
    S.grid(c); S.panel_label(c, "(c)")
    txt = []
    for k in REP3:
        kb = asym[k]["blue"][0].sum(); kr = asym[k]["red"][0].sum(); nn = asym[k]["blue"][1].sum()
        txt.append(r"%s: $f_{\rm blue}-f_{\rm red}=%+.3f$"
                   % (V.LINE_BY_KEY[k].label, (kb - kr) / nn))
        rep[k]["blue_red"] = (kb / nn, kr / nn)
    S.tag(c, "\n".join(txt) + "\n"
          + r"no galaxy-common $\phi$ pattern: see (d)", "lr", fs=7.0)

    # ── (d) is the azimuthal modulation significant? ──────────────────────────────────
    d = ax[1, 1]
    xk = np.arange(len(V.LINES))
    st_rms = np.array([rep[L.key]["rms"] for L in V.LINES])
    st_n95 = np.array([rep[L.key]["n95"] for L in V.LINES])
    wg, sh, sh0, ex, chi2, chi2_0 = [], [], [], [], [], []
    for L in V.LINES:
        vif = vif_per_galaxy(comp, stat, SL, L.key, dvc, sids)
        o, s_, e_, c_ = within_galaxy_excess(rep[L.key]["K"], rep[L.key]["N"], vif)
        _, s0, _, c0 = within_galaxy_excess(rep[L.key]["K"], rep[L.key]["N"])
        wg.append(np.nanmedian(o)); sh.append(np.nanmedian(s_)); sh0.append(np.nanmedian(s0))
        ex.append(np.nanmedian(e_)); chi2.append(np.nanmedian(c_)); chi2_0.append(np.nanmedian(c0))
        rep[L.key].update(wg_rms=np.nanmedian(o), shot=np.nanmedian(s_),
                          shot_naive=np.nanmedian(s0), excess=np.nanmedian(e_),
                          chi2=np.nanmedian(c_), chi2_naive=np.nanmedian(c0),
                          vif=float(np.nanmedian(vif)))
    d.bar(xk - 0.20, st_rms, 0.38, color=S.ACCENT, alpha=0.85,
          label=r"stacked $f(\phi)$: RMS about the mean")
    d.bar(xk + 0.20, wg, 0.38, color=S.CLASS["HVC"], alpha=0.80,
          label=r"single galaxy: RMS of $f(\phi)$ (median)")
    d.plot(xk - 0.20, st_n95, "_", color="0.15", ms=13, mew=2.0, ls="none")
    d.plot(xk + 0.20, sh, "_", color="0.15", ms=13, mew=2.0, ls="none")
    d.plot(xk + 0.20, sh0, "_", color="0.55", ms=13, mew=1.4, ls="none")
    hd, lb = d.get_legend_handles_labels()
    hd = hd[:2] + [plt.Line2D([], [], color="0.15", marker="_", ls="none", ms=13, mew=2.0),
                   plt.Line2D([], [], color="0.55", marker="_", ls="none", ms=13, mew=1.4)]
    lb = lb[:2] + [r"null (shift 95th pct $|$ binomial$\times\sqrt{\rm VIF}$)",
                   r"naive binomial null (assumes independence -- too low)"]
    d.set_xticks(xk)
    d.set_xticklabels([L.label for L in V.LINES], rotation=42, ha="right", fontsize=7.4)
    d.set_ylabel(r"azimuthal modulation amplitude of $f_{\rm c}$ [fraction]")
    d.set_ylim(0, 1.30 * max(np.nanmax(wg), np.nanmax(st_rms)))
    leg_ul(d, handles=hd, labels=lb, fontsize=6.9, framealpha=0.9)
    S.grid(d); S.panel_label(d, "(d)")
    # placed under the axis: the bars fill the panel and any in-panel box collides
    # with the four-entry legend.
    d.set_xlabel(
        r"STACKED bars all sit BELOW their null $\Rightarrow$ no galaxy-common azimuthal "
        r"pattern; $\phi_0$ is arbitrary per galaxy, so none could be detected." + "\n"
        + r"SINGLE-GALAXY bars exceed the correlation-corrected null $\Rightarrow$ real "
        r"WITHIN-galaxy azimuthal structure (the naive null would overstate it).",
        fontsize=6.8, labelpad=6)

    fig.tight_layout()
    D["fig18"] = rep
    return save_fig(fig, "fig18_covfrac_vs_orientation")


# ═══════════════════════════════════════════════════════════════════════════════════════
# fig19 -- per-galaxy breakdown
# ═══════════════════════════════════════════════════════════════════════════════════════
def fig19(D):
    comp, stat, SL, sids, dvc, gal = D["comp"], D["stat"], D["SL"], D["sids"], D["dvcol"], D["gal"]
    g = gal.set_index("sid").loc[sids].reset_index()

    per = {}
    for L in V.LINES:
        k, n = counts_by(comp, stat, SL, L.key, dvc, ["sid"])
        per[L.key] = pd.DataFrame(dict(sid=n.index, k=k.to_numpy(), n=n.to_numpy()))
        per[L.key]["f"] = per[L.key].k / per[L.key].n
    blue, red = {}, {}
    for L in V.LINES:
        kb, nb_ = counts_by(comp, stat, SL, L.key, dvc, ["sid"], signed="blue")
        kr, _ = counts_by(comp, stat, SL, L.key, dvc, ["sid"], signed="red")
        blue[L.key] = (kb / nb_).reindex(sids).to_numpy()
        red[L.key] = (kr / nb_).reindex(sids).to_numpy()

    fig = plt.figure(figsize=(15.6, 9.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.30, 1.0, 1.0], hspace=0.30, wspace=0.28)
    a = fig.add_subplot(gs[:, 0])
    b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[0, 2])
    e = fig.add_subplot(gs[1, 1:])

    # ── (a) per-galaxy HVC covering fraction, sorted by the primary line ──────────────
    order = per[PRIMARY].sort_values("f").sid.to_numpy()
    y = np.arange(len(order))
    for L in [V.LINE_BY_KEY[k] for k in REP]:
        t = per[L.key].set_index("sid").loc[order]
        lo, hi = V.wilson(t.k.to_numpy(), t.n.to_numpy())
        a.errorbar(t.f, y, xerr=[t.f - lo, hi - t.f], fmt="o", ms=4.2, lw=0, elinewidth=1.1,
                   color=L.color, mfc=L.color if L.ls == "-" else "white",
                   mec=L.color, capsize=0, label=L.label)
        a.axvline(t.k.sum() / t.n.sum(), color=L.color, ls="--", lw=1.2, alpha=0.75)
    a.set_yticks(y)
    inmap = g.set_index("sid").rho_in_disk.to_dict()
    dag = r"$^{\dagger}$"
    a.set_yticklabels([str(s) + ("" if inmap[s] else dag) for s in order], fontsize=7.6)
    a.set_ylim(-0.8, len(order) - 0.2)
    # data-driven x limit: a hardcoded 0.86 silently clipped 18/20 H I points in --raw mode
    fmax = max(float(per[k].f.max()) for k in REP)
    a.set_xlim(0, min(1.0, 1.10 * fmax))
    a.set_xlabel(r"HVC covering fraction $f_{\rm c}$ [fraction]")
    a.set_ylabel(r"subhalo ID  ($^{\dagger}$: $\rho$ beyond $R_{\rm disk}$)")
    a.legend(fontsize=7.6, loc="center right", framealpha=0.92)
    S.grid(a); S.panel_label(a, "(a)")
    S.tag(a, "dashed: sample mean\n" + D["stag"], "lr", fs=7.4)

    # ── (b)/(c) f_c vs galaxy property ───────────────────────────────────────────────
    def prop_panel(ax_, xv, xlab, lines, logx=False):
        txt = []
        for L in [V.LINE_BY_KEY[k] for k in lines]:
            t = per[L.key].set_index("sid").loc[sids]
            lo, hi = V.wilson(t.k.to_numpy(), t.n.to_numpy())
            inn = g.rho_in_disk.to_numpy()
            ax_.errorbar(xv, t.f, yerr=[t.f - lo, hi - t.f], fmt="none",
                         ecolor=L.color, elinewidth=1.0, alpha=0.7)
            ax_.scatter(xv[inn], t.f.to_numpy()[inn], s=42, color=L.color, marker="o",
                        zorder=4, label=L.label)
            ax_.scatter(xv[~inn], t.f.to_numpy()[~inn], s=46, facecolor="white",
                        edgecolor=L.color, marker="o", lw=1.5, zorder=4)
            rho_s, p_s = spearmanr(xv, t.f)
            txt.append((L.key, rho_s, p_s))
            D["spearman"].append((L.key, xlab, rho_s, p_s))
        lab = "\n".join([r"%s: $r_{\rm s}=%+.2f$, $p=%.3f$" %
                         (V.LINE_BY_KEY[k].label, r_, p_) for k, r_, p_ in txt])
        tag_at(ax_, lab, 0.115, 0.985, fs=7.2)
        if logx:
            ax_.set_xscale("log")
        ax_.set_xlabel(xlab)
        ax_.set_ylabel(r"HVC covering fraction $f_{\rm c}$ [fraction]")
        yl = ax_.get_ylim(); sp = yl[1] - yl[0]
        ax_.set_ylim(yl[0] - 0.06 * sp, yl[0] + 1.34 * sp)
        S.grid(ax_)

    prop_panel(b, g.logMstar.to_numpy(), r"$\log_{10}(M_\star/M_\odot)$", REP3)
    b.legend(fontsize=7.4, loc="lower right", framealpha=0.9)
    S.panel_label(b, "(b)")
    prop_panel(c, g.SFR_Msun_yr.to_numpy(), r"${\rm SFR}\ [M_\odot\,{\rm yr^{-1}}]$", REP3)
    S.panel_label(c, "(c)")
    c.scatter([], [], s=42, color="0.35", label=r"$\rho<R_{\rm disk}$")
    c.scatter([], [], s=46, facecolor="white", edgecolor="0.35", lw=1.5,
              label=r"$\rho\geq R_{\rm disk}$")
    c.legend(fontsize=7.4, loc="lower right", framealpha=0.9)

    # ── (d) blue-red HVC asymmetry per galaxy vs SFR (and M200c) ─────────────────────
    sfr = g.SFR_Msun_yr.to_numpy()
    txt = []
    for L in [V.LINE_BY_KEY[k] for k in REP3]:
        A = blue[L.key] - red[L.key]
        inn = g.rho_in_disk.to_numpy()
        e.scatter(sfr[inn], A[inn], s=44, color=L.color, marker="o", zorder=4, label=L.label)
        e.scatter(sfr[~inn], A[~inn], s=48, facecolor="white", edgecolor=L.color, lw=1.5,
                  marker="o", zorder=4)
        rs, ps = spearmanr(sfr, A); rm, pm = spearmanr(g.logM200c.to_numpy(), A)
        txt.append((L.key, rs, ps, rm, pm))
        D["spearman"].append((L.key, "blue-red asym vs SFR", rs, ps))
        D["spearman"].append((L.key, "blue-red asym vs logM200c", rm, pm))
    e.axhline(0, color="0.3", ls="--", lw=1.3)
    e.set_xlabel(r"${\rm SFR}\ [M_\odot\,{\rm yr^{-1}}]$")
    e.set_ylabel(r"$f_{\rm blue}-f_{\rm red}$  (HVC) [fraction]")
    yl = e.get_ylim()
    e.set_ylim(yl[0] - 0.055 * (yl[1] - yl[0]), yl[0] + (yl[1] - yl[0]) * 1.30)
    e.set_xlim(*c.get_xlim())          # same SFR range as panel (c) -- they share the axis
    e.legend(fontsize=7.6, loc="lower left", framealpha=0.9, ncol=3)
    S.tag(e, "\n".join([r"%s: SFR $r_{\rm s}=%+.2f\,(p=%.3f)$; $M_{200c}$ $r_{\rm s}=%+.2f\,(p=%.3f)$"
                        % (V.LINE_BY_KEY[k].label, rs, ps, rm, pm)
                        for k, rs, ps, rm, pm in txt]), "ur", fs=7.0)
    S.grid(e); S.panel_label(e, "(d)")

    D["fig19"] = dict(per=per, blue=blue, red=red, gal=g)
    return save_fig(fig, "fig19_covfrac_vs_galaxy")


# ═══════════════════════════════════════════════════════════════════════════════════════
# fig20 -- disk-edge fallback systematics
# ═══════════════════════════════════════════════════════════════════════════════════════
def fig20(D):
    comp, stat, SL, dvc, var = D["comp"], D["stat"], D["SL"], D["dvcol"], D["variant"]
    flag = f"{var}_edge_fallback"
    SLf = SL.copy()
    SLf["fb"] = SLf[flag].astype(bool)
    sub = {"direct": SLf.loc[~SLf.fb, SLK], "fallback": SLf.loc[SLf.fb, SLK]}
    STY = {"direct": dict(color=S.VMODE["direct_cool"], hatch=None),
           "fallback": dict(color=S.VMODE["R95-edge"], hatch="//")}
    # A raw direct-vs-fallback split is CONFOUNDED WITH GALAXY: whether rho lies inside the
    # cold disk is a property of the galaxy, so most galaxies are ~100 per cent one or the
    # other.  The controlled comparison is the paired one, restricted to the galaxies that
    # contain BOTH kinds of sightline.
    ffrac = SLf.groupby("sid").fb.mean()
    mixed = ffrac[(ffrac > 0.05) & (ffrac < 0.95)].index.to_numpy()
    mx = {"direct": SLf.loc[~SLf.fb & SLf.sid.isin(mixed), SLK],
          "fallback": SLf.loc[SLf.fb & SLf.sid.isin(mixed), SLK]}

    fig, ax = plt.subplots(2, 2, figsize=(14.0, 10.0))
    xk = np.arange(len(V.LINES))
    res = {}

    # ── (a) HVC covering fraction: direct vs fallback ────────────────────────────────
    a = ax[0, 0]
    for j, (nm, s_) in enumerate(sub.items()):
        f, lo, hi = [], [], []
        for L in V.LINES:
            k, n = counts_by(comp, stat, SL, L.key, dvc, ["sid"], sl_sub=s_)
            kk, nn = float(k.sum()), float(n.sum())
            f.append(kk / nn); l_, h_ = V.wilson(kk, nn); lo.append(l_); hi.append(h_)
            res.setdefault(L.key, {})[nm] = dict(f=kk / nn, k=kk, n=nn)
        f = np.array(f)
        a.bar(xk + (j - 0.5) * 0.38, f, 0.36, color=STY[nm]["color"], alpha=0.85,
              hatch=STY[nm]["hatch"], edgecolor="white",
              label=(r"direct $v_{\rm ISM}$" if nm == "direct" else r"disk-edge fallback"))
        a.errorbar(xk + (j - 0.5) * 0.38, f, yerr=[f - np.array(lo), np.array(hi) - f],
                   fmt="none", ecolor="0.2", elinewidth=1.1)
    # galaxy-matched (paired) control -- only galaxies holding both kinds of sightline
    for j, (nm, s_) in enumerate(mx.items()):
        fm = []
        for L in V.LINES:
            k, n = counts_by(comp, stat, SL, L.key, dvc, ["sid"], sl_sub=s_)
            fm.append(float(k.sum()) / float(n.sum()))
            res.setdefault(L.key, {}).setdefault("matched", {})[nm] = fm[-1]
            per_g = (k / n).reindex(mixed)
            res[L.key].setdefault("matched_pg", {})[nm] = per_g
        a.plot(xk + (j - 0.5) * 0.38, fm, "D", ms=5.6, mfc="white", mec="0.1", mew=1.4,
               ls="none", zorder=6,
               label=(r"galaxy-matched, POOLED ($n_{\rm gal}=%d$)" % len(mixed))
               if j == 0 else None)
    for i, L in enumerate(V.LINES):
        pg = res[L.key]["matched_pg"]
        dd = (pg["fallback"] - pg["direct"]).dropna()
        res[L.key]["paired"] = (float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(len(dd))))
        top = max(res[L.key]["fallback"]["f"], res[L.key]["direct"]["f"],
                  res[L.key]["matched"]["direct"], res[L.key]["matched"]["fallback"])
        a.text(i, top + 0.035, r"$%+.3f$" % res[L.key]["paired"][0], ha="center",
               va="bottom", fontsize=6.6, color="0.15")
    a.set_xticks(xk)
    a.set_xticklabels([L.label for L in V.LINES], rotation=42, ha="right", fontsize=7.4)
    a.set_ylabel(r"HVC covering fraction $f_{\rm c}$ [fraction]")
    a.set_ylim(0, min(1.0, 1.62 * max(max(res[L.key]["direct"]["f"], res[L.key]["fallback"]["f"])
                                      for L in V.LINES)))
    leg_ul(a, fontsize=7.4, ncol=1, framealpha=0.92)
    S.grid(a); S.panel_label(a, "(a)")
    nfb = int(SLf.fb.sum())
    S.tag(a, (r"$%d/%d$ sightlines use the fallback" % (nfb, len(SLf))) + "\n" + D["stag"],
          "ur", fs=7.0)
    a.set_xlabel(
        r"diamonds: POOLED $f_{\rm c}$ over the matched galaxies.  Numbers: MEAN of the "
        r"PER-GALAXY $f_{\rm c}^{\rm fallback}-f_{\rm c}^{\rm direct}$." + "\n"
        + r"The two estimators differ because galaxies carry unequal $n$; $+$ means the "
        r"fallback subset has the HIGHER $f_{\rm c}$.", fontsize=6.8, labelpad=6)

    # ── (b) blue / red split, direct vs fallback ─────────────────────────────────────
    b = ax[0, 1]
    for nm, s_ in sub.items():
        for sg, col in (("blue", S.INFLOW), ("red", S.OUTFLOW)):
            f, er = [], []
            for L in V.LINES:
                k, n = counts_by(comp, stat, SL, L.key, dvc, ["sid"], signed=sg, sl_sub=s_)
                kk, nn = float(k.sum()), float(n.sum())
                f.append(kk / nn); lo, hi = V.wilson(kk, nn); er.append((kk / nn - lo, hi - kk / nn))
                res[L.key].setdefault("signed", {})[(nm, sg)] = kk / nn
            f = np.array(f); er = np.array(er).T
            off = -0.13 if nm == "direct" else 0.13
            b.errorbar(xk + off, f, yerr=er, fmt="o" if sg == "blue" else "s", ms=5.2,
                       color=col, mfc=col if nm == "direct" else "white", mec=col,
                       elinewidth=1.0, lw=0, capsize=0)
    hb = [plt.Line2D([], [], color=S.INFLOW, marker="o", lw=0, ms=6, label=r"blueshifted"),
          plt.Line2D([], [], color=S.OUTFLOW, marker="s", lw=0, ms=6, label=r"redshifted"),
          plt.Line2D([], [], color="0.3", marker="o", lw=0, ms=6, label=r"direct"),
          plt.Line2D([], [], color="0.3", marker="o", lw=0, ms=6, mfc="white",
                     label=r"fallback")]
    leg_ul(b, handles=hb, fontsize=7.6, ncol=2, framealpha=0.92)
    b.set_xticks(xk)
    b.set_xticklabels([L.label for L in V.LINES], rotation=42, ha="right", fontsize=7.4)
    b.set_ylabel(r"signed HVC covering fraction [fraction]")
    yl = b.get_ylim(); b.set_ylim(yl[0], yl[0] + (yl[1] - yl[0]) * 1.22)
    S.grid(b); S.panel_label(b, "(b)")

    # ── (c) v3a vs v3b agreement within each subset ──────────────────────────────────
    c = ax[1, 0]
    dv = (SLf.v_ism_v3a - SLf.v_ism_v3b)
    bins = np.linspace(-160, 160, 65)
    for nm in sub:
        x = dv[SLf.fb == (nm == "fallback")].dropna().to_numpy()
        h, _ = np.histogram(x, bins=bins, density=True)
        c.step(0.5 * (bins[:-1] + bins[1:]), h, where="mid", color=STY[nm]["color"], lw=2.0,
               label=(r"direct: med $%+.1f$, NMAD $%.1f$" % (np.median(x), 1.4826 * np.median(np.abs(x - np.median(x)))))
               if nm == "direct" else
               (r"fallback: med $%+.1f$, NMAD $%.1f$" % (np.median(x), 1.4826 * np.median(np.abs(x - np.median(x))))))
        res.setdefault("_vism", {})[nm] = (float(np.median(x)),
                                           float(1.4826 * np.median(np.abs(x - np.median(x)))))
    S.shade_classes(c, xmax=160.0, alpha=0.07)
    c.axvline(0, color="0.3", ls=":", lw=1.2)
    c.set_xlim(-160, 160); c.set_yscale("log"); c.set_ylim(1e-5, 1.0)
    c.set_xlabel(r"$v_{\rm ISM}^{\rm v3a}-v_{\rm ISM}^{\rm v3b}\ [\mathrm{km\,s^{-1}}]$")
    c.set_ylabel(r"sightline PDF")
    leg_ul(c, fontsize=7.8, framealpha=0.92)
    S.grid(c); S.panel_label(c, "(c)")
    S.tag(c, r"shading: ISM / IVC / HVC $|\Delta v|$ bands", "ur", fs=7.4)

    # ── (d) f_c(v3a) vs f_c(v3b), per line and subset ────────────────────────────────
    d = ax[1, 1]
    for nm, s_ in sub.items():
        fa, fb2 = [], []
        for L in V.LINES:
            ka, na = counts_by(comp, stat, SL, L.key, "dv_v3a", ["sid"], sl_sub=s_)
            kb, nb2 = counts_by(comp, stat, SL, L.key, "dv_v3b", ["sid"], sl_sub=s_)
            fa.append(ka.sum() / na.sum()); fb2.append(kb.sum() / nb2.sum())
            res[L.key].setdefault("ab", {})[nm] = (fa[-1], fb2[-1])
        d.scatter(fa, fb2, s=58, c=[L.color for L in V.LINES],
                  marker="o" if nm == "direct" else "s",
                  edgecolor="0.2" if nm == "direct" else STY[nm]["color"],
                  linewidths=1.1, alpha=0.9,
                  label=("direct" if nm == "direct" else "fallback"))
    d.plot([0, 1], [0, 1], color="0.35", ls="--", lw=1.2)
    fall = [v for L in V.LINES for nm in sub for v in res[L.key]["ab"][nm]]
    lo_, hi_ = min(fall), max(fall)
    pad = 0.12 * (hi_ - lo_)
    d.set_xlim(lo_ - pad, hi_ + pad); d.set_ylim(lo_ - pad, hi_ + 1.9 * pad)
    d.set_xlabel(r"$f_{\rm c}$ using $\Delta v_{\rm v3a}$")
    d.set_ylabel(r"$f_{\rm c}$ using $\Delta v_{\rm v3b}$")
    leg_ul(d, fontsize=8.0, framealpha=0.92)
    mx = max(abs(res[L.key]["ab"][nm][0] - res[L.key]["ab"][nm][1])
             for L in V.LINES for nm in sub)
    S.tag(d, r"max $|f_{\rm v3a}-f_{\rm v3b}| = %.4f$" % mx, "lr", fs=7.8)
    S.grid(d); S.panel_label(d, "(d)")

    res["_nmixed"] = len(mixed)
    res["_mixed"] = mixed
    fig.tight_layout()
    D["fig20"] = res
    return save_fig(fig, "fig20_edge_fallback_systematics")


# ═══════════════════════════════════════════════════════════════════════════════════════
# fig21 -- sensitivity of the observables to the v_ISM variant
# ═══════════════════════════════════════════════════════════════════════════════════════
VARLAB = {"dv_v3a": r"v3a (1-kpc tube)", "dv_v3b": r"v3b (binned slit)",
          "dv_v1": r"v1 (direct cool gas)"}
VARCOL = {"dv_v3a": S.VMODE["direct_cool"], "dv_v3b": S.ACCENT, "dv_v1": S.CLASS["IVC"]}


def fig21(D):
    comp, stat, SL = D["comp"], D["stat"], D["SL"]
    v1sub = SL.loc[SL.v1_defined, SLK]
    fig, ax = plt.subplots(2, 2, figsize=(14.0, 10.0))
    xk = np.arange(len(V.LINES))
    res = {}

    # ── (a) full sample: v3a vs v3b ──────────────────────────────────────────────────
    a = ax[0, 0]
    for j, dvc in enumerate(["dv_v3a", "dv_v3b"]):
        f, er = [], []
        for L in V.LINES:
            k, n = counts_by(comp, stat, SL, L.key, dvc, ["sid"])
            kk, nn = float(k.sum()), float(n.sum())
            f.append(kk / nn); lo, hi = V.wilson(kk, nn); er.append((kk / nn - lo, hi - kk / nn))
            res.setdefault(L.key, {})[("full", dvc)] = kk / nn
        f = np.array(f)
        a.bar(xk + (j - 0.5) * 0.38, f, 0.36, color=VARCOL[dvc], alpha=0.85,
              edgecolor="white", label=VARLAB[dvc])
        a.errorbar(xk + (j - 0.5) * 0.38, f, yerr=np.array(er).T, fmt="none",
                   ecolor="0.2", elinewidth=1.1)
    for i, L in enumerate(V.LINES):
        d_ = res[L.key][("full", "dv_v3b")] - res[L.key][("full", "dv_v3a")]
        a.text(i, max(res[L.key][("full", "dv_v3a")], res[L.key][("full", "dv_v3b")]) + 0.02,
               f"{d_:+.4f}", ha="center", fontsize=6.6, color="0.25")
    a.set_xticks(xk)
    a.set_xticklabels([L.label for L in V.LINES], rotation=42, ha="right", fontsize=7.4)
    a.set_ylabel(r"HVC covering fraction $f_{\rm c}$ [fraction]")
    a.set_ylim(0, min(1.0, 1.30 * float(np.nanmax(
        [res[L.key][("full", v_)] for L in V.LINES for v_ in ("dv_v3a", "dv_v3b")]))))
    leg_ul(a, fontsize=8.0, framealpha=0.92)
    S.grid(a); S.panel_label(a, "(a)")
    S.tag(a, r"all 14{,}400 sightlines" + "\n" + D["stag"], "ur", fs=7.4)

    # ── (b) v1-defined subset: all three variants on identical sightlines ────────────
    b = ax[0, 1]
    for j, dvc in enumerate(["dv_v3a", "dv_v3b", "dv_v1"]):
        f, er = [], []
        for L in V.LINES:
            k, n = counts_by(comp, stat, SL, L.key, dvc, ["sid"], sl_sub=v1sub)
            kk, nn = float(k.sum()), float(n.sum())
            f.append(kk / nn); lo, hi = V.wilson(kk, nn); er.append((kk / nn - lo, hi - kk / nn))
            res[L.key][("v1sub", dvc)] = kk / nn
            res[L.key]["n_v1sub"] = nn
        f = np.array(f)
        b.bar(xk + (j - 1) * 0.27, f, 0.25, color=VARCOL[dvc], alpha=0.85,
              edgecolor="white", label=VARLAB[dvc])
        b.errorbar(xk + (j - 1) * 0.27, f, yerr=np.array(er).T, fmt="none",
                   ecolor="0.2", elinewidth=1.0)
    b.set_xticks(xk)
    b.set_xticklabels([L.label for L in V.LINES], rotation=42, ha="right", fontsize=7.4)
    b.set_ylabel(r"HVC covering fraction $f_{\rm c}$ [fraction]")
    b.set_ylim(0, min(1.0, 1.30 * float(np.nanmax(
        [res[L.key][("v1sub", v_)] for L in V.LINES
         for v_ in ("dv_v3a", "dv_v3b", "dv_v1")]))))
    leg_ul(b, fontsize=8.0, framealpha=0.92)
    S.grid(b); S.panel_label(b, "(b)")
    S.tag(b, r"common subset: $v_{\rm ISM}^{\rm v1}$ defined ($%d$ sightlines)"
          % int(len(v1sub)), "ur", fs=7.4)

    # ── (c) per-sightline difference between the v_ISM variants ─────────────────────
    c = ax[1, 0]
    pairs = [("v_ism_v3a", "v_ism_v3b", r"v3a $-$ v3b", S.ACCENT),
             ("v_ism_v3a", "v_ism_direct_cool", r"v3a $-$ v1", S.VMODE["direct_cool"]),
             ("v_ism_v3b", "v_ism_direct_cool", r"v3b $-$ v1", S.CLASS["IVC"])]
    bins = np.linspace(-250, 250, 101)
    xcb = 0.5 * (bins[:-1] + bins[1:])
    for p, q, lab, col in pairs:
        x = (SL[p] - SL[q]).dropna().to_numpy()
        h, _ = np.histogram(x, bins=bins, density=True)
        med = float(np.median(x)); nmad = float(1.4826 * np.median(np.abs(x - med)))
        f100 = float((np.abs(x) >= V.HVC).mean())
        c.step(xcb, h, where="mid", color=col, lw=2.1,
               label=lab + r": med $%+.1f$, NMAD $%.1f$" % (med, nmad))
        res.setdefault("_dv", {})[lab] = (med, nmad, f100, len(x))
    S.shade_classes(c, xmax=250.0, alpha=0.07)
    c.axvline(0, color="0.3", ls=":", lw=1.2)
    c.set_xlim(-250, 250); c.set_yscale("log")
    c.set_xlabel(r"$\Delta v_{\rm ISM}$ between variants $[\mathrm{km\,s^{-1}}]$")
    c.set_ylabel(r"sightline PDF")
    leg_ul(c, fontsize=7.6, framealpha=0.92)
    S.grid(c); S.panel_label(c, "(c)")
    S.tag(c, r"shading: ISM / IVC / HVC $|\Delta v|$ bands", "ur", fs=7.4)

    # ── (d) fraction of components that change kinematic class ──────────────────────
    d = ax[1, 1]
    combos = [("dv_v3a", "dv_v3b", "v3a->v3b", r"v3a $\to$ v3b", S.ACCENT),
              ("dv_v3b", "dv_v1", "v3b->v1", r"v3b $\to$ v1", S.CLASS["IVC"])]
    for j, (p, q, key, lab, col) in enumerate(combos):
        fr = []
        for L in V.LINES:
            cc = comp[(comp.line == L.key) & comp[p].notna() & comp[q].notna()]
            ch = (S.classify(cc[p].to_numpy()) != S.classify(cc[q].to_numpy()))
            fr.append(float(ch.mean()) if len(cc) else np.nan)
            res[L.key][("class_change", key)] = fr[-1]
        d.bar(xk + (j - 0.5) * 0.38, fr, 0.36, color=col, alpha=0.85, edgecolor="white",
              label=lab)
        for i, v in enumerate(fr):
            d.text(i + (j - 0.5) * 0.38, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.4,
                   color="0.25", rotation=90)
    d.set_xticks(xk)
    d.set_xticklabels([L.label for L in V.LINES], rotation=42, ha="right", fontsize=7.4)
    d.set_ylabel(r"fraction of components changing ISM/IVC/HVC class")
    d.set_ylim(0, 1.30 * np.nanmax([res[L.key][("class_change", "v3b->v1")] for L in V.LINES]))
    leg_ul(d, fontsize=8.0, framealpha=0.92)
    S.grid(d); S.panel_label(d, "(d)")
    S.tag(d, r"class boundaries at $|\Delta v| = 40,\,100\ \mathrm{km\,s^{-1}}$", "ur", fs=7.4)

    fig.tight_layout()
    D["fig21"] = res
    return save_fig(fig, "fig21_vism_variant_sensitivity")


# ═══════════════════════════════════════════════════════════════════════════════════════
# quantitative report
# ═══════════════════════════════════════════════════════════════════════════════════════
def report(D):
    var = D["variant"]
    print("\n" + "=" * 92)
    print(f"QUANTITATIVE REPORT  --  variant {var}, "
          f"{'clean' if D['clean'] else 'raw'} sample")
    print("=" * 92)

    print("\n[sample] HVC covering fraction, RAW vs CLEAN "
          "(clean = ~b_at_ceiling & ~beyond_common_window)")
    print(f"{'line':14s} {'n_fit':>7s} {'f_raw':>7s} {'f_clean':>8s} {'delta':>7s} "
          f"{'b_ceil':>7s} {'>500':>6s}")
    for L in V.LINES:
        fr, _, n = V.covering_fraction(D["comp_raw"], D["stat"], D["dvcol"][3:], L.key)
        fc, _, _ = V.covering_fraction(D["comp"], D["stat"], D["dvcol"][3:], L.key)
        sub = D["comp_raw"][D["comp_raw"].line == L.key]
        print(f"{L.key:14s} {n:7d} {fr:7.3f} {fc:8.3f} {fc-fr:+7.3f} "
              f"{sub.b_at_ceiling.mean():7.3f} {sub.beyond_common_window.mean():6.3f}")

    r18 = D["fig18"]
    print("\n[fig18] azimuthal modulation of the HVC covering fraction "
          f"({NBIN} bins of {BW:.0f} deg)")
    print(f"{'line':14s} {'<f_c>':>7s} {'stack RMS':>10s} {'null95':>8s} {'p':>7s} "
          f"{'gal RMS':>8s} {'shot':>7s} {'excess':>7s} {'VIF':>5s} "
          f"{'chi2/dof':>9s} {'chi2/dof_naive':>15s}")
    for L in V.LINES:
        r = r18[L.key]
        print(f"{L.key:14s} {r['mean']:7.3f} {r['rms']:10.4f} {r['n95']:8.4f} "
              f"{r['p']:7.3f} {r['wg_rms']:8.4f} {r['shot']:7.4f} {r['excess']:7.4f} "
              f"{r['vif']:5.1f} {r['chi2']:9.2f} {r['chi2_naive']:15.2f}")
    print("  stack RMS  = RMS of the 20-galaxy-stacked f(phi) about its mean")
    print("  null95     = 95th pct of the same statistic when each galaxy's profile is")
    print("               circularly shifted (preserves each galaxy's own binned profile in")
    print("               full; destroys only the inter-galaxy phase alignment).  p >~ 0.3")
    print("               for every line -> NO galaxy-common azimuthal pattern.  NOTE this")
    print("               is EXPECTED: phi=0 is e1 = unit(n_disk x y_box), an arbitrary")
    print("               per-galaxy direction, so a common pattern could not survive the")
    print("               stack.  The stacked panel carries no azimuthal information.")
    print("  gal RMS / shot / excess = median over galaxies of the within-galaxy RMS of")
    print("               f(phi), its shot-noise expectation and the excess in quadrature.")
    print("  VIF        = median variance-inflation factor of a 15-alpha bin mean, measured")
    print("               from the alpha-ordered detection sequence (adjacent sightlines are")
    print("               0.45 kpc apart and correlated).  `shot` and `chi2/dof` INCLUDE it;")
    print("               `chi2/dof_naive` is the independent-Bernoulli value and overstates")
    print("               the within-galaxy significance by exactly this factor.")

    print("\n[fig18b] peak-to-peak of <logN_HVC> vs azimuth (dex)")
    for L in V.LINES:
        y = r18[L.key]["logN"]
        print(f"  {L.key:14s} mean {np.nanmean(y):6.3f}  p2p {np.nanmax(y)-np.nanmin(y):6.3f}")

    print("\n[fig19] per-galaxy HVC covering fraction")
    per = D["fig19"]["per"]
    for L in [V.LINE_BY_KEY[k] for k in REP]:
        t = per[L.key]
        print(f"  {L.key:14s} mean {t.k.sum()/t.n.sum():.3f}  "
              f"range {t.f.min():.3f}-{t.f.max():.3f}  gal-to-gal sigma {t.f.std():.3f}")
    print("\n[fig19] Spearman correlations (galaxy level, n = 20)")
    sp = D["spearman"]
    m_tests = len(sp)
    # Benjamini-Hochberg q over the whole family of tests this figure performs
    order = np.argsort([p for *_, p in sp])
    q = np.empty(m_tests)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        prev = min(prev, sp[idx][3] * m_tests / (m_tests - rank))
        q[idx] = prev
    bonf = 0.05 / m_tests
    for i, (key, xlab, rs, ps) in enumerate(sp):
        star = "  *" if ps < 0.05 else "   "
        star += " BONF" if ps < bonf else "     "
        print(f"  {key:14s} vs {xlab:34s} r_s = {rs:+.3f}  p = {ps:.4f}  "
              f"q_BH = {q[i]:.4f}{star}")
    print(f"  MULTIPLE COMPARISONS: this figure performs {m_tests} Spearman tests, so a raw")
    print(f"  p < 0.05 is NOT significant on its own.  Bonferroni alpha = {bonf:.5f};")
    print("  `BONF` marks tests that survive it, q_BH is the Benjamini-Hochberg FDR value.")
    print("  (Widening the family to all 9 lines x 5 galaxy properties = 45 tests gives")
    print("   Bonferroni alpha = 0.00111, which the C II / Si III vs SFR pair still passes.)")

    print("\n[fig20] disk-edge fallback ({} flag)".format(f"{var}_edge_fallback"))
    r20 = D["fig20"]
    print(f"{'line':14s} {'f_direct':>9s} {'f_fallb':>9s} {'delta':>8s} "
          f"{'d_paired':>9s} {'+-sem':>7s} {'blue_dir':>9s} {'blue_fb':>8s} "
          f"{'red_dir':>8s} {'red_fb':>8s}")
    for L in V.LINES:
        r = r20[L.key]
        s_ = r["signed"]
        print(f"{L.key:14s} {r['direct']['f']:9.3f} {r['fallback']['f']:9.3f} "
              f"{r['fallback']['f']-r['direct']['f']:+8.3f} "
              f"{r['paired'][0]:+9.3f} {r['paired'][1]:7.3f} "
              f"{s_[('direct','blue')]:9.3f} {s_[('fallback','blue')]:8.3f} "
              f"{s_[('direct','red')]:8.3f} {s_[('fallback','red')]:8.3f}")
    print(f"  n_direct = {r20[V.LINES[0].key]['direct']['n']:.0f}, "
          f"n_fallback = {r20[V.LINES[0].key]['fallback']['n']:.0f}")
    print(f"  d_paired = mean over the {r20['_nmixed']} galaxies holding BOTH kinds of "
          f"sightline of (f_fallback - f_direct) within that galaxy.  The unpaired `delta`")
    print("             is confounded with galaxy identity (rho<R_disk is a galaxy property).")
    for nm, (med, nmad) in r20["_vism"].items():
        print(f"  v3a - v3b within {nm:9s}: median {med:+6.1f}, NMAD {nmad:5.1f} km/s")
    print("  max |f_c(v3a) - f_c(v3b)| over lines x subsets = "
          f"{max(abs(r20[L.key]['ab'][nm][0]-r20[L.key]['ab'][nm][1]) for L in V.LINES for nm in ('direct','fallback')):.4f}")

    print("\n[fig21] v_ISM variant sensitivity")
    r21 = D["fig21"]
    print(f"{'line':14s} {'v3a(full)':>10s} {'v3b(full)':>10s} {'|d|':>7s} || "
          f"{'v3a(v1s)':>9s} {'v3b(v1s)':>9s} {'v1(v1s)':>9s} {'v3b-v1':>8s}")
    for L in V.LINES:
        r = r21[L.key]
        print(f"{L.key:14s} {r[('full','dv_v3a')]:10.4f} {r[('full','dv_v3b')]:10.4f} "
              f"{abs(r[('full','dv_v3b')]-r[('full','dv_v3a')]):7.4f} || "
              f"{r[('v1sub','dv_v3a')]:9.4f} {r[('v1sub','dv_v3b')]:9.4f} "
              f"{r[('v1sub','dv_v1')]:9.4f} "
              f"{r[('v1sub','dv_v1')]-r[('v1sub','dv_v3b')]:+8.4f}")
    for lab, (med, nmad, f100, n) in r21["_dv"].items():
        print(f"  dv_ISM {lab}: median {med:+6.1f}, NMAD {nmad:5.1f} km/s, "
              f"frac |diff| >= 100 = {f100:.4f}  (n = {n})")
    print(f"{'line':14s} {'class chg v3a->v3b':>19s} {'class chg v3b->v1':>19s}")
    for L in V.LINES:
        r = r21[L.key]
        print("{:14s} {:19.4f} {:19.4f}".format(
            L.key, r[("class_change", "v3a->v3b")], r[("class_change", "v3b->v1")]))
    print("=" * 92 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════════════
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    variant = args[0] if args else "v3b"
    if variant not in ("v3a", "v3b"):
        raise SystemExit(f"variant must be v3a or v3b, got {variant!r}")
    clean = "--raw" not in sys.argv

    usetex = S.set_style()
    S.FIGDIR = OUTBASE / f"paper_figures_{variant}"
    globals()["FILE_SUFFIX"] = "" if clean else "_rawsample"
    print(f"variant={variant}  usetex={usetex}  FIGDIR={S.FIGDIR}  "
          f"sample={'clean' if clean else 'raw'}  file suffix={FILE_SUFFIX!r}")

    comp, comp_raw, stat, SL, gal, sids = load_all(clean)
    check_geometry(SL, variant)

    D = dict(comp=comp, comp_raw=comp_raw, stat=stat, SL=SL, gal=gal, sids=sids,
             variant=variant, dvcol=f"dv_{variant}", clean=clean, spearman=[],
             stag=SAMPLE_TAG[clean])

    for fn in (fig18, fig19, fig20, fig21):
        p = fn(D)
        print(f"  saved {p}")
    report(D)


if __name__ == "__main__":
    main()
