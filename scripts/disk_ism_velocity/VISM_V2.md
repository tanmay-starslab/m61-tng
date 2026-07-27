# v_ISM v2 — supervisor's 3-tracer rotation-curve method (corrected)

Rebuild of the disk-ISM velocity following the supervisor's recipe exactly, with the one
requested change (cold gas T<10³ → T<10⁴ K). All outputs are in NEW directories; nothing in
the v1 pipeline was overwritten.

## The correctness fix (why this matters)
The v1 fiducial curve was the equal-weight average of {cold gas, SF gas, young stars}, but
cold gas was selected at **T<10³ K, which is empty in TNG** (the star-forming EoS floors gas
at ~10³·⁸ K). Verified: `n_cold_gas = 0` in all 60 radial bins, `contrib = "sf_gas;young_stars"`
everywhere. So the "3-tracer" average had silently been **SF gas + young stars only**. Using
**T<10⁴ K** restores cold gas — the dominant, most spatially extended ISM tracer (SF gas and
young stars thin out beyond ~20 kpc, exactly where the sightlines probe at ρ≈26 kpc).

## Method (as implemented)
`v_ISM(sid,mode,α) = v_rot(R_anchor) × proj`, galaxy rest frame:
- `v_rot(R)` = equal-weight, per-radial-bin average of the **median v_φ** of cold gas (T<10⁴),
  SF gas (SFR>0), young stars (age<300 Myr); a bin contributes only if it holds ≥5 particles.
- `R_anchor` = galactocentric cylindrical radius of the sightline anchor (≈ impact parameter).
- `proj = φ̂·los` projects the circular rotation onto the sightline.
- `v_sys = SubhaloVel·los` is removed via the rotation curve (both gas and spectrum are in the
  same rest frame), so v_ISM is directly comparable to the absorption velocity.

## Answers to the review questions
- **Systemic velocity.** `v_sys = SubhaloVel·los(α)` — the galaxy's bulk peculiar velocity
  projected onto that orientation's sightline. It is strongly orientation-dependent: the
  per-galaxy peak-to-peak over α has median **≈323 km/s**. Subtracted identically from gas and
  spectrum (Tier-1 validation already confirmed the two agree to <1 km/s with this v_sys).
- **Binning.** Curves built at 0.5 / 1 / 2 kpc. The operative v_φ at the sampled radius is
  **bin-insensitive**: the induced change in v_ISM from 0.5→2 kpc bins has median **0.0 km/s**
  (see `diag_bin_sensitivity.png`; the three bin-width curves overlie in `rotation_curves/`).
- **Averaging.** Equal-weight mean of the three tracer medians per bin (now including cold gas).
  Cross-checked against the mass-weighted mean; the tracers agree within their 16–84 bands.

## How well does v_ISM land on the absorption? (the acid test)
Overlaying v_ISM on the Si II 1260 spectrum and testing whether it sits within 30 km/s of a
major absorption trough (`diag_spectra_overlay.py`, gallery + landing CSV per SID):

- **Well-sampled disks: it works.** SID 342448 → **84%** of sightlines land on a major
  component, median offset **14 km/s** (see `gallery_sid342448.png`; the model line sits on the
  deep Si II/C II trough in nearly every orientation).
- **Beyond-disk / noisy-outer-curve galaxies: it degrades.** Compact galaxies whose disk ends
  before ρ≈26 kpc (e.g. 432106, 46%) and galaxies with turbulent outer curves (360923, 30%)
  land far less often, because there is no ordered disk rotation to project at that radius.

**Full 20-SID aggregate (14,400 sightlines):** the model lands on a major Si II component
**56%** of the time (median offset to nearest trough 26 km/s), ranging **0.27→0.84** across
galaxies (best for well-sampled disks, worst for beyond-disk/noisy). Distance to the Si II dip:
**MODEL median 37 km/s** (53% within 40 km/s, 82% within 100) vs **DIRECT-v1 median 11 km/s**
(91% within 40, 98% within 100).

Overall the model is **unbiased** vs the Si II dip (median offset −4 km/s) but has **~46 km/s
scatter** (vs the v1 direct along-ray method's ~15 km/s), and is nearly **one value per galaxy**
(v_rot(≈26 kpc)×proj), clustering in two ±100 km/s bands — it captures the mean disk rotation
sense but not the per-sightline ISM structure the direct method resolves. So the supervisor
model marks a major ISM component **~56% of the time overall (≈80%+ for in-disk well-sampled
sightlines)**, while the direct method does so ~91% of the time.

## Takeaway
The supervisor's method is now implemented correctly (cold gas included) and is a clean,
unbiased estimate of the *mean* disk-ISM rotation velocity at the impact radius. It reliably
marks the major ISM absorption for **in-disk, well-sampled sightlines**, but should be flagged
as unreliable (or replaced by the direct method) where ρ is **beyond the disk edge** — which is
the same in_disk boundary the v1 work already identified.

## Files (all new)
- `build_sid_rc_v2.py` (+ `build_rc_v2_from_disk.py`) → `rotation_curves_v2/`
- `stage_b_vism_v2.py` → `vism_tables_v2/vism_master_v2.csv` (14,400 sightlines)
- `diag_rotation_curves.py`, `diag_vsys_bins_compare.py`, `diag_spectra_overlay.py` →
  `diagnostics_v2/`
- `make_paper_figures_v2.py` → `paper_figures_v2/` (9 HVC figures, dv from the model v_ISM)
