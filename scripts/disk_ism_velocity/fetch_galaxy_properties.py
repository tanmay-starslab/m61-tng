#!/usr/bin/env python3
"""Fetch CATALOG stellar mass, M200c, SFR (and context) for the 20 study subhalos from the
IllustrisTNG public API (TNG50-1, snap 99). Values are group-catalog quantities, not computed:
  Mstar  = SubhaloMassType[stars]  (subhalo 'mass_stars')
  SFR    = SubhaloSFR              (subhalo 'sfr')
  M200c  = Group_M_Crit200 of the parent FoF group  (group info.json)
Units: mass_* and Group_M_* are 1e10 Msun/h; lengths ckpc/h. h=0.6774 (z=0 -> comoving=physical).
Raw JSONs cached under outputs/.../galaxy_properties/cache/.
"""
import json, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

KEY = "7f3b66fdf83ae7c85b47794073f746ff"
H = 0.6774
SIDS = [143881, 143884, 143885, 143886, 167395, 307487, 342448, 348901, 352426, 360923,
        375073, 388544, 398784, 413372, 432106, 438148, 452978, 456326, 482889, 488530]
OUT = Path("/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/galaxy_properties")
CACHE = OUT / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
BASE = "http://www.tng-project.org/api/TNG50-1/snapshots/99"


def get(url, cachefile):
    cf = CACHE / cachefile
    if cf.exists():
        return json.loads(cf.read_text())
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"api-key": KEY}), timeout=45))
            cf.write_text(json.dumps(d)); time.sleep(0.4); return d
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2)


rows = []
for sid in SIDS:
    s = get(f"{BASE}/subhalos/{sid}/", f"sub_{sid}.json")
    grnr = s["grnr"]
    g = get(f"{BASE}/halos/{grnr}/info.json", f"group_{grnr}.json")
    Mstar = s["mass_stars"] * 1e10 / H
    Mgas = s["mass_gas"] * 1e10 / H
    Mtot = s["mass"] * 1e10 / H
    M200c = g["Group_M_Crit200"] * 1e10 / H
    R200c = g["Group_R_Crit200"] / H
    sfr = s["sfr"]
    rows.append(dict(
        sid=sid, central=bool(s["primary_flag"]), grnr=grnr,
        Mstar_Msun=Mstar, logMstar=np.log10(Mstar),
        M200c_Msun=M200c, logM200c=np.log10(M200c), R200c_kpc=R200c,
        SFR_Msun_yr=sfr, sSFR_yr=sfr / Mstar if Mstar > 0 else np.nan,
        Mgas_Msun=Mgas, Mtot_sub_Msun=Mtot, vmax_kms=s["vmax"],
        starZ_Zsun=s["starmetallicity"] / 0.0127, halfmassrad_stars_kpc=s["halfmassrad_stars"] / H))
    print(f"  {sid}: {'central' if s['primary_flag'] else 'SATELLITE'}  logMstar={np.log10(Mstar):.2f}  "
          f"logM200c={np.log10(M200c):.2f}  SFR={sfr:.2f}  sSFR={sfr/Mstar:.2e}")
df = pd.DataFrame(rows)
OUT.mkdir(exist_ok=True)
df.to_csv(OUT / "galaxy_properties_tng.csv", index=False)
print(f"\nsaved {OUT}/galaxy_properties_tng.csv ({len(df)} galaxies)")
