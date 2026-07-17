"""Loader PLAsTiCC con features de FORMA de curva de luz (no solo media).

PLAsTiCC lightcurves: object_id, mjd, passband(0-5), flux, flux_err,
detected_bool. Extrae por passband (6 bandas) estadísticos de FORMA:
  - media, desvío, amplitud (max-min), pendiente (flux vs mjd, poly1),
    tiempo_al_pico (mjd del flux máximo normalizado), n_obs.
Son ~6 stats x 6 passbands = 36 features. Esto separa clases que con solo
media de flujo eran indistinguibles (ver STATUS_HONESTO: límite de features).

Requiere leer las lightcurves (más lento que la metadata). Se cachea en
data/processed/plasticc_features.parquet si existe.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

RAW_LC = "data/raw/plasticc_train_lightcurves.csv.gz"
RAW_META = "data/raw/plasticc_train_metadata.csv.gz"
CACHE = "data/processed/plasticc_features.csv.gz"

PASSBANDS = 6
STATS = ["mean", "std", "amp", "slope", "peakmjd", "nobs"]


def _features_for_object(g: pd.DataFrame) -> dict:
    row = {}
    for pb in range(PASSBANDS):
        sub = g[g["passband"] == pb]
        f = sub["flux"].to_numpy(dtype=float)
        t = sub["mjd"].to_numpy(dtype=float)
        if len(f) == 0:
            for s in STATS:
                row[f"pb{pb}_{s}"] = 0.0
            continue
        row[f"pb{pb}_mean"] = float(np.mean(f))
        row[f"pb{pb}_std"] = float(np.std(f)) if len(f) > 1 else 0.0
        row[f"pb{pb}_amp"] = float(np.ptp(f))  # max - min
        if len(f) > 1:
            # pendiente flux vs mjd (poly1)
            A = np.vstack([t, np.ones_like(t)]).T
            try:
                slope, _ = np.linalg.lstsq(A, f, rcond=None)[0]
            except Exception:
                slope = 0.0
            row[f"pb{pb}_slope"] = float(slope)
            row[f"pb{pb}_peakmjd"] = float(t[np.argmax(f)])  # momento del pico
        else:
            row[f"pb{pb}_slope"] = 0.0
            row[f"pb{pb}_peakmjd"] = float(t[0])
        row[f"pb{pb}_nobs"] = float(len(f))
    return row


def load_plasticc(max_objects: int | None = None, use_cache: bool = True,
                  random_state: int = 0) -> dict:
    """Carga PLAsTiCC con features de forma de curva de luz.

    Devuelve dict con X (n, n_feat), y (n,) target, feature_names.
    """
    if use_cache and os.path.exists(CACHE):
        df = pd.read_csv(CACHE)
    else:
        meta = pd.read_csv(RAW_META)
        lc = pd.read_csv(RAW_LC)
        if max_objects is None:
            objs = meta["object_id"].unique()
        else:
            rng = np.random.RandomState(random_state)
            objs = rng.choice(meta["object_id"].unique(),
                              min(max_objects, meta["object_id"].nunique()),
                              replace=False)
        # agrupar lightcurves por objeto
        recs = []
        for oid in objs:
            g = lc[lc["object_id"] == oid]
            feats = _features_for_object(g)
            feats["object_id"] = oid
            feats["target"] = int(meta.loc[meta["object_id"] == oid, "target"].iloc[0])
            recs.append(feats)
        df = pd.DataFrame(recs)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        df.to_csv(CACHE, index=False, compression="gzip")

    y = df["target"].to_numpy(dtype=int)
    feat_cols = [c for c in df.columns if c not in ("object_id", "target")]
    X = df[feat_cols].to_numpy(dtype=float)
    # limpiar NaN/Inf
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    return {"X": X, "y": y, "feature_names": feat_cols}
