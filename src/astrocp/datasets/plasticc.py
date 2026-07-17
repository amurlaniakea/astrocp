"""Loaders de datasets astronómicos reales para astrocp.

PLAsTiCC (LSST-like transiente classification challenge):
  - metadata real con 14 clases de transitorios (target)
  - lightcurves reales (flux por passband)
Los datos NO se commitean (.gitignore excluye data/raw/).
"""
from __future__ import annotations

import gzip
import csv
import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
META = os.path.join(RAW_DIR, "plasticc_train_metadata.csv.gz")
LC = os.path.join(RAW_DIR, "plasticc_train_lightcurves.csv.gz")

# passbands de PLAsTiCC (u,g,r,i,z,y)
_PASSBANDS = [0, 1, 2, 3, 4, 5]


def _features_from_lightcurves(lc_path: str, object_ids) -> dict:
    """Media de flujo por passband por objeto (feature fija, determinista)."""
    agg = {oid: {pb: [] for pb in _PASSBANDS} for oid in object_ids}
    with gzip.open(lc_path, "rt") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            oid = int(row[0])
            if oid not in agg:
                continue
            pb = int(float(row[2]))  # columna 2 = passband (0..5)
            try:
                flux = float(row[3])  # columna 3 = flux
            except ValueError:
                continue
            agg[oid][pb].append(flux)
    feats = {}
    for oid in object_ids:
        vals = []
        for pb in _PASSBANDS:
            arr = agg[oid][pb]
            vals.append(np.mean(arr) if arr else 0.0)
        feats[oid] = vals
    return feats


def load_plasticc(max_objects: int | None = None) -> dict:
    """Carga PLAsTiCC train como dict con X (features), y (clases), ids.

    Returns
    -------
    dict with keys: X (np.ndarray 2D), y (np.ndarray 1D int),
                    object_id (np.ndarray 1D int), ddf_bool (np.ndarray 1D int)
    """
    if not (os.path.exists(META) and os.path.exists(LC)):
        raise FileNotFoundError(
            "Faltan datos PLAsTiCC en data/raw/. Descargar desde Zenodo 2539456 "
            "(plasticc_train_metadata.csv.gz, plasticc_train_lightcurves.csv.gz)."
        )
    meta = pd.read_csv(META)
    if max_objects is not None:
        meta = meta.sample(frac=1.0, random_state=0).head(max_objects).reset_index(drop=True)
    object_ids = meta["object_id"].astype(int).tolist()
    feats = _features_from_lightcurves(LC, object_ids)
    X = np.array([feats[oid] for oid in object_ids], dtype=float)
    y = meta["target"].astype(int).to_numpy()
    ddf = meta["ddf_bool"].astype(int).to_numpy()
    return {
        "X": X,
        "y": y,
        "object_id": np.array(object_ids, dtype=int),
        "ddf_bool": ddf,
    }
