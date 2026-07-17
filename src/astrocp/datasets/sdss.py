"""Loader SDSS specgals (astroML, empaquetado localmente — SIN red).

Usa fetch_sdss_specgals() de astroML: 661k galaxias con clasificación BPT
real (bptclass: -1,1,2,3,4,5) y features espectroscópicas físicas
(magnitudes modelMag u/g/r/i/z, fluxes de líneas Hα/NII, velDisp, z...).
Las clases SON separables con estas features (a diferencia de PLAsTiCC con
solo media de flujo) → aisla el método AD-MCP de la calidad de features.

NO requiere query SQL en vivo (que da 404 en SDSS actual). astroML trae
el recarray empaquetado.
"""
from __future__ import annotations

import numpy as np
import astroML.datasets as aml


def load_sdss_bpt(max_objects: int | None = 20000, random_state: int = 0) -> dict:
    """Carga SDSS specgals como dict con X (features), y (bptclass).

    bptclass: 1=Star-forming, 2=Composite, 3=AGN/LINER, 4=Low-z/LIER,
              5=no BPT (ambiguo), -1=sin líneas (pasiva).
    Devuelve X (n, n_feat), y (n,), object_mask.
    """
    arr = aml.fetch_sdss_specgals()
    # features espectroscópicas físicas (separables por física real)
    feat_names = [
        "modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
        "h_alpha_flux", "nii_6584_flux", "velDisp", "z",
    ]
    X = np.vstack([arr[f].astype(float) for f in feat_names]).T
    y = arr["bptclass"].astype(int)
    # limpiar NaN/Inf
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if max_objects is not None and len(y) > max_objects:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(y), max_objects, replace=False)
        X, y = X[idx], y[idx]
    return {"X": X, "y": y, "feature_names": feat_names}
