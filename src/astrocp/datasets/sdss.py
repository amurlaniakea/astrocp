# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 Pedro Sordo Martínez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
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
