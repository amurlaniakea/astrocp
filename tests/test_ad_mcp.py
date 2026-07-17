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
"""Test PLAsTiCC: documenta el LÍMITE de features (no un bug del método).

Con features de solo media de flujo por passband (6 dims), las clases
raras de PLAsTiCC (n~30-200 en train) son INDISTINGUIBLES. AD-MCP controla
la cobertura MARGINAL (~0.90) pero NO rescata la cobertura condicional de
las clases raras. Eso es un límite de DATOS/FEATURES, validado por contraposición
con test_sdss_b.py (donde las clases SÍ son separables y AD-MCP sí mejora).

Este test NO se "arregla" para que pase. Confirma el hallazgo empírico:
el gap de librería es real, el método funciona (ver test_sdss_b.py), y el
límite en PLAsTiCC es de features (Opción A del plan: ingeniería de features
de curva de luz).
"""
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from astrocp.datasets.plasticc import load_plasticc
from astrocp.strata.ad_mcp import ADMCP, conditional_coverage_by_class, marginal_coverage


@pytest.fixture(scope="module")
def plasticc_split():
    d = load_plasticc()  # dataset completo para estratos reales
    X, y = d["X"], d["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y)
    return X_tr, X_te, y_tr, y_te


def test_ad_mcp_controla_cobertura_marginal(plasticc_split):
    """AD-MCP mantiene la cobertura marginal cerca del nominal (1-alpha)."""
    X_tr, X_te, y_tr, y_te = plasticc_split
    m = ADMCP(estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
              alpha=0.1, conformity_score="raps", n_bins=5, n_min=30, random_state=0)
    m.fit_conformalize(X_tr, y_tr)
    _, ys = m.predict_set(X_te)
    marg = marginal_coverage(y_te, ys)
    # marginal debe estar en el rango nominal (puede ser levemente mayor por
    # el calibration set pequeño en PLAsTiCC)
    assert 0.80 <= marg <= 0.97, f"marginal PLAsTiCC fuera de rango: {marg:.3f}"


def test_ad_mcp_limite_features_clases_raras(plasticc_split):
    """Documenta (no falla por bug): clases raras de PLAsTiCC con 6 features
    no alcanzan cobertura condicional >=0.80. Es límite de features, no del
    método — ver test_sdss_b.py para validación del método en clases separables.
    """
    X_tr, X_te, y_tr, y_te = plasticc_split
    m = ADMCP(estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
              alpha=0.1, conformity_score="raps", n_bins=5, n_min=30, random_state=0)
    m.fit_conformalize(X_tr, y_tr)
    _, ys = m.predict_set(X_te)
    cc = conditional_coverage_by_class(y_te, ys)
    clases_bajas = [int(k) for k, v in cc.items() if v < 0.80]
    # el hallazgo esperado: hay clases raras por debajo de 0.80
    assert len(clases_bajas) > 0, (
        "SORPRESA: todas las clases raras superaron 0.80 con solo 6 features. "
        "Revisar si el loader o el split cambiaron.")
    # y se documenta cuáles
    print(f"\n[PLAsTiCC límite features] clases <0.80: {clases_bajas}")
    print(f"[PLAsTiCC límite features] cobertura por clase: "
          f"{ {int(k): round(v,3) for k,v in cc.items()} }")
