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
"""Test del selector de lambda por CV (criterio pre-registrado, no a ojo).

Valida que select_lambda:
  - devuelve un lambda DENTRO de la grilla evaluada (reproducible, no mágico),
  - es determinista (misma semilla -> mismo lambda_best),
  - el score compuesto del elegido es <= el de cualquier otro de la grilla.

NO afirma qué lambda debe salir (evitar mover postes de nuevo).
"""
import pytest
from sklearn.model_selection import train_test_split

from astrocp.datasets.sdss import load_sdss_bpt
from astrocp.strata.tune import select_lambda, cv_score_lambda


@pytest.fixture(scope="module")
def sdss_small():
    d = load_sdss_bpt(max_objects=6000)
    X, y = d["X"], d["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    return X_tr, y_tr


def test_select_lambda_devuelve_valor_de_grilla(sdss_small):
    X, y = sdss_small
    lambdas = (0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2)
    r = select_lambda(X, y, lambdas=lambdas, random_state=0, n_jobs=-1)
    assert r["lambda_best"] in lambdas
    # el score del elegido es el mínimo de la tabla
    scores = [t["score"] for t in r["tabla"]]
    assert abs(min(scores) - r["score_mejor"]) < 1e-9


def test_select_lambda_determinista(sdss_small):
    X, y = sdss_small
    r1 = select_lambda(X, y, random_state=0, n_jobs=-1)
    r2 = select_lambda(X, y, random_state=0, n_jobs=-1)
    assert r1["lambda_best"] == r2["lambda_best"]


def test_cv_score_lambda_reproducible(sdss_small):
    X, y = sdss_small
    a = cv_score_lambda(X, y, 0.01, random_state=0)
    b = cv_score_lambda(X, y, 0.01, random_state=0)
    assert abs(a["score"] - b["score"]) < 1e-9
