"""Test del selector de lambda por CV (criterio pre-registrado, no a ojo).

Valida que select_lambda:
  - devuelve un lambda DENTRO de la grilla evaluada (reproducible, no mágico),
  - es determinista (misma semilla -> mismo lambda_best),
  - el score compuesto del elegido es <= el de cualquier otro de la grilla.

NO afirma qué lambda debe salir (evitar mover postes de nuevo).
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
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
