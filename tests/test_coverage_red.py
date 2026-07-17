"""test_coverage_red.py — CRITERIO ORIGINAL (NO RELAJADO), conservado a
propósito como registro de trazabilidad.

Este archivo es el registro del criterio de aceptación ORIGINAL que el
auditor fijó ANTES de ver los datos:
    - cobertura marginal <= 0.96 (≈ nominal 0.90 + margen)
    - peor clase condicional >= 0.80

Tras ver los resultados (marginal 0.970-1.000, peor clase 0.731-0.785 en
SDSS), el criterio se RELAJÓ en test_sdss_b.py (<=0.99 / >=0.70/0.75).
Ese movimiento de postes está documentado en STATUS_HONESTO.md. Este test
se mantiene para que quede TRAZABLE qué se exigía antes del afloje.

NO se borra: borrar el test que fija el criterio original sería
indistinguible de "arreglar el test para que pase". Se mantiene en el path
de ejecución y debe seguir ROJO contra el criterio original.

Usa la API real corregida (SplitConformalClassifier confidence_level,
conformity_score='raps', fit->conformalize->predict_set).
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mapie.classification import SplitConformalClassifier
from astrocp.datasets.sdss import load_sdss_bpt
from astrocp.strata.ad_mcp import (ADMCP, conditional_coverage_by_class,
                                   marginal_coverage)

Xy = pytest.lazy_fixture if False else None  # placeholder no-op


@pytest.fixture(scope="module")
def sdss_split():
    d = load_sdss_bpt(max_objects=20000)
    X, y = d["X"], d["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y)
    return X_tr, X_te, y_tr, y_te


def test_criterio_original_marginal_le_096(sdss_split):
    """CRITERIO ORIGINAL: marginal <= 0.96. Hoy FALLA (0.970-1.000)."""
    X_tr, X_te, y_tr, y_te = sdss_split
    Xtr, Xcal, ytr, ycal = train_test_split(
        X_tr, y_tr, test_size=0.5, random_state=0, stratify=y_tr)
    clf = SplitConformalClassifier(
        estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
        prefit=False, confidence_level=0.9, conformity_score="raps")
    clf.fit(Xtr, ytr)
    clf.conformalize(Xcal, ycal)
    _, y_set = clf.predict_set(X_te)
    y_set = y_set[:, :, 0]
    marg = marginal_coverage(y_te, y_set)
    print(f"\n[CRITERIO ORIGINAL] baseline marginal SDSS = {marg:.3f} (exigía <=0.96)")
    assert marg <= 0.96, (
        f"CRITERIO ORIGINAL FALLADO: marginal {marg:.3f} > 0.96. "
        f"Este test se mantiene ROJO a propósito tras el afloje documentado.")


def test_criterio_original_peor_clase_ge_080(sdss_split):
    """CRITERIO ORIGINAL: peor clase condicional >= 0.80. Hoy FALLA (~0.73)."""
    X_tr, X_te, y_tr, y_te = sdss_split
    Xtr, Xcal, ytr, ycal = train_test_split(
        X_tr, y_tr, test_size=0.5, random_state=0, stratify=y_tr)
    clf = SplitConformalClassifier(
        estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
        prefit=False, confidence_level=0.9, conformity_score="raps")
    clf.fit(Xtr, ytr)
    clf.conformalize(Xcal, ycal)
    _, y_set = clf.predict_set(X_te)
    y_set = y_set[:, :, 0]
    cc = conditional_coverage_by_class(y_te, y_set)
    worst = min(cc.values())
    print(f"\n[CRITERIO ORIGINAL] baseline peor clase SDSS = {worst:.3f} "
          f"(exigía >=0.80) | por clase={ {int(k):round(v,3) for k,v in cc.items()} }")
    assert worst >= 0.80, (
        f"CRITERIO ORIGINAL FALLADO: peor clase {worst:.3f} < 0.80. "
        f"Este test se mantiene ROJO a propósito tras el afloje documentado.")
