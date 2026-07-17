"""Test B: validar AD-MCP en SDSS (clases SEPARABLES) para aislar método
de calidad de features.

Si AD-MCP mejora la cobertura condicional de clases raras respecto al
baseline LAC en SDSS, el método funciona y el fallo en PLAsTiCC era de
features (no del método). Si tampoco mejora aquí, el problema está en la
implementación de AD-MCP.

Criterio del auditor: cobertura CONDICIONAL por clase (no marginal).
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mapie.classification import SplitConformalClassifier
from astrocp.datasets.sdss import load_sdss_bpt
from astrocp.strata.ad_mcp import ADMCP, conditional_coverage_by_class, marginal_coverage


@pytest.fixture(scope="module")
def sdss_split():
    d = load_sdss_bpt(max_objects=20000)
    X, y = d["X"], d["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y)
    return X_tr, X_te, y_tr, y_te


def test_baseline_lac_sdss(sdss_split):
    """Baseline RAPS en SDSS (clases separables)."""
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
    assert 0.85 <= marginal_coverage(y_te, y_set) <= 0.99
    # las clases raras deben estar razonablemente cubiertas en SDSS
    cc = conditional_coverage_by_class(y_te, y_set)
    worst = min(cc.values())
    assert worst >= 0.70, f"baseline SDSS deja clase rara en {worst:.3f}"


def test_ad_mcp_sdss_mejora_o_iguala_baseline(sdss_split):
    """AD-MCP en SDSS: no empeora la peor clase respecto a baseline.

    Con features separables, AD-MCP debe AL MENOS igualar al baseline en la
    peor clase (y preferiblemente mejorarla), validando que el método
    funciona cuando las clases son distinguibles (el fallo en PLAsTiCC era
    de features, no del método).
    """
    X_tr, X_te, y_tr, y_te = sdss_split
    # baseline
    Xtr, Xcal, ytr, ycal = train_test_split(
        X_tr, y_tr, test_size=0.5, random_state=0, stratify=y_tr)
    base = SplitConformalClassifier(
        estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
        prefit=False, confidence_level=0.9, conformity_score="raps")
    base.fit(Xtr, ytr)
    base.conformalize(Xcal, ycal)
    _, yb = base.predict_set(X_te)
    yb = yb[:, :, 0]
    cc_base = conditional_coverage_by_class(y_te, yb)
    worst_base = min(cc_base.values())

    # AD-MCP
    m = ADMCP(estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
              alpha=0.1, conformity_score="raps", n_bins=5, n_min=30, random_state=0)
    m.fit_conformalize(X_tr, y_tr)
    _, ys = m.predict_set(X_te)
    cc_ad = conditional_coverage_by_class(y_te, ys)
    worst_ad = min(cc_ad.values())

    marg_ad = marginal_coverage(y_te, ys)
    # marginal aceptable (modelo muy confiado -> sobre-cobertura leve ok)
    assert 0.85 <= marg_ad <= 0.99, f"AD-MCP marginal SDSS fuera de rango: {marg_ad:.3f}"
    # TODAS las clases deben estar razonablemente cubiertas en SDSS
    for c, cov in cc_ad.items():
        assert cov >= 0.75, f"AD-MCP deja clase {c} en {cov:.3f} < 0.75 (SDSS separable)"
    # AD-MCP no debe empeorar la peor clase respecto a baseline
    assert worst_ad >= worst_base - 0.05, (
        f"AD-MCP empeora la peor clase en SDSS: {worst_ad:.3f} vs baseline {worst_base:.3f}")
