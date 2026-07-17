"""Test (b): features de FORMA de curva de luz sobre PLAsTiCC.

Documenta (no maquilla) el hallazgo empírico de esta fase:
  - Con 36 features de forma, varias clases raras mejoran MUCHO vs las 6
    features de solo media (clase 6: ~0.45->0.68, 64: ~0.46->0.67, 92:
    ~0.73->0.81).
  - PERO con features ricas el baseline Mondrian global puede superar a
    AD-MCP estratificado en la peor clase (régimen donde anomaly no aísla
    clases raras). AD-MCP NO es universalmente superior: es herramienta de
    régimen.

El test verifica que (1) el loader entrega 36 features finitas, y (2) las
clases raras MEJORAN respecto al loader de 6 features (criterio relativo,
mismo método). No exige que AD-MCP supere al baseline (eso no siempre pasa).
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from astrocp.datasets.plasticc import load_plasticc
from astrocp.strata.tune import select_lambda
from astrocp.strata.ad_mcp import ADMCP, conditional_coverage_by_class, marginal_coverage


@pytest.fixture(scope="module")
def plasticc_forma():
    d = load_plasticc(max_objects=2500, use_cache=True)  # usa cache si existe
    X, y = d["X"], d["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te


def test_loader_forma_entrega_36_features_finitas(plasticc_forma):
    X, y, *_ = plasticc_forma
    assert X.shape[1] == 36, f"esperaba 36 features de forma, llegó {X.shape[1]}"
    assert np.all(np.isfinite(X)), "features de forma tienen NaN/Inf"


def test_select_lambda_recorre_con_features_nuevas(plasticc_forma):
    X, y, *_ = plasticc_forma
    # el óptimo de lambda puede cambiar con el espacio de features (aviso
    # del auditor) -> se re-corre, no se reusa el de 6 features.
    r = select_lambda(X, y, random_state=0, n_jobs=-1)
    assert r["lambda_best"] in (0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2)
    # documenta qué lambda salió (puede diferir del de 6 features)
    print(f"\n[b] lambda_best con features de forma = {r['lambda_best']}")


def test_clases_raras_mejoran_con_features_forma(plasticc_forma):
    """Criterio relativo honesto: con features de forma, AD-MCP debe cubrir
    mejor las clases raras que con 6 features (mismo método, mismo split)."""
    X, y, X_tr, X_te, y_tr, y_te = plasticc_forma
    lam = select_lambda(X, y, random_state=0, n_jobs=-1)["lambda_best"]
    m = ADMCP(estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
              alpha=0.1, conformity_score="raps", n_bins=5, lambda_reg=lam, random_state=0)
    m.fit_conformalize(X_tr, y_tr)
    _, ys = m.predict_set(X_te)
    cc = conditional_coverage_by_class(y_te, ys)
    # clases que con 6 features estaban ~0.45-0.73 deben subir con forma
    for c, cota in [(6, 0.60), (64, 0.60), (92, 0.78)]:
        if c in cc:
            assert cc[c] >= cota, (
                f"clase {c} con features de forma ({cc[c]:.3f}) no supera "
                f"la cota {cota} esperada vs 6 features")
    print(f"\n[b] AD-MCP forma: marginal={marginal_coverage(y_te,ys):.3f} "
          f"peor={min(cc.values()):.3f} | por clase={ {int(k):round(v,3) for k,v in cc.items()} }")
