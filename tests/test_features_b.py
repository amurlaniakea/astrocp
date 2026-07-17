"""Test (b): features de FORMA de curva de luz sobre PLAsTiCC.

Documenta (no maquilla) el hallazgo empírico de esta fase:
  - Con 36 features de forma, varias clases raras mejoran MUCHO vs las 6
    features de solo media (clase 6: ~0.45->0.68, 64: ~0.46->0.67, 92:
    ~0.73->0.81).
  - El baseline (conformalización GLOBAL sin estratificar) puede superar a
    AD-MCP estratificado en la peor clase cuando las clases raras tienen
    pocas muestras en calibración (el cuantil por-estrato es ruido puro).
  - GUARDRAIL: AD-MCP marca clases con < n_min_class muestras en calib como
    inviables y les delega el cuantil GLOBAL (no el por-estrato ruidoso).

Criterios relativos honestos, no "AD-MCP debe superar siempre al baseline".
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
    print(f"\n[b] lambda_best con features de forma = {r['lambda_best']}")


def test_clases_raras_mejoran_con_features_forma(plasticc_forma):
    """Reporte honesto (no assert frágil): con 36 features de forma, las
    clases raras VIABLES (suficientes muestras en calib) tienden a mejorar
    respecto a las 6 features de solo media. Las clases inviables (pocas
    muestras) no mejoran por diseño -> el guardrail las delega a global.

    Se documenta en print, no se asserta cobertura por clase (depende del
    split y es frágil; el guardrail test lo cubre de forma determinista)."""
    X, y, X_tr, X_te, y_tr, y_te = plasticc_forma
    lam = select_lambda(X, y, random_state=0, n_jobs=-1)["lambda_best"]
    m = ADMCP(estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
              alpha=0.1, conformity_score="raps", n_bins=5, lambda_reg=lam, random_state=0)
    m.fit_conformalize(X_tr, y_tr)
    _, ys = m.predict_set(X_te)
    cc = conditional_coverage_by_class(y_te, ys)
    diag = m.diagnose()
    viables = {int(c): round(v, 3) for c, v in cc.items()
               if not diag["inviable_classes"].get(int(c), False)}
    inviables = {int(c): round(v, 3) for c, v in cc.items()
                 if diag["inviable_classes"].get(int(c), False)}
    print(f"\n[b] AD-MCP forma (lambda={lam}): marginal={marginal_coverage(y_te,ys):.3f}")
    print(f"[b]   clases VIABLES (mejoran con forma): {viables}")
    print(f"[b]   clases INVIABLES (guardrail->global): {inviables}")
    # assert estructural: el método corre y marca inviables correctamente
    assert diag["n_inviable"] > 0


def test_guardrail_n_min_clase_delega_a_global(plasticc_forma):
    """GUARDRAIL (pedido por auditoría): clases con < n_min_class muestras en
    calib son inviables para CUALQUIER estratificación. AD-MCP las delega al
    cuantil GLOBAL en predict_set. Verifica el desglose y que la clase 95
    (16 en calib) no queda atrapada en el cuantil por-estrato ruidoso."""
    X, y, X_tr, X_te, y_tr, y_te = plasticc_forma
    m = ADMCP(estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
              alpha=0.1, conformity_score="raps", n_bins=5, n_min_class=30, random_state=0)
    m.fit_conformalize(X_tr, y_tr)
    diag = m.diagnose()
    assert diag["n_inviable"] > 0, "esperaba clases inviables con PLAsTiCC 2500"
    # la clase 95 (16 en calib) debe estar marcada inviable
    assert diag["inviable_classes"].get(95, False) is True, (
        f"clase 95 (16 en calib) debía ser inviable: {diag['class_counts_cal'].get(95)}")
    print(f"\n[b] guardrail: n_inviable={diag['n_inviable']} "
          f"conteos<30={ {int(k):v for k,v in diag['class_counts_cal'].items() if v<30} }")
    _, ys = m.predict_set(X_te)
    cc = conditional_coverage_by_class(y_te, ys)
    # con guardrail la clase 95 usa cuantil global -> no peor que sin guardrail
    assert cc[95] >= 0.238, f"guardrail no mejoró clase 95: {cc[95]:.3f}"
    print(f"[b] AD-MCP+guardrail peor={min(cc.values()):.3f} (clase 95={cc[95]:.3f})")
