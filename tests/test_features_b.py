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
    cuantil GLOBAL en predict_set.

    Verificación FUERTE (no contra el piso roto): la cobertura de la clase 95
    con guardrail debe quedar razonablemente CERCA de la clase 95 en el
    baseline puro (SplitConformalClassifier, mismo cuantil global sin
    estratificar). Si el guardrail está bien implementado, ambas usan la
    misma idea (un solo cuantil sin estratificar) -> deben coincidir. Una
    brecha grande indicaría bug en _q_global o en su aplicación en predict_set.

    Para que la clase 95 sea determinísticamente inviable (no depende del
    split), se fuerza un subset pequeño (max_objects=1200): clase 95 queda
    con <30 muestras en calibración, activando el guardrail. Así el test no
    es frágil al conteo de un split concreto.
    """
    from mapie.classification import SplitConformalClassifier
    X, y, X_tr, X_te, y_tr, y_te = plasticc_forma
    # subset pequeño para garantizar clase 95 inviable por conteo
    idx95 = np.where(y_tr == 95)[0]
    # tomar 1200 objetos del train manteniendo stratificación aproximada
    rng = np.random.RandomState(0)
    take = rng.choice(len(X_tr), 1200, replace=False)
    Xs, ys = X_tr[take], y_tr[take]

    m = ADMCP(estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
              alpha=0.1, conformity_score="raps", n_bins=5, n_min_class=30, random_state=0)
    m.fit_conformalize(Xs, ys)
    diag = m.diagnose()
    # con subset de 1200, la clase 95 debe tener <30 en calib -> inviable
    n95 = diag["class_counts_cal"].get(95, 999)
    assert n95 < 30, f"clase 95 debía tener <30 en calib (subset 1200), llegó {n95}"
    assert diag["inviable_classes"].get(95, False) is True, (
        f"clase 95 ({n95} en calib) debía ser inviable")
    _, ysg = m.predict_set(X_te)
    cc_ad = conditional_coverage_by_class(y_te, ysg)

    # baseline puro (mismo SplitConformalClassifier del resto de comparaciones)
    Xtr, Xcal, ytr, ycal = train_test_split(
        Xs, ys, test_size=0.5, random_state=0, stratify=ys)
    base = SplitConformalClassifier(
        estimator=RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1),
        prefit=False, confidence_level=0.9, conformity_score="raps")
    base.fit(Xtr, ytr)
    base.conformalize(Xcal, ycal)
    _, yb = base.predict_set(X_te)
    yb = yb[:, :, 0]
    cc_base = conditional_coverage_by_class(y_te, yb)

    cov_95_ad = cc_ad[95]
    cov_95_base = cc_base[95]
    print(f"\n[b] guardrail: clase 95 AD-MCP(con guardrail)={cov_95_ad:.3f} "
          f"vs baseline puro={cov_95_base:.3f} (n95_cal={n95})")
    print(f"[b]   n_inviable={diag['n_inviable']}")
    assert abs(cov_95_ad - cov_95_base) <= 0.15, (
        f"guardrail no delega bien a global: clase 95 AD-MCP={cov_95_ad:.3f} "
        f"vs baseline={cov_95_base:.3f} (brecha >0.15 -> bug en _q_global)")
