"""Selección de lambda_reg por criterio EXTERNO pre-registrado (no a ojo).

Problema auditado: lambda_reg=0.01 se había fijado a dedo. El auditor pidió
fijarlo por un criterio objetivo pre-registrado (no mirando resultados),
prefiriendo el valor de ASTRANet si lo reportan (independiente de nuestros
datos). Como NO pudimos recuperar el texto del paper (la red no devolvió el
abstract), usamos validación cruzada con un score compuesto pre-registrado:

    score(lambda) = |marginal_cv - (1-alpha)| + max(0, umbral - peor_clase_cv)

Se minimiza sobre una grilla de lambdas. El umbral de peor-clase es un
hiperparámetro del CRITERIO, fijado ANTES de ver datos (default 0.80, el
criterio ORIGINAL del auditor). NO se ajusta lambda mirando la tabla.

Esto es reproducible por el auditor: misma grilla, misma semilla, mismo score.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from .ad_mcp import ADMCP, conditional_coverage_by_class, marginal_coverage


def cv_score_lambda(X, y, lambda_val, alpha=0.1, umbral_peor=0.80,
                    n_bins=5, n_splits=4, random_state=0, n_jobs=-1):
    """Score compuesto pre-registrado para UN lambda dado.

    score = |marginal_cv - (1-alpha)| + max(0, umbral_peor - peor_clase_cv)

    Lower is better. Se evalúa por K-fold (cada fold: fit_conformalize en
    train, predict_set en val, se acumulan coberturas).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    marg_list, worst_list = [], []
    for tr, va in kf.split(X):
        m = ADMCP(estimator=RandomForestClassifier(n_estimators=80,
                                                   random_state=random_state,
                                                   n_jobs=n_jobs),
                  alpha=alpha, conformity_score="raps", n_bins=n_bins,
                  lambda_reg=lambda_val, random_state=random_state)
        m.fit_conformalize(X[tr], y[tr])
        _, ys = m.predict_set(X[va])
        marg_list.append(marginal_coverage(y[va], ys))
        cc = conditional_coverage_by_class(y[va], ys)
        worst_list.append(min(cc.values()))
    marginal = float(np.mean(marg_list))
    worst = float(np.mean(worst_list))
    score = abs(marginal - (1 - alpha)) + max(0.0, umbral_peor - worst)
    return {"lambda": lambda_val, "marginal": marginal, "peor_clase": worst,
            "score": score}


def select_lambda(X, y, lambdas=(0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2),
                  alpha=0.1, umbral_peor=0.80, n_bins=5, n_splits=4,
                  random_state=0, n_jobs=-1):
    """Selecciona lambda minimizando el score compuesto pre-registrado.

    Devuelve dict con 'lambda_best', la tabla completa y el score.
    El criterio (score compuesto + umbral) está fijado ANTES de ver datos.
    """
    tabla = []
    for lam in lambdas:
        r = cv_score_lambda(X, y, lam, alpha=alpha, umbral_peor=umbral_peor,
                            n_bins=n_bins, n_splits=n_splits,
                            random_state=random_state, n_jobs=n_jobs)
        tabla.append(r)
    best = min(tabla, key=lambda d: d["score"])
    return {"lambda_best": best["lambda"], "tabla": tabla, "score_mejor": best["score"]}
