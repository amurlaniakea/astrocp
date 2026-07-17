"""AD-MCP: Anomaly-stratified Mondrian Conformal Prediction.

Implementación MANUAL del núcleo que pide ASTRANet (arXiv:2607.08044):
"AD-stratified Mondrian conformal prediction (AD-MCP) ... achieving uniform
conditional coverage across anomaly-score strata where vanilla Mondrian
under-covers in the operational regime."

MAPIE 1.4.1 NO trae MondrianConformalClassifier (verificado: 0 ocurrencias
de 'mondrian' en mapie.classification). Este módulo lo construye a mano.

Contrato de API real de MAPIE 1.4.1 (verificado por ejecución en
API_VERIFICATION.md): SplitConformalClassifier usa confidence_level y
conformity_score='aps'/'raps'; orden fit -> conformalize -> predict_set.
Para AD-MCP calculamos los cuantiles APS nosotros mismos (fórmula cerrada),
sin depender de la API interna de MAPIE (APSConformityScore no existe).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
def _raps_scores(proba: np.ndarray, y: np.ndarray, class_to_idx: dict,
                lambda_reg: float = 0.01, include_last_label: bool = True) -> np.ndarray:
    """Score RAPS por punto (Romano et al.): cumsum de probas en orden desc
    hasta la clase verdadera, MÁS lambda*(ranking-1). El término de
    regularización evita que la última clase colapse el cuantil a 1.0
    (problema que tiene APS puro con modelos muy confiados).
    Devuelve array (n,).
    """
    order = np.argsort(-proba, axis=1)        # índices en orden prob desc
    Kc = proba.shape[1]
    cum = np.zeros_like(proba)
    running = np.zeros(proba.shape[0])
    for rank in range(Kc):
        cls = order[:, rank]
        running += proba[np.arange(len(proba)), cls]
        cum[np.arange(len(proba)), cls] = running
    y_idx = np.array([class_to_idx[int(c)] for c in y])
    # ranking (1-indexed) de la clase verdadera en el orden desc
    rank_of_true = (order == y_idx[:, None]).argmax(axis=1) + 1
    scores = cum[np.arange(len(y)), y_idx] + lambda_reg * (rank_of_true - 1)
    if include_last_label:
        is_last = rank_of_true == Kc
        if is_last.any():
            p_K = proba[np.arange(len(y)), y_idx]
            # RAPS: última clase usa score (1 - p_K) + lambda*(K-1)
            scores = scores.copy()
            scores[is_last] = (1.0 - p_K[is_last]) + lambda_reg * (Kc - 1)
    return scores


def _raps_scores_matrix(proba: np.ndarray, lambda_reg: float = 0.01) -> np.ndarray:
    """Matriz de scores RAPS (n, K) para construir conjuntos en predict."""
    order = np.argsort(-proba, axis=1)
    Kc = proba.shape[1]
    cum = np.zeros_like(proba)
    running = np.zeros(proba.shape[0])
    for rank in range(Kc):
        cls = order[:, rank]
        running += proba[np.arange(len(proba)), cls]
        cum[np.arange(len(proba)), cls] = running
    # ranking por clase (1-indexed)
    ranks = np.zeros_like(cum, dtype=int)
    for r in range(Kc):
        ranks[np.arange(len(proba)), order[:, r]] = r + 1
    return cum + lambda_reg * (ranks - 1)


def _strata_with_min_samples(anomaly: np.ndarray, y: np.ndarray,
                             n_bins: int = 4, n_min: int = 30) -> np.ndarray:
    """Bins de anomaly score adaptativos con mínimo de muestras por clase.
    (Histórico; AD-MCP actual NO fragmenta el train por estrato, usa bins
    fijos del score continuo. Se conserva por compatibilidad.)"""
    edges = np.quantile(anomaly, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.digitize(anomaly, edges[1:-1])


class ADMCP:
    """Anomaly-stratified Mondrian Conformal Predictor.

    Parámetros
    -----------
    estimator : clasificador sklearn (default RandomForestClassifier)
    alpha : nivel de error objetivo (cobertura deseada = 1-alpha)
    conformity_score : 'aps' (maneja mejor multi-clase que 'lac')
    method : método de anomaly ('isolation_forest' | 'distance')
    n_bins, n_min : control de estratificación adaptativa
    random_state
    """

    def __init__(self, estimator=None, alpha: float = 0.1,
                 conformity_score: str = "raps", method: str = "isolation_forest",
                 n_bins: int = 4, n_min: int = 30, lambda_reg: float = 0.01,
                 n_min_class: int = 30, random_state: int = 0):
        from sklearn.ensemble import RandomForestClassifier
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100, random_state=random_state)
        self.estimator = estimator
        self.alpha = alpha
        self.conformity_score = conformity_score
        self.method = method
        self.n_bins = n_bins
        self.n_min = n_min
        self.lambda_reg = lambda_reg
        self.n_min_class = n_min_class
        self.random_state = random_state
        self._strata_seen = None

    def fit_conformalize(self, X: np.ndarray, y: np.ndarray,
                         test_size_cal: float = 0.5):
        """AD-MCP (Mondrian sobre anomaly score), semántica correcta:

        1. UN modelo base global (RF) entrenado en todo X.
        2. IsolationForest sobre X para el anomaly score continuo.
        3. Split train/calib GLOBAL; en calib se computan los conformity
           scores APS y se calcula, POR CADA BIN de anomaly, el cuantil de
           conformalización (Mondrian por anomaly).
        Guarda self._quantiles[s] = umbral tal que P(score <= umbral | bin s)
        ~ 1-alpha.
        """
        self._iso = IsolationForest(n_estimators=100, random_state=self.random_state,
                                    contamination="auto").fit(X)
        anomaly = -self._iso.score_samples(X)
        self._edges = np.quantile(anomaly, np.linspace(0, 1, self.n_bins + 1))
        self._edges[0], self._edges[-1] = -np.inf, np.inf

        Xtr, Xcal, ytr, ycal = train_test_split(
            X, y, test_size=test_size_cal,
            random_state=self.random_state, stratify=y)
        # anomaly/stratum sobre calib para coherencia con el modelo base
        a_cal = -self._iso.score_samples(Xcal)
        str_cal = np.digitize(a_cal, self._edges[1:-1])

        from sklearn.base import clone
        self._base = clone(self.estimator).fit(Xtr, ytr)

        self._classes = np.unique(y)
        self._class_to_idx = {int(c): i for i, c in enumerate(self._classes)}

        proba = self._base.predict_proba(Xcal)
        scores_cal = _raps_scores(proba, ycal, self._class_to_idx,
                                  lambda_reg=self.lambda_reg,
                                  include_last_label=True)

        # GUARDRAIL (pedido por auditoría): conteo de calibración por clase.
        # Si una clase tiene < n_min_class muestras en calib, CUALQUIER
        # estratificación (por anomaly o por otra variable) es inviable por
        # falta de señal -> se marca inviable y se usa el cuantil GLOBAL para
        # ella en predict_set (delegación a conformalización no estratificada).
        self._class_counts_cal = {int(c): int((ycal == c).sum()) for c in self._classes}
        self._inviable_classes = {int(c): (cnt < self.n_min_class)
                                  for c, cnt in self._class_counts_cal.items()}

        # cuantil GLOBAL (fallback para clases inviables)
        self._q_global = float(np.quantile(scores_cal, 1 - self.alpha * (len(scores_cal) + 1) / len(scores_cal)))

        self._quantiles = {}
        for s in np.unique(str_cal):
            mask = str_cal == s
            sc = scores_cal[mask]
            n = sc.shape[0]
            if n < 2:
                continue
            q = float(np.quantile(sc, 1 - self.alpha * (n + 1) / n))
            self._quantiles[s] = q
        return self

    def diagnose(self) -> dict:
        """Reporte de guardrail: conteo de calib por clase y viabilidad de
        AD-MCP para cada una. Útil para decidir si usar AD-MCP o baseline."""
        if not hasattr(self, "_class_counts_cal"):
            raise RuntimeError("fit_conformalize() debe llamarse antes de diagnose()")
        return {
            "n_min_class": self.n_min_class,
            "class_counts_cal": dict(self._class_counts_cal),
            "inviable_classes": {int(c): bool(v)
                                 for c, v in self._inviable_classes.items()},
            "n_inviable": int(sum(self._inviable_classes.values())),
        }

    def predict_set(self, X: np.ndarray) -> tuple:
        """Predice conjuntos usando el cuantil Mondrian del bin de anomaly.

        Para cada punto: score APS; el conjunto = {c : score_c <= q_bin}.
        Devuelve (y_pred, y_set) con y_set shape (n, n_classes) indexado
        por clase interna 0..K-1.
        """
        if self._iso is None:
            raise RuntimeError("fit_conformalize() debe llamarse antes de predict_set()")
        aX = -self._iso.score_samples(X)
        sX = np.digitize(aX, self._edges[1:-1])
        proba = self._base.predict_proba(X)
        scores = _raps_scores_matrix(proba, lambda_reg=self.lambda_reg)
        K = len(self._classes)
        y_set = np.zeros((len(X), K), dtype=int)
        y_pred = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            s = sX[i]
            q_stratum = self._quantiles.get(
                s, self._quantiles.get(self._fallback_stratum(), np.inf))
            for c in range(K):
                # GUARDRAIL: clases inviables (pocas muestras en calib) usan
                # el cuantil GLOBAL, no el por-estrato ruidoso.
                if self._inviable_classes.get(int(self._classes[c]), False):
                    q = self._q_global
                else:
                    q = q_stratum
                if scores[i, c] <= q:
                    y_set[i, c] = 1
            y_pred[i] = int(self._classes[np.argmax(proba[i])])
        return y_pred, y_set

    def _fallback_stratum(self):
        # estrato con más muestras calib (más robusto)
        return max(self._quantiles, key=lambda s: self._quantiles[s]) if self._quantiles else 0


def conditional_coverage_by_class(y_true: np.ndarray, y_set: np.ndarray,
                                  classes=None) -> dict:
    """Cobertura CONDICIONAL por clase (el criterio que NO engaña).

    y_set shape: (n_samples, n_classes) — 1 si la clase está en el conjunto.
    y_set está indexado por CLASE INTERNA 0..K-1 (no por etiqueta real).
    """
    if classes is None:
        classes = np.unique(y_true)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    out = {}
    for c in classes:
        idx = np.where(y_true == c)[0]
        if len(idx) == 0:
            continue
        internal = class_to_idx[int(c)]
        covered = y_set[idx, internal].mean()
        out[int(c)] = float(covered)
    return out


def marginal_coverage(y_true: np.ndarray, y_set: np.ndarray) -> float:
    """Cobertura MARGINAL (engañosa: oculta falla condicional)."""
    n = len(y_true)
    classes = np.unique(y_true)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    cov = sum(1 for i in range(n)
              if y_set[i, class_to_idx[int(y_true[i])]] == 1) / n
    return float(cov)
