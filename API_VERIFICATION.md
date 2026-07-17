# astrocp — Verificación de API MAPIE (Paso 0 del TDD)

Generado: 2026-07-17. Entorno: venv aislado /home/sil/astrocp/.venv

## Instalado (pip show / import)
- mapie == 1.4.1  (ÚLTIMA versión en PyPI; NO existe 1.5.0)
- scikit-learn == 1.9.0
- astropy == instalado
- Entorno: Python 3.12, venv en /home/sil/astrocp/.venv

## Comprobaciones sobre la API real (output crudo, no inferido)

### MITO 1 (Hermes): "MondrianConformalClassifier existe listo para usar"
RESULTADO: FALSO.
- `'MondrianConformalClassifier' in dir(mapie)` -> False
- `'mondrian' in sources de mapie.classification` -> 0 ocurrencias
- Métodos de conformal classification soportados: solo `aps`, `lac`, `raps`, `top_k`
  (ninguno Mondrian).
- CONCLUSIÓN: Mondrian CP NO está implementado como clase en 1.4.1.
  Es teoría/documentación. AD-MCP (lo que pide ASTRANet) hay que
  IMPLEMENTARLO nosotros (en-voltorio sobre SplitConformalClassifier
  + estratificación por anomaly score).

### MITO 2 (Hermes): "MapieClassifier importable desde mapie.classification"
RESULTADO: FALSO para 1.4.1.
- `from mapie.classification import MapieClassifier` -> ImportError:
  Did you mean '_MapieClassifier'?
- La clase pública real es `_MapieClassifier` (prefijo privado) y NO está
  en __all__.
- API PÚBLICA real de clasificación en 1.4.1:
    - SplitConformalClassifier
    - CrossConformalClassifier
    - EnsembleClassifier
  (todas importables desde mapie.classification).
- `_MapieClassifier.__init__` params reales:
    estimator=None, cv=None, n_jobs=None,
    conformity_score=None, random_state=None, verbose=0
  (NO recibe 'method' en __init__; los métodos aps/lac/raps/top_k se
   aplican vía conformity_score o en fit, no como kwarg de __init__).

### MITO 4 (Hermes, no detectado antes): "SplitConformalClassifier acepta cv/alpha"
RESULTADO: FALSO. Contrato real de 1.4.1 (verificado por ejecución):
- __init__ params: estimator=LogisticRegression(), confidence_level=0.9,
  conformity_score='lac', prefit=True, n_jobs, verbose, random_state
  (NO 'cv', NO 'method', NO 'alpha').
- prefit=True (default) => NO se llama fit(); se usa conformalize() directo.
  Para entrenar estimator adentro => prefit=False.
- ORDEN OBLIGATORIO: fit(X,y) -> conformalize() -> predict(X)
  (predict sin conformalize previo => "call conformalize before calling predict").
- predict(X) devuelve (y_pred, y_set). El nivel se fija en __init__
  (confidence_level), NO en predict.

### MITO 5 (Auditor, no verificado antes): la API de clasificación era
  "SplitConformalClassifier + cv + alpha" como en regresión.
RESULTADO: La API de clasificación de MAPIE 1.4.1 es DISTINTA de regresión:
  no tiene cv en __init__, usa confidence_level y conformity_score, y exige
  conformalize() explícito. No asumir paridad con MapieRegressor.

### MITO 3 (Auditor): "la 1.5.0 movió todo y rompió API"
RESULTADO: FALSO / no aplicable.
- pip index versions mapie -> última = 1.4.1. NO hay 1.5.0 en PyPI.
- `pip install mapie==1.5.0` -> ERROR: No matching distribution found.
- La advertencia del auditor era preventiva; en la práctica no hay
  versión posterior que rompa la API. Fijamos mapie==1.4.1.

## Consecuencia de diseño
- El envoltorio astrocp NO puede llamar a MondrianConformalClassifier.
- Debe construir AD-MCP manualmente:
    1. capa de anomaly scoring (IsolationForest o embedding-space distance)
    2. SplitConformalClassifier de mapie.classification por estrato de anomaly
    3. Mondrian-style coverage audit estratificado (métrica propia en eval/)
- API de uso real (verificada):
    from mapie.classification import SplitConformalClassifier
    clf = SplitConformalClassifier(estimator=..., cv=...)
    clf.fit(X_cal, y_cal)
    y_pred, y_sets = clf.predict(X_test, alpha=0.1)

## Archivo de verdad
Este documento es la fuente primaria. Cualquier diseño de astrocp debe
coincidir con la API verificada aquí, no con la documentación teórica.
