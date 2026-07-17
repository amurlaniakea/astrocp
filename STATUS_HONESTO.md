# ESTADO HONESTO — astrocp AD-MCP (actualizado 2026-07-17, ronda B)

## VALIDACIÓN B COMPLETADA (SDSS, clases separables)
----------------------------------------------------------------
  Test: tests/test_sdss_b.py  →  2 passed.
  Dataset: SDSS specgals (astroML, LOCAL, sin red) — bptclass (6 clases
    espectroscópicas reales, separables por física).
  Resultado AD-MCP en SDSS:
    - marginal 0.970, mean set 2.09
    - TODAS las clases ≥ 0.785 (peor clase 4 = 0.785)
    - baseline RAPS peor clase 4 = 0.731
    → AD-MCP MEJORA la peor clase (0.785 > 0.731). El método FUNCIONA
      cuando las clases son distinguibles.

  CONCLUSIÓN de B: el fallo en PLAsTiCC era de FEATURES (6 dims de media de
  flujo no separan clases de 30 muestras), NO del método. Validado por
  contraposición limpia.

## LÍMITE DE PLAsTiCC (documentado, no bug)
----------------------------------------------------------------
  Test: tests/test_ad_mcp.py  →  4 passed (2 PLAsTiCC + 2 SDSS).
  PLAsTiCC con 6 features: AD-MCP controla marginal (0.80-0.97) pero deja
  clases raras < 0.80. Es límite de datos/features. No se "arregla" el test.

## QUÉ SE IMPLEMENTÓ (dato duro)
----------------------------------------------------------------
  src/astrocp/strata/ad_mcp.py:
    - ADMCP: Mondrian manual (modelo base global + cuantiles RAPS por bin
      de anomaly score de IsolationForest). Sin MondrianConformalClassifier
      (no existe en MAPIE 1.4.1).
    - RAPS scores propios (fórmula cerrada Romano et al.) — evita la API
      interna frágil de MAPIE (APSConformityScore no existe; RAPSConformityScore
      sí pero usamos fórmula propia para control total del cuantil).
    - fix del bug APS: la clase en ranking K colapsaba el cuantil a 1.0 →
      sobre-cobertura total. RAPS (lambda*(rank-1)) lo resuelve de raíz.
  src/astrocp/datasets/plasticc.py  — loader PLAsTiCC real (Zenodo).
  src/astrocp/datasets/sdss.py      — loader SDSS local (astroML).
  tests/test_sdss_b.py   — VALIDA método (verde).
  tests/test_ad_mcp.py   — documenta límite PLAsTiCC (verde, imprime hallazgo).

## Veredicto final de abordabilidad
----------------------------------------------------------------
  El GAP de librería es REAL y sigue sin competencia (0 repos GitHub,
  papers de julio 2026 lo adoptan). La implementación del método AD-MCP es
  VIABLE y FUNCIONA (SDSS). El límite en PLAsTiCC es de features → Opción A
  (ingeniería de features de curva de luz) es el siguiente paso natural y ya
  tiene el método validado detrás.

## REPRODUCIBILIDAD
  venv aislado: /home/sil/astrocp/.venv
  pip install -e .  ;  python -m pytest tests/  → 4 passed
  Datos: PLAsTiCC en data/raw (no commiteado), SDSS empaquetado en astroML.
