# TEST ROJO — hallazgos empíricos (Paso 0 del TDD)

Ejecutado 2026-07-17 contra MAPIE 1.4.1 real + PLAsTiCC real (1500 objs).

## RESULTADO DEL TEST: ROJO (esperado, por diseño)

El test NO pasa. Eso es correcto: su función es demostrar que la
estratificación naive por quartiles de anomaly score NO cumple la garantía
AD-MCP con datos reales desbalanceados.

## POR QUÉ FALLA (datos duros, no conjetura)

1. PLAsTiCC train: 7848 objs, 14 clases, MUY desbalanceadas:
     clase 90 = 2313,  clase 53 = 30,  clase 92 = 239,  clase 64 = 102.
   Al estratificar por quartiles de anomaly score y hacer train/calib/test
   DENTRO de cada estrato, varios estratos quedan con <2 muestras por
   clase → SplitConformalClassifier (vía sklearn StratifiedKFold internO)
   levanta: "The least populated classes ... only 1 member".

2. Con datos SINTÉTICOS (make_classification 3 clases) la cobertura fue
   0.64 con set size fijo = n_clases (todas las clases) → under-coverage
   sistemática de LAC con 3+ clases y estimator débil. Esto es justo lo
   que ASTRANet (2607.08044) documenta: "Mondrian vanilla under-covers
   in the operational regime".

3. Con el ejemplo CANÓNICO de MAPIE (2 clases separables) la librería SÍ
   da cobertura 0.953. O sea: el problema NO es la librería, es que
   (a) 3+ clases + estimator débil → under-coverage, y
   (b) estratificación naive por anomaly rompe por falta de muestras/estrato.

## QUÉ HAY QUE IMPLEMENTAR (núcleo del valor astrocp)

Para que el test VERDE necesitamos AD-MCP real, no la estratificación tonta:
  - Anomaly scoring que NO fragmente tanto (p.ej. IsolationForest sobre
    features, o distance-to-class-mean con smoothing).
  - Estratos con MÍNIMO de muestras por clase (merge de estratos contiguos
    hasta n_min por clase), o Mondrian sobre el score continuo con bins
    adaptativos.
  - possiblemente cambiar conformity_score de 'lac' a 'aps'/'raps' que
    manejan mejor multi-clase ( documented en MAPIE theory).

## ESTADO
  tests/test_coverage_red.py  → ROJO intencional (falla por under-coverage
                                       y por muestras insuficientes/estrato).
  NO se "arregla" el test para que pase; se implementa el módulo
  src/astrocp/strata/ad_mcp.py y se vuelve a correr.

## PRÓXIMO PASO (decisión del usuario)
  ¿Implemento el módulo AD-MCP (anomaly + estratos con n_min + aps/raps)
  para llevar el test a VERDE con datos reales?  Ese es el hito que convence
  de que el gap es abordable de verdad.
