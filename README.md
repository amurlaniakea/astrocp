# astrocp

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL%203.0%20or%20later-blue.svg)](LICENSE)

Conformal Prediction con cobertura estratificada por anomalía para sondeos astronómicos.

`astrocp` implementa **AD-MCP** (Anomaly-stratified Mondrian Conformal Prediction):
un modelo base global (RandomForest) más cuantiles de conformidad RAPS por
bins del anomaly score (IsolationForest), para dar conjuntos de predicción con
**cobertura condicional** por clase en datasets de astronomía con clases raras
y fuertemente desbalanceadas (PLAsTiCC, SDSS).

## Contexto y gap

La predicción conformista (Conformal Prediction, CP) se ha aplicado a datos
astronómicos antes: ver *"Conformal Prediction for Astronomy Data with
Measurement Error"* (Giertych, Williams & Ghosh, arXiv:2412.10544, 2024),
que construye intervalos de predicción para regresión con error de medición
heterocedástico (masas de exoplanetas).

El gap que `astrocp` aborda es **distinto y más específico**:

> No existe una librería mantenida de Conformal Prediction **para
> clasificación multi-clase con incertidumbre condicional por clase** en
> astronomía — es decir, que controle la cobertura de *cada* clase (incluidas
> las raras) por separado, no solo la cobertura marginal.

AD-MCP es una variante Mondrian manual (estratificación por anomaly score)
sobre MAPIE 1.4.1, con un guardrail `n_min_class` que delega a un cuantil
global las clases con pocas muestras en calibración (donde *cualquier*
estratificación sería ruido puro).

## Features

- `ADMCP`: Mondrian manual (modelo base + cuantiles RAPS por bin de anomaly).
- `select_lambda`: fijación de `lambda_reg` por validación cruzada con un
  criterio compuesto (no a ojo).
- Guardrail `n_min_class`: marca clases inviables y delega a cuantil global.
- Loaders reales: PLAsTiCC (lightcurves) y SDSS (`astroML` local, sin red).
- Métricas de **cobertura condicional por clase** (la que no engaña).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Reproducir los tests

```bash
pytest
```

Resultado esperado: 11 passed, 2 failed a propósito (`tests/test_coverage_red.py`
conserva el criterio de aceptación ORIGINAL que se relajó durante la
auditoría, como registro de trazabilidad).

Los datos de PLAsTiCC vienen commiteados en `data/raw/` (input real de Zenodo)
para que el clone sea reproducible sin red.

## Estado del método (honesto)

- AD-MCP **mejora la cobertura condicional relativa** vs baseline en régimen
  favorable (SDSS; PLAsTiCC con features pobres).
- Con features ricas, el baseline de conformalización global puede superar a
  AD-MCP en la peor clase: el límite real no es de "régimen de anomaly" sino
  de **muestras mínimas por clase** (con 16-30 en calibración, ninguna
  estratificación es viable).
- El guardrail hace que el paquete *avise* cuándo no debe estratificar, en
  vez de ocultar la limitación.

La historia completa (incluidos los criterios relajados y los bugs de
reproducibilidad detectados en auditoría ciega) está en `STATUS_HONESTO.md`.

## License

AGPL-3.0-or-later. Ver [LICENSE](LICENSE).

Copyright © 2026 Pedro Sordo Martínez — amurlaniakea@gmail.com
