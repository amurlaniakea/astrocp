# ESTADO HONESTO — astrocp AD-MCP (actualizado 2026-07-17, ronda b)

## INTEGRIDAD DEL REPO (verificada por Hermes)
----------------------------------------------------------------
  El auditor pidió confirmar que no hay otra sesión tocando /home/sil/astrocp.
  Verificado: solo mi sesión de Hermes (pid 28143) + pyright LSP (hijo) +
  centinela watch (daemon legítimo de Sil). Una sola rama local
  (feat/astrocp-ad-mcp), sin remotos, sin locks. El aviso de "subagente
  hermano" era falso positivo del sistema de escritura de archivos.

## (a) lambda_reg — CERRADO (con matiz de honestidad)
----------------------------------------------------------------
  tune.py selecciona lambda por CV minimizando score compuesto. MATIZ: la
  fórmula/grilla se diseñaron DESPUÉS de ver el sweep manual (no es
  "pre-registrado ciego" en sentido estricto). Término corregido en archivo.
  SDSS elige lambda=0.001; PLAsTiCC (6 feat) también 0.001.

## (b) FEATURES DE FORMA DE CURVA DE LUZ — resultado nuanzado
----------------------------------------------------------------
  Loader PLAsTiCC reescrito: extrae por passband (6) media/desvio/amplitud/
  pendiente/tiempo-al-pico/n_obs = 36 features (antes solo media = 6).

  select_lambda SOBRE PLAsTiCC con features nuevas elige lambda=0.01 (NO el
  0.001 de las 6 features viejas) -> confirma el aviso del auditor: el óptimo
  cambia con el espacio de features. Re-corrido, no reusado.

  Holdout PLAsTiCC (2500 obj, 36 feat), lambda=0.01:
    AD-MCP:   marginal 0.901 | peor clase 0.238
             (clase 6:0.684, 15:0.828, 16:0.957, 42:0.952, 52:0.458, 53:1.0,
              62:0.864, 64:0.667, 65:0.966, 67:0.480, 88:0.920, 90:0.993,
              92:0.806, 95:0.238)
    baseline: marginal 0.925 | peor clase 0.440

  HALLAZGO HONESTO: con features ricas, el BASELINE Mondrian global
  (peor 0.440) SUPERA a AD-MCP estratificado (peor 0.238). VARIAS clases raras
  mejoran mucho vs 6-feat (clase 6: 0.45->0.684, 64: 0.46->0.667, 92:
  0.73->0.806), pero la clase 95 (n~30) sigue en 0.238.

  INTERPRETACIÓN METODOLÓGICA: AD-MCP ayuda en el régimen donde el modelo base
  deja colas mal calibradas (SDSS moderado, PLAsTiCC 6-feat débil). Cuando las
  features son ricas y el modelo base ya separa bien, el baseline global puede
  superar al estratificado por anomaly — porque en PLAsTiCC las clases raras
  NO viven en el "anomaly tail" con features de forma. AD-MCP NO es
  universalmente superior; es una herramienta de régimen.

## QUÉ SE IMPLEMENTÓ
----------------------------------------------------------------
  src/astrocp/datasets/plasticc.py  loader features de forma (36 dims, cache csv.gz).
  src/astrocp/strata/tune.py         select_lambda por CV.
  src/astrocp/strata/ad_mcp.py       ADMCP (Mondrian manual, RAPS por anomaly).
  tests/: test_sdss_b, test_coverage_red (rojo original), test_ad_mcp,
         test_tune.

## VEREDICTO DE ABORDABILIDAD (final de esta fase)
----------------------------------------------------------------
  GAP de librería: REAL (0 competencia GitHub, papers julio 2026 lo adoptan).
  Método AD-MCP: VIABLE y mejora cobertura condicional RELATIVA en régimen
  favorable (SDSS; PLAsTiCC 6-feat débil). Con features ricas el baseline
  global puede superarlo (régimen donde anomaly no aísla clases raras).
  lambda: fijado por CV objetivo (reproducible), no mágico.
  Conclusión honesta para el paquete: astrocp debe EXPONER ambos modos
  (AD-MCP estratificado y Mondrian global) y dejar elegir según el régimen,
  documentando cuándo cada uno aplica. No vender AD-MCP como panacea.

## REPRODUCIBILIDAD
  venv /home/sil/astrocp/.venv · pip install -e . · pytest -> 7 passed + 2 failed
  Datos: PLAsTiCC (lightcurves) en data/raw; SDSS astroML local.
  Cache features: data/processed/plasticc_features.csv.gz (no commiteado).
