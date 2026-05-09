# IIC2440-T1

Diseno e implementacion de un data warehouse sobre una coleccion masiva de articulos de noticias, y la posterior construccion de analitica distribuida usando el paradigma MapReduce.

## Parte 1: Data Warehouse

El pipeline ETL esta en `Data Warehouse/main.py`. Genera la carpeta `warehouse/` con dimensiones y tabla de hechos particionada por ano y mes.

```powershell
python "Data Warehouse/main.py"
```

## Parte 2: Analitica MapReduce

El script de la parte 2 esta en `MapReduce Analytics/main.py`. Lee la carpeta `warehouse/` generada por la parte 1 y calcula:

- Top-K terminos mensuales.
- Frecuencias de palabras por region.
- Divergencia de vocabulario por fuente.
- Deteccion de peaks diarios y una visualizacion SVG.

```Terminal
python "MapReduce Analytics/main.py" --warehouse "warehouse" --output "MapReduce Analytics/output" --min-global-count 500 --min-regional-count 200 --min-source-terms 1000 --region-limit 30
```

Resultados principales:

- `top_terms_monthly.csv`
- `regional_word_distribution.csv`
- `source_vocabulary_divergence.csv`
- `daily_peaks.csv`
- `daily_volume_peaks.svg`

## Visualizaciones

Luego de ejecutar la parte 2, se pueden generar graficos de apoyo para el informe:

```powershell
python "MapReduce Analytics/visualizations.py" --output "MapReduce Analytics/output" --figures "MapReduce Analytics/figures"
```

Figuras generadas:

- `global_top_terms.png`
- `monthly_leading_terms.png`
- `source_vocabulary_divergence.png`
- `daily_peaks.png`
- `regional_terms_<region>.png`
