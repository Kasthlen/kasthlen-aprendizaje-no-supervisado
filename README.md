# Análisis PCA + Clustering (Mushroom dataset)

Resumen
- Dataset: UCI Mushroom dataset (https://archive.ics.uci.edu/ml/datasets/mushroom)
- Objetivo: explorar clustering no supervisado (PCA + KMeans) y comparar con etiquetas reales; entrenar modelo supervisado (RandomForest) y explicar con SHAP.
- Notebook principal: `kasthlen_clustering_pca.ipynb`

Estructura del repositorio
- kasthlen_clustering_pca.ipynb - notebook principal con todo el análisis
- run_pipeline.py - script para ejecutar el pipeline (preprocesamiento, PCA, clustering, RF, SHAP) y guardar artefactos
- outputs/
  - rf_top_importances.csv
  - rf_misclassified_full.csv
  - shap_summary.png
  - shap_by_cluster_mean.csv
  - force_plots/*.html (opcional)
- requirements.txt - dependencias

Cómo reproducir (local)
1. Crear entorno:
   - python -m venv venv
   - source venv/bin/activate
   - pip install -r requirements.txt
2. Ejecutar:
   - python run_pipeline.py
3. Resultados en `outputs/`

Buenas prácticas y notas
- Mantén `outputs/` en el repo solo si los archivos son pequeños. Para artefactos grandes (varios MBs de HTML) considera subir a Releases o a un bucket.
- Define un RANDOM_STATE (por ejemplo 42) para reproducibilidad.
- Si compartes el notebook, considera limpiar salidas largas o añadir un README con una guía rápida.