# Datos locales

Los CSV no se versionan en Git: el dataset de origen se descarga de Kaggle y
los archivos procesados y de inferencia se generan localmente.

## Dataset de origen

- Nombre: Telco Customer Churn.
- Página de descarga: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>.
- Autor que figura en Kaggle: BlastChar; la descripción atribuye el conjunto a
  IBM Sample Data Sets.
- Licencia mostrada por Kaggle: `Data files © Original Authors`. Revisa las
  condiciones de la página antes de redistribuir los datos; este proyecto no
  vuelve a publicar el CSV.
- Archivo requerido: `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

Descarga el archivo desde la página de Kaggle y guárdalo en
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`. La carpeta `data/raw` se puede
crear desde PowerShell:

```powershell
New-Item -ItemType Directory -Force .\data\raw | Out-Null
```

Los notebooks `notebooks/01_exploracion_inicial_churn.ipynb` y
`notebooks/02_preparacion_datos_churn.ipynb` leen ese archivo. El notebook 02
produce `data/processed/telco_churn_preparado.csv`. Los CSV de entrada/salida
de inferencia se guardan en `data/inference/`.

No pongas en Git archivos `.env`, datos reales de clientes ni los CSV locales.
