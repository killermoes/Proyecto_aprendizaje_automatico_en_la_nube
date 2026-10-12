# Arquitectura de la solución

```mermaid
flowchart LR
    Kaggle["Kaggle / IBM Sample Data"] --> Raw["data/raw<br/>CSV descargado localmente"]
    Raw -->     Notebook["notebooks/ 01-02<br/>exploración y preparación"]
    Notebook --> Prepared["data/processed<br/>CSV preparado localmente"]
    Prepared -->     Prefect["src/churn_mlops/pipeline.py<br/>Prefect: comparación + Optuna"]
    Prefect --> MLflow["MLflow Tracking<br/>parámetros, métricas y modelos"]
    MLflow --> Registry["MLflow Model Registry<br/>versión y alias champion"]
    Registry --> Batch["src/churn_mlops/batch_predict.py<br/>inferencia por lotes"]
    Input["CSV de entrada"] --> Batch
    Batch --> Output["CSV de predicciones"]
    Batch --> Monitor["Métricas batch<br/>MLflow: conteos, positivos y faltantes"]
    Docker["Docker image"] --> Batch
```

El pipeline selecciona el candidato por F1 medio de validación cruzada,
registra la versión en MLflow Model Registry y actualiza el alias `champion`.
La inferencia resuelve el alias, valida el esquema y emite métricas agregadas
del lote. No se guardan filas de clientes ni el CSV de salida como artefactos
de MLflow.

## Límites

- El flujo es batch; no expone un endpoint online.
- Las métricas de inferencia describen volumen, faltantes y distribución de
  predicciones. No miden desempeño real porque el proceso no dispone de
  etiquetas posteriores al período de predicción.
- Prefect, MLflow y la imagen Docker se ejecutan localmente para la
  demostración; no hay infraestructura productiva ni alertas configuradas.
