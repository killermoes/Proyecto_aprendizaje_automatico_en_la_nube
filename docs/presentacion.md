# Guion de presentación

Borrador de diapositivas para trasladar a la plantilla solicitada por el
docente. Actualiza los campos entre corchetes después de ejecutar el README;
no presentes como verificados los valores o capturas que aún no se hayan
comprobado.

## 1. Problema y objetivo

- Proyecto: predicción de cancelación de clientes de telecomunicaciones.
- La variable objetivo es `Churn`; `Yes` representa cancelación.
- Objetivo analítico: identificar clientes con riesgo de churn y evaluar el
  modelo mediante F1 de la clase `Yes`.
- No se afirma un ahorro financiero medido; ese impacto requiere validación
  operativa con la empresa.

## 2. Datos y calidad

- Dataset: Telco Customer Churn, atribuido en su ficha de Kaggle a IBM Sample
  Data Sets.
- 7.043 filas y 21 columnas en el archivo original.
- `TotalCharges` se convierte a numérico; variables faltantes y tipos se
  documentan en los notebooks 01 y 02.
- La ficha de fuente y licencia figura en [`../data/README.md`](../data/README.md).

## 3. Modelos y evaluación

- Baseline y comparación: Dummy, Logistic Regression, Random Forest y
  XGBoost.
- Partición estratificada 80/20, semilla 31; selección por F1 medio de
  validación cruzada estratificada de 3 folds.
- Referencia guardada: Logistic Regression F1 (`Yes`) 0.551; XGBoost optimizado
  F1 CV 0.641 y F1 test 0.611.
- Sustituye estos valores por los resultados de la ejecución final si difieren.

## 4. Seguimiento y orquestación

- MLflow registra parámetros, métricas y modelos.
- Prefect orquesta carga, comparación, optimización y registro del candidato.
- Evidencias capturadas: experimento y trials de MLflow en
  [`../evidencias/`](../evidencias/), y ejecución completada de Prefect.

## 5. Arquitectura

- Muestra el diagrama de [`arquitectura.md`](arquitectura.md).
- Menciona que el dataset y artefactos locales están excluidos de Git y que el
  CSV de origen debe descargarse por separado.

## 6. Registro y empaquetado

- MLflow Model Registry mantiene las versiones del modelo
  `telco-churn-classifier`; el alias `champion` apunta al candidato vigente.
- La solución se empaqueta como imagen Docker para ejecutar inferencia batch.
- Evidencia de versión, alias e imagen en [`../evidencias/`](../evidencias/).

## 7. Demostración funcional

- Ejecuta una inferencia batch pequeña desde un CSV sin `Churn`.
- Muestra el CSV de salida y las métricas agregadas `batch_inference` en MLflow.
- Las capturas de predicción y monitoreo están en [`../evidencias/`](../evidencias/).

## 8. Monitoreo, límites y futuro

- Se registran cantidad de filas, conteos/tasa de churn predicha y valores
  faltantes.
- Sin etiquetas posteriores no se mide el desempeño real ni se dispara una
  alerta automática.
- Trabajo futuro: integrar etiquetas reales, definir umbrales de drift y
  rendimiento con el área usuaria, y desplegar con almacenamiento/servicios
  administrados.

## 9. Cierre

- Resume la reproducibilidad, los resultados medidos y las limitaciones.
- Inserta nombres del equipo y agradecimientos según la guía del curso.
