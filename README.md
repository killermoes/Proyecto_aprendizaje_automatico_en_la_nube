# Telco Customer Churn - Proyecto MLOps

Proyecto reproducible para estimar si un cliente de telecomunicaciones
cancelará el servicio (`Churn = Yes`). La métrica principal es F1 de la clase
positiva `Yes`. El flujo compara modelos, optimiza XGBoost, registra el
candidato en MLflow y permite ejecutar inferencia batch localmente o dentro
de Docker.

## Requisitos

- Windows 10/11, PowerShell y conexión a Internet.
- Python 3.14.
- Docker Desktop con el motor Linux basado en WSL 2 para la opción Docker.
- Dataset Telco Customer Churn descargado siguiendo [`data/README.md`](data/README.md).

Ejecuta los comandos desde la carpeta raíz del proyecto. En VS Code, abre
**Terminal → New Terminal** y comprueba que el prompt termina en `PROYECTO`.
Los comandos se muestran para PowerShell.

## Estructura del proyecto

```text
PROYECTO/
├── README.md
├── data/                 # Instrucciones y datos locales ignorados por Git
├── notebooks/            # Exploración, preparación y experimentación
├── src/churn_mlops/      # Pipeline y servicio de inferencia batch
├── configs/              # Configuración de MLflow, datos y experimentos
├── tests/                # Pruebas automatizadas
├── docs/                 # Arquitectura y guion de presentación
├── evidencias/           # Capturas de la ejecución
├── Dockerfile
├── pyproject.toml        # Dependencias y configuración del paquete
├── requirements-docker.txt # Dependencias reducidas para inferencia en Docker
└── .env.example          # Ejemplo sin credenciales
```

## Flujo de trabajo recomendado

Sigue las secciones en este orden:

1. Descargar el dataset.
2. Crear el entorno e instalar dependencias.
3. Ejecutar pruebas automatizadas.
4. Ejecutar los notebooks 01 y 02 para explorar y preparar los datos.
5. Iniciar MLflow y Prefect.
6. Ejecutar una sola vez el pipeline de entrenamiento y optimización.
7. Comprobar candidatos, ganador, versión y alias en MLflow.
8. Ejecutar inferencia batch y revisar las métricas.
9. Construir la imagen Docker, exportar el modelo versionado y ejecutar
   inferencia en el contenedor.

El pipeline del paso 6 ya ejecuta el baseline, la comparación de modelos y las
20 pruebas Optuna. Los notebooks 03, 04 y 05 contienen el recorrido educativo
de esas etapas, pero volverlos a ejecutar entrena modelos otra vez. Para evitar
trabajo duplicado, usa el pipeline como la ejecución de entrenamiento
reproducible principal y no ejecutes esos notebooks en la misma pasada.

## 1. Dataset

1. Descarga `WA_Fn-UseC_-Telco-Customer-Churn.csv` desde la fuente y siguiendo
   las indicaciones de [`data/README.md`](data/README.md).
2. Crea la carpeta local para el archivo:

   ```powershell
   New-Item -ItemType Directory -Force .\data\raw | Out-Null
   ```

3. Copia el archivo descargado a:

   ```text
   data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

4. Confirma la ruta:

   ```powershell
   Test-Path .\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

   Debe devolver `True`. Los CSV locales están excluidos de Git y no se
   incluyen en la imagen Docker.

## 2. Entorno Python

Si todavía no existe `.venv`, crea el entorno e instala las dependencias:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Comprueba Python y las dependencias principales:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip show pandas scikit-learn mlflow prefect optuna xgboost
```

`pyproject.toml` fija las dependencias del proyecto. `.env.example` documenta
los endpoints locales; no contiene secretos y no se carga automáticamente.
Configura las variables requeridas en cada terminal según los comandos de
inicio de MLflow y Prefect.

Si `.venv` ya existe y las dependencias están instaladas, no es necesario
recrear el entorno ni reinstalar paquetes.

## 3. Pruebas automatizadas

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La salida debe terminar en `OK`. Estas pruebas usan modelos simulados: no
requieren iniciar MLflow, Docker o Prefect y no vuelven a entrenar.

## 4. Exploración y preparación de datos

1. Desde el explorador de VS Code, abre
   `notebooks/01_exploracion_inicial_churn.ipynb`.
2. Selecciona el kernel `.venv` (Python 3.14) y elige **Run All**. Revisa las
   tablas y gráficos; la ejecución debe terminar sin error.
3. Abre `notebooks/02_preparacion_datos_churn.ipynb` y elige **Run All**.
4. Confirma que se generó el conjunto preparado:

   ```powershell
   Test-Path .\data\processed\telco_churn_preparado.csv
   ```

   Debe devolver `True`. El dataset original tiene 7.043 filas y 21 columnas.

Los notebooks 03, 04 y 05 son material de aprendizaje para ejecutar baseline,
tracking y comparación avanzada por separado. Ejecutarlos puede volver a
entrenar y optimizar modelos. En el flujo principal de este README, pasa al
pipeline de Prefect, que integra esas etapas en una única ejecución.

## 5. Iniciar MLflow

Abre una terminal PowerShell nueva en la raíz y ejecuta el servidor:

```powershell
.\.venv\Scripts\mlflow.exe server `
  --host 0.0.0.0 `
  --port 5000 `
  --backend-store-uri "sqlite:///mlflow.db" `
  --default-artifact-root ".\mlruns" `
  --allowed-hosts "localhost,127.0.0.1,127.0.0.1:5000,host.docker.internal,host.docker.internal:5000"
```

Deja esta terminal abierta. La interfaz debe estar disponible en
<http://127.0.0.1:5000>. Se usa `0.0.0.0` para permitir la conexión local desde
Docker mediante `host.docker.internal`; no expongas el puerto 5000 a Internet.

MLflow conserva su base de datos y los artefactos localmente en `mlflow.db` y
`mlruns/`. Estos archivos no se suben al repositorio. En una instalación nueva,
el Model Registry se crea al ejecutar el pipeline.

## 6. Iniciar Prefect

Abre una segunda terminal PowerShell en la raíz:

```powershell
.\.venv\Scripts\prefect.exe server start
```

Deja esta terminal abierta. La interfaz de Prefect debe estar disponible en
<http://127.0.0.1:4200>.

## 7. Ejecutar el pipeline una sola vez

Abre una tercera terminal PowerShell en la raíz. Configura Prefect y ejecuta el
flujo:

```powershell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
.\.venv\Scripts\python.exe -m churn_mlops.pipeline
```

Espera a que termine; el flujo puede tardar por el entrenamiento y las 20
pruebas de Optuna. Debe finalizar correctamente. **No ejecutes de nuevo este
comando ni los notebooks 03, 04 o 05 para esta misma validación.**

El flujo realiza lo siguiente:

- Divide los datos de forma estratificada en entrenamiento y prueba, con
  `test_size=0.2` y semilla `31`.
- Compara Dummy, Logistic Regression y Random Forest.
- Optimiza XGBoost con Optuna durante 20 trials y validación cruzada
  estratificada de 3 folds.
- Registra parámetros, métricas, modelos y runs de Optuna en el experimento
  `telco-churn-advanced`.
- Selecciona el modelo por F1 medio de validación cruzada para `Yes`; reserva
  test para la evaluación final.
- Registra el candidato ganador como una nueva versión de
  `telco-churn-classifier` y asigna el alias `champion`.

La URI de MLflow predeterminada es `http://127.0.0.1:5000`. Para usar otro
servidor, configura `MLFLOW_TRACKING_URI` antes de iniciar los procesos.

## 8. Revisar runs, candidatos y modelo registrado

En MLflow, visita <http://127.0.0.1:5000>:

1. Abre **Experiments → `telco-churn-advanced` → Runs**.
2. Revisa los runs de los modelos comparados y el run
   `xgboost_optuna_f1_yes`. Sus runs hijos contienen los trials de Optuna.
3. En el run de XGBoost revisa `best_cv_f1_yes` y las métricas de test.
4. Abre **Model registry → `telco-churn-classifier`**.
5. Confirma que existe una versión registrada y que el alias `champion` apunta
   a la versión recién creada.

Los modelos candidatos y trials se guardan como runs del experimento. El
pipeline registra como versión del Model Registry solamente al ganador.

## 9. Crear un CSV pequeño para inferencia

En una terminal PowerShell nueva, desde la raíz, crea un CSV de cinco filas a
partir del conjunto procesado. Se excluye `Churn` para evitar usar la respuesta
real como predictor:

```powershell
New-Item -ItemType Directory -Force .\data\inference | Out-Null
@'
import pandas as pd

data = pd.read_csv("data/processed/telco_churn_preparado.csv")
data.drop(columns=["Churn"]).head(5).to_csv(
    "data/inference/churn_batch_input.csv",
    index=False,
)
'@ | .\.venv\Scripts\python.exe -
```

## 10. Ejecutar inferencia batch en Windows

Con MLflow todavía en ejecución, ejecuta:

```powershell
.\.venv\Scripts\python.exe -m churn_mlops.batch_predict `
  --input .\data\inference\churn_batch_input.csv `
  --output .\data\inference\churn_batch_predictions.csv
```

La terminal debe informar `Predictions written: 5 rows`. El CSV de salida
contiene `churn_prediction` (`Yes` o `No`) y `customerID` si estaba en la
entrada.

El script carga `models:/telco-churn-classifier@champion` y registra un run
`batch_inference` en `telco-churn-advanced` con métricas agregadas: cantidad de
filas, conteos y tasa de predicciones positivas y cantidad/tasa de valores
faltantes. No registra filas de clientes ni predicciones individuales como
artefactos. En MLflow abre el run `batch_inference` para revisar las métricas.

## 11. Docker Desktop: instalación y comprobación

Para usar Docker, instala Docker Desktop para Windows y selecciona el motor
basado en WSL 2. Si Docker Desktop indica que falta **Virtual Machine
Platform**, habilita esa característica de Windows desde PowerShell como
administrador y reinicia cuando se solicite. En equipos administrados, consulta
al área de TI antes de modificar características del sistema.

Abre Docker Desktop y espera a que el motor indique que está en ejecución.
Abre una terminal PowerShell nueva desde la raíz del proyecto y localiza la
CLI, tanto si está en `PATH` como si se instaló en la carpeta del usuario:

```powershell
$docker = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $docker) {
    $docker = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"
}
if (-not (Test-Path $docker)) {
    throw "No se encontró docker.exe. Comprueba la instalación de Docker Desktop."
}
& $docker --version
& $docker info
```

La versión y la información del servidor deben mostrarse sin error de daemon.
No continúes hasta que el motor esté activo.

## 12. Construir la imagen Docker

Desde la misma terminal, en la raíz del proyecto:

```powershell
& $docker build -t telco-churn-batch:local .
```

Debe terminar correctamente y etiquetar la imagen como
`telco-churn-batch:local`. El `.dockerignore` evita incluir notebooks, datos,
documentos, evidencias, entornos locales, el repositorio anidado y los
artefactos de MLflow.

## 13. Exportar el modelo para el contenedor

Los artefactos registrados por MLflow en esta instalación usan una ruta local
de Windows (`file:`). Un contenedor Linux no puede abrir directamente esa ruta.
Por eso, antes de ejecutar Docker, descarga el modelo `champion` a una carpeta
que se compartirá como volumen:

1. Asegúrate de que MLflow esté abierto en `http://127.0.0.1:5000`.
2. En la terminal PowerShell del host, desde la raíz, ejecuta:

   ```powershell
   $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
   New-Item -ItemType Directory -Force .\data\inference\champion_model | Out-Null
   .\.venv\Scripts\mlflow.exe artifacts download `
     --artifact-uri "models:/telco-churn-classifier@champion" `
     --dst-path ".\data\inference\champion_model"
   ```

3. Comprueba que la descarga contiene el archivo de definición del modelo:

   ```powershell
   Test-Path .\data\inference\champion_model\MLmodel
   ```

   Debe devolver `True`. Esta descarga no entrena ni vuelve a registrar el
   modelo. Los artefactos descargados se mantienen locales y no se incluyen en
   Git ni en la imagen.

## 14. Ejecutar inferencia dentro de Docker

Mantén MLflow activo para que el contenedor pueda registrar las métricas batch.
En la terminal donde definiste `$docker`, desde la raíz:

```powershell
& $docker run --rm `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  -v "${PWD}\data\inference:/data" `
  telco-churn-batch:local `
  --input /data/churn_batch_input.csv `
  --output /data/churn_batch_predictions_docker.csv `
  --model-uri /data/champion_model
```

Debe informar `Predictions written: 5 rows`. Comprueba que el archivo de salida
se creó en el host:

```powershell
Test-Path .\data\inference\churn_batch_predictions_docker.csv
```

Debe devolver `True`. Opcionalmente, compara ambas salidas:

```powershell
Import-Csv .\data\inference\churn_batch_predictions.csv
Import-Csv .\data\inference\churn_batch_predictions_docker.csv
```

La inferencia Docker no entrena el modelo. Registra otro run `batch_inference`
en MLflow; revisa que el run se encuentre bajo `telco-churn-advanced`.

## 15. Pruebas, monitoreo y evidencias

- Las pruebas automatizadas se ejecutan con el comando de la sección 3.
- La inferencia registra conteos y proporciones del lote. Sin las etiquetas
  reales posteriores no se calculan métricas productivas como precisión y
  recall ni se generan alertas automáticas.
- El diagrama y los límites de la solución están en
  [`docs/arquitectura.md`](docs/arquitectura.md).
- Usa [`evidencias/README.md`](evidencias/README.md) para capturar
  pruebas, flujo Prefect, runs, alias, inferencia y Docker. No captures datos
  personales, secretos ni credenciales.
- Usa [`docs/presentacion.md`](docs/presentacion.md) como base para las
  diapositivas. Sustituye los campos pendientes por valores y evidencias
  comprobados durante tu ejecución.

## 16. Git y seguridad

Los datos fuente, procesados y de inferencia, las bases locales de MLflow y
Prefect, los entornos, artefactos y el repositorio anidado
`churn-end-to-end/` se excluyen de Git. La carpeta anidada es otro repositorio
local; este proyecto no la modifica ni la necesita. No agregues `.env`, tokens,
contraseñas, credenciales ni datos privados.

Antes de preparar un commit:

```powershell
git status --short
git check-ignore -v data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv data\processed\telco_churn_preparado.csv data\inference\churn_batch_input.csv
```

Revisa manualmente los archivos que se vayan a publicar. La raíz del proyecto
debe conectarse al repositorio GitHub de entrega que el equipo elija; no
incluyas la carpeta `.git` del repositorio `churn-end-to-end/`.

## Referencias de resultados

Los notebooks guardados incluyen resultados de análisis previos. En la última
ejecución documentada del flujo, XGBoost obtuvo F1 CV aproximado de `0.641` y
F1 test aproximado de `0.611`; los resultados exactos se consultan en los runs
de MLflow de cada ejecución.
