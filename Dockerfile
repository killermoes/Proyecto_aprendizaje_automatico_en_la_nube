FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

WORKDIR /app

COPY requirements-docker.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-docker.txt

COPY configs/ ./configs/
COPY src/ ./src/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "churn_mlops.batch_predict"]
