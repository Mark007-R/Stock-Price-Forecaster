# StockAI serving image — FastAPI honest-forecasting API + Streamlit ops
# dashboard (one image, two commands; compose picks the entrypoint per service).
#
# Slim by design: no TensorFlow, no Prophet (see requirements-serving.txt).
# The champion the API serves is ARIMA; the deep models stay a research
# concern in requirements.txt.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

# libgomp: OpenMP runtime that xgboost's wheel links against; curl: healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
# Heavy numeric wheels first, as their own layer, with a persistent pip
# cache mount: on an unreliable network a failed attempt keeps every wheel
# it completed, so retries only fetch what's missing. Pins mirror
# requirements-serving.txt.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 \
        numpy==1.26.2 pandas==3.0.3 scipy==1.16.3 \
        scikit-learn==1.9.0 statsmodels==0.14.6
# xgboost with --no-deps: its Linux wheel otherwise drags in
# nvidia-nccl-cu12 (~300 MB of GPU collective-comms runtime) — dead weight
# in a CPU-only container. Its actual runtime needs (numpy, scipy) are
# already installed above.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 --no-deps xgboost==3.2.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 -r requirements-serving.txt

# Serving code + the indicator math the API reuses from the Flask app.
COPY src/ src/
COPY historical.py .
# Sprint result tables the dashboard renders (small CSVs; plots excluded
# via .dockerignore).
COPY results/ results/

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
