
# ==========================================
# Stage 1: Build React frontend
# ==========================================
FROM node:20-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package*.json ./

RUN npm ci --no-audit --no-fund

COPY frontend/ ./

RUN npm run build


# ==========================================
# Stage 2: Python backend runtime
# ==========================================
FROM python:3.11-slim

WORKDIR /app

# System dependencies required by ONNX Runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 \
       libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ ./backend/

# Copy ML utilities
COPY ml/ ./ml/

# Copy ML artifacts
COPY artifacts/ ./artifacts/

# Copy built React frontend
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Python configuration
ENV PYTHONPATH=/app

ENV MODEL_PATH=/app/artifacts/isl_mobilenet.onnx

ENV PORT=10000

EXPOSE 10000

# Start FastAPI
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
