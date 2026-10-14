FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch first (~200MB instead of 4GB CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, API backend, and model checkpoints
COPY src/ src/
COPY backend/ backend/
COPY results/ results/

ENV PYTHONPATH="/app/src"
ENV PORT=8000
EXPOSE 8000
EXPOSE 7860

# Runs on Render ($PORT), Hugging Face Spaces (7860), or local container (8000)
CMD ["sh", "-c", "python -m uvicorn app:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
