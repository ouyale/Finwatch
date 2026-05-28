FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer - only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY finwatch/    ./finwatch/
COPY api/         ./api/
COPY dashboard/   ./dashboard/
COPY data/processed/ ./data/processed/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
