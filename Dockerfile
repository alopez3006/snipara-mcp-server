# syntax=docker/dockerfile:1

# ============ BUILD STAGE ============
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Generate Prisma client
COPY prisma ./prisma
RUN prisma generate


# ============ RUNTIME STAGE ============
FROM python:3.12-slim as runtime

WORKDIR /app

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Prisma binaries from builder and set correct permissions
COPY --from=builder /root/.cache/prisma-python /home/appuser/.cache/prisma-python
RUN chown -R appuser:appgroup /home/appuser/.cache

# Copy application code
COPY src ./src

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set home directory for Prisma to find binaries
ENV HOME=/home/appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
