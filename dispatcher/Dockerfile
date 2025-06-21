# Start from a minimal Python image
FROM python:3.9-slim

# Set a working directory
WORKDIR /dispatcher

# Prevent Python from buffering stdout/stderr (so logs appear immediately)
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch

# Install any OS-level deps (here none beyond the base is strictly required)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy in your dispatcher code and config
COPY dispatcher/dispatcher.py .
COPY config/service.cfg config/service.cfg

# Expose the port your Flask app listens on
EXPOSE 8080


# Health check
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8080/health || exit 1


# Run the dispatcher
CMD ["python", "dispatcher.py"]