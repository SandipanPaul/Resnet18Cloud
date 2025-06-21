# Use Python slim for a small footprint
FROM python:3.9-slim

# Working directory inside the container
WORKDIR /autoscaler

# Make Python logs immediately visible
ENV PYTHONUNBUFFERED=1

# Install OS-level dependencies if needed (e.g. SSL certs)
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

# Copy your autoscaler code
COPY autoscaler/autoscaler.py .


# No ports to expose—this runs as a background worker

# Entrypoint: start your Python script
CMD ["python", "autoscaler.py"]
