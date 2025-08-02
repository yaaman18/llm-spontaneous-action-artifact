# NewbornAI 2.0 Production Dockerfile
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r newbornai && useradd -r -g newbornai newbornai

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .
COPY README.md .

# Install application
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache /app/backups && \
    chown -R newbornai:newbornai /app

# Copy configuration files
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown newbornai:newbornai /entrypoint.sh

# Switch to non-root user
USER newbornai

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["newborn-ai", "serve", "--host", "0.0.0.0", "--port", "8000"]


# ====================
# Development Stage
# ====================
FROM base as development

USER root

# Install development dependencies
RUN pip install -e ".[dev,docs,performance,security]"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    tree \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy development configuration
COPY docker/dev-entrypoint.sh /dev-entrypoint.sh
RUN chmod +x /dev-entrypoint.sh

USER newbornai

# Development command
CMD ["newborn-ai", "serve", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# ====================
# Testing Stage  
# ====================
FROM base as testing

USER root

# Install test dependencies
RUN pip install -e ".[dev]"

# Install testing tools
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy test configuration
COPY tests/ ./tests/
COPY docker/test-entrypoint.sh /test-entrypoint.sh
RUN chmod +x /test-entrypoint.sh && \
    chown -R newbornai:newbornai /app

USER newbornai

ENTRYPOINT ["/test-entrypoint.sh"]
CMD ["pytest", "tests/", "-v", "--cov=newborn_ai_2"]


# ====================
# Production Stage
# ====================
FROM base as production

# Remove unnecessary packages for smaller image
USER root
RUN apt-get update && apt-get remove -y \
    build-essential \
    git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

USER newbornai

# Production optimized command
CMD ["newborn-ai", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]