FROM python:3.11-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml /app/
COPY README.md /app/

# Install requirements
RUN pip install --no-cache-dir -e .

# Copy app code
COPY . /app/

# Expose Streamlit port
EXPOSE 8501

# Add healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Run streamlit app
ENTRYPOINT ["streamlit", "run", "/app/nf_binder_vis/app.py"]