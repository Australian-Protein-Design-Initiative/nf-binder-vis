FROM python:3.11-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Node.js and npm
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml /app/
COPY README.md /app/

# Install other requirements
RUN pip install --no-cache-dir .[all]

# Copy the local streamlit-file-browser package
# COPY streamlit-file-browser /app/streamlit-file-browser
# Or clone the remote version
RUN git clone https://github.com/pansapiens/streamlit-file-browser.git /app/streamlit-file-browser

# Install npm dependencies for streamlit-file-browser and build
RUN cd /app/streamlit-file-browser/streamlit_file_browser/frontend && \
    npm install --legacy-peer-deps && \
    npm install @types/minimatch@^3.0.5 --legacy-peer-deps && \
    export NODE_OPTIONS=--openssl-legacy-provider && \
    npm run build && \
    cd ../.. && \
    python setup.py sdist && \
    cd /app

# Install streamlit-file-browser directly from local copy
RUN pip install --no-cache-dir --force-reinstall \
    ./streamlit-file-browser/dist/streamlit_file_browser-*.tar.gz

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
ENTRYPOINT ["streamlit", "run", "/app/app.py"]