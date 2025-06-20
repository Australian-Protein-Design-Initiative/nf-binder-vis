FROM python:3.11-slim

WORKDIR /app

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for the build process and just
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    just \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install uv, our Python package manager, using the official script
ENV UV_HOME=/opt/uv
ENV PATH="/opt/uv/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy the entire application into the container
COPY . /app/

# Use just to install all dependencies. This will create a .venv directory.
RUN just install

# Expose the default Streamlit port
EXPOSE 8501

# Add a healthcheck to ensure the Streamlit application is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Configure Streamlit to run in headless mode and disable usage stats gathering
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# The command to run when the container starts.
# We execute streamlit from the virtual environment created by 'just install'.
ENTRYPOINT ["/app/.venv/bin/streamlit", "run", "/app/app.py"]