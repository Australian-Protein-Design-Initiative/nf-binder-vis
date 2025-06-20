# justfile for nf-binder-vis

# This justfile streamlines the setup and execution of the nf-binder-vis application.
# For more information on just, see: https://github.com/casey/just

# Default recipe to run when 'just' is invoked without arguments.
default: install

# Variables for frequently used paths.
SFLB_DIR := "streamlit-file-browser"

# Check the Node.js version for compatibility.
NODE_MAJOR_VERSION := `node --version | cut -d. -f1 | sed 's/v//'`

# Check and handle conda environment conflicts.
check-conda:
    #!/usr/bin/env bash
    if [ -n "$CONDA_DEFAULT_ENV" ]; then \
        echo "ERROR: Conda environment '$CONDA_DEFAULT_ENV' is currently active."; \
        echo "This will cause conflicts with uv package management."; \
        echo ""; \
        echo "Please deactivate conda first by running:"; \
        echo "  conda deactivate"; \
        echo "  conda deactivate  # Run twice to ensure base env is also deactivated"; \
        echo ""; \
        echo "Then run 'just install' again."; \
        exit 1; \
    fi

# Clone or pull the streamlit-file-browser dependency.
clone:
    @if [ ! -d "{{SFLB_DIR}}" ]; then \
        echo "Cloning streamlit-file-browser..."; \
        git clone -b symlinks https://github.com/pansapiens/streamlit-file-browser.git "{{SFLB_DIR}}"; \
    else \
        echo "Directory '{{SFLB_DIR}}' already exists, pulling latest changes."; \
        (cd "{{SFLB_DIR}}" && git pull); \
    fi

# Build the streamlit-file-browser frontend and create a Python source distribution (sdist).
# This recipe depends on 'clone' to ensure the repository is available.
build: clone
    #!/usr/bin/env bash
    set -ex
    echo "Building streamlit-file-browser frontend..."
    cd "{{SFLB_DIR}}/streamlit_file_browser/frontend"
    
    # Clean existing node_modules and package-lock.json to avoid version conflicts
    rm -rf node_modules package-lock.json
    
    # Use --legacy-peer-deps to handle TypeScript version conflicts.
    npm install --legacy-peer-deps
    
    # Set NODE_OPTIONS for compatibility with Node.js >= 17.
    if [ "{{NODE_MAJOR_VERSION}}" -ge 17 ]; then
        export NODE_OPTIONS=--openssl-legacy-provider
    fi
    npm run build

    echo "Building streamlit-file-browser sdist..."
    cd ../../.. # Back to the project root from the frontend directory.
    cd "{{SFLB_DIR}}"
    # Install build dependencies into the virtual environment.
    uv pip install setuptools wheel
    # Create the source distribution.
    python3 setup.py sdist

# Set up the Python virtual environment using uv.
venv: check-conda
    @if [ ! -d ".venv" ]; then \
        echo "Creating virtual environment with uv..."; \
        uv venv; \
    fi

# Install all necessary dependencies for the project.
# This depends on 'venv' to ensure the environment exists and 'build' to ensure dependencies are built.
install: venv build
    #!/usr/bin/env bash
    set -ex
    echo "Installing python dependencies..."
    # Clean out any old versions of streamlit-file-browser.
    uv pip uninstall -y streamlit-file-browser || true
    uv cache clean

    # Install nf-binder-vis in editable mode.
    uv pip install -e .

    # Install the locally built streamlit-file-browser package.
    uv pip install --force-reinstall "{{SFLB_DIR}}/dist/streamlit-file-browser-"*.tar.gz

# Run the main Streamlit application.
run path:
    @if [ -z "{{path}}" ]; then \
        echo "Error: path argument is required. Usage: just run <path>"; \
        exit 1; \
    fi
    @echo "Running: streamlit run app.py -- --path {{path}}"
    .venv/bin/streamlit run app.py -- --path "{{path}}"

# Run the BindCraft visualization Streamlit application.
run-bindcraft path:
    @if [ -z "{{path}}" ]; then \
        echo "Error: path argument is required. Usage: just run-bindcraft <path>"; \
        exit 1; \
    fi
    @echo "Running: streamlit run bindcraft.py -- --path {{path}}"
    .venv/bin/streamlit run bindcraft.py -- --path "{{path}}"

# Clean up build artifacts from streamlit-file-browser.
clean-build:
    @echo "Cleaning streamlit-file-browser build artifacts..."
    @if [ -d "{{SFLB_DIR}}" ]; then \
        rm -rf "{{SFLB_DIR}}/streamlit_file_browser/frontend/node_modules"; \
        rm -rf "{{SFLB_DIR}}/streamlit_file_browser/frontend/package-lock.json"; \
        rm -rf "{{SFLB_DIR}}/streamlit_file_browser/frontend/build"; \
        rm -rf "{{SFLB_DIR}}/dist"; \
        rm -rf "{{SFLB_DIR}}/build"; \
        rm -rf "{{SFLB_DIR}}/*.egg-info"; \
    fi

# Clean up the project directory by removing cloned dependencies and the virtual environment.
clean: clean-build
    @echo "Cleaning up..."
    rm -rf "{{SFLB_DIR}}" .venv
    uv cache clean
    @echo "Cleaned."
