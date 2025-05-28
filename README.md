# nf-binder-vis

Results visualization for the nf-binder-design workflow.


## Installation

### Clone the Repository

First, clone this repository and the streamlit-file-browser dependency:

```bash
# Clone nf-binder-vis
git clone https://github.com/Australian-Protein-Design-Initiative/nf-binder-vis.git
cd nf-binder-vis

# Clone streamlit-file-browser in the symlinks branch
git clone -b symlinks https://github.com/pansapiens/streamlit-file-browser.git
```

### Install Node.js Dependencies

The streamlit-file-browser component requires Node.js dependencies:

```bash
# You'll need npm - if you don't have an npm/node installation, do
# curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# source ~/.bashrc
# nvm install node
# nvm use  node

# Install Node.js dependencies
cd streamlit-file-browser/streamlit_file_browser/frontend/

# Use --legacy-peer-deps to handle TypeScript version conflict
npm install --legacy-peer-deps

# Build the React app for streamlit-file-browser
# If using Node.js >= 17, you need to set OpenSSL legacy provider
export NODE_OPTIONS=--openssl-legacy-provider
npm run build

cd ../../../
```

### Install Python Dependencies

Using UV:

```bash
# Install UV if you don't have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=~/.local/bin:${PATH}
# or pip install uv

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Clean out any old streamlit-file-browser versions and cache
uv pip uninstall streamlit-file-browser || true # Allow failure if not installed
uv cache clean

# Install the nf-binder-vis package and its dependencies
uv pip install -e .

# Install the local streamlit-file-browser
cd streamlit-file-browser/streamlit_file_browser/frontend/
# Use --legacy-peer-deps to handle TypeScript version conflict (if still needed)
npm install --legacy-peer-deps
# If using Node.js >= 17, you might need to set OpenSSL legacy provider
export NODE_OPTIONS=--openssl-legacy-provider
npm run build
cd ../../ # Back to streamlit-file-browser root

# Ensure setuptools is available for setup.py
uv pip install setuptools wheel
# Build the sdist package using setup.py
python setup.py sdist
# Install the built package from its dist directory
uv pip install --force-reinstall dist/streamlit_file_browser-*.tar.gz
cd ..
```

## Usage

### Using UV

After following the installation steps above:

```bash
# From activated venv:
streamlit run app.py -- --path /path/to/results

# Or using UV directly:
uv run streamlit run app.py -- --path /path/to/results
```

Or the BindCraft version:
```bash
uv run streamlit run bindcraft.py -- --path /path/to/bindcraft/runs
# This will detect mutliple bindcraft runs recursively under this path and aggregate the results for Accepted models
#
# The thumbs up/thumbs down populates a 'good' column in /path/to/bindcraft/runs/bindcraft_summary.tsv
# that can be used to select on-target and off-target binders manually
```

This will use the local version of streamlit-file-browser with its Node.js dependencies properly installed.

Open http://localhost:8501/


## Building a Docker/Apptainer Image

```bash
docker build -t nf-binder-vis:latest .
apptainer build nf-binder-vis.sif docker-daemon://nf-binder-vis:latest
```

### Running with Apptainer

```bash
apptainer run nf-binder-vis.sif --server.port 8502 -- --path /path/to/results
```
> The `--server.port` flag can be used to change the default port. Note the bare `--` before the `--path` flag - this is not a typo.

Open http://localhost:8502/

### Running with Docker

```bash
docker run -it --rm -v /abs/path/to/results:/results nf-binder-vis:latest -- --path /results
```

Open http://localhost:8501/



## TODO

- Look at https://github.com/PDBeurope/pdb-images#pdbimages - can we pre-render a galley of images for each pdb file?
- BindCraft support

