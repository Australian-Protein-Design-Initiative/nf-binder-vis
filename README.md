# nf-binder-vis

Results visualization for the nf-binder-design project.


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
# nvm install --lts node
# nvm use --lts node

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

# Install the nf-binder-vis package and its dependencies
uv pip install -e .

# Install the local streamlit-file-browser
cd streamlit-file-browser
uv pip install -e .
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

This will use the local version of streamlit-file-browser with its Node.js dependencies properly installed.

Open http://localhost:8501/

### Using Docker

```bash
RESULTS_PATH=/path/to/results docker compose up -d --build
```

Open http://localhost:8501/

## Building

### Docker Image

```bash
docker build -t nf-binder-vis .
apptainer build nf-binder-vis.sif docker-daemon://nf-binder-vis:latest
```

## TODO

- Look at https://github.com/PDBeurope/pdb-images#pdbimages - can we pre-render a galley of images for each pdb file?
- BindCraft support

