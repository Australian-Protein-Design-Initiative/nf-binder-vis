# nf-binder-vis

Results visualization for the nf-binder-design workflow.

## Installation

This project uses `just` to simplify the build and installation process. You can find instructions for installing `just` [here](https://github.com/casey/just).

### Quickstart

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Australian-Protein-Design-Initiative/nf-binder-vis.git
    cd nf-binder-vis
    ```

2.  **Build and Install**

    This will clone the required `streamlit-file-browser` dependency, build its frontend components, create a Python virtual environment, and install all necessary dependencies.

    ```bash
    just install
    ```

    If you only want to build the dependencies without installing, you can run:

    ```bash
    just build
    ```

## Usage

After following the installation steps above:

```bash
# From the project root, this will use the virtual environment managed by just/uv.
just run

# To run the BindCraft visualization:
just run-bindcraft
```

This will use the local version of `streamlit-file-browser` with its Node.js dependencies properly installed and run the Streamlit application. By default, it will be available at http://localhost:8501/.

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
- Merge `app.y` and `bindcraft.py` to support generic binder pipelines (scores table + PDBs) ?

