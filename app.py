#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair>=5.2.0",
#     "matplotlib",
#     "pandas",
#     "seaborn",
#     "streamlit>=1.32.0",
#     "streamlit-molstar",
# ]
# ///

import sys
import os
import re
import uuid
import json
import logging
from pathlib import Path
import argparse
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
from typing import Optional, List, Any, Tuple, Dict
from streamlit_molstar import st_molstar, st_molstar_rcsb, st_molstar_remote
from streamlit_molstar.auto import st_molstar_auto

logger = logging.getLogger(__name__)

# --- Constants for column names ---
DEFAULT_SORT_COLUMN = "Average_i_pTM"
DEFAULT_SORT_ASCENDING = False
DEFAULT_SCATTER_X_COL = "Average_pLDDT"  # Typical pLDDT column
DEFAULT_DIST_METRIC = DEFAULT_SORT_COLUMN
# ----------------------------------


def get_target_sequence(
    pdb_file: str, method: str, binder_sequence: Optional[str] = None
) -> Optional[str]:
    """Extract the target sequence from a PDB file based on the method.

    Args:
        pdb_file: Path to the PDB file
        method: The method type ("bindcraft", "rfd", or other)
        binder_sequence: The binder sequence (used for methods other than bindcraft/rfd)

    Returns:
        The target sequence as a string, or None if not found
    """

    if not pdb_file or not os.path.exists(pdb_file):
        return None

    try:
        # Note: get_chain_sequences functionality would need to be implemented
        # For now, return None as this is not critical for the main functionality
        chain_sequences = None

        # For now, return None as sequence extraction is not implemented
        # This functionality can be added later if needed
        return None

    except Exception as e:
        logger.error(f"Error extracting target sequence from {pdb_file}: {str(e)}")
        return None


def extract_backbone_id(design_id: str, method: str) -> str:
    """Extract backbone_id from design_id by removing MPNN variant suffixes.

    For bindcraft: removes _mpnn{n} suffix
    For rfd: removes _mpnn{n} suffix and _af2pred suffix

    Args:
        design_id: The design identifier
        method: The method type ("bindcraft" or "rfd")

    Returns:
        The backbone_id with MPNN variant suffixes removed
    """
    if not design_id:
        return design_id

    # bindcraft_design_111_l93_s308700_mpnn10 -> bindcraft_design_111_l93_s308700
    # design_ppi_1Ty4GSo_6_dldesign_0_cycle1_mpnn1_af2pred -> design_ppi_1Ty4GSo_6_dldesign_0_cycle1

    # Remove _mpnn{n} pattern, optionally followed by _af2pred
    # This handles both bindcraft (_mpnn{n}) and RFD (_mpnn{n}_af2pred) patterns
    backbone_id = re.sub(r"_mpnn\d+(?:_af2pred)?$", "", design_id)

    return backbone_id


# Run folder signatures for declarative run detection
# Each signature defines the structure required to identify a run type
run_folder_signatures = [
    {
        "method": "bindcraft",
        "submethod": "nf-binder-design",
        "priority": 1,  # Higher priority = checked first
        "required_files": ["results/bindcraft/final_design_stats.csv"],
        "required_dirs": ["results/bindcraft/accepted"],
        "results_table": "results/bindcraft/final_design_stats.csv",
        "pdb_pattern": "results/bindcraft/accepted/*.pdb",
        "skip_dirs": ["results/bindcraft/batches"],  # Skip walking into batches
        "params_files": ["results/params.json"],
        # Design parsing configuration
        "design_id_columns": ["Design"],
        "primary_score_columns": ["Average_i_pTM"],
        "sort_ascending": False,
        "pdb_search_patterns": [
            "{design_id}.pdb",
            "{design_id}_*.pdb",
            "{design_id}*.pdb",
        ],
    },
    {
        "method": "rfd",
        "submethod": "nf-binder-design",
        "priority": 2,
        "required_files": ["results/combined_scores.tsv"],
        "required_dirs": [
            "results/af2_initial_guess",
            "results/proteinmpnn",
            "results/rfdiffusion",
        ],
        "results_table": "results/combined_scores.tsv",
        "pdb_pattern": "results/af2_initial_guess/pdbs/*.pdb",
        "skip_dirs": [],
        "params_files": ["results/params.json"],
        # Design parsing configuration
        "design_id_columns": ["description"],
        "primary_score_columns": ["pae_interaction"],
        "sort_ascending": True,
        "pdb_search_patterns": ["{design_id}.pdb"],
    },
    {
        "method": "bindcraft",
        "submethod": "regular",
        "priority": 3,
        "required_files": ["final_design_stats.csv"],
        "required_dirs": ["Accepted"],
        "results_table": "final_design_stats.csv",
        "pdb_pattern": "Accepted/*.pdb",
        "skip_dirs": [],
        "params_files": ["../settings.json"],
        # Design parsing configuration
        "design_id_columns": ["Design"],
        "primary_score_columns": ["Average_i_pTM"],
        "sort_ascending": False,
        "pdb_search_patterns": [
            "{design_id}.pdb",
            "{design_id}_*.pdb",
            "{design_id}*.pdb",
        ],
    },
    {
        "method": "rfd",
        "submethod": "regular",
        "priority": 4,
        "required_dirs": ["af2_initial_guess"],
        "results_table": "combined_scores.tsv",
        "pdb_pattern": "af2_initial_guess/pdbs/*.pdb",
        "skip_dirs": [],
        # Design parsing configuration
        "design_id_columns": ["description"],
        "primary_score_columns": ["pae_interaction"],
        "sort_ascending": True,
        "pdb_search_patterns": ["{design_id}.pdb"],
    },
]


def guess_project_id(path: Path) -> str:
    run_name = guess_run_name(path)
    disallowed_patterns = [
        r"^runs$",
        r"^results.*$",
        r"^batch.*$",
        r"^bindcraft$",
        r"^rfd$",
        r"^\d+$",
    ]

    current_path = path
    found_run_name = False
    while current_path != current_path.parent:
        name = current_path.name
        if name == run_name:
            found_run_name = True
            current_path = current_path.parent
            continue
        if found_run_name:
            is_disallowed = any(
                re.match(pattern, name) for pattern in disallowed_patterns
            )
            if not is_disallowed:
                return name
        current_path = current_path.parent
    return ""


def guess_run_name(path: Path) -> str:
    disallowed_patterns = [r"^results.*$", r"^bindcraft$", r"^batches$", r"^\d+$"]
    current_path = path
    while current_path != current_path.parent:
        name = current_path.name
        is_disallowed = any(re.match(pattern, name) for pattern in disallowed_patterns)
        if not is_disallowed:
            return name
        current_path = current_path.parent
    return path.name


def _check_required_files(path: Path, required_files: List[str]) -> bool:
    """Check if all required files exist for the given run signature."""
    for file_path_str in required_files:
        file_path = path / file_path_str
        if not file_path.is_file():
            return False
    return True


def _check_required_dirs(path: Path, required_dirs: List[str]) -> bool:
    """Check if all required directories exist for the given run signature."""
    for dir_path_str in required_dirs:
        dir_path = path / dir_path_str
        if not dir_path.is_dir():
            return False
    return True


def _check_required_patterns(path: Path, required_patterns: List[str]) -> bool:
    """Check if any files match the required patterns for the given run signature."""
    for pattern in required_patterns:
        matches = list(path.glob(pattern))
        if matches:
            return True
    return False


def _find_pdb_file_for_design(
    run_path: Path, design_id: str, pdb_search_patterns: List[str], pdb_base_dir: str
) -> Optional[str]:
    """Find the PDB file for a design using the search patterns."""
    pdb_dir = run_path / pdb_base_dir
    if not pdb_dir.exists():
        return None

    for pattern in pdb_search_patterns:
        search_pattern = pattern.format(design_id=design_id)
        matches = list(pdb_dir.glob(search_pattern))
        if matches:
            return str(matches[0])
    return None


def detect_run_type(path: Path) -> Optional[Dict[str, Any]]:
    """Detect the run type using declarative signatures.

    Returns the matching signature with run_name extracted, or None if no match.
    """
    if not path.is_dir():
        return None

    # Sort signatures by priority (higher priority first)
    sorted_signatures = sorted(run_folder_signatures, key=lambda x: x["priority"])

    for signature in sorted_signatures:
        # The current directory is the run_name
        run_name = path.name

        # Check required files
        if "required_files" in signature:
            if not _check_required_files(path, signature["required_files"]):
                continue

        # Check required directories
        if "required_dirs" in signature:
            if not _check_required_dirs(path, signature["required_dirs"]):
                continue

        # Check required patterns (alternative to required_files for some cases)
        if "required_patterns" in signature:
            if not _check_required_patterns(path, signature["required_patterns"]):
                continue

        # Special case for regular RFD: check if combined_scores.tsv exists OR .cs files exist
        if signature["method"] == "rfd" and signature["submethod"] == "regular":
            combined_file = path / "combined_scores.tsv"
            cs_files_scores = list((path / "af2_initial_guess" / "scores").glob("*.cs"))
            cs_files_root = list((path / "af2_initial_guess").glob("*.cs"))

            if not (combined_file.is_file() or cs_files_scores or cs_files_root):
                continue

        # If we get here, this signature matches
        return {**signature, "run_name": run_name, "detected_path": str(path)}

    return None


def find_runs_recursive(root_path: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=True):
        current_dir = Path(dirpath)
        if current_dir.name == "work":
            dirnames[:] = []
            continue

        # Skip directories that are inside batches subdirectories of nf-binder-design runs
        path_parts = current_dir.parts
        if "batches" in path_parts:
            # Check if this is inside an nf-binder-design run by looking for the pattern
            # {run_name}/results/bindcraft/batches/{n}/...
            try:
                batches_index = path_parts.index("batches")
                if (
                    batches_index >= 2
                    and path_parts[batches_index - 1] == "bindcraft"
                    and path_parts[batches_index - 2] == "results"
                ):
                    # This is inside a batches directory of an nf-binder-design run, skip it
                    dirnames[:] = []
                    continue
            except ValueError:
                pass  # "batches" not found in path_parts

        # Use declarative detection
        detected_run = detect_run_type(current_dir)
        if detected_run:
            run_id = str(uuid.uuid4())
            guessed_project_id = guess_project_id(current_dir)
            run_name = detected_run["run_name"]

            # Use the results table and PDB pattern directly from the signature
            results_table = detected_run["results_table"]
            pdb_pattern = detected_run["pdb_pattern"]

            # Find PDB files using the pattern
            pdb_files = [str(p) for p in current_dir.glob(pdb_pattern)]

            # Determine if this is an nf-binder-design run
            is_nf_binder_design = detected_run["submethod"] == "nf-binder-design"

            runs.append(
                {
                    "run_id": run_id,
                    "project_id": guessed_project_id,
                    "path": str(current_dir),
                    "method": detected_run["method"],
                    "submethod": detected_run["submethod"],
                    "results_table": results_table,
                    "pdb_files": pdb_files,
                    "is_nf_binder_design": is_nf_binder_design,
                    "signature": detected_run,  # Store the full signature for use in parse_designs_from_run
                    "metadata": {
                        "name": run_name,
                        "original_name": current_dir.name,
                        "parent_path": str(current_dir.parent),
                        "pdb_count": len(pdb_files),
                    },
                }
            )

            # Handle directory skipping based on signature
            if "skip_dirs" in detected_run and detected_run["skip_dirs"]:
                # Remove directories that should be skipped from dirnames
                skip_dir_names = []
                for skip_path_str in detected_run["skip_dirs"]:
                    skip_path = current_dir / skip_path_str
                    if skip_path.is_dir():
                        skip_dir_names.append(skip_path.name)
                dirnames[:] = [d for d in dirnames if d not in skip_dir_names]
            else:
                # Stop walking this directory tree since we found a run
                dirnames[:] = []
    return runs


def load_run_table(run_metadata: Dict[str, Any]) -> Optional[pd.DataFrame]:
    try:
        merged_paths = run_metadata.get("merged_paths", [run_metadata["path"]])
        results_table = run_metadata.get("results_table")
        if not results_table:
            return None
        all_dfs: List[pd.DataFrame] = []
        for run_path in merged_paths:
            path = Path(run_path)
            table_path = path / results_table
            if not table_path.exists():
                logger.warning(f"Results table not found: {table_path}")
                continue
            try:
                if table_path.suffix.lower() == ".csv":
                    df = pd.read_csv(table_path)
                elif table_path.suffix.lower() == ".tsv":
                    df = pd.read_csv(table_path, sep="\t")
                else:
                    logger.warning(f"Unsupported table format: {table_path}")
                    continue
                # Coerce numeric-like object columns to numeric dtype where fully parseable
                # This avoids returning numbers as strings, without forcing partial coercion.
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = pd.to_numeric(df[col], errors="ignore")
                df = _standardise_dataframe_columns(df, run_metadata.get("method", ""))
                df["source_path"] = str(path)
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading table from {table_path}: {str(e)}")
                continue
        if not all_dfs:
            logger.warning(
                f"No valid tables found for run: {run_metadata.get('path', 'unknown')}"
            )
            return None
        if len(all_dfs) == 1:
            return all_dfs[0]
        else:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            logger.info(
                f"Combined {len(all_dfs)} tables for merged run, total rows: {len(combined_df)}"
            )
            return combined_df
    except Exception as e:
        logger.error(f"Error loading run table: {str(e)}")
        return None


def _standardise_dataframe_columns(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    canonical_groups: Dict[str, List[str]] = {
        "Sequence": [
            "Sequence",
            "sequence",
            "binder_sequence",
            "binder_seq",
            "seq",
        ],
        "Length": ["Length", "length", "len", "binder_length"],
    }
    lower_to_original: Dict[str, str] = {col.lower(): col for col in df.columns}
    result_df = df.copy()
    for target, variants in canonical_groups.items():
        source_cols: List[str] = []
        for v in variants:
            original = lower_to_original.get(v.lower())
            if original and original not in source_cols:
                source_cols.append(original)
        if not source_cols:
            continue
        if target in result_df.columns:
            for src in source_cols:
                if src == target:
                    continue
                result_df[target] = result_df[target].fillna(result_df[src])
        else:
            series = None
            for idx, src in enumerate(source_cols):
                if idx == 0:
                    series = result_df[src]
                else:
                    series = series.where(series.notna(), result_df[src])  # type: ignore
            if series is not None:
                result_df[target] = series
        for src in source_cols:
            if src != target and src in result_df.columns:
                result_df = result_df.drop(columns=[src])
    return result_df


def parse_run_params(run_metadata: Dict[str, Any]) -> Optional[Any]:
    """Parse parameter/settings file for a run and return the raw JSON content.

    - Parse the first existing file in signature["params_files"], if provided.

    Returns the parsed JSON (dict/list/primitive) or None if not found or error.
    """
    try:
        run_path_str = run_metadata.get("path", "")
        if not run_path_str:
            return None
        run_path = Path(run_path_str)

        signature: Dict[str, Any] = run_metadata.get("signature", {})
        submethod = signature.get("submethod", run_metadata.get("submethod", ""))

        json_path: Optional[Path] = None

        params_files: List[str] = signature.get("params_files", [])
        for rel in params_files:
            candidate = run_path / rel
            if candidate.is_file():
                json_path = candidate
                break

        if not json_path:
            return None

        logger.info(f"Parsing run params from {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error parsing run params: {str(e)}")
        return None


def parse_designs_from_run(run_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        df = load_run_table(run_metadata)
        if df is None or df.empty:
            return []

        designs: List[Dict[str, Any]] = []
        run_path = run_metadata["path"]
        run_name = run_metadata["metadata"]["name"]
        signature = run_metadata.get("signature", {})

        # Get configuration from signature
        design_id_columns = signature.get("design_id_columns", [])
        primary_score_columns = signature.get("primary_score_columns", [])
        sort_ascending = signature.get("sort_ascending", True)
        pdb_search_patterns = signature.get("pdb_search_patterns", ["{design_id}.pdb"])

        # Find design ID column
        design_id_col = None
        for col_name in design_id_columns:
            if col_name in df.columns:
                design_id_col = col_name
                break

        # Fallback: look for common design ID column names
        if not design_id_col:
            for col in df.columns:
                if col.lower() in ["design", "description", "name", "id"]:
                    design_id_col = col
                    break

        # Find primary score column
        primary_score_col = None
        for col_name in primary_score_columns:
            if col_name in df.columns:
                primary_score_col = col_name
                break

        # Fallback: use first numeric column
        if not primary_score_col:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                primary_score_col = numeric_cols[0]

        # Sort by primary score if available
        if primary_score_col and primary_score_col in df.columns:
            df = df.sort_values(
                primary_score_col, ascending=sort_ascending
            ).reset_index(drop=True)

        # Determine PDB base directory from the pdb_pattern
        pdb_pattern = signature.get("pdb_pattern", "")
        pdb_base_dir = pdb_pattern.split("/*.pdb")[0] if "/*.pdb" in pdb_pattern else ""

        # Parse any run-wide parameters/settings; will be attached as a single 'params' field
        run_params: Optional[Any] = parse_run_params(run_metadata)

        for index, row in df.iterrows():
            design_id = (
                str(row.get(design_id_col, f"design_{index}"))
                if design_id_col
                else f"design_{index}"
            )

            # Find PDB file using signature configuration
            pdb_file = _find_pdb_file_for_design(
                Path(run_path), design_id, pdb_search_patterns, pdb_base_dir
            )

            # Extract backbone_id for MPNN filtering
            backbone_id = extract_backbone_id(design_id, run_metadata["method"])

            # Get binder sequence for target sequence extraction
            binder_sequence = None
            for seq_col in ["Sequence", "sequence", "binder_sequence"]:
                if seq_col in df.columns:
                    try:
                        val = row[seq_col]
                        # Check if val is not null and not empty
                        if val is not None and str(val).strip():
                            binder_sequence = str(val)
                            break
                    except (KeyError, AttributeError):
                        continue

            # Extract target sequence from PDB file
            target_sequence = (
                get_target_sequence(pdb_file, run_metadata["method"], binder_sequence)
                if pdb_file
                else None
            )

            design: Dict[str, Any] = {
                "design_id": design_id,
                "backbone_id": backbone_id,
                "run_id": run_metadata["run_id"],
                "project_id": run_metadata.get("project_id", ""),
                "run_name": run_name,
                "method": run_metadata["method"],
                "run_path": run_path,
                "pdb_file": pdb_file,
                "target_sequence": target_sequence,
                **{
                    col: row[col]
                    for col in df.columns
                    if col != design_id_col
                    and not bool(
                        (
                            pd.isna(row[col])
                            if hasattr(pd, "isna")
                            else (row[col] is None)
                        )
                    )
                },
            }
            # Attach raw params JSON (applies to all designs in the run)
            if run_params is not None:
                design["params"] = run_params
            designs.append(design)
        return designs
    except Exception as e:
        logger.error(
            f"Error parsing designs from run {run_metadata['run_id']}: {str(e)}"
        )
        return []


def update_good_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and updates the 'good_rank' column based on the existing sort order.
    The input DataFrame is expected to be sorted already.
    """
    # Ensure the 'good' column exists
    if "good" not in df.columns:
        df["good_rank"] = pd.NA
        return df.astype({"good_rank": "Int64"})

    # Get the indices of rows marked as 'good'
    good_indices = df[df["good"]].index

    # Create a Series with ranks for the 'good' rows
    # The rank is based on the existing order in the DataFrame
    good_rank_series = pd.Series(
        np.arange(1, len(good_indices) + 1), index=good_indices
    )

    # Assign this series to the new 'good_rank' column.
    # Rows not in good_indices will get NaN.
    df["good_rank"] = good_rank_series

    # Convert to nullable integer type to support NAs
    df["good_rank"] = df["good_rank"].astype("Int64")

    return df


def load_all_designs(
    root_path: Path, exclude_folders: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """Load and aggregate designs from all runs under root_path."""
    if exclude_folders is None:
        exclude_folders = []

    if not root_path.exists() or not root_path.is_dir():
        st.error(f"Root search path {root_path} does not exist or is not a directory.")
        return None

    # Find all runs
    runs = find_runs_recursive(root_path)
    if not runs:
        st.warning(f"No runs found under {root_path}.")
        return None

    st.info(f"Found {len(runs)} run(s). Loading designs...")

    # Parse designs from each run
    all_designs = []
    for run in runs:
        designs = parse_designs_from_run(run)
        all_designs.extend(designs)

    if not all_designs:
        st.warning("No designs found in any of the runs.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_designs)

    # Add good column, default False
    if "good" not in df.columns:
        df.insert(1, "good", False)
    else:
        df["good"] = df["good"].fillna(False).astype(bool)

    # Implement dual-score sorting
    # Sort by Average_i_pTM desc (NaN last), then pae_interaction asc (NaN last)
    sort_columns = []
    sort_ascending = []

    if "Average_i_pTM" in df.columns:
        sort_columns.append("Average_i_pTM")
        sort_ascending.append(False)  # Higher is better

    if "pae_interaction" in df.columns:
        sort_columns.append("pae_interaction")
        sort_ascending.append(True)  # Lower is better

    if sort_columns:
        df = df.sort_values(
            by=sort_columns, ascending=sort_ascending, na_position="last"
        ).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Add Rank column
    df["Rank"] = np.arange(1, len(df) + 1).astype(int)

    # Position Rank column after 'good'
    cols = df.columns.tolist()
    if "good" in cols:
        cols.remove("Rank")
        good_index = cols.index("good")
        cols.insert(good_index + 1, "Rank")
        df = df[cols]

    # Add/update good_rank
    df = update_good_rank(df)

    return df


def plot_distribution(df: pd.DataFrame, metric: str) -> None:
    """Create a distribution plot for a single selected metric."""
    if not metric or metric not in df.columns:
        st.warning(f"Metric '{metric}' not found in DataFrame for distribution plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))  # Single plot
    sns.kdeplot(data=df, x=metric, ax=ax, fill=True, warn_singular=False)
    sns.rugplot(data=df, x=metric, ax=ax, alpha=0.5)
    ax.set_title(f"Distribution of {metric}")
    plt.tight_layout()
    st.pyplot(fig)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[Any]:
    """Create interactive scatter plot using Altair."""
    # Create a selection condition
    selection = alt.selection_point(
        name="selected_point", fields=["design_id", x_col, y_col]
    )

    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(x_col, scale=alt.Scale(zero=False)),
            y=alt.Y(y_col, scale=alt.Scale(zero=False)),
            tooltip=["design_id", "run_name", x_col, y_col, "pdb_file"],
            color=alt.condition(
                selection, alt.value("steelblue"), alt.value("lightgray")
            ),
        )
        .properties(width=500, height=500)
        .add_params(selection)
        .interactive()
    )

    # Use streamlit's event system to handle point clicks
    chart_event = st.altair_chart(
        chart, use_container_width=True, key="scatter", on_select="rerun"
    )

    return chart_event


def main():
    # Set up page config for wide mode and collapsed sidebar
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    st.title("Protein Binder Design Results Viewer")

    # Command line argument for default path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="Path to results directory", default="."
    )
    args = parser.parse_args()

    # Define the summary file path
    summary_file_path = Path(args.path).resolve() / "designs_summary.tsv"
    st.markdown(f"Saving summary data to: `{summary_file_path}`")

    # Initialize session state for path if not exists
    if "root_search_path" not in st.session_state:
        st.session_state["root_search_path"] = Path(args.path).resolve()

    # Initialize session state for selected structure index (now a list for multi-select)
    if "selected_df_indices" not in st.session_state:
        st.session_state.selected_df_indices = []

    # Initialize df in session state if it's not already there to prevent AttributeError
    if "df" not in st.session_state:
        st.session_state.df = None

    # Sidebar for path input and regenerate button
    with st.sidebar:
        st.header("Results Directory")
        st.info(f"Searching in: {st.session_state.root_search_path}")
        st.markdown(
            "To change the search directory, please restart the app with the desired `--path` argument."
        )
        st.divider()

        # Exclude folders input
        st.subheader("Scan Settings")
        exclude_folders_text = st.text_input(
            "Exclude folders (comma separated)",
            value="work",
            help="Folders to exclude when scanning for results. Separate multiple folders with commas.",
        )
        # Parse the exclude folders text into a list
        exclude_folders = [
            folder.strip()
            for folder in exclude_folders_text.split(",")
            if folder.strip()
        ]
        st.divider()

        if st.button(
            "🔄 Regenerate Summary File",
            help="Re-scans the directory, merges existing 'good' ratings, and overwrites designs_summary.tsv",
        ):
            with st.spinner("Regenerating summary file..."):
                # Force a re-scan and merge by temporarily setting df to None in session_state
                current_summary_df = None
                if summary_file_path.exists():
                    try:
                        current_summary_df = pd.read_csv(summary_file_path, sep="\t")
                    except Exception as e:
                        st.warning(
                            f"Could not read existing summary file for regeneration: {e}"
                        )

                fresh_df = load_all_designs(
                    st.session_state.root_search_path, exclude_folders
                )
                if fresh_df is not None:
                    if "good" not in fresh_df.columns:
                        insert_pos = 1 if len(fresh_df.columns) > 0 else 0
                        fresh_df.insert(insert_pos, "good", False)
                    else:
                        fresh_df["good"] = fresh_df["good"].fillna(False).astype(bool)

                    if (
                        current_summary_df is not None
                        and "design_id" in current_summary_df.columns
                        and "run_path" in current_summary_df.columns
                        and "good" in current_summary_df.columns
                    ):
                        good_stamps_to_merge = current_summary_df[
                            ["design_id", "run_path", "good"]
                        ].drop_duplicates(
                            subset=["design_id", "run_path"], keep="first"
                        )

                        if "good" in fresh_df.columns:
                            fresh_df = fresh_df.drop(columns=["good"])

                        fresh_df = pd.merge(
                            fresh_df,
                            good_stamps_to_merge,
                            on=["design_id", "run_path"],
                            how="left",
                        )
                        fresh_df["good"] = fresh_df["good"].fillna(False).astype(bool)

                    # With 'good' status merged, we need to recalculate good_rank
                    fresh_df = update_good_rank(fresh_df)

                    st.session_state.df = fresh_df
                    try:
                        st.session_state.df.drop(
                            columns=["good_rank"], errors="ignore"
                        ).to_csv(summary_file_path, sep="\t", index=False)
                        st.success("Summary file regenerated and saved.")
                    except Exception as e:
                        st.error(f"Error saving regenerated summary file: {e}")

                    st.session_state.selected_df_indices = []
                    if "good" in st.session_state.df.columns:
                        st.session_state.good_values = pd.Series(
                            st.session_state.df["good"],
                            index=st.session_state.df.index,
                            dtype=bool,
                        )
                    else:
                        st.session_state.good_values = pd.Series(
                            [False] * len(st.session_state.df),
                            index=st.session_state.df.index,
                            dtype=bool,
                        )
                    st.rerun()
                else:
                    st.error("Failed to load data during regeneration scan.")

    # Attempt to load data: either from existing session_state.df (if already loaded/regenerated),
    # or from summary file, or by scanning if summary doesn't exist.
    if st.session_state.df is None:
        if summary_file_path.exists():
            try:
                st.session_state.df = pd.read_csv(summary_file_path, sep="\t")
                # Always remove Rank if it exists, to re-calculate it based on current sort.
                if "Rank" in st.session_state.df.columns:
                    st.session_state.df = st.session_state.df.drop(columns=["Rank"])

                if "good" in st.session_state.df.columns:
                    st.session_state.df["good"] = (
                        st.session_state.df["good"].fillna(False).astype(bool)
                    )
                else:
                    st.session_state.df["good"] = False

                if "pdb_file" in st.session_state.df.columns:
                    st.session_state.df["pdb_file"] = st.session_state.df[
                        "pdb_file"
                    ].astype(pd.StringDtype())
                if "run_path" in st.session_state.df.columns:
                    st.session_state.df["run_path"] = st.session_state.df[
                        "run_path"
                    ].astype(pd.StringDtype())

                st.session_state.df = st.session_state.df.convert_dtypes(
                    convert_string=False
                )

                # Implement dual-score sorting
                sort_columns = []
                sort_ascending = []

                if "Average_i_pTM" in st.session_state.df.columns:
                    sort_columns.append("Average_i_pTM")
                    sort_ascending.append(False)

                if "pae_interaction" in st.session_state.df.columns:
                    sort_columns.append("pae_interaction")
                    sort_ascending.append(True)

                if sort_columns:
                    st.session_state.df = st.session_state.df.sort_values(
                        by=sort_columns, ascending=sort_ascending, na_position="last"
                    ).reset_index(drop=True)
                else:
                    st.session_state.df = st.session_state.df.reset_index(drop=True)

                # Add Rank column from 1 to N and position it after 'good'.
                st.session_state.df["Rank"] = np.arange(
                    1, len(st.session_state.df) + 1
                ).astype(int)
                cols = st.session_state.df.columns.tolist()
                if "good" in cols:
                    cols.remove("Rank")
                    good_index = cols.index("good")
                    cols.insert(good_index + 1, "Rank")
                    st.session_state.df = st.session_state.df[cols]

                # Update good_rank after loading and sorting
                st.session_state.df = update_good_rank(st.session_state.df)

                st.info(f"Loaded data from summary file: {summary_file_path}")
            except Exception as e:
                st.error(
                    f"Error loading summary file {summary_file_path}: {e}. Will perform initial scan."
                )
                st.session_state.df = None

        # If df is still None (summary didn't exist or failed to load), then perform initial scan and create summary.
        if st.session_state.df is None:
            st.session_state.df = load_all_designs(
                st.session_state.root_search_path, exclude_folders
            )
            if st.session_state.df is not None:
                # Ensure 'good' column exists before first save
                if "good" not in st.session_state.df.columns:
                    insert_pos = 1 if len(st.session_state.df.columns) > 0 else 0
                    st.session_state.df.insert(insert_pos, "good", False)
                else:
                    st.session_state.df["good"] = (
                        st.session_state.df["good"].fillna(False).astype(bool)
                    )
                try:
                    st.session_state.df.drop(
                        columns=["good_rank"], errors="ignore"
                    ).to_csv(summary_file_path, sep="\t", index=False)
                    st.success(f"Initial summary file created: {summary_file_path}")
                except Exception as e:
                    st.error(
                        f"Could not create initial summary file {summary_file_path}: {e}"
                    )

    df = st.session_state.df

    if df is None:
        st.warning(
            "No design data loaded. Please ensure the specified path contains valid run directories."
        )
        return

    # Initialize or sync 'good_values' session state with the current df's 'good' column
    if "good" in df.columns:
        st.session_state.good_values = pd.Series(df["good"], index=df.index, dtype=bool)
    else:
        st.session_state.good_values = pd.Series(
            [False] * len(df), index=df.index, dtype=bool
        )
        df["good"] = st.session_state.good_values

    # Ensure df also reflects the most current 'good' values from session_state
    df["good"] = st.session_state.good_values.reindex(
        df.index, fill_value=False
    ).astype(bool)

    # Get numeric columns for plotting
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Initialize session state for column selection if not exists
    if "selected_cols" not in st.session_state:
        # Prioritize these columns, then add the rest
        default_table_display_order = [
            "good",  # Make this the first *data* column displayed by default
            "Rank",
            "good_rank",
            "Average_i_pTM",  # BindCraft score
            "pae_interaction",  # RFdiffusion score
            "Average_pLDDT",  # More specific RMSD column
            "plddt_binder",  # RFdiffusion pLDDT
            "design_id",
            "run_name",
            "project_id",
            "pdb_file",
            "run_path",
        ]

        actual_default_table_cols = [
            col for col in default_table_display_order if col in df.columns
        ]
        remaining_table_cols = [
            col for col in df.columns if col not in actual_default_table_cols
        ]
        st.session_state.selected_cols = (
            actual_default_table_cols + remaining_table_cols
        )

    # Initialize session state for scatter plot axes
    default_x_col = None
    if numeric_cols:
        # Try both pLDDT columns
        if "Average_pLDDT" in numeric_cols:
            default_x_col = "Average_pLDDT"
        elif "plddt_binder" in numeric_cols:
            default_x_col = "plddt_binder"
        else:
            default_x_col = numeric_cols[0]

    default_y_col = None
    if numeric_cols:
        # Try both score columns
        if "Average_i_pTM" in numeric_cols:
            default_y_col = "Average_i_pTM"
        elif "pae_interaction" in numeric_cols:
            default_y_col = "pae_interaction"
        elif len(numeric_cols) > 1:
            default_y_col = numeric_cols[1]
        elif numeric_cols:
            default_y_col = numeric_cols[0]

    if "scatter_x_col" not in st.session_state:
        st.session_state.scatter_x_col = default_x_col
    if "scatter_y_col" not in st.session_state:
        st.session_state.scatter_y_col = default_y_col

    # Initialize session state for auto_next checkbox
    if "auto_next_on_thumbs_click" not in st.session_state:
        st.session_state.auto_next_on_thumbs_click = False

    # Initialize session state for "Show good only" filter
    if "show_good_only" not in st.session_state:
        st.session_state.show_good_only = False

    # Prepare DataFrame for the data_editor: add a selection column
    df_for_editor = df.copy()
    selection_col_name = "✔️ Select"
    df_for_editor[selection_col_name] = False
    if st.session_state.selected_df_indices:
        for idx in st.session_state.selected_df_indices:
            if 0 <= idx < len(df_for_editor):
                df_for_editor.loc[idx, selection_col_name] = True

    # Update 'good' column in df_for_editor from the main df
    if "good" in df.columns and "good" in df_for_editor.columns:
        df_for_editor["good"] = df["good"]

    # Create a view of the dataframe that can be filtered for display
    df_display = df_for_editor
    if st.session_state.show_good_only:
        if "good" in df_display.columns:
            df_display = df_display[df_display["good"] == True]

    # Create tabs for different views
    table_tab, scatter_tab, dist_tab = st.tabs(
        ["Data Table", "Scatter Plot", "Distributions"]
    )

    # These will capture the direct output of widgets for selection processing
    scatter_event_result = None
    edited_df_from_editor = None

    with table_tab:
        if st.session_state.show_good_only:
            st.info(
                f"Displaying {len(df_display)} of {len(df_for_editor)} total rows (good only)."
            )
        else:
            st.info(f"Total rows: {len(df_for_editor)}")

        editor_column_config = {
            selection_col_name: st.column_config.CheckboxColumn(
                "View",
                help="Select a row to view structure",
                default=False,
            ),
            "good": st.column_config.CheckboxColumn(
                "Good",
                help="Mark as a good design",
                default=False,
            ),
            **{
                col: st.column_config.NumberColumn(col, format="%.3f")
                for col in numeric_cols
                if col in df_for_editor.columns
            },
        }

        # Override Rank column to ensure integer display
        if "Rank" in df_for_editor.columns:
            editor_column_config["Rank"] = st.column_config.NumberColumn(
                "Rank", format="%d"
            )

        # Override good_rank column to ensure integer display and add help text
        if "good_rank" in df_for_editor.columns:
            editor_column_config["good_rank"] = st.column_config.NumberColumn(
                "Good Rank", format="%d", help="Rank among only 'Good' designs"
            )

        # Put selection column first.
        columns_to_display_in_editor = [selection_col_name] + [
            col
            for col in st.session_state.selected_cols
            if col in df_for_editor.columns and col != selection_col_name
        ]

        # Filter df_for_editor to only include columns that will be displayed
        df_display_subset = df_display[columns_to_display_in_editor]

        edited_df_from_editor = st.data_editor(
            df_display_subset,
            hide_index=True,
            use_container_width=True,
            column_config=editor_column_config,
            disabled=[
                col for col in df.columns if col not in [selection_col_name, "good"]
            ],
            key="data_editor_table",
        )

        # Table controls below the dataframe
        st.caption("Table Controls")
        with st.expander("Column Display Settings"):
            st.checkbox(
                "Show good only",
                key="show_good_only",
                help="Filter the table to show only designs marked as 'Good'.",
            )
            all_cols = df.columns.tolist()
            # Show both score types first in options
            if "Average_i_pTM" in all_cols:
                all_cols.remove("Average_i_pTM")
                all_cols.insert(0, "Average_i_pTM")
            if "pae_interaction" in all_cols:
                all_cols.remove("pae_interaction")
                all_cols.insert(1, "pae_interaction")
            st.session_state.selected_cols = st.multiselect(
                "Select columns to display",
                options=all_cols,
                default=st.session_state.selected_cols,
            )

    with scatter_tab:
        col1, col2 = st.columns(2)
        with col1:
            if numeric_cols:
                current_x_idx = 0
                if (
                    st.session_state.scatter_x_col
                    and st.session_state.scatter_x_col in numeric_cols
                ):
                    current_x_idx = numeric_cols.index(st.session_state.scatter_x_col)
                selected_x = st.selectbox(
                    "X-axis",
                    options=numeric_cols,
                    index=current_x_idx,
                    key="x_axis_selector",
                )
                if selected_x:
                    st.session_state.scatter_x_col = selected_x
            else:
                st.warning("No numeric columns available for X-axis.")
        with col2:
            if numeric_cols:
                current_y_idx = 0
                if (
                    st.session_state.scatter_y_col
                    and st.session_state.scatter_y_col in numeric_cols
                ):
                    current_y_idx = numeric_cols.index(st.session_state.scatter_y_col)
                elif len(numeric_cols) > 1 and numeric_cols[0] != st.session_state.get(
                    "scatter_x_col"
                ):
                    current_y_idx = 0
                elif len(numeric_cols) > 1:
                    current_y_idx = 1

                selected_y = st.selectbox(
                    "Y-axis",
                    options=numeric_cols,
                    index=current_y_idx,
                    key="y_axis_selector",
                )
                if selected_y:
                    st.session_state.scatter_y_col = selected_y
            else:
                st.warning("No numeric columns available for Y-axis.")

        if (
            st.session_state.scatter_x_col
            and st.session_state.scatter_y_col
            and st.session_state.scatter_x_col in df.columns
            and st.session_state.scatter_y_col in df.columns
        ):
            scatter_event_result = plot_scatter(
                df, st.session_state.scatter_x_col, st.session_state.scatter_y_col
            )
        elif numeric_cols:
            st.warning("Please select valid X and Y axes for the scatter plot.")

    with dist_tab:
        if numeric_cols:
            current_dist_metric_idx = 0
            if (
                st.session_state.scatter_y_col
                and st.session_state.scatter_y_col in numeric_cols
            ):
                current_dist_metric_idx = numeric_cols.index(
                    st.session_state.scatter_y_col
                )
            elif "Average_i_pTM" in numeric_cols:
                current_dist_metric_idx = numeric_cols.index("Average_i_pTM")
            elif "pae_interaction" in numeric_cols:
                current_dist_metric_idx = numeric_cols.index("pae_interaction")

            selected_dist_metric = st.selectbox(
                "Select metric for distribution plot",
                options=numeric_cols,
                index=current_dist_metric_idx,
                key="dist_metric_selector",
            )
            if selected_dist_metric:
                st.session_state.scatter_y_col = selected_dist_metric
                if st.session_state.scatter_y_col in df.columns:
                    plot_distribution(df, st.session_state.scatter_y_col)
                else:
                    st.warning(
                        f"Selected metric '{st.session_state.scatter_y_col}' not found in DataFrame."
                    )
        else:
            st.info("No numeric columns available for distribution plots.")

    # Process selections to update st.session_state.selected_df_indices
    new_selection_made_in_this_run = False

    # 1. Process selection from data_editor
    if edited_df_from_editor is not None:
        # Process 'good' column changes from the editor
        if "good" in edited_df_from_editor.columns and "good" in df.columns:
            changed_good_series = edited_df_from_editor["good"]
            for original_df_idx in changed_good_series.index:
                if original_df_idx in df.index:
                    new_good_value = changed_good_series.loc[original_df_idx]
                    new_good_value_bool = bool(new_good_value)

                    current_good_value_in_state = bool(
                        st.session_state.good_values.loc[original_df_idx]
                    )

                    if current_good_value_in_state != new_good_value_bool:
                        df.loc[original_df_idx, "good"] = new_good_value_bool
                        st.session_state.good_values.loc[original_df_idx] = (
                            new_good_value_bool
                        )
                        new_selection_made_in_this_run = True

                        # A 'good' value changed, so we need to re-calculate the good_rank
                        df = update_good_rank(df)
                        st.session_state.df = df

                        # Save DataFrame to TSV after editor change
                        try:
                            df.drop(columns=["good_rank"], errors="ignore").to_csv(
                                summary_file_path, sep="\t", index=False
                            )
                        except Exception as e:
                            st.error(f"Error saving summary file: {e}")

        # Check if the selection column exists in the output of data_editor
        if selection_col_name in edited_df_from_editor.columns:
            selected_in_editor_series = edited_df_from_editor[selection_col_name]

            # Find rows that are checked True in the editor
            currently_checked_indices = selected_in_editor_series[
                selected_in_editor_series == True
            ].index.tolist()

            if currently_checked_indices:
                # User has at least one box checked in the editor
                if set(st.session_state.selected_df_indices) != set(
                    currently_checked_indices
                ):
                    st.session_state.selected_df_indices = sorted(
                        currently_checked_indices
                    )
                    new_selection_made_in_this_run = True

            elif not currently_checked_indices and st.session_state.selected_df_indices:
                # No boxes are checked in the editor, but there was a selection.
                st.session_state.selected_df_indices = []
                new_selection_made_in_this_run = True

    # 2. Process selection from scatter plot
    if scatter_event_result:
        selection_data = scatter_event_result.get("selection")
        if selection_data and "selected_point" in selection_data:
            selected_points = selection_data["selected_point"]
            if selected_points and len(selected_points) > 0:
                # Use "design_id" field for matching
                selected_design_id = selected_points[0].get("design_id")
                if selected_design_id:
                    matching_rows = df[df["design_id"] == selected_design_id]
                    if not matching_rows.empty:
                        new_idx_from_scatter = int(matching_rows.index[0])
                        # Scatter click replaces current selection
                        if st.session_state.selected_df_indices != [
                            new_idx_from_scatter
                        ]:
                            st.session_state.selected_df_indices = [
                                new_idx_from_scatter
                            ]
                            new_selection_made_in_this_run = True
                else:  # Fallback if "design_id" is not in point data
                    point_data = selected_points[0]
                    if (
                        st.session_state.scatter_x_col in point_data
                        and st.session_state.scatter_y_col in point_data
                    ):
                        mask = (
                            df[st.session_state.scatter_x_col]
                            == point_data[st.session_state.scatter_x_col]
                        ) & (
                            df[st.session_state.scatter_y_col]
                            == point_data[st.session_state.scatter_y_col]
                        )
                        if mask.any():
                            # Ensure the index from the original df is used
                            original_df_index = df[mask].index[0]
                            new_idx_from_scatter = int(original_df_index)
                            # Scatter click replaces current selection
                            if st.session_state.selected_df_indices != [
                                new_idx_from_scatter
                            ]:
                                st.session_state.selected_df_indices = [
                                    new_idx_from_scatter
                                ]
                                new_selection_made_in_this_run = True

    # If any selection change happened (editor or scatter), rerun to ensure UI consistency
    if new_selection_made_in_this_run:
        st.rerun()

    # Structure Viewer Section
    st.header("Structure Viewer")

    # New layout for structure controls
    if not df.empty:
        # Columns for: Prev, Thumbs Up, Thumbs Down, Info, Next
        col_prev_btn, col_thumb_up, col_thumb_down, col_info_display, col_next_btn = (
            st.columns([1.5, 1, 1, 6, 1.5])
        )

        primary_idx_for_nav = (
            st.session_state.selected_df_indices[0]
            if st.session_state.selected_df_indices
            else None
        )

        with col_prev_btn:
            prev_disabled = primary_idx_for_nav is None or primary_idx_for_nav == 0
            if st.button(
                "⬅️",
                help="Previous Structure",
                disabled=prev_disabled,
                use_container_width=True,
            ):
                if primary_idx_for_nav is not None and primary_idx_for_nav > 0:
                    st.session_state.selected_df_indices = [primary_idx_for_nav - 1]
                    st.rerun()

        with col_thumb_up:
            thumb_up_disabled = not (
                primary_idx_for_nav is not None
                and len(st.session_state.selected_df_indices) == 1
            )
            if st.button(
                "👍",
                help="Mark as Good",
                disabled=thumb_up_disabled,
                use_container_width=True,
            ):
                if (
                    "good" in df.columns
                    and primary_idx_for_nav is not None
                    and primary_idx_for_nav in df.index
                ):
                    idx_to_update = int(primary_idx_for_nav)
                    df.loc[idx_to_update, "good"] = True
                    if idx_to_update in st.session_state.good_values.index:
                        st.session_state.good_values.at[idx_to_update] = True

                    # A 'good' value changed, so we need to re-calculate the good_rank
                    df = update_good_rank(df)
                    st.session_state.df = df  # also update session state

                    try:
                        df.drop(columns=["good_rank"], errors="ignore").to_csv(
                            summary_file_path, sep="\t", index=False
                        )
                    except Exception as e:
                        st.error(f"Error saving summary file: {e}")
                    if (
                        st.session_state.auto_next_on_thumbs_click
                        and idx_to_update < len(df) - 1
                    ):
                        st.session_state.selected_df_indices = [idx_to_update + 1]
                    st.rerun()

        with col_thumb_down:
            thumb_down_disabled = not (
                primary_idx_for_nav is not None
                and len(st.session_state.selected_df_indices) == 1
            )
            if st.button(
                "👎",
                help="Mark as Not Good",
                disabled=thumb_down_disabled,
                use_container_width=True,
            ):
                if (
                    "good" in df.columns
                    and primary_idx_for_nav is not None
                    and primary_idx_for_nav in df.index
                ):
                    idx_to_update = int(primary_idx_for_nav)
                    df.loc[idx_to_update, "good"] = False
                    if idx_to_update in st.session_state.good_values.index:
                        st.session_state.good_values.at[idx_to_update] = False

                    # A 'good' value changed, so we need to re-calculate the good_rank
                    df = update_good_rank(df)
                    st.session_state.df = df  # also update session state

                    try:
                        df.drop(columns=["good_rank"], errors="ignore").to_csv(
                            summary_file_path, sep="\t", index=False
                        )
                    except Exception as e:
                        st.error(f"Error saving summary file: {e}")
                    if (
                        st.session_state.auto_next_on_thumbs_click
                        and idx_to_update < len(df) - 1
                    ):
                        st.session_state.selected_df_indices = [idx_to_update + 1]
                    st.rerun()

        with col_info_display:
            if (
                primary_idx_for_nav is not None
                and len(st.session_state.selected_df_indices) == 1
            ):
                selected_row_for_info = df.iloc[primary_idx_for_nav]
                design_id = selected_row_for_info.get("design_id", "")
                run_name = selected_row_for_info.get("run_name", "")
                project_id = selected_row_for_info.get("project_id", "")
                rank_value = selected_row_for_info.get("Rank", primary_idx_for_nav + 1)

                # Get both score types if present
                ipTM_value = selected_row_for_info.get("Average_i_pTM", "N/A")
                pae_value = selected_row_for_info.get("pae_interaction", "N/A")

                ipTM_text = "N/A"
                if isinstance(ipTM_value, (float, int)):
                    ipTM_text = f"{ipTM_value:.3f}"
                elif ipTM_value != "N/A":
                    ipTM_text = str(ipTM_value)

                pae_text = "N/A"
                if isinstance(pae_value, (float, int)):
                    pae_text = f"{pae_value:.3f}"
                elif pae_value != "N/A":
                    pae_text = str(pae_value)

                good_status = selected_row_for_info.get("good", False)
                good_status_emoji = "✅" if good_status else "❌"

                table_data = {
                    "Attribute": [
                        "Rank:",
                        "Design:",
                        "Run:",
                        "Project:",
                        "Average_i_pTM:",
                        "PAE Interaction:",
                        f"Good ({len(df[df['good'] == True])}/{len(df)}):",
                    ],
                    "Value": [
                        f"{rank_value}/{len(df)}",
                        design_id,
                        run_name,
                        project_id,
                        ipTM_text,
                        pae_text,
                        good_status_emoji,
                    ],
                }
                df_info_table = pd.DataFrame(table_data)
                st.markdown(
                    df_info_table.to_html(
                        index=False, header=False, border=0, classes=["no-header-table"]
                    ),
                    unsafe_allow_html=True,
                )
            elif len(st.session_state.selected_df_indices) > 1:
                st.markdown(
                    f"_{len(st.session_state.selected_df_indices)} structures selected_"
                )
            else:
                st.markdown("_No structure selected_")

        with col_next_btn:
            next_disabled = (
                primary_idx_for_nav is None or primary_idx_for_nav >= len(df) - 1
            )
            if st.button(
                "➡️",
                help="Next Structure",
                disabled=next_disabled,
                use_container_width=True,
            ):
                if (
                    primary_idx_for_nav is not None
                    and primary_idx_for_nav < len(df) - 1
                ):
                    st.session_state.selected_df_indices = [primary_idx_for_nav + 1]
                    st.rerun()
    else:
        st.info("No data loaded to display structures.")

    # Structure viewer and auto-next checkbox
    if st.session_state.selected_df_indices and not df.empty:
        pdb_paths_to_view = []
        # Collect PDB paths for all selected structures
        for idx in st.session_state.selected_df_indices:
            if isinstance(idx, int) and 0 <= idx < len(df):
                row = df.iloc[idx]
                pdb_file_path_str = row.get("pdb_file")
                design_id_for_warning = row.get("design_id", f"Row index {idx}")
                if pdb_file_path_str:
                    pdb_path = Path(pdb_file_path_str)
                    if pdb_path.exists():
                        pdb_paths_to_view.append(str(pdb_path))
                    else:
                        st.warning(
                            f"PDB file not found for {design_id_for_warning}: {pdb_path}"
                        )
                else:
                    st.warning(
                        f"Row for {design_id_for_warning} (index {idx}) is missing 'pdb_file' information."
                    )
            else:
                st.warning(f"Invalid index {idx} in selection list.")

        if pdb_paths_to_view:
            molstar_key = "mol_viewer_" + "_".join(
                sorted([Path(p).stem for p in pdb_paths_to_view])
            )
            st_molstar_auto(pdb_paths_to_view, key=molstar_key, height="600px")

            st.checkbox(
                "Auto next on 👍/👎 click",
                key="auto_next_on_thumbs_click",
                value=True,
                help="If checked, clicking thumbs up or down will automatically advance to the next structure.",
            )

        elif st.session_state.selected_df_indices:
            st.info("Selected item(s) have no valid PDB files to display a structure.")

    elif not df.empty:
        st.info(
            "Select structure(s) from the table or a single structure from the scatter plot/nav buttons."
        )


if __name__ == "__main__":
    main()
