#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair>=5.2.0",
#     "matplotlib",
#     "pandas",
#     "seaborn",
#     "streamlit>=1.32.0",
#     # "streamlit-file-browser @ git+https://github.com/pansapiens/streamlit-file-browser@symlinks", # Removed
#     "streamlit-molstar",
# ]
# ///

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import numpy as np # Import numpy
from typing import Optional, List, Any, Tuple
from streamlit_molstar import st_molstar, st_molstar_rcsb, st_molstar_remote
from streamlit_molstar.auto import st_molstar_auto

# Removed: from streamlit_file_browser import st_file_browser

# --- Constants for column names ---
DEFAULT_SORT_COLUMN = "Average_i_pTM"
DEFAULT_SORT_ASCENDING = False
DEFAULT_SCATTER_X_COL = "Average_pLDDT" # Typical pLDDT column
# DEFAULT_SCATTER_Y_COL will be DEFAULT_SORT_COLUMN
DEFAULT_DIST_METRIC = DEFAULT_SORT_COLUMN
# ----------------------------------

def update_good_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and updates the 'good_rank' column based on the existing sort order.
    The input DataFrame is expected to be sorted already.
    """
    # Ensure the 'good' column exists
    if 'good' not in df.columns:
        df['good_rank'] = pd.NA
        return df.astype({'good_rank': 'Int64'})

    # Get the indices of rows marked as 'good'
    good_indices = df[df['good']].index

    # Create a Series with ranks for the 'good' rows
    # The rank is based on the existing order in the DataFrame
    good_rank_series = pd.Series(np.arange(1, len(good_indices) + 1), index=good_indices)

    # Assign this series to the new 'good_rank' column.
    # Rows not in good_indices will get NaN.
    df['good_rank'] = good_rank_series

    # Convert to nullable integer type to support NAs
    df['good_rank'] = df['good_rank'].astype('Int64')

    return df


def is_bindcraft_results(path: Path) -> bool:
    """Check if a directory contains BindCraft results (final_design_stats.csv and Accepted/ folder)."""
    if not path.is_dir():
        return False
    stats_file = path / "final_design_stats.csv"
    accepted_folder = path / "Accepted"
    return stats_file.is_file() and accepted_folder.is_dir()


def find_bindcraft_results(root_path: Path) -> List[Path]:
    """Recursively find all BindCraft result directories within root_path."""
    bindcraft_result_dirs: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        current_dir = Path(dirpath)
        if is_bindcraft_results(current_dir):
            bindcraft_result_dirs.append(current_dir)
            # Optional: If a directory is a results dir, don't recurse further into it?
            # For now, it will, which means nested results dirs would be found.
            # If this is not desired, `dirnames[:] = []` could be used to stop further recursion in this branch.
            dirnames[:] = []  # Stop recursion into this directory
    return bindcraft_result_dirs


def parse_score_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Parse a single .cs file and return a pandas DataFrame."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        score_lines = [line.strip()[6:] for line in lines if line.startswith("SCORE:")]

        if not score_lines:
            st.warning(f"Could not find SCORE: lines in {file_path}")
            return None

        df = pd.DataFrame([x.split() for x in score_lines])
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        df["source_file"] = file_path.name

        return df

    except Exception as e:
        st.error(f"Error processing {file_path}: {str(e)}")
        return None


def load_bindcraft_data(root_search_path: Path) -> Optional[pd.DataFrame]:
    """Load and aggregate BindCraft results from subdirectories, using glob for PDBs."""
    if not root_search_path.exists() or not root_search_path.is_dir():
        st.error(f"Root search path {root_search_path} does not exist or is not a directory.")
        return None

    result_dirs = find_bindcraft_results(root_search_path)

    if not result_dirs:
        st.warning(f"No BindCraft result directories found under {root_search_path}.")
        return None

    st.info(f"Found {len(result_dirs)} BindCraft result director(y/ies). Aggregating data...")

    all_dfs: List[pd.DataFrame] = []
    for res_dir in result_dirs:
        stats_file = res_dir / "final_design_stats.csv"
        accepted_dir = res_dir / "Accepted" # Define accepted_dir for this result directory
        try:
            df_single = pd.read_csv(stats_file)
            # Add the relative path of the result directory
            df_single["results_dir_path"] = str(res_dir.relative_to(root_search_path) if res_dir.is_relative_to(root_search_path) else res_dir)
            
            pdb_file_paths = []
            if "Design" not in df_single.columns:
                st.error(f"'Design' column not found in {stats_file}. Cannot locate PDB files.")
                # Fill with None if 'Design' column is missing for this file
                pdb_file_paths = [None] * len(df_single)
            else:
                for design_name in df_single["Design"]:
                    if pd.isna(design_name): # Handle cases where design_name might be NaN
                        pdb_file_paths.append(None)
                        st.warning(f"Encountered a missing Design name in {stats_file}, cannot search for PDB.")
                        continue

                    # Use a more specific approach to find PDB files for the design
                    # Ensure accepted_dir is valid before searching
                    if not accepted_dir.is_dir():
                        st.warning(f"Accepted directory not found: {accepted_dir}. Cannot search PDB for {design_name}.")
                        pdb_file_paths.append(None)
                        continue
                    
                    # First, try an exact match with the design name
                    exact_match_path = accepted_dir / f"{design_name}.pdb"
                    
                    if exact_match_path.exists():
                        # Found exact match
                        chosen_pdb_path = str(exact_match_path.resolve())
                    else:
                        # Try pattern with underscore separator to avoid matching unrelated designs
                        # e.g., Design "cxcr2_l142_s959184_mpnn1" should match "cxcr2_l142_s959184_mpnn1_model2.pdb"
                        # but not "cxcr2_l142_s959184_mpnn10_model2.pdb"
                        potential_pdbs = sorted(list(accepted_dir.glob(f"{design_name}_*.pdb")))
                        
                        if not potential_pdbs:
                            # As a last resort, try the original glob pattern
                            potential_pdbs = sorted(list(accepted_dir.glob(f"{design_name}*.pdb")))
                        
                        if not potential_pdbs:
                            st.warning(f"No PDB file found for Design '{design_name}' in {accepted_dir}")
                            chosen_pdb_path = None
                        elif len(potential_pdbs) == 1:
                            chosen_pdb_path = str(potential_pdbs[0].resolve())
                        else: # Multiple PDBs found
                            st.warning(f"Multiple PDBs found for Design '{design_name}' in {accepted_dir}: {[p.name for p in potential_pdbs]}. Using the first one: {potential_pdbs[0].name}")
                            chosen_pdb_path = str(potential_pdbs[0].resolve())
                    
                    pdb_file_paths.append(chosen_pdb_path)
            
            df_single["pdb_file"] = pdb_file_paths
            all_dfs.append(df_single)
        except Exception as e:
            st.error(f"Error processing data from result directory {res_dir} (stats file: {stats_file}): {str(e)}")
            # Optionally, you might want to skip this df_single or handle more gracefully

    if not all_dfs:
        st.error("Failed to load or process data from any of the found BindCraft result directories.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Ensure 'pdb_file' column exists even if all were None, convert to string type for consistency
    if 'pdb_file' not in combined_df.columns:
        combined_df['pdb_file'] = pd.Series([None] * len(combined_df), dtype=pd.StringDtype())
    else:
        combined_df['pdb_file'] = combined_df['pdb_file'].astype(pd.StringDtype())
        
    combined_df = combined_df.convert_dtypes(convert_string=False) # Avoid converting our StringDtype pdb_file back

    # Add 'good' column, default to False
    if 'good' not in combined_df.columns:
        # Insert 'good' as the second column (index 1) if possible, or first if df is narrow.
        # The actual display order in the table is controlled later.
        insert_pos = 1 if len(combined_df.columns) > 0 else 0
        combined_df.insert(insert_pos, 'good', False) # Pandas will broadcast False to all rows
    else: # Ensure it's boolean
        combined_df['good'] = combined_df['good'].astype(bool)

    # Always attempt to sort by DEFAULT_SORT_COLUMN (Average_i_pTM) if it exists
    if DEFAULT_SORT_COLUMN in combined_df.columns:
        combined_df = combined_df.sort_values(DEFAULT_SORT_COLUMN, ascending=DEFAULT_SORT_ASCENDING).reset_index(drop=True)
    else:
        st.warning(f"'{DEFAULT_SORT_COLUMN}' column not found in the aggregated data. Cannot sort by it.")
        combined_df = combined_df.reset_index(drop=True)

    # Add Rank column from 1 to N and position it after 'good'.
    combined_df['Rank'] = np.arange(1, len(combined_df) + 1).astype(int)
    cols = combined_df.columns.tolist()
    if 'good' in cols:
        cols.remove('Rank')
        good_index = cols.index('good')
        cols.insert(good_index + 1, 'Rank')
        combined_df = combined_df[cols]

    # Add/update good_rank
    combined_df = update_good_rank(combined_df)

    return combined_df


def plot_distribution(df: pd.DataFrame, metric: str) -> None:
    """Create a distribution plot for a single selected metric."""
    if not metric or metric not in df.columns:
        st.warning(f"Metric '{metric}' not found in DataFrame for distribution plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5)) # Single plot
    sns.kdeplot(data=df, x=metric, ax=ax, fill=True, warn_singular=False)
    sns.rugplot(data=df, x=metric, ax=ax, alpha=0.5)
    ax.set_title(f"Distribution of {metric}")
    plt.tight_layout()
    st.pyplot(fig)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[Any]:
    """Create interactive scatter plot using Altair."""
    # Create a selection condition
    # Assuming 'Design' column will replace 'description'
    selection = alt.selection_point(name="selected_point", fields=["Design", x_col, y_col])

    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(x_col, scale=alt.Scale(zero=False)),
            y=alt.Y(y_col, scale=alt.Scale(zero=False)),
            tooltip=["Design", x_col, y_col, "pdb_file"], # Added pdb_file to tooltip
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

    st.title("BindCraft Results Viewer") # Updated title

    # Command line argument for default path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="Path to results directory", default="."
    )
    args = parser.parse_args()

    # Define the summary file path
    summary_file_path = Path(args.path).resolve() / "bindcraft_summary.tsv"

    # Initialize session state for path if not exists
    if "root_search_path" not in st.session_state: # Renamed for clarity
        st.session_state["root_search_path"] = Path(args.path).resolve()

    # Initialize session state for selected structure index (now a list for multi-select)
    if "selected_df_indices" not in st.session_state: # Changed from selected_df_index
        st.session_state.selected_df_indices = [] # Initialize as an empty list

    # Initialize df in session state if it's not already there to prevent AttributeError
    if "df" not in st.session_state:
        st.session_state.df = None

    # Sidebar for path input and regenerate button
    with st.sidebar:
        st.header("Results Directory")
        st.info(f"Searching in: {st.session_state.root_search_path}")
        st.markdown("To change the search directory, please restart the app with the desired `--path` argument.")
        st.divider()

        if st.button("🔄 Regenerate Summary File", 
                      help="Re-scans the directory, merges existing 'good' ratings, and overwrites bindcraft_summary.tsv"):
            with st.spinner("Regenerating summary file..."):
                # Force a re-scan and merge by temporarily setting df to None in session_state
                # The logic below will then handle the full scan, merge, and save.
                # This ensures that even if a summary file exists, we are truly regenerating it.
                current_summary_df = None
                if summary_file_path.exists():
                    try:
                        current_summary_df = pd.read_csv(summary_file_path, sep="\t")
                    except Exception as e:
                        st.warning(f"Could not read existing summary file for regeneration: {e}")

                fresh_df = load_bindcraft_data(st.session_state.root_search_path)
                if fresh_df is not None:
                    if 'good' not in fresh_df.columns:
                        insert_pos = 1 if len(fresh_df.columns) > 0 else 0
                        fresh_df.insert(insert_pos, 'good', False)
                    else:
                        fresh_df['good'] = fresh_df['good'].fillna(False).astype(bool)

                    if current_summary_df is not None and \
                       "Design" in current_summary_df.columns and \
                       "results_dir_path" in current_summary_df.columns and \
                       "good" in current_summary_df.columns:
                        
                        good_stamps_to_merge = current_summary_df[
                            ["Design", "results_dir_path", "good"]
                        ].drop_duplicates(subset=["Design", "results_dir_path"], keep='first')
                        
                        if 'good' in fresh_df.columns: # Should exist due to above ensure block
                            fresh_df = fresh_df.drop(columns=['good'])
                        
                        fresh_df = pd.merge(
                            fresh_df, 
                            good_stamps_to_merge, 
                            on=["Design", "results_dir_path"], 
                            how='left'
                        )
                        fresh_df['good'] = fresh_df['good'].fillna(False).astype(bool)
                    
                    # With 'good' status merged, we need to recalculate good_rank
                    fresh_df = update_good_rank(fresh_df)

                    st.session_state.df = fresh_df
                    try:
                        st.session_state.df.drop(columns=['good_rank'], errors='ignore').to_csv(summary_file_path, sep="\t", index=False)
                        st.success("Summary file regenerated and saved.")
                    except Exception as e:
                        st.error(f"Error saving regenerated summary file: {e}")
                    
                    st.session_state.selected_df_indices = []
                    if "good" in st.session_state.df.columns:
                        st.session_state.good_values = pd.Series(st.session_state.df["good"], index=st.session_state.df.index, dtype=bool)
                    else:
                        st.session_state.good_values = pd.Series([False] * len(st.session_state.df), index=st.session_state.df.index, dtype=bool)
                    st.rerun()
                else:
                    st.error("Failed to load data during regeneration scan.")

    # Attempt to load data: either from existing session_state.df (if already loaded/regenerated),
    # or from summary file, or by scanning if summary doesn't exist.
    if st.session_state.df is None: # Only proceed if df is not already set (e.g., by regeneration button)
        if summary_file_path.exists():
            try:
                st.session_state.df = pd.read_csv(summary_file_path, sep="\t")
                # Always remove Rank if it exists, to re-calculate it based on current sort.
                if 'Rank' in st.session_state.df.columns:
                    st.session_state.df = st.session_state.df.drop(columns=['Rank'])

                if "good" in st.session_state.df.columns:
                    st.session_state.df["good"] = st.session_state.df["good"].fillna(False).astype(bool)
                else:
                    st.session_state.df["good"] = False # Add good column if missing, pandas will broadcast
                
                if 'pdb_file' in st.session_state.df.columns:
                    st.session_state.df['pdb_file'] = st.session_state.df['pdb_file'].astype(pd.StringDtype())
                if 'results_dir_path' in st.session_state.df.columns:
                    st.session_state.df['results_dir_path'] = st.session_state.df['results_dir_path'].astype(pd.StringDtype())
                
                st.session_state.df = st.session_state.df.convert_dtypes(convert_string=False)
                if DEFAULT_SORT_COLUMN in st.session_state.df.columns:
                    st.session_state.df = st.session_state.df.sort_values(DEFAULT_SORT_COLUMN, ascending=DEFAULT_SORT_ASCENDING).reset_index(drop=True)
                else:
                    st.session_state.df = st.session_state.df.reset_index(drop=True)

                # Add Rank column from 1 to N and position it after 'good'.
                st.session_state.df['Rank'] = np.arange(1, len(st.session_state.df) + 1).astype(int)
                cols = st.session_state.df.columns.tolist()
                if 'good' in cols:
                    cols.remove('Rank')
                    good_index = cols.index('good')
                    cols.insert(good_index + 1, 'Rank')
                    st.session_state.df = st.session_state.df[cols]

                # Update good_rank after loading and sorting
                st.session_state.df = update_good_rank(st.session_state.df)

                st.info(f"Loaded data from summary file: {summary_file_path}")
            except Exception as e:
                st.error(f"Error loading summary file {summary_file_path}: {e}. Will perform initial scan.")
                st.session_state.df = None # Explicitly set to None to trigger scan below

        # If df is still None (summary didn't exist or failed to load), then perform initial scan and create summary.
        if st.session_state.df is None:
            st.session_state.df = load_bindcraft_data(st.session_state.root_search_path)
            if st.session_state.df is not None:
                # Ensure 'good' column exists before first save
                if 'good' not in st.session_state.df.columns:
                    insert_pos = 1 if len(st.session_state.df.columns) > 0 else 0
                    st.session_state.df.insert(insert_pos, 'good', False)
                else:
                    st.session_state.df['good'] = st.session_state.df['good'].fillna(False).astype(bool)
                try:
                    st.session_state.df.drop(columns=['good_rank'], errors='ignore').to_csv(summary_file_path, sep="\t", index=False)
                    st.success(f"Initial summary file created: {summary_file_path}")
                except Exception as e:
                    st.error(f"Could not create initial summary file {summary_file_path}: {e}")
            # If df is STILL None after scan, then there's nothing to load or create.

    df = st.session_state.df

    if df is None:
        st.warning("No BindCraft data loaded. Please ensure the specified path contains valid BindCraft results or a summary file.")
        return

    # Initialize or sync 'good_values' session state with the current df's 'good' column
    # This should happen regardless of how df was loaded (summary or scan)
    if "good" in df.columns:
        st.session_state.good_values = pd.Series(df["good"], index=df.index, dtype=bool)
    else: # Should not happen if load_bindcraft_data or summary loading works
        st.session_state.good_values = pd.Series([False] * len(df), index=df.index, dtype=bool)
        df["good"] = st.session_state.good_values # Add good column to df if it's missing

    # Ensure df also reflects the most current 'good' values from session_state
    # (This might be redundant if just initialized, but good for consistency)
    df["good"] = st.session_state.good_values.reindex(df.index, fill_value=False).astype(bool)
    
    # Get numeric columns for plotting
    # Use np.number to select all numeric types, which is more robust for linters.
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Initialize session state for column selection if not exists
    if "selected_cols" not in st.session_state:
        # Prioritize these columns, then add the rest
        # Updated to use more likely column names based on user-provided CSV header
        default_table_display_order = [
            "good", # Make this the first *data* column displayed by default
            "Rank",
            "good_rank",
            DEFAULT_SORT_COLUMN,
            DEFAULT_SCATTER_X_COL,
            "Average_Binder_RMSD", # More specific RMSD column
            "Design", 
            "pdb_file", 
            "results_dir_path"
        ]
        
        actual_default_table_cols = [col for col in default_table_display_order if col in df.columns]
        remaining_table_cols = [col for col in df.columns if col not in actual_default_table_cols]
        st.session_state.selected_cols = actual_default_table_cols + remaining_table_cols

    # Initialize session state for scatter plot axes
    default_x_col = None
    if numeric_cols:
        # New defaults for BindCraft
        if "mean_plddt" in numeric_cols:
            default_x_col = "mean_plddt"
        elif "plddt" in numeric_cols: # common alternative
            default_x_col = "plddt"
        else:
            default_x_col = numeric_cols[0]
    
    default_y_col = None
    if numeric_cols:
        # New defaults for BindCraft
        if "Average_i_pTM" in numeric_cols:
            default_y_col = "Average_i_pTM"
        elif len(numeric_cols) > 1:
            default_y_col = numeric_cols[1]
        elif numeric_cols: # only one numeric col
            default_y_col = numeric_cols[0]

    if "scatter_x_col" not in st.session_state:
        st.session_state.scatter_x_col = default_x_col
    if "scatter_y_col" not in st.session_state:
        st.session_state.scatter_y_col = default_y_col
    
    # Initialize session state for auto_next checkbox
    if "auto_next_on_thumbs_click" not in st.session_state:
        st.session_state.auto_next_on_thumbs_click = False

    # Initialize session state for "Show good only" filter
    if 'show_good_only' not in st.session_state:
        st.session_state.show_good_only = False

    # Prepare DataFrame for the data_editor: add a selection column
    # This df_for_editor will be updated based on st.session_state.selected_df_indices
    df_for_editor = df.copy()
    selection_col_name = "✔️ Select"
    df_for_editor[selection_col_name] = False # Initialize all to False
    if st.session_state.selected_df_indices: # If the list is not empty
        for idx in st.session_state.selected_df_indices:
            if 0 <= idx < len(df_for_editor):
                df_for_editor.loc[idx, selection_col_name] = True
    
    # Update 'good' column in df_for_editor from the main df (which is synced with session_state)
    if 'good' in df.columns and 'good' in df_for_editor.columns:
        df_for_editor['good'] = df['good']

    # Create a view of the dataframe that can be filtered for display
    df_display = df_for_editor
    if st.session_state.show_good_only:
        if 'good' in df_display.columns:
            df_display = df_display[df_display['good'] == True]

    # Create tabs for different views
    table_tab, scatter_tab, dist_tab = st.tabs(
        ["Data Table", "Scatter Plot", "Distributions"]
    )

    # These will capture the direct output of widgets for selection processing
    scatter_event_result = None
    edited_df_from_editor = None # To store the result from st.data_editor

    with table_tab:
        if st.session_state.show_good_only:
            st.info(f"Displaying {len(df_display)} of {len(df)} total rows (good only).")
        else:
            st.info(f"Total rows: {len(df_for_editor)}")
        
        editor_column_config = {
            selection_col_name: st.column_config.CheckboxColumn(
                "View",
                help="Select a row to view structure",
                default=False,
            ),
            "good": st.column_config.CheckboxColumn( # Added configuration for 'good' column
                "Good",
                help="Mark as a good design",
                default=False,
            ),
            **{
                col: st.column_config.NumberColumn(col, format="%.3f")
                for col in numeric_cols if col in df_for_editor.columns
            },
        }

        # Override Rank column to ensure integer display
        if 'Rank' in df_for_editor.columns:
            editor_column_config['Rank'] = st.column_config.NumberColumn("Rank", format="%d")

        # Override good_rank column to ensure integer display and add help text
        if 'good_rank' in df_for_editor.columns:
            editor_column_config['good_rank'] = st.column_config.NumberColumn(
                "Good Rank",
                format="%d",
                help="Rank among only 'Good' designs"
            )

        # Put selection column first.
        columns_to_display_in_editor = [selection_col_name] + [col for col in st.session_state.selected_cols if col in df_for_editor.columns and col != selection_col_name]

        # Filter df_for_editor to only include columns that will be displayed
        df_display_subset = df_display[columns_to_display_in_editor]

        edited_df_from_editor = st.data_editor(
            df_display_subset, # Pass the subset
            hide_index=True,
            use_container_width=True,
            column_config=editor_column_config, # Config should now align with df_display_subset
            disabled=[col for col in df.columns if col not in [selection_col_name, "good"]], # Allow 'good' to be edited
            key="data_editor_table",
        )

        # Table controls below the dataframe
        st.caption("Table Controls")
        # Put column selector in expander
        with st.expander("Column Display Settings"):
            st.checkbox(
                "Show good only", 
                key="show_good_only",
                help="Filter the table to show only designs marked as 'Good'."
            )
            all_cols = df.columns.tolist()
            # Show ipTM first in options
            if "Average_i_pTM" in all_cols:
                all_cols.remove("Average_i_pTM")
                all_cols.insert(0, "Average_i_pTM")
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
                if st.session_state.scatter_x_col and st.session_state.scatter_x_col in numeric_cols:
                    current_x_idx = numeric_cols.index(st.session_state.scatter_x_col)
                selected_x = st.selectbox(
                    "X-axis", options=numeric_cols, index=current_x_idx, key="x_axis_selector"
                )
                if selected_x: 
                    st.session_state.scatter_x_col = selected_x
            else:
                st.warning("No numeric columns available for X-axis.")
        with col2:
            if numeric_cols:
                current_y_idx = 0 # Default to first option
                # Try to find a sensible default for Y, different from X if possible
                if st.session_state.scatter_y_col and st.session_state.scatter_y_col in numeric_cols:
                    current_y_idx = numeric_cols.index(st.session_state.scatter_y_col)
                elif len(numeric_cols) > 1 and numeric_cols[0] != st.session_state.get("scatter_x_col"):
                    current_y_idx = 0
                elif len(numeric_cols) > 1:
                    current_y_idx = 1
                # else: current_y_idx remains 0 (only one numeric col, or failed to find a different one)

                selected_y = st.selectbox(
                    "Y-axis", options=numeric_cols, index=current_y_idx, key="y_axis_selector"
                )
                if selected_y:
                    st.session_state.scatter_y_col = selected_y
            else:
                st.warning("No numeric columns available for Y-axis.")
        
        if st.session_state.scatter_x_col and st.session_state.scatter_y_col and st.session_state.scatter_x_col in df.columns and st.session_state.scatter_y_col in df.columns:
            scatter_event_result = plot_scatter(df, st.session_state.scatter_x_col, st.session_state.scatter_y_col)
        elif numeric_cols: 
            st.warning("Please select valid X and Y axes for the scatter plot.")

    with dist_tab:
        if numeric_cols:
            current_dist_metric_idx = 0
            if st.session_state.scatter_y_col and st.session_state.scatter_y_col in numeric_cols:
                current_dist_metric_idx = numeric_cols.index(st.session_state.scatter_y_col)
            elif DEFAULT_DIST_METRIC in numeric_cols:
                current_dist_metric_idx = numeric_cols.index(DEFAULT_DIST_METRIC)
            # If numeric_cols is not empty, current_dist_metric_idx is valid.
            
            selected_dist_metric = st.selectbox(
                "Select metric for distribution plot",
                options=numeric_cols,
                index=current_dist_metric_idx,
                key="dist_metric_selector"
            )
            if selected_dist_metric:
                st.session_state.scatter_y_col = selected_dist_metric
                if st.session_state.scatter_y_col in df.columns:
                    plot_distribution(df, st.session_state.scatter_y_col)
                else:
                    st.warning(f"Selected metric '{st.session_state.scatter_y_col}' not found in DataFrame.")
        else:
            st.info("No numeric columns available for distribution plots.")

    # Process selections to update st.session_state.selected_df_indices
    new_selection_made_in_this_run = False

    # 1. Process selection from data_editor
    if edited_df_from_editor is not None:
        # data_editor returns a DataFrame with the current state of its display, including our selection column.
        # We need to map its potentially filtered/reordered view back to original `df` indices if it was modified.
        # However, since we pass df_for_editor and it has the same index as df, and num_rows="fixed" (default),
        # the indices in edited_df_from_editor (if it has an index) should align with df_for_editor.
        # Let's assume edited_df_from_editor refers to the state of the columns we passed to it.
        
        # Process 'good' column changes from the editor
        if "good" in edited_df_from_editor.columns and "good" in df.columns:
            # Iterate through the edited_df_from_editor to find changes in the 'good' column
            # The index of edited_df_from_editor should align with df if no sorting/filtering was applied
            # by the data_editor itself on the passed data, which is the case here.
            changed_good_series = edited_df_from_editor["good"]
            for original_df_idx in changed_good_series.index: # original_df_idx here is the index from the DataFrame
                if original_df_idx in df.index: # Ensure index is valid for original df
                    new_good_value = changed_good_series.loc[original_df_idx]
                    # Ensure new_good_value is a boolean
                    new_good_value_bool = bool(new_good_value)

                    # Check against the session state directly or the df synced with session state
                    current_good_value_in_state = bool(st.session_state.good_values.loc[original_df_idx])
                    
                    if current_good_value_in_state != new_good_value_bool:
                        df.loc[original_df_idx, "good"] = new_good_value_bool
                        st.session_state.good_values.loc[original_df_idx] = new_good_value_bool
                        new_selection_made_in_this_run = True # Mark that a change was made

                        # A 'good' value changed, so we need to re-calculate the good_rank
                        df = update_good_rank(df)
                        st.session_state.df = df # also update session state

                        # Save DataFrame to TSV after editor change
                        try:
                            df.drop(columns=['good_rank'], errors='ignore').to_csv(summary_file_path, sep="\t", index=False)
                        except Exception as e:
                            st.error(f"Error saving summary file: {e}")

        # Check if the selection column exists in the output of data_editor
        if selection_col_name in edited_df_from_editor.columns:
            selected_in_editor_series = edited_df_from_editor[selection_col_name]
            
            # Find rows that are checked True in the editor
            currently_checked_indices = selected_in_editor_series[selected_in_editor_series == True].index.tolist()

            if currently_checked_indices:
                # User has at least one box checked in the editor
                # Update session state to match all currently checked boxes
                if set(st.session_state.selected_df_indices) != set(currently_checked_indices):
                    st.session_state.selected_df_indices = sorted(currently_checked_indices)
                    new_selection_made_in_this_run = True

            elif not currently_checked_indices and st.session_state.selected_df_indices:
                # No boxes are checked in the editor, but there was a selection.
                # This means the user manually unchecked all active rows.
                st.session_state.selected_df_indices = []
                new_selection_made_in_this_run = True

    # 2. Process selection from scatter plot
    if scatter_event_result: 
        selection_data = scatter_event_result.get("selection")
        if selection_data and "selected_point" in selection_data:
            selected_points = selection_data["selected_point"]
            if selected_points and len(selected_points) > 0:
                # Use "Design" field for matching
                selected_design = selected_points[0].get("Design") 
                if selected_design:
                    matching_rows = df[df["Design"] == selected_design] # Changed "description" to "Design"
                    if not matching_rows.empty:
                        new_idx_from_scatter = int(matching_rows.index[0])
                        # Scatter click replaces current selection
                        if st.session_state.selected_df_indices != [new_idx_from_scatter]:
                            st.session_state.selected_df_indices = [new_idx_from_scatter]
                            new_selection_made_in_this_run = True 
                else: # Fallback if "Design" is not in point data (should be, based on tooltip and selection fields)
                    point_data = selected_points[0]
                    if st.session_state.scatter_x_col in point_data and st.session_state.scatter_y_col in point_data:
                        mask = (df[st.session_state.scatter_x_col] == point_data[st.session_state.scatter_x_col]) & \
                               (df[st.session_state.scatter_y_col] == point_data[st.session_state.scatter_y_col])
                        if mask.any():
                            # Ensure the index from the original df is used
                            original_df_index = df[mask].index[0] 
                            new_idx_from_scatter = int(original_df_index)
                             # Scatter click replaces current selection
                            if st.session_state.selected_df_indices != [new_idx_from_scatter]:
                                st.session_state.selected_df_indices = [new_idx_from_scatter]
                                new_selection_made_in_this_run = True
    
    # If any selection change happened (editor or scatter), rerun to ensure UI consistency
    # This is especially for data_editor to reflect the single selected checkbox.
    if new_selection_made_in_this_run:
        st.rerun()


    # Structure Viewer Section
    st.header("Structure Viewer")

    # Define col_info here to ensure it's in scope if df is not empty
    # col_prev, col_info, col_next = (None, None, None) # Old column definition
    
    # New layout for structure controls
    if not df.empty:
        # Columns for: Prev, Thumbs Up, Thumbs Down, Info, Next
        col_prev_btn, col_thumb_up, col_thumb_down, col_info_display, col_next_btn = st.columns([1.5, 1, 1, 6, 1.5])

        primary_idx_for_nav = st.session_state.selected_df_indices[0] if st.session_state.selected_df_indices else None

        with col_prev_btn:
            prev_disabled = primary_idx_for_nav is None or primary_idx_for_nav == 0
            if st.button("⬅️", help="Previous Structure", disabled=prev_disabled, use_container_width=True):
                if primary_idx_for_nav is not None and primary_idx_for_nav > 0:
                    st.session_state.selected_df_indices = [primary_idx_for_nav - 1]
                    st.rerun()
        
        with col_thumb_up:
            thumb_up_disabled = not (primary_idx_for_nav is not None and len(st.session_state.selected_df_indices) == 1)
            if st.button("👍", help="Mark as Good", disabled=thumb_up_disabled, use_container_width=True):
                if "good" in df.columns and primary_idx_for_nav is not None and primary_idx_for_nav in df.index:
                    idx_to_update = int(primary_idx_for_nav)
                    df.loc[idx_to_update, "good"] = True
                    if idx_to_update in st.session_state.good_values.index:
                        st.session_state.good_values.at[idx_to_update] = True
                    
                    # A 'good' value changed, so we need to re-calculate the good_rank
                    df = update_good_rank(df)
                    st.session_state.df = df # also update session state

                    try:
                        df.drop(columns=['good_rank'], errors='ignore').to_csv(summary_file_path, sep="\t", index=False)
                    except Exception as e:
                        st.error(f"Error saving summary file: {e}")
                    if st.session_state.auto_next_on_thumbs_click and idx_to_update < len(df) - 1:
                        st.session_state.selected_df_indices = [idx_to_update + 1]
                    st.rerun()

        with col_thumb_down:
            thumb_down_disabled = not (primary_idx_for_nav is not None and len(st.session_state.selected_df_indices) == 1)
            if st.button("👎", help="Mark as Not Good", disabled=thumb_down_disabled, use_container_width=True):
                if "good" in df.columns and primary_idx_for_nav is not None and primary_idx_for_nav in df.index:
                    idx_to_update = int(primary_idx_for_nav)
                    df.loc[idx_to_update, "good"] = False
                    if idx_to_update in st.session_state.good_values.index:
                        st.session_state.good_values.at[idx_to_update] = False

                    # A 'good' value changed, so we need to re-calculate the good_rank
                    df = update_good_rank(df)
                    st.session_state.df = df # also update session state

                    try:
                        df.drop(columns=['good_rank'], errors='ignore').to_csv(summary_file_path, sep="\t", index=False)
                    except Exception as e:
                        st.error(f"Error saving summary file: {e}")
                    if st.session_state.auto_next_on_thumbs_click and idx_to_update < len(df) - 1:
                        st.session_state.selected_df_indices = [idx_to_update + 1]
                    st.rerun()
        
        with col_info_display:
            if primary_idx_for_nav is not None and len(st.session_state.selected_df_indices) == 1:
                selected_row_for_info = df.iloc[primary_idx_for_nav]
                design_name = selected_row_for_info.get("Design", "") 
                rank_value = selected_row_for_info.get("Rank", primary_idx_for_nav + 1)
                metric_value_for_display = selected_row_for_info.get(DEFAULT_SORT_COLUMN, 'N/A') 
                metric_text = "N/A"
                try:
                    metric_float = float(metric_value_for_display)
                    metric_text = f"{metric_float:.3f}"
                except (ValueError, TypeError):
                    metric_text = str(metric_value_for_display) if pd.notna(metric_value_for_display) else "N/A"
                results_dir_path_text = selected_row_for_info.get("results_dir_path", "N/A")
                good_status = selected_row_for_info.get("good", False)
                good_status_emoji = "✅" if good_status else "❌"
                
                table_data = {
                    "Attribute": ["Rank:", "Model:", "Run:", f"{DEFAULT_SORT_COLUMN}", f"Good ({len(df[df['good'] == True])}/{len(df)}):"],
                    "Value": [f"{rank_value}/{len(df)}", design_name, results_dir_path_text, metric_text, good_status_emoji]
                }
                df_info_table = pd.DataFrame(table_data)
                st.markdown(df_info_table.to_html(index=False, header=False, border=0, classes=["no-header-table"]), unsafe_allow_html=True)
            elif len(st.session_state.selected_df_indices) > 1:
                st.markdown(f"_{len(st.session_state.selected_df_indices)} structures selected_" )
            else:
                st.markdown("_No structure selected_")

        with col_next_btn:
            next_disabled = primary_idx_for_nav is None or primary_idx_for_nav >= len(df) - 1
            if st.button("➡️", help="Next Structure", disabled=next_disabled, use_container_width=True):
                if primary_idx_for_nav is not None and primary_idx_for_nav < len(df) - 1:
                    st.session_state.selected_df_indices = [primary_idx_for_nav + 1]
                    st.rerun()
    else:
        st.info("No data loaded to display structures.")

    # "Auto next" checkbox below the structure viewer, if structures are being viewed
    # This logic for checkbox placement is now dependent on pdb_paths_to_view having content.

    if st.session_state.selected_df_indices and not df.empty:
        pdb_paths_to_view = []
        # The info display logic previously here is now moved into col_info_display above.
        # We still need to collect PDB paths for the viewer itself.
        for idx in st.session_state.selected_df_indices:
            if isinstance(idx, int) and 0 <= idx < len(df):
                row = df.iloc[idx]
                pdb_file_path_str = row.get("pdb_file")
                design_name_for_warning = row.get("Design", f"Row index {idx}")
                if pdb_file_path_str:
                    pdb_path = Path(pdb_file_path_str)
                    if pdb_path.exists():
                        pdb_paths_to_view.append(str(pdb_path))
                    else:
                        st.warning(f"PDB file not found for {design_name_for_warning}: {pdb_path}")
                else:
                    st.warning(f"Row for {design_name_for_warning} (index {idx}) is missing 'pdb_file' information.")
            else:
                st.warning(f"Invalid index {idx} in selection list.")
        
        if pdb_paths_to_view:
            molstar_key = "mol_viewer_" + "_".join(sorted([Path(p).stem for p in pdb_paths_to_view]))
            st_molstar_auto(pdb_paths_to_view, key=molstar_key, height="600px")
            
            st.checkbox("Auto next on 👍/👎 click", key="auto_next_on_thumbs_click", value=True,
                        help="If checked, clicking thumbs up or down will automatically advance to the next structure.")

        elif st.session_state.selected_df_indices: 
            st.info("Selected item(s) have no valid PDB files to display a structure.")

    elif not df.empty:
        st.info("Select structure(s) from the table or a single structure from the scatter plot/nav buttons.")


if __name__ == "__main__":
    main()

    ###
    # We cannot run this way - custom components fail to load
    ###

    # import os
    # import sys

    # # Disable Streamlit telemetry
    # os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    # os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

    # from streamlit.web import cli as stcli
    # from streamlit import runtime

    # if runtime.exists():
    #     main()
    # else:
    #     # Pass through all arguments including --path
    #     sys.argv = ["streamlit", "run", sys.argv[0]] + ["--"] + sys.argv[1:]
    #     sys.exit(stcli.main()) 