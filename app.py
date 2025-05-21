#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair>=5.2.0",
#     "matplotlib",
#     "pandas",
#     "seaborn",
#     "streamlit>=1.32.0",
#     "streamlit-file-browser @ git+https://github.com/pansapiens/streamlit-file-browser@symlinks",
#     "streamlit-molstar",
# ]
# ///

import sys
from pathlib import Path
import argparse
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from typing import Optional, List, Any
from streamlit_molstar import st_molstar, st_molstar_rcsb, st_molstar_remote
from streamlit_molstar.auto import st_molstar_auto

from streamlit_file_browser import st_file_browser


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


def load_data(path: Path) -> Optional[pd.DataFrame]:
    """Load and combine all .cs files from the given directory."""
    if not path.exists():
        st.error(f"Path {path} does not exist")
        return None

    # First check if combined scores file exists
    combined_file = path / "combined_scores.tsv"
    if combined_file.exists():
        st.info(f"Found combined scores file: {combined_file}")
        try:
            combined_df = pd.read_csv(combined_file, sep="\t")
            # Automatically convert to best possible dtypes
            combined_df = combined_df.convert_dtypes()
            return combined_df
        except Exception as e:
            st.warning(
                f"Error loading combined scores file: {str(e)}. Falling back to individual .cs files."
            )

    else:
        # If no combined file was found, fall back to parsing individual .cs files
        # new location in af2_initial_guess/scores/
        cs_files = list((path / "af2_initial_guess" / "scores").glob("*.cs"))
        # DEPRECATED old location
        cs_files.extend(list((path / "af2_initial_guess").glob("*.cs")))
        if not cs_files:
            st.warning(f"No .cs files found in {path}")
        else:
            st.info(f"Found {len(cs_files)} .cs files")
            dfs = []
            for file_path in cs_files:
                df = parse_score_file(file_path)
                if df is not None:
                    dfs.append(df)

            combined_df = pd.concat(dfs, ignore_index=True)
            # Automatically convert to best possible dtypes
            combined_df = combined_df.convert_dtypes()
            return combined_df


def plot_distribution(df: pd.DataFrame, metrics: List[str]) -> None:
    """Create distribution plots for selected metrics."""
    if not metrics:
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 2.5 * n_metrics), squeeze=False) # squeeze=False ensures axes is always 2D

    for i, metric in enumerate(metrics):
        ax = axes[i, 0] # Access the Axes object from the 2D array
        sns.kdeplot(data=df, x=metric, ax=ax)
        sns.rugplot(data=df, x=metric, ax=ax, alpha=0.2)
        ax.set_title(f"Distribution of {metric}")

    plt.tight_layout()
    st.pyplot(fig)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[Any]:
    """Create interactive scatter plot using Altair."""
    # Create a selection condition
    selection = alt.selection_point(name="selected_point", fields=["description", x_col, y_col])

    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(x_col, scale=alt.Scale(zero=False)),
            y=alt.Y(y_col, scale=alt.Scale(zero=False)),
            tooltip=["description", x_col, y_col],
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

    st.title("nf-binder-design Results Viewer")

    # Command line argument for default path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, help="Path to results directory", default="."
    )
    args = parser.parse_args()

    # Initialize session state for path if not exists
    if "current_path" not in st.session_state:
        st.session_state["current_path"] = Path(args.path)

    # Initialize session state for selected structure index (now a list for multi-select)
    if "selected_df_indices" not in st.session_state: # Changed from selected_df_index
        st.session_state.selected_df_indices = [] # Initialize as an empty list

    # Sidebar for path input
    with st.sidebar:
        st.header("Select run folder")
        file_select_event = st_file_browser(
            st.session_state.current_path,
            show_preview=False,
            show_download_file=True,
            key="run_selector",
        )

        # Update path if a new folder is selected
        if file_select_event and file_select_event["type"] == "SELECT_FOLDER":
            selected_path = file_select_event.get("target", {}).get("path")
            st.write(selected_path)
            if selected_path:
                # Create absolute path from the selected path
                st.session_state["current_path"] = Path(
                    args.path, selected_path
                ).resolve()

    st.info("Run path: " + str(st.session_state["current_path"]))

    # Use the current path from session state
    df = load_data(st.session_state.current_path)
    if df is None:
        return

    # Get numeric columns for plotting
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Initialize session state for column selection if not exists
    if "selected_cols" not in st.session_state:
        cols = df.columns.tolist()
        # Put pae_interaction first if present
        if "pae_interaction" in cols:
            cols.remove("pae_interaction")
            cols.insert(0, "pae_interaction")
        st.session_state.selected_cols = cols

    # Initialize session state for scatter plot axes
    default_x_col = None
    if numeric_cols:
        default_x_col = "plddt_binder" if "plddt_binder" in numeric_cols else numeric_cols[0]
    
    default_y_col = None
    if numeric_cols:
        if "pae_interaction" in numeric_cols:
            default_y_col = "pae_interaction"
        elif len(numeric_cols) > 1:
            default_y_col = numeric_cols[1]
        elif numeric_cols: # only one numeric col
            default_y_col = numeric_cols[0]

    if "scatter_x_col" not in st.session_state:
        st.session_state.scatter_x_col = default_x_col
    if "scatter_y_col" not in st.session_state:
        st.session_state.scatter_y_col = default_y_col
    
    # Sort dataframe by pae_interaction ascending
    if "pae_interaction" in df.columns:
        df = df.sort_values("pae_interaction", ascending=True).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True) # Ensure 0-based index

    # Prepare DataFrame for the data_editor: add a selection column
    # This df_for_editor will be updated based on st.session_state.selected_df_indices
    df_for_editor = df.copy()
    selection_col_name = "✔️ Select"
    df_for_editor[selection_col_name] = False # Initialize all to False
    if st.session_state.selected_df_indices: # If the list is not empty
        for idx in st.session_state.selected_df_indices:
            if 0 <= idx < len(df_for_editor):
                df_for_editor.loc[idx, selection_col_name] = True
            # else: Malformed index in list, could add a warning or clean-up step


    # Create tabs for different views
    table_tab, scatter_tab, dist_tab = st.tabs(
        ["Data Table", "Scatter Plot", "Distributions"]
    )

    # These will capture the direct output of widgets for selection processing
    scatter_event_result = None
    edited_df_from_editor = None # To store the result from st.data_editor

    with table_tab:
        st.info(f"Total rows: {len(df_for_editor)}")
        
        editor_column_config = {
            selection_col_name: st.column_config.CheckboxColumn(
                "View",
                help="Select a row to view structure",
                default=False,
            ),
            **{
                col: st.column_config.NumberColumn(col, format="%.3f")
                for col in numeric_cols if col in df_for_editor.columns
            },
        }

        # Ensure that the list of columns passed to data_editor only contains what's intended.
        # Put selection column first.
        columns_to_display_in_editor = [selection_col_name] + [col for col in st.session_state.selected_cols if col != selection_col_name]
        
        # Filter df_for_editor to only include columns that will be displayed
        # This ensures that column_config only needs to be concerned with these displayed columns.
        df_display_subset = df_for_editor[columns_to_display_in_editor]

        edited_df_from_editor = st.data_editor(
            df_display_subset, # Pass the subset
            hide_index=True,
            use_container_width=True,
            column_config=editor_column_config, # Config should now align with df_display_subset
            disabled=df.columns.tolist(), # Still disable original data columns from df
            key="data_editor_table",
        )

        # Table controls below the dataframe
        st.caption("Table Controls")
        # Put column selector in expander
        with st.expander("Column Display Settings"):
            all_cols = df.columns.tolist()
            # Show pae_interaction first in options
            if "pae_interaction" in all_cols:
                all_cols.remove("pae_interaction")
                all_cols.insert(0, "pae_interaction")
            st.session_state.selected_cols = st.multiselect(
                "Select columns to display",
                options=all_cols,
                default=st.session_state.selected_cols,
            )

    with scatter_tab:
        col1, col2 = st.columns(2)
        with col1:
            selected_x = st.selectbox(
                "X-axis", 
                options=numeric_cols, 
                index=numeric_cols.index(st.session_state.scatter_x_col) if st.session_state.scatter_x_col and st.session_state.scatter_x_col in numeric_cols else 0,
                key="x_axis_selector"
            )
            if selected_x: # Update session state if a valid selection is made
                 st.session_state.scatter_x_col = selected_x
        with col2:
            selected_y = st.selectbox(
                "Y-axis",
                options=numeric_cols,
                index=numeric_cols.index(st.session_state.scatter_y_col) if st.session_state.scatter_y_col and st.session_state.scatter_y_col in numeric_cols else (1 if len(numeric_cols) > 1 else 0),
                key="y_axis_selector"
            )
            if selected_y: # Update session state if a valid selection is made
                st.session_state.scatter_y_col = selected_y
        
        if st.session_state.scatter_x_col and st.session_state.scatter_y_col:
            scatter_event_result = plot_scatter(df, st.session_state.scatter_x_col, st.session_state.scatter_y_col)
        else:
            st.warning("Please select X and Y axes for the scatter plot.")

    with dist_tab:
        selected_metrics = st.multiselect(
            "Select metrics for distribution plots",
            options=numeric_cols,
            default=["pae_interaction", "pae_binder", "plddt_binder", "rg"],
        )
        if selected_metrics:
            plot_distribution(df, selected_metrics)

    # Process selections to update st.session_state.selected_df_indices
    new_selection_made_in_this_run = False

    # 1. Process selection from data_editor
    if edited_df_from_editor is not None:
        # data_editor returns a DataFrame with the current state of its display, including our selection column.
        # We need to map its potentially filtered/reordered view back to original `df` indices if it was modified.
        # However, since we pass df_for_editor and it has the same index as df, and num_rows="fixed" (default),
        # the indices in edited_df_from_editor (if it has an index) should align with df_for_editor.
        # Let's assume edited_df_from_editor refers to the state of the columns we passed to it.
        
        # Check if the selection column exists in the output of data_editor
        # It should, as we added it to df_for_editor and configured it.
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
                selected_description = selected_points[0].get("description")
                if selected_description:
                    matching_rows = df[df["description"] == selected_description]
                    if not matching_rows.empty:
                        new_idx_from_scatter = int(matching_rows.index[0])
                        # Scatter click replaces current selection
                        if st.session_state.selected_df_indices != [new_idx_from_scatter]:
                            st.session_state.selected_df_indices = [new_idx_from_scatter]
                            new_selection_made_in_this_run = True 
                else:
                    point_data = selected_points[0]
                    if st.session_state.scatter_x_col in point_data and st.session_state.scatter_y_col in point_data:
                        mask = (df[st.session_state.scatter_x_col] == point_data[st.session_state.scatter_x_col]) & \
                               (df[st.session_state.scatter_y_col] == point_data[st.session_state.scatter_y_col])
                        if mask.any():
                            new_idx_from_scatter = int(df[mask].index[0])
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
    col_prev, col_info, col_next = (None, None, None) 
    if not df.empty:
        col_prev, col_info, col_next = st.columns([2, 8, 2])

        # Determine primary index for Next/Previous logic (e.g., first selected or last if any)
        primary_idx_for_nav = st.session_state.selected_df_indices[0] if st.session_state.selected_df_indices else None

        with col_prev:
            prev_disabled = primary_idx_for_nav is None or primary_idx_for_nav == 0
            if st.button("⬅️ Previous", disabled=prev_disabled, use_container_width=True):
                if primary_idx_for_nav is not None and primary_idx_for_nav > 0:
                    st.session_state.selected_df_indices = [primary_idx_for_nav - 1]
                    st.rerun() 

        with col_next:
            next_disabled = primary_idx_for_nav is None or primary_idx_for_nav >= len(df) - 1
            if st.button("Next ➡️", disabled=next_disabled, use_container_width=True):
                if primary_idx_for_nav is not None and primary_idx_for_nav < len(df) - 1:
                    st.session_state.selected_df_indices = [primary_idx_for_nav + 1]
                    st.rerun() 
    else:
        st.info("No data loaded to display structures.")


    if st.session_state.selected_df_indices and not df.empty:
        pdb_paths_to_view = []
        first_selected_idx = st.session_state.selected_df_indices[0]

        # Display info for the first selected structure
        if isinstance(first_selected_idx, int) and 0 <= first_selected_idx < len(df):
            selected_row_for_info = df.iloc[first_selected_idx]
            description = selected_row_for_info.get("description", "")
            pae_value = selected_row_for_info.get('pae_interaction', 'N/A')
            pae_text = "N/A"
            if isinstance(pae_value, (float, int)):
                pae_text = f"{pae_value:.3f}"
            elif pae_value != 'N/A':
                pae_text = str(pae_value)

            if col_info is not None: 
                with col_info:
                    info_text = f"<b>Viewing:</b> {description}"
                    if len(st.session_state.selected_df_indices) > 1:
                        info_text += f" (+{len(st.session_state.selected_df_indices) - 1} more)"
                    info_text += f"<br><b>PAE Interaction (first):</b> {pae_text}"
                    st.markdown(f"<div style='text-align: center;'>{info_text}</div>", unsafe_allow_html=True)
            else: 
                st.markdown(f"**Viewing (first):** {description}")
                if len(st.session_state.selected_df_indices) > 1:
                    st.caption(f"(Total {len(st.session_state.selected_df_indices)} structures selected)")
                st.markdown(f"**PAE Interaction (first):** {pae_text}")

        # Collect PDB paths for all selected structures
        for idx in st.session_state.selected_df_indices:
            if isinstance(idx, int) and 0 <= idx < len(df):
                row = df.iloc[idx]
                desc = row.get("description", "")
                if desc:
                    pdb_path = (
                        st.session_state.current_path
                        / "af2_initial_guess"
                        / "pdbs"
                        / f"{desc}.pdb"
                    )
                    if pdb_path.exists():
                        pdb_paths_to_view.append(str(pdb_path))
                    else:
                        st.warning(f"PDB file not found for {desc}: {pdb_path}")
            else:
                st.warning(f"Invalid index {idx} in selection list.")
        
        if pdb_paths_to_view:
            # Create a dynamic key based on the PDB paths to ensure st_molstar_auto updates
            molstar_key = "mol_viewer_" + "_".join(sorted([Path(p).stem for p in pdb_paths_to_view]))
            st_molstar_auto(pdb_paths_to_view, key=molstar_key, height="800px")
        elif st.session_state.selected_df_indices: # Selected items exist, but no PDBs found for them
            st.info("Selected item(s) have no 'description' or valid PDB files to display a structure.")

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
