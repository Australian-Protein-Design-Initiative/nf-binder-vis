#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from typing import Optional, List
from streamlit_molstar import st_molstar


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

    cs_files = list(path.glob("**/*.cs"))
    if not cs_files:
        st.error(f"No .cs files found in {path}")
        return None

    st.info(f"Found {len(cs_files)} .cs files")

    dfs = []
    for file_path in cs_files:
        df = parse_score_file(file_path)
        if df is not None:
            dfs.append(df)

    if not dfs:
        st.error("No valid data found in any of the .cs files")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)

    numeric_columns = [
        "binder_aligned_rmsd",
        "pae_binder",
        "pae_interaction",
        "pae_target",
        "plddt_binder",
        "plddt_target",
        "plddt_total",
        "target_aligned_rmsd",
        "time",
    ]

    for col in numeric_columns:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    return combined_df


def plot_distribution(df: pd.DataFrame, metrics: List[str]) -> None:
    """Create distribution plots for selected metrics."""
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.5 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        sns.kdeplot(data=df, x=metric, ax=axes[i])
        sns.rugplot(data=df, x=metric, ax=axes[i], alpha=0.2)
        axes[i].set_title(f"Distribution of {metric}")

    plt.tight_layout()
    st.pyplot(fig)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Create interactive scatter plot using Altair."""
    # Create a selection condition
    selection = alt.selection_point(name="selected_point", fields=[x_col, y_col])

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

    # Sidebar for path input
    path_input = st.sidebar.text_input("Results Directory Path", value=args.path)
    path = Path(path_input)

    df = load_data(path)
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

    # Sort dataframe by pae_interaction ascending
    if "pae_interaction" in df.columns:
        df = df.sort_values("pae_interaction", ascending=True)

    # Create tabs for different views
    table_tab, scatter_tab, dist_tab = st.tabs(
        ["Data Table", "Scatter Plot", "Distributions"]
    )

    with table_tab:
        st.info(f"Total rows: {len(df)}")
        # Display the full dataset with selected columns and enable selection
        selection = st.dataframe(
            df[st.session_state.selected_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "_selected_": st.column_config.CheckboxColumn(
                    "View",
                    help="Select a row to view structure",
                    default=False,
                ),
                **{
                    col: st.column_config.NumberColumn(col, format="%.3f")
                    for col in numeric_cols
                },
            },
            selection_mode=["single-row"],
            on_select="rerun",
            key="data",
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
            x_col = st.selectbox(
                "X-axis", options=numeric_cols, index=numeric_cols.index("plddt_binder")
            )
        with col2:
            y_col = st.selectbox(
                "Y-axis",
                options=numeric_cols,
                index=numeric_cols.index("pae_interaction"),
            )
        # Get selected points from scatter plot
        chart_event = plot_scatter(df, x_col, y_col)

    with dist_tab:
        selected_metrics = st.multiselect(
            "Select metrics for distribution plots",
            options=numeric_cols,
            default=["pae_interaction", "pae_binder", "plddt_binder"],
        )
        if selected_metrics:
            plot_distribution(df, selected_metrics)

    # Function to show the molecular viewer
    def show_structure(description: str):
        if description:
            pdb_path = path / "af2_initial_guess" / "pdbs" / f"{description}.pdb"
            if pdb_path.exists():
                st_molstar(str(pdb_path), key="mol_viewer", height="800px")
            else:
                st.warning(f"PDB file not found: {pdb_path}")

    # Show structure based on either table or scatter plot selection
    st.header("Structure Viewer")
    if selection.selection and selection.selection.rows:
        # Table selection takes precedence
        selected_row = df.iloc[selection.selection.rows[0]]
        show_structure(selected_row.get("description", ""))
    elif chart_event.selection and "selected_point" in chart_event.selection:
        # Scatter plot selection
        selected_data = chart_event.selection["selected_point"]
        if selected_data and len(selected_data) > 0:
            # Find the row in df that matches the selected point
            mask = (df[x_col] == selected_data[0][x_col]) & (
                df[y_col] == selected_data[0][y_col]
            )
            if any(mask):
                selected_row = df[mask].iloc[0]
                show_structure(selected_row.get("description", ""))


if __name__ == "__main__":
    main()
