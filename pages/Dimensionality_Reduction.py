import streamlit as st
import streamlit.components.v1 as components
from pyvis import network as net
from pyvis import options as pyvis_options
import pandas as pd
import numpy as np
import json
import requests
import plotly
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from utils import custom_css, format_proteins_as_strings
from streamlit.errors import StreamlitAPIException
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import plotly.graph_objects as go

from pages.Plot_Spectra import get_peaks, get_peaks_from_db_result


#####
# A note abote streamlit session states:
# All session states related to this page begin with "dm_" to reduce the 
# chance of collisions with other pages. 
#####

# Set Page Configuration
st.set_page_config( 
                    page_title="IDBac - Dimensionality Reduction", 
                    page_icon="assets/idbac_logo_square.png", 
                    layout="wide",
                    initial_sidebar_state="auto",
                    menu_items=None
                )
custom_css()


st.title("Dimensionality Reduction")
dm_n_components = 3

# Get List of Options
spectra_df = st.session_state.get('query_only_spectra_df')
metadata_df = st.session_state.get('metadata_df', pd.DataFrame())
bin_size = st.session_state['workflow_params'].get('bin_size')

if spectra_df is None:
    st.error("No spectra data available.")
    st.stop()

if bin_size is None:
    st.error("bin_size parameter not available. Please rerun the workflow.")
    st.stop()


options = format_proteins_as_strings(spectra_df)

# Dropdown to Select Spectra (Select Many)
st.subheader("Select Spectra")
st.session_state.dm_selected_spectra = st.multiselect(
                                                        "Select spectra",
                                                        options=options,
                                                        default=options,
                                                        help="Select the spectra you want to include in the analysis."
                                                      )

# Select Method Parameters
st.subheader("Select Method Parameters")
st.session_state.dm_method = st.selectbox(  
                                            "Select Method",
                                            options=["PCA", "t-SNE"],   # "PCOA"
                                            index=0,
                                            help="Select the method you want to use for dimensionality reduction."
                                        )

if st.session_state.dm_method == "t-SNE":
    st.session_state.dm_perplexity = st.slider(
        "Perplexity",
        min_value=1,
        max_value=len(st.session_state.dm_selected_spectra) - 1,
        value=min(30, len(st.session_state.dm_selected_spectra) - 1),
        step=1,
        help="Perplexity parameter for t-SNE. Higher values lead to more global structure being preserved."
    )
    st.session_state.dm_learning_rate = st.slider(
        "Learning Rate",
        min_value=10,
        max_value=1000,
        value=200,
        step=10,
        help="Learning rate parameter for t-SNE. Higher values lead to faster convergence."
    )
    st.session_state.dm_max_iter = st.slider(
        "Number of Iterations",
        min_value=100,
        max_value=1000,
        value=500,
        step=10,
        help="Number of iterations for t-SNE. Higher values lead to more accurate results."
    )

# Dropdown for metadata coloring options (any column in metadata_df)
plot_colors = "black"
if not metadata_df.empty:
    metadata_df = metadata_df.set_index('Filename')
    st.session_state.dm_metadata_coloring = st.selectbox(
        "Select Metadata Column for Coloring",
        options=["None"] + list(metadata_df.columns),
        index=0,
        help="Select a metadata column to color the spectra in the plot."
    )
    if st.session_state.dm_metadata_coloring != "None":
        plot_colors = metadata_df.loc[st.session_state.dm_selected_spectra, st.session_state.dm_metadata_coloring]

        # Convert plot_colors to a list of hex values
        if len(plot_colors.unique()) <= 10:
            cmap = plt.get_cmap("tab10")
            # Map to category
            plot_colors = plot_colors.astype("category").cat.codes

            color_mapping = {key: cmap(i) for key, i in enumerate(plot_colors.unique())}
            plot_colors = [color_mapping.get(color, "black") for color in plot_colors]
        elif len(plot_colors.unique()) <= 20:
            cmap = plt.get_cmap("tab20")
            # Map to category
            plot_colors = plot_colors.astype("category").cat.codes

            color_mapping = {key: cmap(i) for key, i in enumerate(plot_colors.unique())}
            plot_colors = [color_mapping.get(color, "black") for color in plot_colors]
        else:
            cmap = plt.get_cmap("viridis")

            # If not castable to float, map to a number and then normalize
            if not np.issubdtype(plot_colors.dtype, np.number):
                color_map = {color: i for i, color in enumerate(plot_colors.unique())}
                plot_colors = plot_colors.apply(lambda x: color_map.get(x, "black"))

            # Normalize the continuous values
            norm = plt.Normalize(vmin=plot_colors.min(), vmax=plot_colors.max())
            plot_colors = [cmap(norm(i)) for i in plot_colors]

# Add Toggle for Displaying Filename as Text
st.session_state.dm_display_filename = st.checkbox(
    "Display Filename as Text",
    value=False,
    help="Check this box to display the filename as text above each point."
)

# Spectra DataFrame
_spectra_df = spectra_df.set_index('filename')
spectra = _spectra_df.loc[st.session_state.dm_selected_spectra, _spectra_df.columns.str.startswith("BIN_")].values


def plot_reduced_data(reduced_data, plot_colors, selected_spectra, display_filename, n_components, method):
    """Helper function to plot the reduced data for both PCA and t-SNE"""
    fig = go.Figure()

    if n_components == 2:
        fig.add_trace(go.Scatter(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            mode='markers',
            marker=dict(size=10, color=plot_colors),  # Use plot_colors directly
            hovertext=selected_spectra,
            hoverinfo="text",
            text=selected_spectra if display_filename else None,
            textposition="top center"
        ))

        fig.update_layout(
            title=f"{method} Results",
            xaxis_title=f"{method}1",
            yaxis_title=f"{method}2",
            template="plotly_white"
        )

    elif n_components == 3:
        text = selected_spectra if display_filename else None
        fig.add_trace(go.Scatter3d(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            z=reduced_data[:, 2],
            mode='markers+text' if display_filename else 'markers',
            marker=dict(size=5, color=plot_colors),  # Use plot_colors directly
            hovertext=selected_spectra,
            text=selected_spectra if display_filename else None,
            textposition="top center"
        ))

        fig.update_layout(
            title=f"{method} Results",
            scene=dict(
                xaxis_title=f"{method}1",
                yaxis_title=f"{method}2",
                zaxis_title=f"{method}3"
            ),
            template="plotly_white"
        )

    st.plotly_chart(fig, use_container_width=True, height=800)

# Apply relevant clustering method
if st.session_state.dm_method == "PCA":
    try:
        # Perform PCA
        pca = PCA(n_components=dm_n_components)
        reduced_data = pca.fit_transform(spectra)

        # Display PCA results
        st.subheader("PCA Results")
        explained_variance = pd.DataFrame(pca.explained_variance_ratio_)
        explained_variance.index = [f'PC{i+1}' for i in range(len(explained_variance))]
        explained_variance.columns = ['Explained Variance Ratio']
        st.write(explained_variance)

        # Plot PCA results
        plot_reduced_data(reduced_data, plot_colors, st.session_state.dm_selected_spectra, st.session_state.dm_display_filename, dm_n_components, "PCA")

    except Exception as e:
        st.error(f"An error occurred during PCA: {e}")

elif st.session_state.dm_method == "t-SNE":
    try:
        tsne = TSNE(
            n_components=dm_n_components,
            perplexity=st.session_state.dm_perplexity,
            learning_rate=st.session_state.dm_learning_rate,
            max_iter=st.session_state.dm_max_iter
        )
        reduced_data = tsne.fit_transform(spectra)

        # Display t-SNE results
        st.subheader("t-SNE Results")
        st.write("Reduced Data:")
        st.dataframe(pd.DataFrame(reduced_data, columns=[f"t-SNE{i+1}" for i in range(dm_n_components)]))

        # Plot t-SNE results
        plot_reduced_data(reduced_data, plot_colors, st.session_state.dm_selected_spectra, st.session_state.dm_display_filename, dm_n_components, "t-SNE")

    except Exception as e:
        st.error(f"An error occurred during t-SNE: {e}")

else:
    st.error("Invalid method selected.")
    st.stop()