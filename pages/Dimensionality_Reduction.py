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

_spectra_df = spectra_df.set_index('filename')
# Extract all features from our spectra (all columns that start with "BIN_")
spectra = _spectra_df.loc[st.session_state.dm_selected_spectra, _spectra_df.columns.str.startswith("BIN_")].values


# Apply relevant clustering method
if st.session_state.dm_method == "PCA":
    # Perform PCA
    try:
        pca = PCA(n_components=dm_n_components)
        reduced_data = pca.fit_transform(spectra)

        # Display results
        st.subheader("PCA Results")
        explained_variance = pd.DataFrame(pca.explained_variance_ratio_)
        explained_variance.index=[f'PC{i+1}' for i in range(len(explained_variance))]
        explained_variance.columns=['Explained Variance Ratio']
        st.write(explained_variance)

        # Plot results
        if dm_n_components == 2:
            fig = plotly.graph_objects.Figure()
            fig.add_trace(plotly.graph_objects.Scatter(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            mode='markers',
            marker=dict(size=10, color='black'),  # Uniformly black
            hovertext=st.session_state.dm_selected_spectra,
            hoverinfo="text"
            ))
            fig.update_layout(
            title="PCA Results",
            xaxis_title="PC1",
            yaxis_title="PC2",
            template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif dm_n_components == 3:
            fig = plotly.graph_objects.Figure()
            fig.add_trace(plotly.graph_objects.Scatter3d(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            z=reduced_data[:, 2],
            mode='markers',
            marker=dict(size=5, color='black'),  # Uniformly black
            hovertext=st.session_state.dm_selected_spectra,
            ))
            fig.update_layout(
            title="PCA Results",
            scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
            ),
            template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True, height=800)
    except Exception as e:
        st.error(f"An error occurred during PCA: {e}")

elif st.session_state.dm_method == "PCOA":
    # Perform PCOA with dm_n_components and plot as scatter
    # Note: PCOA is not directly available in sklearn, but can be implemented using MDS
    mds = MDS(n_components=dm_n_components, dissimilarity='precomputed')
    raise NotImplementedError("PCOA is not implemented yet.")
    # The following is incorrect, that's not a distance matrix:
    distance_matrix = squareform(input_spectra)
    reduced_data = mds.fit_transform(distance_matrix)
    # Display results
    st.subheader("PCOA Results")
    st.write("Reduced Data:")
    st.dataframe(pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(dm_n_components)]))
    # Plot results
    if dm_n_components == 2:
        fig, ax = plt.subplots()
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)
    elif dm_n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        st.pyplot(fig)

elif st.session_state.dm_method == "t-SNE":
    # Perform t-SNE
    try:
        tsne = TSNE(
                        n_components=dm_n_components,
                        perplexity=st.session_state.dm_perplexity,
                        learning_rate=st.session_state.dm_learning_rate,
                        max_iter=st.session_state.dm_max_iter
                    )
        reduced_data = tsne.fit_transform(spectra)

        # Display results
        st.subheader("t-SNE Results")
        st.write("Reduced Data:")
        st.dataframe(pd.DataFrame(reduced_data, columns=[f"t-SNE{i+1}" for i in range(dm_n_components)]))

        # Plot results
        if dm_n_components == 2:
            fig = plotly.graph_objects.Figure()
            fig.add_trace(plotly.graph_objects.Scatter(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                mode='markers',
                marker=dict(size=10, color='black'),  # Uniformly black
                hovertext=st.session_state.dm_selected_spectra,
                hoverinfo="text"
            ))
            fig.update_layout(
                title="t-SNE Results",
                xaxis_title="t-SNE1",
                yaxis_title="t-SNE2",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif dm_n_components == 3:
            fig = plotly.graph_objects.Figure()
            fig.add_trace(plotly.graph_objects.Scatter3d(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                z=reduced_data[:, 2],
                mode='markers',
                marker=dict(size=5, color='black'),  # Uniformly black
                hovertext=st.session_state.dm_selected_spectra,
            ))
            fig.update_layout(
                title="t-SNE Results",
                scene=dict(
                    xaxis_title="t-SNE1",
                    yaxis_title="t-SNE2",
                    zaxis_title="t-SNE3"
                ),
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True, height=800)
    except Exception as e:
        st.error(f"An error occurred during t-SNE: {e}")

else:
    st.error("Invalid method selected.")
    st.stop()