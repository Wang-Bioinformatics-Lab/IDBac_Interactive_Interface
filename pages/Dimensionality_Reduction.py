import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import custom_css, format_proteins_as_strings
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc

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
displayed_as_categorical=True
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
            displayed_as_categorical = False
            cmap = plt.get_cmap("viridis")

            # If not castable to float, map to a number and then normalize
            if not np.issubdtype(plot_colors.dtype, np.number):
                color_map = {color: i for i, color in enumerate(plot_colors.unique())}
                plot_colors = plot_colors.apply(lambda x: color_map.get(x, "black"))

            # Normalize the continuous values
            norm = plt.Normalize(vmin=plot_colors.min(), vmax=plot_colors.max())
            plot_colors = [cmap(norm(i)) for i in plot_colors]

with st.expander("Preprocessing", expanded=True):
    # Options for Log Normalization
    st.session_state.dm_log_normalization = st.selectbox(
        "Select Log Normalization Method",
        options=["None", "Log2", "Log10", ],
        index=1,
        help="Select the log normalization method to apply to the spectra before dimensionality reduction. All logs first "
    )
    # Options for Centering
    st.session_state.dm_centering = st.selectbox(
        "Select Centering Method",
        options=["None", "Mean", "Median"],
        index=1,
        help="Select the centering method to apply to the spectra before dimensionality reduction."
    )
    # Options for Scaling
    st.session_state.dm_scaling = st.selectbox(
        "Select Scaling Method",
        options=["None", "Standardization", "Min-Max Scaling"],
        index=2,
        help="Select the scaling method to apply to the spectra before dimensionality reduction."
    )

# Add Toggle for Displaying Filename as Text
st.session_state.dm_display_filename = st.checkbox(
    "Display Filename as Text",
    value=False,
    help="Check this box to display the filename as text above each point."
)
# Add slider for font-size multiplier
st.session_state.dm_font_size_multiplier = st.slider(
    "Font Size Multiplier",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Adjust the font size of the plot text elements."
)
# Add dropdown for font color 
st.session_state.dm_font_color = st.selectbox(
    "Select Font Color",
    options=[
        ("Black", "black"),
        ("Dark Slate Gray", "darkslategray"),
        ("Dim Gray", "dimgray"),
        ("Gray", "gray"),
        ("Light Gray", "lightgray"),
        ("White", "white")
    ],
    index=0,
    format_func=lambda x: x[0],
    help="Select the font color for the plot text elements."
)
# Store only the color value in session state
st.session_state.dm_font_color = st.session_state.dm_font_color[1]
# Add dropdown for font family
st.session_state.dm_font_family = st.selectbox(
    "Select Font Family",
    options=[
        ("Arial", "Arial"),
        ("Courier New", "Courier New"),
        ("Georgia", "Georgia"),
        ("Times New Roman", "Times New Roman"),
        ("Verdana", "Verdana"),
    ],
    index=0,
    format_func=lambda x: x[0],
    help="Select the font family for the plot text elements."
)
# Store only the font family value in session state
st.session_state.dm_font_family = st.session_state.dm_font_family[1]

# Spectra DataFrame
_spectra_df = spectra_df.set_index('filename')
spectra = _spectra_df.loc[st.session_state.dm_selected_spectra, _spectra_df.columns.str.startswith("BIN_")].values

# Apply Log Normalization
if st.session_state.dm_log_normalization == "Log2":
    spectra = np.log2(spectra + 1)
elif st.session_state.dm_log_normalization == "Log10":
    spectra = np.log10(spectra + 1)
elif st.session_state.dm_log_normalization == "None":
    pass

# Apply Centering
if st.session_state.dm_centering == "Mean":
    spectra = spectra - np.mean(spectra, axis=1, keepdims=True)
elif st.session_state.dm_centering == "Median":
    spectra = spectra - np.median(spectra, axis=1, keepdims=True)
elif st.session_state.dm_centering == "None":
    pass

# Apply Scaling
if st.session_state.dm_scaling == "Standardization":
    spectra = (spectra - np.mean(spectra, axis=1, keepdims=True)) / np.std(spectra, axis=1, keepdims=True)
elif st.session_state.dm_scaling == "Min-Max Scaling":
    spectra = (spectra - np.min(spectra, axis=1, keepdims=True)) / (np.max(spectra, axis=1, keepdims=True) - np.min(spectra, axis=1, keepdims=True))
elif st.session_state.dm_scaling == "None":
    pass

def plot_reduced_data(reduced_data, selected_spectra, display_filename, n_components, method, metadata_df=None):
    # Basic plotting dataframe
    plot_df = pd.DataFrame(reduced_data, columns=[f"{method}{i+1}" for i in range(n_components)])
    plot_df['Filename'] = selected_spectra

    # Metadata coloring
    coloring_col = st.session_state.get("dm_metadata_coloring", "None")
    has_metadata = metadata_df is not None and not metadata_df.empty and coloring_col != "None"

    if has_metadata:
        plot_df[coloring_col] = metadata_df.loc[selected_spectra, coloring_col].values
        plot_df['hover'] = plot_df['Filename'] + "<br>" + coloring_col + ": " + plot_df[coloring_col].astype(str)
    else:
        plot_df['hover'] = plot_df['Filename']
        coloring_col = None

    # Choose Plotly Express for legend + color handling
    if n_components == 2:
        fig = px.scatter(
            plot_df,
            x=f"{method}1",
            y=f"{method}2",
            color=coloring_col if has_metadata else None,
            hover_name='hover',
            text='Filename' if display_filename else None,
            color_discrete_sequence=px.colors.qualitative.Plotly if has_metadata and plot_df[coloring_col].nunique() <= 10 else px.colors.qualitative.Light24,
            color_continuous_scale='viridis' if has_metadata and not pd.api.types.is_categorical_dtype(plot_df[coloring_col]) and plot_df[coloring_col].dtype.kind in 'if' else None
        )
    else:
        fig = px.scatter_3d(
            plot_df,
            x=f"{method}1",
            y=f"{method}2",
            z=f"{method}3",
            color=coloring_col if has_metadata else None,
            hover_name='hover',
            text='Filename' if display_filename else None,
            color_discrete_sequence=px.colors.qualitative.Plotly if has_metadata and plot_df[coloring_col].nunique() <= 10 else px.colors.qualitative.Light24,
            color_continuous_scale='viridis' if has_metadata and not pd.api.types.is_categorical_dtype(plot_df[coloring_col]) and plot_df[coloring_col].dtype.kind in 'if' else None
        )

    fig.update_traces(marker=dict(size=6 if n_components == 3 else 10))

    # Global font scaling
    base_size = 12  # Default base font size
    font_size_multiplier = st.session_state.get('dm_font_size_multiplier', 1.0)
    font_color = st.session_state.get('dm_font_color', 'black')
    font_family = st.session_state.get('dm_font_family', 'Arial')
    fig.update_layout(
        template="plotly_white",
        font=dict(
            size=base_size * font_size_multiplier,
            color=font_color,
            family=font_family
        ),
        legend=dict(
            font=dict(
                size=base_size * font_size_multiplier * 0.9,
                color=font_color,
                family=font_family
            )
        ),
        # title=dict(font=dict(size=base_size * font_size_multiplier * 1.2, color=font_color, family=font_family))
    )

    config = {
        'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'IDBac_Dimensionality_Reduction_Protein',
        'scale':5 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    st.plotly_chart(fig, use_container_width=True, height=800, config=config)

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
        plot_reduced_data(
                            reduced_data=reduced_data,
                            selected_spectra=st.session_state.dm_selected_spectra,
                            display_filename=st.session_state.dm_display_filename,
                            n_components=dm_n_components,
                            method="PCA",
                            metadata_df=metadata_df
                        )

    except Exception as e:
        raise e
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
        plot_reduced_data(
                    reduced_data=reduced_data,
                    selected_spectra=st.session_state.dm_selected_spectra,
                    display_filename=st.session_state.dm_display_filename,
                    n_components=dm_n_components,
                    method="t-SNE",
                    metadata_df=metadata_df
                )

    except Exception as e:
        st.error(f"An error occurred during t-SNE: {e}")

else:
    st.error("Invalid method selected.")
    st.stop()