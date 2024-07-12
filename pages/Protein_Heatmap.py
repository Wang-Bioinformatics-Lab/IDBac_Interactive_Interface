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
from utils import custom_css

# import StreamlitAPIException
from streamlit.errors import StreamlitAPIException

#####
# A note abote streamlit session states:
# All session states related to this page begin with "phm_" to reduce the 
# chance of collisions with other pages. 
#####

# Set Page Configuration
st.set_page_config(page_title="IDBac - Protein Heatmap", page_icon=None, layout="wide", initial_sidebar_state="collapsed", menu_items=None)
custom_css()

def format_proteins_as_strings(df):
    output = []
    for row in df.to_dict(orient="records"):
        if row['db_search_result']:
            output.append(f"DB Result - {row['filename']}")
        else:   
            output.append(row['filename'])
            
    return output

def basic_dendrogram(disabled=False):
    """
    This function generates a basic dendrogram for the protein heatmap page. 
    """
    st.slider("Coloring Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01, key="phm_coloring_threshold")
    clustering_options = ["ward", "single", "complete", "average", "weighted", "centroid", "median"]
    st.selectbox("Clustering Method", clustering_options, key="phm_clustering_method")

    if disabled:
        return None, None
    if st.session_state['query_spectra_numpy_data'].shape[0] <= 1:
        st.warning("There are not enough spectra to create a dendrogram. \n \
                   Please check number of input spectra and database search results file.")
        return None, None

    def _dist_fun(x):
        return squareform(st.session_state['distance_measure'](x))

    dendro = ff.create_dendrogram(st.session_state['query_spectra_numpy_data'],
                                orientation='bottom',
                                labels=st.session_state['query_only_spectra_df'].filename.values, # We will use the labels as a unique identifier
                                distfun=_dist_fun,
                                linkagefun=lambda x: linkage(x, method=st.session_state["phm_clustering_method"],),
                                color_threshold=st.session_state["phm_coloring_threshold"])
    
    st.plotly_chart(dendro, use_container_width=True)
    # Sadly the only way to get the actual clusters (to plot the graph) is to recompute the linkage with scipy
    # (TODO: Just used scipy to plot it)
    dist_matrix = _dist_fun(st.session_state['query_spectra_numpy_data'])
    linkage_matrix = linkage(dist_matrix,
                            method=st.session_state["phm_clustering_method"])
    sch_dendro = dendrogram(linkage_matrix,
                            labels=st.session_state['query_only_spectra_df'].filename.values,
                            no_plot=True,
                            color_threshold=st.session_state["phm_coloring_threshold"])
    
    # Note that C0 means unclustered here:
    cluster_dict = {filename: color for filename, color in zip(sch_dendro['ivl'], sch_dendro['leaves_color_list'])}
    # Replace C0, C1, C2 with integers 
    cluster_dict = {filename: int(color[1:]) for filename, color in cluster_dict.items()}
    # Add 1 if value >= 1 to reserve 1 for small molecules, 0 for unclustered
    cluster_dict = {filename: color + 1 if color >= 1 else 0 for filename, color in cluster_dict.items()}
    
    return cluster_dict, dendro

def draw_protein_heatmap(all_spectra_df, bin_size, cluster_dict=None):
    st.subheader("Protein Spectra m/z Heatmap")
    add_filters_1, add_filters_2, add_filters_3 = st.columns([0.45, 0.10, 0.45])
    # Options
    all_options = format_proteins_as_strings(all_spectra_df)

    # Initialze all protins as selected
    st.session_state['phm_selected_proteins'] = all_options

    #### Select Strains ####
    with st.form(key="phm_mz_filters", border=False):
        # Protein Cluster Selection
        disabled=False
        if cluster_dict is None:
            cluster_dict = [None]
            add_filters_1.multiselect("Select Clusters to Add", [], disabled=True, key='phm_selected_clusters')
            
        else:
            inverted_cluster_dict = {}
            for filename, cluster_id in cluster_dict.items():
                if inverted_cluster_dict.get(cluster_id) is None:
                    inverted_cluster_dict[cluster_id] = [filename]
                else:
                    inverted_cluster_dict[cluster_id].append(filename)
            cluster_display_dict = {tuple(set(filenames)): f"Cluster {cluster_id-1}: {tuple(set(filenames))}".replace(',','') for cluster_id, filenames in inverted_cluster_dict.items()}
            unclustered_key = inverted_cluster_dict.get(0)
            if unclustered_key is not None:
                unclustered_key = tuple(set(unclustered_key))
                cluster_display_dict[unclustered_key] = cluster_display_dict[unclustered_key].replace("Cluster -1", "Unclustered")
            
            add_filters_1.multiselect("Select Clusters to Add", 
                            list(set(cluster_display_dict.keys())),
                            format_func=cluster_display_dict.get,
                            key='phm_selected_clusters')
            
        # Metadata Selection Options
        metadata_selection_col1, metadata_selection_col2 = add_filters_1.columns([0.5, 0.5])
        if st.session_state.get('metadata_df') is None:
            metadata_selection_col1.selectbox("Metadata Criteria",
                                              ["No Metadata Available"],
                                              key="phm_metadata_criteria",
                                              disabled=True)
            metadata_selection_col2.multiselect("Metadata Values",
                                                [],
                                                key="phm_metadata_values",
                                                disabled=True)
            
        else:
            metadata_selection_col1.selectbox("Metadata Criteria",
                                                st.session_state["metadata_df"].columns,
                                                key="phm_metadata_criteria")
            metadata_selection_col2.multiselect("Metadata Values",
                                                st.session_state["metadata_df"][st.session_state["phm_metadata_criteria"]].unique(),
                                                key="phm_metadata_values")
                    
        # Initialize selected proteins and clusters states if not initialized
        if 'phm_selected_clusters' not in st.session_state:
            st.session_state['phm_selected_clusters'] = []
        if 'phm_selected_proteins' not in st.session_state:
            st.session_state['phm_selected_proteins'] = []
        if 'phm_metadata_criteria' not in st.session_state:
            st.session_state['phm_metadata_criteria'] = None
        if 'phm_metadata_values' not in st.session_state:
            st.session_state['phm_metadata_values'] = []

        # Button to Move Clusters to Individual Protein List
        with st.container():
            (_, ab, _) =  add_filters_2.columns([1,1,1])    # Allows the arrow button to stay centered
            ab.markdown('<div class="button-label centered">Add Selection</div>', unsafe_allow_html=True)
            add_button = ab.button(":arrow_forward:", key="Add")

        # Individual Protein Selection
        try:
            phm_selected_proteins = add_filters_3.multiselect(
                "Select Strains",
                all_options,
                default=st.session_state['phm_selected_proteins']
            )
        except StreamlitAPIException as _:
            # This is caused by a task change, so we just reset the selected proteins
            phm_selected_proteins = []
            st.session_state['phm_selected_proteins'] = []
            phm_selected_proteins = add_filters_3.multiselect(
                "Select Strains",
                all_options,
                default=st.session_state['phm_selected_proteins']
            )

            
        phm_selected_prot_submitted = st.form_submit_button("Apply/Update Filters")
        
    if phm_selected_prot_submitted:
        st.session_state['phm_selected_proteins'] = phm_selected_proteins

    # Handle add button click
    if add_button:
        # Add from clusters
        for cluster in st.session_state['phm_selected_clusters']:
            to_add = set(cluster) - set(st.session_state['phm_selected_proteins'])
            st.session_state['phm_selected_proteins'].extend(to_add)
        st.session_state['phm_selected_clusters'].clear()
        # Add from metadata
        if st.session_state['phm_metadata_criteria'] is not None:
            relevant_ids = st.session_state['metadata_df'][st.session_state['metadata_df'][st.session_state['phm_metadata_criteria']].isin(st.session_state['phm_metadata_values'])].Filename
            to_add = set(relevant_ids) - set(st.session_state['phm_selected_proteins'])
            st.session_state['phm_selected_proteins'].extend(to_add)
        st.session_state['phm_metadata_values'].clear()

        st.rerun()  # Refresh the UI to reflect the updated selection
    #########################
    
    min_count = st.slider("Minimum m/z Count", min_value=0, max_value=max(1,len(st.session_state['phm_selected_proteins'])), step=1, value=1,
                         help="The minimum number of times an m/z value must be present \
                               in the selected proteins to be displayed.")
    min_intensity = st.slider("Minimum Relative Intensity", min_value=0.0, max_value=1.0, step=0.01, value=0.40,
                              help="The minimum relative intensity value to display.")
    
    
    # Remove "DB Result - " from the selected proteins
    selected_proteins = [x.replace("DB Result - ", "") for x in st.session_state['phm_selected_proteins']]
       
    # Set index to filename
    all_spectra_df = all_spectra_df.set_index("filename")
    all_spectra_df = all_spectra_df.loc[selected_proteins, :]
    
    bin_columns = [col for col in all_spectra_df.columns if col.startswith("BIN_")]
    bin_columns = sorted(bin_columns, key=lambda x: int(x.split("_")[-1]))
    all_spectra_df = all_spectra_df.loc[:, bin_columns]
    # Normalize Intensity (Normalize Across Row)
    all_spectra_df = all_spectra_df.div(all_spectra_df.max(axis=1), axis=0)
    # Set zeros to nan
    all_spectra_df = all_spectra_df.replace(0, np.nan)
    # Set all values less than min_intensity to nan
    all_spectra_df = all_spectra_df.where(all_spectra_df > min_intensity)
    # Filter bins by count
    bin_columns = [col for col in bin_columns if all_spectra_df[col].notna().sum() >= min_count]
    all_spectra_df = all_spectra_df.loc[:, bin_columns]
    
    def _convert_bin_to_mz(bin_name):
        bin = int(bin_name.split("_")[-1])
        
        return f"[{bin * bin_size}, {(bin + 1) * bin_size})"
    
    # Remove rows with all nan
    all_spectra_df = all_spectra_df.dropna(how='all', axis='columns')
    
    if len(all_spectra_df.columns) != 0:
        # Note: We transpose the dataframe so that the proteins are on the x-axis
        st.markdown("Common m/z values between selected proteins and their relative intensities.")
        # Draw Heatmap
        dynamic_height = max(500, len(all_spectra_df.columns) * 24) # Dyanmic height based on number of m/z values
        
        # If we're suppled a dendrogram, use it to reorder the heatmap
        x = None
        if False:   # The below code is copied from the small molecule heatmap, but it doesn't work for proteins.
                    # I've left it here in case we want to try to get it working in the future.
           
            # Remove any rows where the filename is not currently selected
            all_filenames = st.session_state['query_only_spectra_df'].filename.values
            all_data      = st.session_state['query_spectra_numpy_data']
            
            # Get the indices of the selected proteins
            selected_indices = [i for i, filename in enumerate(all_filenames) if filename in st.session_state["phm_selected_proteins"]]
            # Get the data for the selected proteins
            numpy_data = all_data[selected_indices]
            
            # Unfortunately, we have to recalculate the dendrogram, because things may cluster differently 
            # depending on the selected proteins.
            # Note though, that we share parameters with the above dendrogram.
            dendro = ff.create_dendrogram(numpy_data,
                                orientation='bottom',
                                labels=st.session_state["phm_selected_proteins"],
                                distfun=st.session_state['distance_measure'],
                                linkagefun=lambda x: linkage(x, method=st.session_state["phm_clustering_method"],),
                                color_threshold=st.session_state["phm_coloring_threshold"])
            
            # Reorder the dataframe based on the dendrogram
            reordered_df = all_spectra_df.reindex(index=dendro.layout.xaxis.ticktext)
            reordered_df = reordered_df.reindex(columns=dendro.layout.yaxis.ticktext)
            all_spectra_df = reordered_df
            # Also us the X values from the dendrogram
            x = dendro.layout.xaxis.tickvals
        
        heatmap = plotly.express.imshow(all_spectra_df.T.values,    # Transpose so m/zs are rows
                                        x=x,
                                        aspect ='auto', 
                                        width=1500, 
                                        height=dynamic_height,
                                        color_continuous_scale='Bluered',)
        # Update axis text (we do this here otherwise spacing is not even)
        heatmap.update_layout(
            xaxis=dict(title="Protein", ticktext=list(all_spectra_df.index.values), tickvals=list(range(len(all_spectra_df.index))), side='top'),
            yaxis=dict(title="m/z", ticktext=[_convert_bin_to_mz(x) for x in all_spectra_df.columns], tickvals=list(range(len(all_spectra_df.columns)))),
            margin=dict(t=5, pad=0),
        )
        
        heatmap.update_coloraxes(cmin=0.0, cmax=1.0, cmid=0.5)
        
        if False: # The below code is copied from the small molecule heatmap, but it doesn't work for proteins.
            #  I've left it here in case we want to try to get it working in the future.
            
            dendrogram_height = 200
            dendrogram_height_as_percent = dendrogram_height / (dynamic_height + dendrogram_height)
            
            fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                                shared_xaxes=True,
                                                vertical_spacing=0.02,
                                                row_heights=[dendrogram_height_as_percent, 1-dendrogram_height_as_percent])
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0), width=1500, height=dynamic_height + dendrogram_height)
        
            for trace in dendro.data:
                fig.add_trace(trace, row=1, col=1)
            
            # Add x-axis labels from dendrogram
            fig.update_xaxes(ticktext=dendro.layout.xaxis.ticktext, tickvals=dendro.layout.xaxis.tickvals, row=1, col=1)
            fig.update_xaxes(ticktext=dendro.layout.xaxis.ticktext, tickvals=dendro.layout.xaxis.tickvals, row=2, col=1)
            # Add y labels to dendrogram
            fig.update_yaxes(ticktext=dendro.layout.yaxis.ticktext, tickvals=dendro.layout.yaxis.tickvals, row=1, col=1, title="Dendrogram Distance")
            # Add y labels to heatmap
            fig.update_yaxes(ticktext=heatmap.layout.yaxis.ticktext, tickvals=heatmap.layout.yaxis.tickvals, row=2, col=1,title="m/z")
            
            for trace in heatmap.data:
                fig.add_trace(trace, row=2, col=1)
            
        else:
            fig = heatmap
            
        fig.update_layout(showlegend=False,
                    coloraxis_colorbar=dict(title="Relative Intensity", 
                                            len=min(500, dynamic_height), 
                                            lenmode="pixels", 
                                            y=0.75)
                                        )
        fig.update_coloraxes(cmin=0.0, cmax=1.0, cmid=0.5,colorscale='Bluered')
        
        st.plotly_chart(fig,use_container_width=True)

        # Rename the indices to be more human readable using _convert_bin_to_mz
        all_spectra_df.columns = [_convert_bin_to_mz(x) for x in all_spectra_df.columns]

        # Add a button to download the heatmap
        st.download_button("Download Current Heatmap Data", all_spectra_df.T.to_csv(), "protein_heatmap.csv", help="Download the data used to generate the heatmap.")

cluster_dict = None
st.subheader("Spectral Similarity Options")
with st.popover(label='Set Spectral Clustering Parameters'):
    cluster_dict, _ = basic_dendrogram()

# Use "query_only_spectra_df" because database spectra may be binned to a different size
draw_protein_heatmap(st.session_state['query_only_spectra_df'], st.session_state['workflow_params']['bin_size'], cluster_dict)