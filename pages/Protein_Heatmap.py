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
from utils import custom_css, parse_numerical_input

# import StreamlitAPIException
from streamlit.errors import StreamlitAPIException

#####
# A note abote streamlit session states:
# All session states related to this page begin with "phm_" to reduce the 
# chance of collisions with other pages. 
#####

# Set Page Configuration
st.set_page_config(page_title="IDBac - Protein Heatmap", page_icon="assets/idbac_logo_square.png", layout="wide", initial_sidebar_state="collapsed", menu_items=None)
custom_css()

def format_proteins_as_strings(df):
    output = []
    for row in df.to_dict(orient="records"):
        if row['db_search_result']:
            output.append(f"DB Result - {row['filename']}")
        else:   
            output.append(row['filename'])
            
    return output

def basic_dendrogram(spectrum_df=None, disabled=False, display=True):
    """
    This function generates a basic dendrogram for the protein heatmap page. 
    """

    # If spectrum_df is specified, use that, otherwise use the query spectra
    if spectrum_df is None:
        spectrum_df = st.session_state['query_only_spectra_df']
        query_spectra_numpy_data = st.session_state['query_spectra_numpy_data']
    else:
        # We need to subset the query_spectrum_numpy_data to match the selected proteins
        selected_proteins = spectrum_df.index
        query_spectra_numpy_data = st.session_state['query_spectra_numpy_data']
        query_only_df = st.session_state['query_only_spectra_df']
        # Select and reorder the numpy data by order in selected_proteins
        indices = []
        for protein in selected_proteins:
            indices.append(np.where(query_only_df.filename == protein)[0][0])
        query_spectra_numpy_data = query_spectra_numpy_data.take(indices, axis=0)

    # Check if options have been initialized, if so skip
    if 'phm_coloring_threshold' not in st.session_state:
        st.slider("Coloring Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01, key="phm_coloring_threshold")
        clustering_options = ["average", "single", "complete", "weighted"]
        if st.session_state['distance_measure'] == "euclidean":
            clustering_options += ['ward', 'median', 'centroid']
        st.selectbox("Clustering Method", clustering_options, key="phm_clustering_method")

    if disabled:
        return None, None, None
    if query_spectra_numpy_data.shape[0] <= 1:
        st.warning("There are not enough spectra to create a dendrogram. \n \
                   Please check number of input spectra and database search results file.")
        return None, None, None

    def _dist_fun(x):
        return squareform(st.session_state['distance_measure'](x), force='tovector')

    # Sadly the only way to get the actual clusters (to plot the graph) is to recompute the linkage with scipy
    # (TODO: Just used scipy to plot it)
    dist_matrix = _dist_fun(query_spectra_numpy_data)
    linkage_matrix = linkage(dist_matrix,
                            method=st.session_state["phm_clustering_method"])
    sch_dendro = dendrogram(linkage_matrix,
                            labels=spectrum_df.filename.values,
                            no_plot=True,
                            color_threshold=st.session_state["phm_coloring_threshold"])
    
    # Note that C0 means unclustered here:
    # Note that the colors will repeat, cluster_id should be unique
    cluster_dict = {}   # Dictionary of ['filename'] {'cluster_id', 'color'}
    filenames = sch_dendro['ivl']
    colors_list = sch_dendro['leaves_color_list']

    prev_color = colors_list[0]
    curr_cluster = 1
    for filename, color in zip(filenames, colors_list):
        # If the last color doesn't equal this color, we have a new cluster
        if color != prev_color and color != 'C0':   # Don't increment the cluster if it's unclustered
            curr_cluster += 1
            prev_color = color
        elif color == 'C0':     # To handle the edge case where to clusters of the same color are split by an unclustered protein
            prev_color = None
        cluster_dict[filename] = {
                                  'cluster': curr_cluster if color != 'C0' else 0,  # If the color is C0, it's unclustered
                                  'color': int(color[1:])
                                  }
    
    dendro = ff.create_dendrogram(query_spectra_numpy_data,
                                orientation='bottom',
                                labels=spectrum_df.filename.values, # We will use the labels as a unique identifier
                                distfun=_dist_fun,
                                linkagefun=lambda x: linkage(x, method=st.session_state["phm_clustering_method"],),
                                color_threshold=st.session_state["phm_coloring_threshold"])
    
    # Add clusters to dendrogram labels
    for i, label in enumerate(dendro.layout.xaxis.ticktext):
        cluster = cluster_dict.get(label)
        if cluster is not None:
            cluster = cluster['cluster']
            if cluster == 0:
                dendro.layout.xaxis.ticktext[i] = f"Unclustered - {label}"
            else:
                dendro.layout.xaxis.ticktext[i] = f"Cluster {cluster} - {label}"
    
    if display == False:
        return cluster_dict, dendro, filenames
    st.plotly_chart(dendro, use_container_width=True)

    return cluster_dict, dendro, filenames

def draw_protein_heatmap(all_spectra_df, bin_size, all_clusters_dict):     # , cluster_dict=None, dendro=None, dendro_ordering=None'
    st.subheader("Protein Spectra m/z Heatmap")
    add_filters_1, add_filters_2, add_filters_3 = st.columns([0.45, 0.10, 0.45])
    # Options
    all_options = format_proteins_as_strings(all_spectra_df)

    # Initialze all protins as selected
    if 'phm_selected_proteins' not in st.session_state:
        st.session_state['phm_selected_proteins'] = all_options

    #### Select Strains ####
    with st.form(key="phm_mz_filters", border=False):
        # Protein Cluster Selection

        if all_clusters_dict is None:
            all_clusters_dict = [None]
            add_filters_1.multiselect("Select Clusters to Add", [], disabled=True, key='phm_selected_clusters')
            
        else:
            inverted_cluster_dict = {}
            for filename, dict_for_filename in all_clusters_dict.items():
                cluster_id = dict_for_filename['cluster']
                if inverted_cluster_dict.get(cluster_id) is None:
                    inverted_cluster_dict[cluster_id] = [filename]
                else:
                    inverted_cluster_dict[cluster_id].append(filename)
            cluster_display_dict = {tuple(set(filenames)): f"Cluster {cluster_id}: {tuple(set(filenames))}".replace(',','') for cluster_id, filenames in inverted_cluster_dict.items()}
            unclustered_key = inverted_cluster_dict.get(0)
            if unclustered_key is not None:
                unclustered_key = tuple(set(unclustered_key))
                cluster_display_dict[unclustered_key] = cluster_display_dict[unclustered_key].replace("Cluster 0", "Unclustered")
            
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
                default=[]
            )

            
        phm_selected_prot_submitted = st.form_submit_button("Apply/Update Filters")
        
    if phm_selected_prot_submitted:
        st.session_state['phm_selected_proteins'] = phm_selected_proteins
        st.rerun()

    # Handle add button click
    if add_button:
        # Add currently select protein
        st.session_state['phm_selected_proteins'] = phm_selected_proteins

        # Add from clusters
        for cluster in st.session_state['phm_selected_clusters']:
            to_add = set(cluster) - set(st.session_state['phm_selected_proteins'])
            st.session_state['phm_selected_proteins'].extend(to_add)

        # Add from metadata
        if st.session_state['phm_metadata_criteria'] is not None:
            relevant_ids = st.session_state['metadata_df'][st.session_state['metadata_df'][st.session_state['phm_metadata_criteria']].isin(st.session_state['phm_metadata_values'])].Filename
            to_add = set(relevant_ids) - set(st.session_state['phm_selected_proteins'])
            st.session_state['phm_selected_proteins'].extend(to_add)

        st.rerun()  # Refresh the UI to reflect the updated selection
    #########################
    
    # m/z range slection
    st.text_input("Filter m/z Values", key="phm_selected_mzs", help="Enter m/z values seperated by commas. Ranges can be entered as [125.0-130.0] or as open ended (e.g., [127.0-]). \n \
                                                                If either end of the range falls within the heatmap bin, the bin will be displayed. \n \
                                                                No value will show all m/z values.")
    try:
        if st.session_state.get("phm_selected_mzs"):
            st.session_state["phm_parsed_selected_mzs"] = parse_numerical_input(st.session_state["phm_selected_mzs"])
        else:
            st.session_state["phm_parsed_selected_mzs"] = []
    except Exception as e:
        st.error("Please enter valid m/z values." + str(e))
        st.stop()

    min_count = st.slider("Minimum m/z Count", min_value=0, max_value=max(1,len(st.session_state['phm_selected_proteins'])), step=1, value=1,
                         help="The minimum number of times an m/z value must be present \
                               in the selected proteins to be displayed.")
    min_intensity = st.slider("Minimum Relative Intensity", min_value=0.0, max_value=1.0, step=0.01, value=0.40,
                              help="The minimum relative intensity value to display.")
    
    # Whether to overlay dendrogram
    st.checkbox("Overlay Dendrogram", key="phm_overlay_dendrogram")
    
    metadata_options = ["None", "Dendrogram Cluster"] + list(st.session_state.get("metadata_df").columns)
    st.selectbox("Display Metadata", metadata_options, key="phm_display_metadata")

    if st.session_state['phm_overlay_dendrogram']:
        st.selectbox("Sort Proteins By", ["Dendrogram Clustering"], key="phm_sort_proteins_by", disabled=True)
    else:
        st.selectbox("Sort Proteins By", ["Protein Name", "Dendrogram Clustering", "Metadata"], key="phm_sort_proteins_by")

    # Remove "DB Result - " from the selected proteins -- DB Results are currently deprecated
    selected_proteins = [x.replace("DB Result - ", "") for x in st.session_state['phm_selected_proteins']]

    # Set index to filename
    all_spectra_df = all_spectra_df.set_index("filename")
    all_spectra_df['filename'] = all_spectra_df.index

    # Recompute dendrogram with selected values only
    subset_cluster_dict, local_dendro, local_dendro_ordering = basic_dendrogram(all_spectra_df.loc[selected_proteins, :], display=False)

    if st.session_state["phm_display_metadata"] != "None" and\
        st.session_state["phm_display_metadata"] != "Dendrogram Cluster":
        # Map filenames to metadata
        metadata_df = st.session_state["metadata_df"]
        metadata_df = metadata_df.set_index("Filename")
        all_spectra_df["_metadata"] = metadata_df.loc[all_spectra_df.index, st.session_state["phm_display_metadata"]]

    if st.session_state["phm_sort_proteins_by"] == "Dendrogram Clustering":
        if local_dendro_ordering is None:
            st.error("Unable to order the proteins by dendrogram clustering, please see dendrogram.")

        # get the local_dendro_ordering subset by selected_proteins
        selected_proteins = [x for x in local_dendro_ordering if x in selected_proteins]

    elif st.session_state["phm_sort_proteins_by"] == "Metadata":
        if st.session_state["phm_display_metadata"] == "None":
            st.error("Please select a displayed metadata value to sort proteins by.")
            st.stop()
        
        selected_proteins = all_spectra_df["_metadata"].sort_values().index

    elif st.session_state["phm_sort_proteins_by"] == "Protein Name":
        selected_proteins = sorted(selected_proteins)


    # Select and order relevant columns
    all_spectra_df = all_spectra_df.loc[selected_proteins, :]

    # Add metadata to name if selected
    if st.session_state["phm_display_metadata"] != "None":
        index = all_spectra_df.index.values
        if st.session_state["phm_display_metadata"] == "Dendrogram Cluster":
            # Subtract 1 from cluster number to match the subset_cluster_dict
            index = [f"Cluster {subset_cluster_dict[filename]['cluster']} - {filename}" for filename in index]
            index = [x.replace("Cluster 0", "Unclustered") for x in index]
        else:
            # Append metadata to names if not nan
            index[all_spectra_df["_metadata"].notna().values] = all_spectra_df["_metadata"][all_spectra_df["_metadata"].notna()].values + \
                                                                " - " + index[all_spectra_df["_metadata"].notna().values]
        all_spectra_df.index = index
    
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
    
    # Remove mzs with all nan
    all_spectra_df = all_spectra_df.dropna(how='all', axis='columns')
    all_spectra_df.columns = [_convert_bin_to_mz(x) for x in all_spectra_df.columns]

    # Remove all mzs not in the selected m/z range
    if st.session_state.get("phm_parsed_selected_mzs"):
        if len(st.session_state["phm_parsed_selected_mzs"]) > 0:
            # Query columns that are included in the selected m/z values
            all_mz_bins = all_spectra_df.columns
            mz_filtered_indices = set()
            for mz_bin in all_mz_bins:
                lower_bin, upper_bin = mz_bin[1:-1].split(", ")
                lower_bin = float(lower_bin)
                upper_bin = float(upper_bin)
                for filter in st.session_state["phm_parsed_selected_mzs"]:
                    if isinstance(filter, float):
                        if lower_bin <= filter < upper_bin:
                            mz_filtered_indices.add(mz_bin)
                    elif isinstance(filter, tuple):
                        start, end = filter
                        # If the ranges overlap at all, add the bin
                        if start > upper_bin or end < lower_bin:
                            continue
                        start_in_bin = start >= lower_bin and start < upper_bin
                        end_in_bin = end >= lower_bin and end < upper_bin
                        
                        # Case 1: start is lower than bin, end falls within bin
                        if start_in_bin:
                            mz_filtered_indices.add(mz_bin)
                        # Case 2: start falls within bin, end is higher than bin
                        elif end_in_bin:
                            mz_filtered_indices.add(mz_bin)
                        # Case 3: start is lower than bin, end is higher than bin
                        if start <= lower_bin and end >= upper_bin:
                            mz_filtered_indices.add(mz_bin)

            # Sort selection by average of range
            mz_filtered_indices = sorted(list(mz_filtered_indices), key=lambda x: (float(x[1:-1].split(", ")[0]) + float(x[1:-1].split(", ")[1])) / 2)
            all_spectra_df = all_spectra_df[mz_filtered_indices]
    
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
        
        if local_dendro is not None and st.session_state['phm_overlay_dendrogram']:
            # Map x-axis values to local_dendrogram values
            x = local_dendro.layout.xaxis.tickvals

        heatmap = plotly.express.imshow(all_spectra_df.T.values,    # Transpose so m/zs are rows
                                        x=x,
                                        aspect ='auto', 
                                        width=1500, 
                                        height=dynamic_height,
                                        color_continuous_scale='Bluered',)
        # Update axis text (we do this here otherwise spacing is not even)
        heatmap.update_layout(
            xaxis=dict(title="Protein", ticktext=list(all_spectra_df.index.values), tickvals=list(range(len(all_spectra_df.index))), side='top'),
            yaxis=dict(title="m/z", ticktext=all_spectra_df.columns, tickvals=list(range(len(all_spectra_df.columns)))),
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
            fig.update_xaxes(ticktext=dendro.layout.xaxis.ticktext, tickvals=dendro.layout.xaxis.tickvals, row=1, col=1, tickangle=90)
            fig.update_xaxes(ticktext=dendro.layout.xaxis.ticktext, tickvals=dendro.layout.xaxis.tickvals, row=2, col=1, tickangle=90)
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

        if st.session_state['phm_overlay_dendrogram']:
            merged_fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                              shared_xaxes=True,
                                              row_width=[0.9, 0.1],
                                              vertical_spacing=0.07,)
            
            # Add dendro
            for trace in local_dendro.data:
                merged_fig.add_trace(trace, row=1, col=1)

            # Set heatmap x-ticks to match dendrogram (already sorted in the same order)
            fig.update_xaxes(ticktext=local_dendro.layout.xaxis.ticktext, tickvals=local_dendro.layout.xaxis.tickvals)

            # Add heatmap
            for trace in fig.data:
                merged_fig.add_trace(trace, row=2, col=1)

            # Update heatmap y-axis
            merged_fig.update_yaxes(ticktext=fig.layout.yaxis.ticktext, tickvals=fig.layout.yaxis.tickvals, row=2, col=1)

            # Show x-labels between plots
            merged_fig.update_xaxes(showticklabels=True, row=2, col=1, side='top', 
                                    ticktext=['']*len(local_dendro.layout.xaxis.ticktext),
                                    tickvals=local_dendro.layout.xaxis.tickvals,
                                    ticklen=5)
            
            merged_fig.update_xaxes(showticklabels=True, row=1, col=1,
                                    ticktext=[x[:20] + "..." if len(x)>=20 else x for x in local_dendro.layout.xaxis.ticktext], # Trim text
                                    tickvals=local_dendro.layout.xaxis.tickvals,
                                    ticklen=5)

            # Hide the legend for the dendrogram
            merged_fig.update_layout(showlegend=False)

            # Update height
            merged_fig.update_layout(height=dynamic_height + 400)

            fig = merged_fig

        # Update color axis
        fig.update_coloraxes(cmin=0.0, cmax=1.0, cmid=0.5,colorscale='Bluered')
        
        st.plotly_chart(fig,use_container_width=True)

        # Add a button to download the heatmap
        st.download_button("Download Current Heatmap Data", all_spectra_df.T.to_csv(), "protein_heatmap.csv", help="Download the data used to generate the heatmap.")

all_clusters_dict = None
st.subheader("Spectral Similarity Options")
with st.popover(label='Set Spectral Clustering Parameters'):
    all_clusters_dict, dendro, dendro_ordering = basic_dendrogram()

# Use "query_only_spectra_df" because database spectra may be binned to a different size
draw_protein_heatmap(st.session_state['query_only_spectra_df'], st.session_state['workflow_params']['bin_size'], all_clusters_dict=all_clusters_dict) # , cluster_dict, dendro=dendro, dendro_ordering=dendro_ordering