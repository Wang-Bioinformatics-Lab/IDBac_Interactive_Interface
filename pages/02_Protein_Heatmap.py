import streamlit as st
import numpy as np
import plotly
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from utils import custom_css, parse_numerical_input, _convert_bin_to_mz, _convert_bin_to_mz_tuple
import logging

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

# Init logging
logging.basicConfig(level=logging.INFO)

st.title("Protein m/z Heatmap")
st.markdown("""
    This page allows you to visualize protein MS spectra as a heatmap, facilitating the rapid identification of biomarkers driving protein MS clustering \
            
    To get started, select a set of isolates below and click "Apply/Update Filters". Then, adjust the heatmap parameters as needed.
    """)

def basic_dendrogram(spectrum_df=None, disabled=False, display=True, add_opts=False):
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
    # if 'phm_coloring_threshold' not in st.session_state:
    if add_opts:
        st.slider("Coloring Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01, key="phm_coloring_threshold")
        clustering_options = ["average", "single", "complete", "weighted"]
        if st.session_state['distance_measure'] == "euclidean":
            clustering_options += ['ward', 'median', 'centroid']
        st.selectbox("Clustering Method", clustering_options, key="phm_clustering_method")

    if disabled:
        logging.debug("Dendrogram disabled returning.")
        return None, None, None
    if query_spectra_numpy_data.shape[0] <= 1:
        st.warning("There are not enough spectra to create a dendrogram. \n \
                   Please check number of input spectra and database search results file.")
        logging.debug("Dendrogram disabled (not enough spectra) returning.")
        return None, None, None

    def _dist_fun(x):
        distances = st.session_state['distance_measure'](x)
        if distances.shape[0] != distances.shape[1]:
            raise ValueError("Distance matrix must be square.")
        # Quantize distance matrix to 1e-6 to prevent symetric errors
        distances = np.round(distances, 6)
        dist_matrix = squareform(distances, force='tovector')
        return dist_matrix

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

    logging.debug("Exiting basic_dendrogram().")
    return cluster_dict, dendro, filenames

@st.cache_data(max_entries=20, show_spinner="Counting Replicates...", ttl=3600*12)
def compute_number_of_replicates(bin_counts, replicate_counts, selected_proteins):
    bin_size = st.session_state['workflow_params']['bin_size']
    def __convert_bin_to_mz_tuple(bin_name):
        return _convert_bin_to_mz_tuple(bin_name, bin_size)

    if bin_counts is None or replicate_counts is None:
        st.error("Bin counts and replicate counts are required to generate the heatmap. Rerun this task to generate the heatmap.")
        st.stop()


    bin_counts = bin_counts.loc[:, selected_proteins]
    bin_counts = bin_counts.fillna(0)
    aggregated_bin_counts = bin_counts.copy(deep=True)
    # Set to all zeros
    aggregated_bin_counts.loc[:, :] = 0

    bin_counts['bin_mz_tuple'] = bin_counts.index.map(__convert_bin_to_mz_tuple)
    bin_counts['lb'] = bin_counts['bin_mz_tuple'].apply(lambda x: x[0])
    bin_counts['ub'] = bin_counts['bin_mz_tuple'].apply(lambda x: x[1])

    columns_to_aggregate = aggregated_bin_counts.columns

    if 'db_search_result' in columns_to_aggregate:
        columns_to_aggregate = columns_to_aggregate.drop('db_search_result')
    if 'filename' in columns_to_aggregate:
        columns_to_aggregate = columns_to_aggregate.drop('filename')

    if 'db_search_result' in columns_to_aggregate:
        columns_to_aggregate = columns_to_aggregate.drop('db_search_result')
    if 'filename' in columns_to_aggregate:
        columns_to_aggregate = columns_to_aggregate.drop('filename')

    if st.session_state["phm_replicate_tolerance_mode"] == "m/z":
        for bin_tuple in bin_counts['bin_mz_tuple'].unique():   # This is effectively iterating over rows, which could be slow
            lb = bin_tuple[0] - st.session_state["phm_mz_tolerance"]
            ub = bin_tuple[1] + st.session_state["phm_mz_tolerance"]
            # Columnwise sum for all bins within the tolerance
            mask = ((bin_counts['lb'] >= lb) & (bin_counts['ub'] <= ub))

            aggregated_bin_counts.loc[mask, columns_to_aggregate] = bin_counts.loc[mask, columns_to_aggregate].sum(axis=0).values

    elif st.session_state["phm_replicate_tolerance_mode"] == "ppm":
        for bin_tuple in bin_counts['bin_mz_tuple'].unique():
            lb = bin_tuple[0] - (bin_tuple[0] * st.session_state["phm_ppm_tolerance"] / 1e6)
            ub = bin_tuple[1] + (bin_tuple[1] * st.session_state["phm_ppm_tolerance"] / 1e6)
            # Columnwise sum for all bins within the tolerance
            mask = ((bin_counts['lb'] >= lb) & (bin_counts['ub'] <= ub))
            aggregated_bin_counts.loc[mask, columns_to_aggregate] = bin_counts.loc[mask, columns_to_aggregate].sum(axis=0).values   # values required?
    
    # Divide by the number of replicates per file
    aggregated_bin_counts.loc[:, columns_to_aggregate] = aggregated_bin_counts.loc[:, columns_to_aggregate].div(replicate_counts.loc[columns_to_aggregate, 'replicates'].values, axis=1)
    
    return aggregated_bin_counts

def draw_protein_heatmap(all_spectra_df, bin_counts, replicate_counts, bin_size, all_clusters_dict):     # , cluster_dict=None, dendro=None, dendro_ordering=None'
    """Draw a heatmap for the selected proteins.

    Args:
        all_spectra_df (pandas.DataFrame): The dataframe containing all of the bin spectra (files as rows).
        bin_counts (pandas.DataFrame): The dataframe containing the bin counts (files as columns).
        replicate_counts (pandas.DataFrame): The dataframe containing the replicate counts for each file (files as rows).
        bin_size (float): The size of the bins.
        all_clusters_dict (dict): The dictionary containing the cluster information for each file.

    Returns:
        None
    """
    
    
    st.subheader("Build a Protein Heatmap")
    st.markdown("""
        Select a set of isolates to build your heatmap based on protein dendrogram clusters or metadata categories. Then, use the arrow to move the selection to the right. You can also directly select individual isolates in the right panel. \
            
        After making your selection, click "Apply/Update Filters" to generate the heatmap. \
    """)
    add_filters_1, add_filters_2, add_filters_3 = st.columns([0.45, 0.10, 0.45])
    # Options
    all_options = all_spectra_df['filename'].values

    # Initialze all protins as selected
    if 'phm_selected_proteins' not in st.session_state:
        st.session_state['phm_selected_proteins'] = all_options

    #### Create heatmap by strain ####
    with st.form(key="phm_mz_filters", border=False):
        # Protein Cluster Selection

        if all_clusters_dict is None:
            all_clusters_dict = [None]
            add_filters_1.multiselect("Create heatmap by protein dendrogram clusters", [], disabled=True, key='phm_selected_clusters')
            
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
            
            add_filters_1.multiselect("Create heatmap by protein dendrogram clusters", 
                            list(set(cluster_display_dict.keys())),
                            format_func=cluster_display_dict.get,
                            key='phm_selected_clusters')
            
        # Metadata Selection Options
        metadata_selection_col1, metadata_selection_col2 = add_filters_1.columns([0.5, 0.5])
        if st.session_state.get('metadata_df') is None:
            metadata_selection_col1.selectbox("Create Heatmap by metadata category",
                                              [None],
                                              format_func=lambda x: "No Metadata Available",
                                              key="phm_metadata_criteria",
                                              disabled=True,)
            metadata_selection_col2.multiselect("Metadata Values",
                                                [],
                                                key="phm_metadata_values",
                                                disabled=True)
            # st.session_state['phm_metadata_criteria'] = None
            
        else:
            metadata_selection_col1.selectbox("Create Heatmap by metadata category",
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
                "Create heatmap by strain",
                all_options,
                default=st.session_state['phm_selected_proteins']
            )
        except StreamlitAPIException as _:
            # This is caused by a task change, so we just reset the selected proteins
            phm_selected_proteins = []
            st.session_state['phm_selected_proteins'] = []
            phm_selected_proteins = add_filters_3.multiselect(
                "Create heatmap by strain",
                all_options,
                default=[]
            )

            
        phm_selected_prot_submitted = st.form_submit_button("Apply/Update Filters")

    # Horizontal line
    st.markdown(f"---")
        
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
            logging.debug(f"Adding {len(to_add)} proteins from cluster selection.")

        # Add from metadata
        if st.session_state['phm_metadata_criteria'] is not None:
            relevant_ids = st.session_state['metadata_df'][st.session_state['metadata_df'][st.session_state['phm_metadata_criteria']].isin(st.session_state['phm_metadata_values'])].Filename
            to_add = set(relevant_ids) - set(st.session_state['phm_selected_proteins'])
            st.session_state['phm_selected_proteins'].extend(to_add)
            logging.debug(f"Adding {len(to_add)} proteins from metadata selection.")

        st.rerun()  # Refresh the UI to reflect the updated selection
    #########################
    
    st.subheader("Heatmap Parameters")
    # Add option to filter down by number of replicates
    st.selectbox("Replicate Tolerance Mode", ['ppm', 'm/z'], key="phm_replicate_tolerance_mode")
    if st.session_state["phm_replicate_tolerance_mode"] == "ppm":
        st.session_state["phm_ppm_tolerance"] = st.number_input("ppm tolerance", min_value=0.0, max_value=None, value=1000.0)
    else:
        st.session_state["phm_mz_tolerance"] = None
    if st.session_state["phm_replicate_tolerance_mode"] == "m/z":
        st.session_state["phm_mz_tolerance"] = st.number_input("m/z tolerance", min_value=0.0, max_value=None, value=1.0)
    else:
        st.session_state["phm_mz_tolerance"] = None
    st.slider("Required Presence Percentage", min_value=0.0, max_value=1.0, value=0.5, key="phm_replicate_threshold")

    min_count = st.slider("Minimum m/z Count", min_value=0, max_value=max(1,len(st.session_state['phm_selected_proteins'])), step=1, value=1,
                         help="The minimum number of times an m/z value must be present \
                               in the selected strains to be displayed.")
    min_intensity = st.slider("Minimum Relative Intensity", min_value=0.0, max_value=1.0, step=0.01, value=0.40,
                              help="The minimum relative intensity value to display.")
    
    # m/z range slection
    st.text_input("Search for specific m/z's", key="phm_selected_mzs", help="Enter m/z values seperated by commas. Ranges can be entered as [125.0-130.0] or as open ended (e.g., [127.0-]). \n \
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

    # Whether to overlay dendrogram
    st.checkbox("Overlay Dendrogram", key="phm_overlay_dendrogram")
    
    metadata_options = ["None", "Dendrogram Cluster"]
    metadata_options += st.session_state["metadata_df"].columns.tolist() if st.session_state.get("metadata_df") is not None else []
    if st.session_state['phm_overlay_dendrogram']:
        st.selectbox("Select metadata to be listed as text next to the strain ID", ['Dendrogram Cluster'], key="phm_display_metadata", disabled=True)
    else:
        st.selectbox("Select metadata to be listed as text next to the strain ID", metadata_options, key="phm_display_metadata")

    if st.session_state['phm_overlay_dendrogram']:
        st.selectbox("Sort strains by", ["Dendrogram Clustering"], key="phm_sort_proteins_by", disabled=True)
    else:
        st.selectbox("Sort strains by", ["Strain Name", "Dendrogram Clustering", "Metadata"], key="phm_sort_proteins_by")

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
    

    if bin_counts is None or replicate_counts is None:
        st.error("Bin counts and replicate counts are required to generate the heatmap. Rerun this task to generate the heatmap.")
        st.stop()
    
    # Select relevant columns
    logging.debug(f"Starting with {len(all_spectra_df.index)} proteins for heatmap.")
    logging.debug(f"Starting with  {len(all_spectra_df.columns)} bins for heatmap.")
    all_spectra_df = all_spectra_df.loc[selected_proteins, :]
    logging.debug(f"Got {len(all_spectra_df.index)} proteins for heatmap.")

    # Compute (and cache) replicate count, then filter by threshold
    aggregated_bin_counts = compute_number_of_replicates(bin_counts, replicate_counts, selected_proteins)
    if logging.getLevelName(logging.getLogger().getEffectiveLevel()) == "DEBUG":
        # Download button
        st.download_button("DEBUG: Download Aggregated Bin Counts", aggregated_bin_counts.to_csv(), "aggregated_bin_counts.csv", help="Download the aggregated bin counts.")
    aggregated_bin_counts[aggregated_bin_counts < st.session_state["phm_replicate_threshold"]] = np.nan
    if logging.getLevelName(logging.getLogger().getEffectiveLevel()) == "DEBUG":
        # Download button
        st.download_button("DEBUG: Download Filtered Aggregated Bin Counts", aggregated_bin_counts.to_csv(), "filtered_aggregated_bin_counts.csv", help="Download the filtered aggregated bin counts.")
        st.download_button("DEBUG: Binarized Filtered Aggregated Bin Counts", aggregated_bin_counts.T.notna().to_csv(), "binarized_filtered_aggregated_bin_counts.csv", help="Download the binarized filtered aggregated bin counts.")
    logging.debug(f"Filtered by replicate threshold: {st.session_state['phm_replicate_threshold']}")
    logging.debug(f"Setting {aggregated_bin_counts.isna().sum().sum()} values to nan based on replicate threshold.")
    logging.debug(f"All aggregated bin counts are nan {aggregated_bin_counts.T.notna().values.all()}")
    all_spectra_df = all_spectra_df.where(aggregated_bin_counts.T.notna())
    if logging.getLevelName(logging.getLogger().getEffectiveLevel()) == "DEBUG":
        # Download button
        st.download_button("DEBUG: Download Replicate Count Filtered Protein Spectra", all_spectra_df.to_csv(), "filtered_protein_spectra.csv", help="Download the filtered protein spectra.")

    # Order the proteins
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

    elif st.session_state["phm_sort_proteins_by"] == "Strain Name":
        logging.debug("Sorting proteins by strain name.")
        selected_proteins = sorted(selected_proteins)
    
    logging_start_size = len(all_spectra_df.index)
    all_spectra_df = all_spectra_df.loc[selected_proteins, :]
    logging.debug(f"Removed {logging_start_size - len(all_spectra_df.index)} proteins based on filters.")
    logging.debug(f"Selected {len(all_spectra_df.index)} proteins for heatmap.")

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
    logging.debug(f"Found {len(bin_columns)} bins.")
    all_spectra_df = all_spectra_df.loc[:, bin_columns]
    # Normalize Intensity (Normalize Across Row)
    all_spectra_df = all_spectra_df.div(all_spectra_df.max(axis=1, skipna=True), axis=0)
    if logging.getLevelName(logging.getLogger().getEffectiveLevel()) == "DEBUG":
        maxes = all_spectra_df.max(axis=1, skipna=True)
        logging.debug(f"Max intensity for each protein: {maxes}")

    # Set zeros to nan
    all_spectra_df = all_spectra_df.replace(0, np.nan)
    # Set all values less than min_intensity to nan
    all_spectra_df = all_spectra_df.where(all_spectra_df > min_intensity)
    # Filter bins by count
    debug_inital_bin_cols = len(bin_columns)
    bin_columns = [col for col in bin_columns if all_spectra_df[col].notna().sum() >= min_count]
    logging.debug(f"Filtered {debug_inital_bin_cols - len(bin_columns)} bins based on count threshold.")
    all_spectra_df = all_spectra_df.loc[:, bin_columns]
    
    def __convert_bin_to_mz(bin):
        return _convert_bin_to_mz(bin, bin_size)

    # Remove mzs with all nan
    logging_start_size = len(all_spectra_df.columns)
    all_spectra_df = all_spectra_df.dropna(how='all', axis='columns')
    logging.debug(f"Removed {logging_start_size - len(all_spectra_df.columns)} m/z values with all nan values.")
    all_spectra_df.columns = [__convert_bin_to_mz(x) for x in all_spectra_df.columns]

    # Remove all mzs not in the selected m/z range
    if st.session_state.get("phm_parsed_selected_mzs"):
        logging.debug("Beginning m/z filtering.")
        if len(st.session_state["phm_parsed_selected_mzs"]) > 0:
            # Query columns that are included in the selected m/z values
            all_mz_bins = all_spectra_df.columns
            mz_filtered_indices = set()
            for mz_bin in all_mz_bins:
                lower_bin, upper_bin = mz_bin[1:-1].split(", ")
                lower_bin = float(lower_bin)
                upper_bin = float(upper_bin)
                for filter in st.session_state["phm_parsed_selected_mzs"]:
                    logging.debug(f"Filtering based on mz: {filter}")
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
    
    logging.debug(f"Drawing heatmap with {len(all_spectra_df.columns)} m/z values and {len(all_spectra_df.index)} proteins.")
    if logging.getLevelName(logging.getLogger().getEffectiveLevel()) == "DEBUG":
        # Createa a download button for the filtered data
        st.download_button("DEBUG: Download Filtered Data", all_spectra_df.to_csv(), "filtered_protein_heatmap.csv", help="Download the data used to generate the heatmap.")

    if len(all_spectra_df.columns) != 0:
        # Note: We transpose the dataframe so that the proteins are on the x-axis
        st.markdown("Common m/z values between selected proteins and their relative intensities.")
        # Draw Heatmap
        dynamic_height = max(500, len(all_spectra_df.columns) * 24) # Dyanmic height based on number of m/z values
        
        # If we're suppled a dendrogram, use it to reorder the heatmap
        x = None        
        if local_dendro is not None and st.session_state['phm_overlay_dendrogram']:
            # Map x-axis values to local_dendrogram values
            x = local_dendro.layout.xaxis.tickvals

        heatmap = plotly.express.imshow(all_spectra_df.T.values,    # Transpose so m/zs are rows
                                        x=x,
                                        aspect ='auto', 
                                        width=1500, 
                                        height=dynamic_height,
                                        color_continuous_scale='Bluered')
        

        # Update axis text (we do this here otherwise spacing is not even)
        heatmap.update_layout(
            xaxis=dict(title="Protein", ticktext=list(all_spectra_df.index.values), tickvals=list(range(len(all_spectra_df.index))), side='top'),
            yaxis=dict(title="m/z", ticktext=all_spectra_df.columns, tickvals=list(range(len(all_spectra_df.columns)))),
            margin=dict(t=5, pad=0),
        )
        
        heatmap.update_coloraxes(cmin=0.0, cmax=1.0, cmid=0.5)
        
        fig = heatmap

        fig.update_layout(showlegend=False,
                    coloraxis_colorbar=dict(title="Relative Intensity", 
                                            len=min(500, dynamic_height), 
                                            lenmode="pixels", 
                                            y=0.75)
                                        )

        if st.session_state['phm_overlay_dendrogram']:
            logging.debug("Overlaying dendrogram.")
            # Height with added room
            dynamic_height = dynamic_height + 400
            space_required_for_labels = 125
            merged_fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                              shared_xaxes=True,
                                              row_width=[0.9, 0.1],
                                              vertical_spacing=space_required_for_labels/dynamic_height,)
            
            # Add dendro
            for trace in local_dendro.data:
                merged_fig.add_trace(trace, row=1, col=1)

            # Set heatmap x-ticks to match dendrogram (already sorted in the same order)
            fig.update_xaxes(ticktext=local_dendro.layout.xaxis.ticktext, tickvals=local_dendro.layout.xaxis.tickvals)

            # Add heatmap
            for trace in fig.data:
                merged_fig.add_trace(trace, row=2, col=1)

            # Update heatmap y-axis
            merged_fig.update_yaxes(ticktext=fig.layout.yaxis.ticktext, tickvals=fig.layout.yaxis.tickvals, row=2, col=1,
                                    autorange='reversed')

            # Show x-labels between plots
            merged_fig.update_xaxes(showticklabels=True, row=2, col=1, side='top', 
                                    ticktext=['']*len(local_dendro.layout.xaxis.ticktext),
                                    tickvals=local_dendro.layout.xaxis.tickvals,
                                    ticklen=5,
                                    tickangle=90)
            
            merged_fig.update_xaxes(showticklabels=True, row=1, col=1,
                                    ticktext=[x[:20] + "..." if len(x)>=20 else x for x in local_dendro.layout.xaxis.ticktext], # Trim text
                                    tickvals=local_dendro.layout.xaxis.tickvals,
                                    ticklen=5,
                                    tickangle=90)

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
        logging.debug("Exiting draw_protein_heatmap().")

def check_preconditions():
    if st.session_state.get('heatmap_binned_spectra') is None:
        st.error("No protein spectra were found for this task. Please check the task parameters.")
        st.stop()

check_preconditions()

all_clusters_dict = None

with st.popover(label='Reference protein dendrogram clusters'):
    all_clusters_dict, dendro, dendro_ordering = basic_dendrogram(add_opts=True)

# Use "heatmap_binned_spectra" because query/database spectra may be binned to a different size
draw_protein_heatmap(st.session_state['heatmap_binned_spectra'],
                     st.session_state["bin_counts_df"],
                     st.session_state['replicate_count_df'],
                     st.session_state['workflow_params']['bin_size'],
                     all_clusters_dict=all_clusters_dict) 