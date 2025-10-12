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
from utils import convert_to_mzml
from typing import Dict, List, Tuple
import io

#####
# A note abote streamlit session states:
# All session states related to this page begin with "sma_" to reduce the 
# chance of collisions with other pages. 
#####

# Set Page Configuration
st.set_page_config(page_title="IDBac - Small Molecule Association", page_icon="assets/idbac_logo_square.png", layout="wide", initial_sidebar_state="collapsed", menu_items=None)
custom_css()

def basic_dendrogram(disabled=False):
    """
    This function generates a basic dendrogram for the small molecule association page. 
    """
    st.slider("Coloring Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01, key="sma_coloring_threshold")
    clustering_options = ["average", "single", "complete", "weighted"]
    if st.session_state['distance_measure'] == "euclidean":
        clustering_options += ['ward', 'median', 'centroid']
    st.selectbox("Clustering Method", clustering_options, key="sma_clustering_method")

    if disabled:
        return None, None
    if st.session_state['query_spectra_numpy_data'].shape[0] <= 1:
        st.warning("There are not enough spectra to create a dendrogram. \n \
                   Please check number of input spectra and database search results file.")
        return None, None

    def _dist_fun(x):
        distances = st.session_state['distance_measure'](x)
        if distances.shape[0] != distances.shape[1]:
            raise ValueError("Distance matrix must be square.")
        # Quantize distance matrix to 1e-6 to prevent symetric errors
        distances = np.round(distances, 6)
        dist_matrix = squareform(distances, force='tovector')
        
        return dist_matrix

    dendro = ff.create_dendrogram(st.session_state['query_spectra_numpy_data'],
                                orientation='bottom',
                                labels=st.session_state['query_only_spectra_df'].filename.values, # We will use the labels as a unique identifier
                                distfun=_dist_fun,
                                linkagefun=lambda x: linkage(x, method=st.session_state["sma_clustering_method"],),
                                color_threshold=st.session_state["sma_coloring_threshold"])
       
    st.plotly_chart(dendro, use_container_width=True)
    # Sadly the only way to get the actual clusters (to plot the graph) is to recompute the linkage with scipy
    # (TODO: Just used scipy to plot it)
    dist_matrix = _dist_fun(st.session_state['query_spectra_numpy_data'])
    linkage_matrix = linkage(dist_matrix,
                            method=st.session_state["sma_clustering_method"])
    sch_dendro = dendrogram(linkage_matrix,
                            labels=st.session_state['query_only_spectra_df'].filename.values,
                            no_plot=True,
                            color_threshold=st.session_state["sma_coloring_threshold"])
    
    cluster_dict = {}
    filenames    = sch_dendro['ivl']
    colors_list  = sch_dendro['leaves_color_list']

    prev_color = colors_list[0]
    curr_cluster = 1
    for color, filename in zip(colors_list, filenames):
        if color != prev_color and color != 'C0':
            curr_cluster += 1
            prev_color = color
        elif color == 'C0':
            # To handle the edge case where to clusters of the same color are split by an unclustered protein
            prev_color = None
        cluster_dict[filename] = {
                                  'cluster': curr_cluster if color != 'C0' else 0,  # If the color is C0, it's unclustered
                                  'color': int(color[1:])
                                  }

    # Add 1 if value >= 1 to reserve 1 for small molecules, 0 for unclustered
    for filename in cluster_dict:
        if cluster_dict[filename]['color'] >= 1:
            cluster_dict[filename]['color'] += 1
    
    return cluster_dict, dendro

def get_small_molecule_dict():
    if st.session_state['task_id'].startswith("DEV-"):
        base_url = "http://ucr-lemon.duckdns.org:4000"
        task_id = st.session_state['task_id'].replace("DEV-", "")
    elif st.session_state['task_id'].startswith('BETA-'):
        base_url = "https://beta.gnps2.org"
        task_id = st.session_state['task_id'].replace("BETA-", "")
    else:
        base_url = "https://gnps2.org"
        task_id = st.session_state['task_id']
    url = f"{base_url}/resultfile?task={task_id}&file=nf_output/small_molecule/summary.json"
    
    response = requests.get(url, timeout=(120,120))
    if response.status_code != 200:
        st.error(f"Error loading small molecule summary file: {response.status_code}")
        st.stop()
    
    response_dict = json.loads(response.content)
    
    output_dict = {}
    
    all_filenames = [d['filename'] for d in response_dict]
    
    # Map scans to their filenames
    for scan in response_dict:
        if output_dict.get(scan['filename']) is None:
            output_dict[scan['filename']] = [scan]
        else: 
            output_dict[scan['filename']].append(scan)
            
    # For each filename, combine the m/z and intensity arrays
    for filename, scan_list in output_dict.items():
        mz_intensity_dict = {}
        mz_frequency_dict = {}  # For each m/z, what percent of scans is it in?
        for scan in scan_list:
            mz_array        = scan['m/z array']
            intensity_array = scan['intensity array']
            
            for mz, intensity in zip(mz_array, intensity_array):
                _mz = np.round(float(mz), 0)
                if _mz in mz_intensity_dict:
                    mz_intensity_dict[_mz].append(float(intensity))
                else:
                    mz_intensity_dict[_mz] = [float(intensity)]
            
        for mz, intensities in mz_intensity_dict.items():
            mz_intensity_dict[mz] = sum(intensities) / len(intensities)
            mz_frequency_dict[mz] = len(intensities) / len(scan_list)
                
        mz_array           = sorted(list(mz_intensity_dict.keys()))
        intensity_array    = [mz_intensity_dict[mz] for mz in mz_array]
        mz_frequency_array = [mz_frequency_dict[mz] for mz in mz_array]
        
        output_dict[filename] = {
            'm/z array': mz_array,
            'intensity array': intensity_array,
            'frequency array': mz_frequency_array
        }
    
    return output_dict

def filter_small_molecule_dict(small_molecule_dict)->Dict[str, Dict[str, List[float]]]:
    """ Applies intensity and frequency filters to the small molecule dictionary. Note that the replicate frequency is applied
    independently of the intensity filter.
    
    Session State Parameters:
    sma_relative_intensity_threshold: Relative intensity threshold required to pass filtering
    sma_replicate_frequency_threshold: Frequency threshold required to pass filtering
    sma_parsed_selected_mzs: List of m/z values to filter by

    Parameters:
    small_molecule_dict (dict): A dictionary of small molecule data

    Returns:
    dict: A filtered dictionary of small molecule data {filename: {m/z array: [float], intensity array: [float], frequency array: [float]}}
    """
    
    output = {}
    
    for k, d in small_molecule_dict.items():
        mz_array        = [float(x) for x in d['m/z array']]
        intensity_array = [float(x) for x in d['intensity array']]
        frequency_array = [float(x) for x in d['frequency array']]
        
        # Get indices where intensity is above threshold
        indices = [i for i, (intensity, frequency) in enumerate(zip(intensity_array, frequency_array)) if intensity > st.session_state.get("sma_relative_intensity_threshold", 0.1) and frequency > st.session_state.get("sma_replicate_frequency_threshold", 0.7)]
        # Get indices where m/z is within tolerance
        if len(st.session_state.get("sma_parsed_selected_mzs")) > 0:
            mz_filtered_indices = set()
            
            for i, mz in enumerate(mz_array):
                for filter in st.session_state.get("sma_parsed_selected_mzs"):
                    if isinstance(filter, float):
                        if abs(mz - filter) <= st.session_state.get("sma_mz_tolerance", 0.1):
                            mz_filtered_indices.add(i)
                    else:
                        start, end = filter
                        if start <= mz <= end:
                            mz_filtered_indices.add(i)
                            
            indices = list(set(indices).intersection(mz_filtered_indices))
        
        # Filter mz_array and intensity_array
        mz_array = [mz_array[i] for i in indices]
        intensity_array = [intensity_array[i] for i in indices]
        
        d['m/z array'] = mz_array
        d['intensity array'] = intensity_array
    
        output[k] = d
    return output

class ShapeMap():
    def __init__(self):
        self.shape_map = {0:'circle', 1:'box', 2:'ellipse', 3:'diamond', 4:'dot', 5:'star', 6:'triangle', 7:'triangleDown', 8:'square'}
        
    def get_shape(self, index):
        """This function returns a shape for a node based on the index. If the index is less than zero, a ValueError is raised. If 
        the index is greater than 8, the index is modded by 8 and a warning is returned.

        Parameters:
        index (int): The index of the shape to return
        
        Returns:
        str: The shape of the node
        bool: A warning if the index was out of bounds
        
        Raises:
        ValueError: If the index is less than 0
        """
        warning = False # If the index is out of bounds, we'll reuse shapes and return an error
        if index < 0:
            raise ValueError("Index must be greater than 0")
        if index > 8: 
            warning = True
            index = index % 8
        return self.shape_map[index], warning
    

def generate_network(cluster_dict:dict=None, height=1000, width=600)->Tuple[Dict[str, Dict[str, List[float]]], nx.Graph]:
    """ This function generates a network graph of the small molecule data. It uses the pyvis library to create an interactive graph.
    
    Parameters:
    cluster_dict (dict): A dictionary of small molecule data {filename: {m/z array: [float], intensity array: [float], frequency array: [float]}}
    height (int): The height of the graph in pixels
    width (int): The width of the graph in pixels

    Returns:
    dict: A dictionary of small molecule data {filename: {m/z array: [float], intensity array: [float], frequency array: [float]}}
    networkx.Graph: A networkx graph object
    """
    # TODO: Right now we don't integrate all_spectra_df which means there could be nodes that aren't truly in the network
    if st.session_state.get("metadata_df") is None:
        st.error("Please upload a metadata file first.")
        st.stop()
    
    df = st.session_state["metadata_df"]
    if 'Small molecule file name' not in df.columns:
        st.error("Please upload a metadata file with a 'Small molecule file name' column.")
        st.stop()
    
    nx_G = nx.Graph()
    cmap = plt.get_cmap(st.session_state.get("sma_node_color_map"))
    shape_map = ShapeMap()
    
    # Add nodes from df['Filename'] and df['Small molecule file name']
    all_filenames = df.loc[~ df['Filename'].isna()].Filename.tolist()
    all_small_molecule_filenames = df.loc[~ df['Small molecule file name'].isna()]['Small molecule file name'].tolist()
    for filename in all_filenames:
        nx_G.add_node(filename, 
                      title=filename, color=colors.to_hex(cmap(0)), 
                      type="Protein",
                      shape=shape_map.get_shape(0)[0])
    # for small_molecule_filename in all_small_molecule_filenames:
    #     graph.add_node(small_molecule_filename, title=small_molecule_filename)
        
    small_mol_dict = filter_small_molecule_dict(get_small_molecule_dict())
    all_mzs = [small_mol_dict.get('m/z array', []) for small_mol_dict in small_mol_dict.values()]
    all_mzs = [mz for sublist in all_mzs for mz in sublist] # Flatten
    mz_value_counts = pd.Series(all_mzs).value_counts()
    all_mzs = np.unique(all_mzs)
    for mz in all_mzs:
        small_molecule_shape = shape_map.get_shape(1)[0]
        if st.session_state.get("sma_node_shapes") == "Circular":
            small_molecule_shape = shape_map.get_shape(0)[0]
            
        
        nx_G.add_node(f'{int(mz)}',                     # Displayed on node
                      title=f'{int(mz)} m/z',           # Displayed on hover
                      color=colors.to_hex(cmap(1)), 
                      type="Small Molecule",
                      shape=small_molecule_shape)
    
    missing_summaries = []

    # Add edges from df['Filename] to m/z's associated with df['Small molecule file name']
    for index, row in df.iterrows():
        if not pd.isna(row['Filename']) and not pd.isna(row['Small molecule file name']):
            if row['Small molecule file name'] not in small_mol_dict:
                # This happens because users may have uploaded a metadata file that 
                # references small molecules that are not in the summary.json file
                # (likely because it was not uploaded). These should raise a warning
                # We'll just skip them here.
                missing_summaries.append(row['Small molecule file name'])
                continue
            
            for mz in small_mol_dict[row['Small molecule file name']]['m/z array']:
                weight = (1/mz_value_counts.get(mz, 1))+0.5      # Weight inversely propotional to frequency per file
                nx_G.add_edge(row['Filename'], f'{int(mz)}', weight=weight) 
                
    if len(missing_summaries) > 0:
        st.warning(f"The following small molecules files were referenced in the metadata, but were not \
                    found in the small molecule data. Please check and ensure that these files were included \
                    in the workflow: {missing_summaries}")
    
    # Calculate Coloring
    if st.session_state.get("sma_node_coloring") == "Network Community Detection" or \
        st.session_state.get("sma_node_shapes") == "Network Community Detection":
        community_fn_mapping = {"Louvain": nx.algorithms.community.greedy_modularity_communities,
                                "Greedy Modularity": nx.algorithms.community.greedy_modularity_communities}
        communities = community_fn_mapping[st.session_state.get("sma_cluster_method")](nx_G)
        
        if st.session_state.get("sma_node_coloring") == "Network Community Detection":
            node_color_map = {}
        if st.session_state.get("sma_node_shapes") == "Network Community Detection":
            node_shape_map = {}
            warning_flag_set = False
        
        # Assign color/shape based on community
        for i, community in enumerate(communities):
            for node in community:
                if st.session_state.get("sma_node_coloring") == "Network Community Detection":
                    node_color_map[node] = colors.to_hex(cmap(i))
                if st.session_state.get("sma_node_shapes") == "Network Community Detection":
                    color, warn = shape_map.get_shape(i)
                    node_shape_map[node] = color
                    if warn:
                        warning_flag_set = True
                        
        if st.session_state.get("sma_node_coloring") == "Network Community Detection":
            nx.set_node_attributes(nx_G, node_color_map, 'color')
        if st.session_state.get("sma_node_shapes") == "Network Community Detection":
            nx.set_node_attributes(nx_G, node_shape_map, 'shape')
            if warning_flag_set:
                st.warning("More than 8 communities detected. Some shapes will be reused.")
                
    if st.session_state.get("sma_node_coloring") == "Protein Dendrogram Clusters" or \
        st.session_state.get("sma_node_shapes") == "Protein Dendrogram Clusters":
            communities = cluster_dict
            color_warning_flag_set = False
            warning_flag_set = False
            
            if st.session_state.get("sma_node_coloring") == "Protein Dendrogram Clusters":
                node_color_map = {}
                
            if st.session_state.get("sma_node_shapes") == "Protein Dendrogram Clusters":
                node_shape_map = {}
                
            # Assign color/shape based on community
            for node, data in communities.items():
                c_id = data['color']
                if st.session_state.get("sma_node_coloring") == "Protein Dendrogram Clusters":
                    if c_id != 0:
                        node_color_map[node] = colors.to_hex(cmap(c_id))
                    else:
                        node_color_map[node] = "#ffffff"
                        color_warning_flag_set = True
                if st.session_state.get("sma_node_shapes") == "Protein Dendrogram Clusters":
                    shape, warn = shape_map.get_shape(c_id)
                    node_shape_map[node] = shape
                    if warn:
                        warning_flag_set = True
                        
            if st.session_state.get("sma_node_coloring") == "Protein Dendrogram Clusters":
                nx.set_node_attributes(nx_G, node_color_map, 'color')
            if st.session_state.get("sma_node_shapes") == "Protein Dendrogram Clusters":
                nx.set_node_attributes(nx_G, node_shape_map, 'shape')
                if warning_flag_set:
                    st.warning("More than 8 clusters detected. Some shapes will be reused.")
                    
            if color_warning_flag_set:
                st.warning("Unclustered nodes are white.")
                     
    # Remove anything that isn't in the selected clusters
    # Must happen before layout
    if st.session_state['sma_selected_proteins'] != []:
        for node, node_type in list(nx_G.nodes.data('type')):
            if node_type == "Protein":
                if node not in st.session_state['sma_selected_proteins']:
                    print(f"DEBUG: Removing node {node} of type {node_type} because it is not in the selected proteins.", flush=True)
                    nx_G.remove_node(node)
       
        # Remove singleton m/z values
        for node, node_type in list(nx_G.nodes.data('type')):
            if node_type == "Small Molecule":
                if nx_G.degree(node) == 0:
                    nx_G.remove_node(node)
           
    # Perform Layout
    pos=None
    pyvis_options.Layout(randomSeed=42)
    if st.session_state.get("sma_network_layout") != 'Default':  # If default, we'll use the defauly pyvis layout
        layout_fn_mapping = {"Spring": nx.drawing.layout.spring_layout,
                             "Circular": nx.drawing.layout.circular_layout,
                             "Spectral": nx.drawing.layout.spectral_layout,
                             "Kamada-Kawai": nx.drawing.layout.kamada_kawai_layout,
                             "Bipartite Layout": nx.drawing.layout.bipartite_layout}        
        layout_default_params = {"Spring": {"k": 2/np.sqrt(len(nx_G)), "seed":42, "iterations": 30, "scale":1},   # k defaults to 1/sqrt(n)
                                 "Circular": {},
                                 "Spectral": {},
                                 "Kamada-Kawai": {}}

        layout_scale = st.session_state.get("sma_layout_scale", 1.0)

        if st.session_state['sma_spectral_similarity_layout'] == 'Yes':
            # Add edges between protein nodes based on clustering of proteins.
            added_edges = []
            # "Transpose Dict"
            spectral_communities = {}

            for node, metadata in cluster_dict.items():
                cluster = metadata['cluster']
                if cluster != 0:
                    if cluster in spectral_communities:
                        spectral_communities[cluster].append(node)
                    else:
                        spectral_communities[cluster] = [node]
                    
            for cluster, nodes in spectral_communities.items():
                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i+1:]:
                        if (node1, node2) not in added_edges and \
                            node1 in nx_G.nodes and \
                            node2 in nx_G.nodes:
                                nx_G.add_edge(node1, node2, weight=0.2)
                                added_edges.append((node1, node2))
        
        # Apply layout
        layout_fn = layout_fn_mapping[st.session_state.get("sma_network_layout")]
        layout_params = layout_default_params[st.session_state.get("sma_network_layout")]
        layout_params['scale'] = layout_scale  # Update scale parameter
        pos = layout_fn(nx_G, **layout_params)
        
        if st.session_state['sma_spectral_similarity_layout'] == 'Yes':
            # Remove edges between protein nodes, we don't want them displayed
            for edge in added_edges:
                if edge in nx_G.edges:
                    nx_G.remove_edge(*edge)
    
    # Convert to PyVis Graph
    physics = (st.session_state.get("sma_physics", "No") == "Yes")
    graph = net.Network(height=f'{height}px', width='100%')
    graph.options.edges.smooth.enabled = False
    graph.toggle_physics(physics)   # For many nodes, things won't render if this is True, independent of how it's set per node/edge
    # Get max and min x 
    x_pos = [pos.get(node)[0] for node in nx_G.nodes]
    y_pos = [pos.get(node)[1] for node in nx_G.nodes]

    for node in nx_G.nodes:
        if pos is not None:
            x_pos = pos.get(node)[0]
            y_pos = pos.get(node)[1]

            # Adjust x,y by width and height
            x_pos = x_pos * width
            y_pos = y_pos * height

        else:
            x_pos = None
            y_pos = None
        graph.add_node(node,
                       title=nx_G.nodes[node].get('title', ''),
                       color=nx_G.nodes[node].get('color', '#000000'),  # Black to spot errors
                       x=x_pos,
                       y=y_pos,
                       shape=nx_G.nodes[node].get('shape'),
                       )
    for edge in nx_G.edges:
        graph.add_edge(edge[0], edge[1])
        
    # Generate HTML code for the graph
    html_graph = graph.generate_html()
    
    # html_graph = update_graph_html(html_graph)
    # print(html_graph)
    
    components.html(html_graph, height=height)

    return small_mol_dict, nx_G
    
def make_heatmap():
    """
    Make a heatmap that shows which m/z values are associated with each protein file with which intensity
    """
    
    mapping = st.session_state["metadata_df"]
          
    small_mol_dict = filter_small_molecule_dict(get_small_molecule_dict())
    
    all_mzs = [small_mol_dict.get('m/z array', []) for small_mol_dict in small_mol_dict.values()]
    all_mzs = [mz for sublist in all_mzs for mz in sublist] # Flatten
    all_mzs = np.sort(np.unique(all_mzs))
       
    heatmap = np.ones((len(st.session_state["sma_selected_proteins"]), len(all_mzs))) * np.nan
    
    df = pd.DataFrame(heatmap, columns=all_mzs, index=st.session_state["sma_selected_proteins"])
    
    for filename in st.session_state["sma_selected_proteins"]:
        relevant_mapping = mapping[mapping['Filename'] == filename]
        all_small_molecule_filenames = relevant_mapping['Small molecule file name'].tolist()
        
        for small_molecule_filename in all_small_molecule_filenames:
            mz_array = small_mol_dict[small_molecule_filename]['m/z array']
            intensity_array = small_mol_dict[small_molecule_filename]['intensity array']
            
            for mz, intensity in zip(mz_array, intensity_array):
                if intensity > st.session_state.get("sma_relative_intensity_threshold", 0.1):
                    df.at[filename, mz] = intensity
    
    # Remove columns that don't meet the frequency threshold
    if st.session_state.get("sma_min_mz_frequency") > 1:
        df = df.loc[:, (df.count() >= st.session_state.get("sma_min_mz_frequency"))]
        
    # Remove cols with all nans
    df = df.loc[:, (df.notna()).any(axis=0)]
    
    if len(df) > 1:
        st.selectbox("Overlay Dendrogram on Heatmap", ["Yes", "No"], key="sma_show_dendrogram")
    else:
        st.selectbox("Overlay Dendrogram on Heatmap", ["No"], key="sma_show_dendrogram", disabled=True)
    
    if len(df) != 0:
        # Note: We transpose the dataframe so that the proteins are on the x-axis
        st.markdown("Common m/z values between selected strains and their intensities. Note: The graph filters are applied here.")
        # Draw Heatmap
        dynamic_height = max(500, len(df.columns) * 24) # Dyanmic height based on number of m/z values
        
        # If we're suppled a dendrogram, use it to reorder the heatmap
        x = None
        if st.session_state['sma_show_dendrogram'] == 'Yes':
            # Remove any rows where the filename is not currently selected
            all_filenames = st.session_state['query_only_spectra_df'].filename.values
            all_data      = st.session_state['query_spectra_numpy_data']
            
            # Get the indices of the selected proteins
            selected_indices = [i for i, filename in enumerate(all_filenames) if filename in st.session_state["sma_selected_proteins"]]
            # Get the data for the selected proteins
            numpy_data = all_data[selected_indices]
            
            # Unfortunately, we have to recalculate the dendrogram, because things may cluster differently 
            # depending on the selected proteins.
            # Note though, that we share parameters with the above dendrogram.
            dendro = ff.create_dendrogram(numpy_data,
                                orientation='bottom',
                                labels=st.session_state["sma_selected_proteins"],
                                distfun=st.session_state['distance_measure'],
                                linkagefun=lambda x: linkage(x, method=st.session_state["sma_clustering_method"],),
                                color_threshold=st.session_state["sma_coloring_threshold"])
            
            # Reorder the dataframe based on the dendrogram
            reordered_df = df.reindex(index=dendro.layout.xaxis.ticktext)
            reordered_df = reordered_df.reindex(columns=dendro.layout.yaxis.ticktext)
            df = reordered_df
            # Also us the X values from the dendrogram
            x = dendro.layout.xaxis.tickvals
        
        heatmap = plotly.express.imshow(df.values.T,
                                        x=x,
                                        aspect ='auto', 
                                        width=1500, 
                                        height=dynamic_height,
                                        color_continuous_scale='Bluered',)
        # Update axis text (we do this here otherwise spacing is not even)
        heatmap.update_layout(
            xaxis=dict(title="Protein", ticktext=list(df.index.values), tickvals=list(range(len(df.index))), side='top'),
            yaxis=dict(title="m/z", ticktext=[str(x) for x in df.columns], tickvals=list(range(len(df.columns)))),
            margin=dict(t=5, pad=0),
        )
        
        heatmap.update_coloraxes(cmin=0.0, cmax=1.0, cmid=0.5)
        
        if st.session_state['sma_show_dendrogram'] == 'Yes':
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
        
        # Add a button to download the heatmap
        st.download_button("Download Current Heatmap Data", df.T.to_csv(), "small_molecule_heatmap.csv", help="Download the data used to generate the heatmap.")

if st.session_state["metadata_df"] is None:
    st.error("Please upload a metadata file first.")
    st.stop()

st.title("Metabolite Association Network Visualization")
st.markdown("""
    This page allows you to visualize the associations between proteins and small molecules based on MS data. One analysis approach is to assess patterns of specialized metabolite production as a function of phylogenetic relatedness, 
    which can provide a means to discriminate between strains at the subspecies level ([Jensen, 2010](https://doi.org/10.1007/s10295-009-0683-z)). \
            
    To get started: set a relative intensity and replicate frequency threshold, and select a network layout. You can also choose to color nodes by protein/small molecule type, network community detection, or protein dendrogram clusters.
    The network is downloadable as a .graphml file for easy import into Cytoscape or other network visualization software. 
    """)

with st.expander("Small Molecule Filters", expanded=True):
    # Add a slider for the relative intensity threshold
    st.slider("Relative Intensity Threshold", min_value=0.00, max_value=1.0, value=0.15, step=0.01, 
            key="sma_relative_intensity_threshold")
    st.slider("Replicate Frequency Threshold", min_value=0.00, max_value=1.0, value=0.70, step=0.05, 
            key="sma_replicate_frequency_threshold", help="Only show m/z values that occur in at least this percentage of replicates.")

    # Add text input to select certain m/z's (comma-seperated)
    mz_col1, mz_col2 = st.columns([3, 1])
    mz_col1.text_input("Search for specific m/z's", key="sma_selected_mzs", value="[200-2000]", help="Enter m/z values seperated by commas. Ranges can be entered as [125.0-130.0] or as open ended (e.g., [127.0-]). No value will show all m/z values.")
    mz_col2.number_input("Tolerance (m/z)", key="sma_mz_tolerance", value=0.1, help="Tolerance for the selected m/z values. Does not apply to ranges.")
    try:
        if st.session_state.get("sma_selected_mzs"):
            st.session_state["sma_parsed_selected_mzs"] = parse_numerical_input(st.session_state["sma_selected_mzs"])
        else:
            st.session_state["sma_parsed_selected_mzs"] = []
    except:
        st.error("Please enter valid m/z values.")
        st.stop()

with st.expander("Metabolite Association Network Options", expanded=True):
#### Network Layout Options
    st.selectbox("Network Layout", ["Spring", "Kamada-Kawai", "Circular", "Spectral",], key="sma_network_layout")   # "Default" is another, unused option here
    if st.session_state.get("sma_network_layout") == 'Default':
        # Enable physics by default because it helps with the layout
        st.selectbox("Physics", ["Yes", "No"], key="sma_physics")
    else:
        st.session_state["sma_physics"] = "No"
        # We can only do this using our custom layout, so we'll have to disable it for the Default layout
        options = ['Yes', 'No']
        disabled=False
        if st.session_state.get('spectra_df') is None or \
            len(st.session_state['spectra_df']) == 0:
                options = ['No']
                disabled = True
        st.selectbox("Incorporate Protein Spectral Similarity into Node Layout", options, key="sma_spectral_similarity_layout", disabled=disabled, help="Proteins with similar MS profiles will be placed closer together in the network layout.")

    #### Network Coloring Option 
    st.selectbox("Node Coloring", ["Protein/Small Molecule", "Network Community Detection", "Protein Dendrogram Clusters"], key="sma_node_coloring")
    st.selectbox("Node Color Map", ['tab10', 'tab20', 'tab20b','tab20c',
                                    'Pastel1', 'Pastel2', 'Paired', 
                                    'Accent', 'Dark2', 'Set1', 'Set2', 'Set3'], 
                                    key="sma_node_color_map", help='See available color maps: \
                                    https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative')
    st.selectbox("Node Shapes", ["Circular","Protein/Small Molecule", "Network Community Detection", "Protein Dendrogram Clusters"], key="sma_node_shapes")

    #### Scale
    # Add a slider to adjust the layout scale
    st.slider("Adjust Layout Scale", min_value=0.5, max_value=5.0, value=1.0, step=0.1, key="sma_layout_scale")

    #### Network Community Detection Options
    # Options for Network Community Detection Node Properties
    if st.session_state.get("sma_node_coloring") == "Network Community Detection" or \
    st.session_state.get("sma_node_shapes") == "Network Community Detection":
        st.subheader("Network Community Detection Options")
        st.selectbox("Cluster Method", ["Louvain", "Greedy Modularity",], key="sma_cluster_method")
        if st.session_state.get("sma_cluster_method") == "Louvain":
            st.text("https://doi.org/10.1088/1742-5468/2008/10/P10008")
        elif st.session_state.get("sma_cluster_method") == "Greedy Modularity":
            st.text("https://doi.org/10.1103/PhysRevE.70.066111")
    # Options for Spectral Similarty Node Properties
    cluster_dict = None
    barebones_dendro = None

with st.expander("Visualize Small Molecule Data", expanded=True):
    with st.popover(label='Reference Protein Dendrogram Clusters'):
        if st.session_state.get("query_spectra_numpy_data") is not None:
            cluster_dict, _ = basic_dendrogram()
        else:
            cluster_dict, _ = basic_dendrogram(disabled=True) 
                
    # Options to show only certain proteins/clusters
    add_filters_1, add_filters_2, add_filters_3 = st.columns([0.46, 0.08, 0.46])
        # First column is a selectbox of protein names
        # Second is a selectbox of cluster names
        # Third is an import button that adds the values of the clusters to the protein name selection

    with st.form(key="sma_mz_filters", border=False):
        # Protein Cluster Selection
        disabled=False
        if cluster_dict is None:
            cluster_dict = [None]
            add_filters_1.multiselect("Populate by protein dendrogram clusters", [], disabled=True, key='sma_selected_clusters')
            
        else:
            inverted_cluster_dict = {}
            for filename, metadata in cluster_dict.items():
                cluster_id = metadata['cluster']
                if inverted_cluster_dict.get(cluster_id) is None:
                    inverted_cluster_dict[cluster_id] = [filename]
                else:
                    inverted_cluster_dict[cluster_id].append(filename)
            cluster_display_dict = {tuple(set(filenames)): f"Cluster {cluster_id-1}: {tuple(set(filenames))}".replace(',','') for cluster_id, filenames in inverted_cluster_dict.items()}
            unclustered_key = inverted_cluster_dict.get(0)
            if unclustered_key is not None:
                unclustered_key = tuple(set(unclustered_key))
                cluster_display_dict[unclustered_key] = cluster_display_dict[unclustered_key].replace("Cluster -1", "Unclustered")
            
            add_filters_1.multiselect("Populate by protein dendrogram clusters", 
                            list(set(cluster_display_dict.keys())),
                            format_func=cluster_display_dict.get,
                            key='sma_selected_clusters')
                    

        if 'sma_selected_clusters' not in st.session_state:
            st.session_state['sma_selected_clusters'] = []
        if 'sma_selected_proteins' not in st.session_state:
            st.session_state['sma_selected_proteins'] = []

        # Button to Move Clusters to Individual Protein List
        add_filters_2.markdown('<div class="button-label">Add Clusters</div>', unsafe_allow_html=True)
        add_button = add_filters_2.button(":arrow_forward:", key="Add")

        # Individual Protein Selection
        sma_selected_proteins = add_filters_3.multiselect(
            "Populate by strain",
            list(st.session_state["metadata_df"]['Filename']),
            default=st.session_state['sma_selected_proteins']
        )
            
        sma_selected_prot_submitted = st.form_submit_button("Apply Filters")
        
    if sma_selected_prot_submitted:
        st.session_state['sma_selected_proteins'] = sma_selected_proteins

    # Handle add button click
    if add_button:
        for cluster in st.session_state['sma_selected_clusters']:
            to_add = set(cluster) - set(st.session_state['sma_selected_proteins'])
            st.session_state['sma_selected_proteins'].extend(to_add)
        st.session_state['sma_selected_clusters'].clear()
        st.rerun()  # Refresh the UI to reflect the updated selection

small_molecule_dict = None
nx_G = None
small_molecule_dict, nx_G = generate_network(cluster_dict)

if nx_G is not None:
    # Write the graphml file to bytesIO
    graphml_bytes = io.BytesIO()
    nx.write_graphml(nx_G, graphml_bytes)
    graphml_bytes.seek(0)
    st.download_button("Download Network as .graphml", graphml_bytes, "IDBac_MAN.graphml", help="Download the network graph as a graphml file.")

# Download Options
if small_molecule_dict is not None:
    choices = sorted(list(small_molecule_dict.keys()))
    print("choices", choices, flush=True)
    download_choice = st.selectbox("Download merged and thresholded small molecule data", choices)
    download_bytes = convert_to_mzml(small_molecule_dict[download_choice])
    st.download_button("Download mzML", download_bytes, f"{download_choice}", help="Download the mzML file for the selected small molecule.")

st.header("Small Molecule Heatmap")
# st.multiselect("Populate by strain", st.session_state["metadata_df"]['Filename'].unique(), key='sma_selected_proteins')

# Option for minimum instance frequency
curr_num_proteins = len(st.session_state.get("sma_selected_proteins", []))

if curr_num_proteins > 1:
    st.slider("Display m/z values that are associated with this many strains", min_value=1, max_value=curr_num_proteins, value=1, step=1, key="sma_min_mz_frequency")
else:
    # Display disabled
    st.slider("Display m/z values that are associated with this many strains", min_value=0, max_value=1, value=1, step=1, key="sma_min_mz_frequency", disabled=True)

make_heatmap()
