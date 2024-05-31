import streamlit as st
import pandas as pd
import requests
import numpy as np
import io
import plotly.figure_factory as ff
import plotly
# Now lets do pairwise cosine distance
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import time
import textwrap

import plotly.graph_objects as go

import numpy as np

from utils import write_job_params, write_warnings, enrich_genbank_metadata
from Protein_Dendrogram_Components import draw_mirror_plot, draw_protein_heatmap

class np_data_wrapper():
    def __init__(self, data_np, spectrum_data_df, db_distance_dict):
        """
        A wrapper around a numpy array that contains metadata and database distance information.
        
        Parameters:
        - data_np (numpy.ndarray): The input data as a numpy array where each row represents binary binned peaks.
        - spectrum_data_df (pandas.DataFrame): The dataframe containing columns ['filename','db_search_result']. 
            'filename' denotes the name of the id of the spectrum. 'db_search_result' denotes whether the spectrum is a database search result.
        - db_distance_dict (dict): The dictionary containing the database distance information.
        """
        self.data_np = data_np
        self.spectrum_data_df = spectrum_data_df
        self.db_distance_dict = db_distance_dict

    def __getitem__(self, index):
        return self.data_np[index]
    
    def __getattr__(self, name):
        return getattr(self.data_np, name)

def get_dist_function_wrapper(distfun):
    """
    A function that returns a wrapper around the distance function that allows us to pass in a numpy array with metadata and a dictionary of database 
    distance information. The goal here is we want to use precomputed distances when given a database search result, but want to compute distances
    between non-database search results.
    
    Parameters:
    - distfun (function): The distance function to be used for calculating distances between data points.
    
    Returns:
    - dist_function_wrapper (function): The wrapped distance function.
    """
    def dist_function_wrapper(wrapped_np_array):
        """
        A wrapper around the distance function that allows us to pass in a numpy array with metadata and a dictionary of database distance information.
        
        Parameters:
        - wrapped_np_array (np_data_wrapper): The numpy array with metadata and database distance information. Contains 
            a numpy array, a dataframe with columns ['filename','db_search_result'], and a dictionary of database distance information.
        
        Returns:
        - distance_matrix (numpy.ndarray): The distance matrix.
        """
        data_np = wrapped_np_array.data_np
        spectrum_data_df = wrapped_np_array.spectrum_data_df
        db_distance_dict = wrapped_np_array.db_distance_dict
        
        # Select rows that are not databse search results and send to the distance function
        non_db_search_result_filenames = spectrum_data_df.filename[spectrum_data_df['db_search_result'] == False].tolist()
        non_db_search_result_indices   = spectrum_data_df.index[spectrum_data_df['db_search_result'] == False].tolist()
        
        # Note that this is only going to work if the database search results are in the bottom of the dataframe
        if len(non_db_search_result_indices) != 0:  # Allows us to hide all queries and ignore this check
            begins_at_zero = non_db_search_result_indices[0] == 0
            is_contiguous = non_db_search_result_indices == list(range(non_db_search_result_indices[0], non_db_search_result_indices[-1] + 1))
            if not begins_at_zero or not is_contiguous:
                raise ValueError("To compute distances, database search results should be at the bottom of the dataframe")
        
        num_inputs = len(non_db_search_result_indices)
        
        if len(non_db_search_result_indices) != 0:
            computed_distances = distfun(data_np[non_db_search_result_indices])
        else: 
            computed_distances = np.zeros((0, num_inputs))
        
        # Add database search results
        db_search_result_filenames = spectrum_data_df.filename[spectrum_data_df['db_search_result'] == True].tolist()
        num_db_search_results = len(db_search_result_filenames)
        
        # Shortcut out to speed up computation
        if num_db_search_results == 0:
            return squareform(computed_distances)
        
        # In theory this should never happen, but it's a good sanity check
        if num_db_search_results + num_inputs != spectrum_data_df.shape[0]:
            raise Exception("Error in creating distance matrix")
        
        start_tile = time.time()
        db_distance_matrix = np.ones((num_inputs, num_db_search_results))
        for i, filename in enumerate(non_db_search_result_filenames):
            db_dist_dict = db_distance_dict.get(filename)
            if db_dist_dict is None:
                continue
            for j, db_filename in enumerate(db_search_result_filenames):
                this_dist = db_dist_dict.get(db_filename)
                if this_dist is not None:
                    # Deal with numerical percision error due to subtractive cancellation
                    if this_dist > 0.999:
                        db_distance_matrix[i, j] = 1
                    else:
                        db_distance_matrix[i, j] = this_dist # 1-sim because we want distance
        print("Time to compute db distances", time.time() - start_tile, flush=True)
        
        start_time = time.time()
        db_db_distance_matrix = np.ones((num_db_search_results, num_db_search_results))
        if 'database_id' in spectrum_data_df.columns: # Only included when there are database search results present
            db_search_database_ids = spectrum_data_df.database_id[spectrum_data_df['db_search_result'] == True].tolist()     
            for i, db_id_1 in enumerate(db_search_database_ids):
                db_dist_dict = db_distance_dict.get(db_id_1)
                if db_dist_dict is None:
                    continue    # Should only happen for old jobs
                for j in range(i+1, len(db_search_database_ids)):
                    db_id_2 = db_search_database_ids[j]
                    this_dist = db_dist_dict.get(db_id_2)
                    if this_dist is not None:
                        # Deal with numerical percision error due to subtractive cancellation
                        if this_dist > 0.999:
                            db_db_distance_matrix[i, j] = 1
                            db_db_distance_matrix[j, i] = 1
                        else:
                            db_db_distance_matrix[i, j] = this_dist
                            db_db_distance_matrix[j, i] = this_dist
                    else:
                        raise ValueError("Missing", db_id_1, db_id_2, db_dist_dict, flush=True)
        print("Time to compute db-db distances", time.time() - start_time, flush=True)

        # Create a single matrix to join everything together
        distance_matrix = np.ones((num_inputs + num_db_search_results, num_inputs + num_db_search_results)) * np.inf    # Multiply by inf to allow for a sanity check
        distance_matrix[:num_inputs, :num_inputs] = computed_distances
        distance_matrix[:num_inputs, num_inputs:] = db_distance_matrix
        distance_matrix[num_inputs:, :num_inputs] = db_distance_matrix.T
        distance_matrix[num_inputs:, num_inputs:] = db_db_distance_matrix
        
        for i in range(num_inputs + num_db_search_results):
            distance_matrix[i,i] = 0
        
        
        if np.max(distance_matrix) > 1:
            raise ValueError("Something went wrong during distance caluclation")

        # The bottom right corner is all ones
        # assert np.max(distance_matrix) < 1.000001, f"Maximum distnace is {np.max(distance_matrix)}"
        return squareform(distance_matrix)

    return dist_function_wrapper

def create_dendrogram(data_np, all_spectra_df, db_distance_dict, 
                      plotted_metadata=[],
                      db_label_column=None,
                      metadata_df=None,
                      db_search_columns=None,
                      cluster_method="ward",
                      coloring_threshold=None,
                      cutoff=None,
                      show_annotations=True,
                      ):
    """
    Create a dendrogram using the given data and parameters.

    Parameters:
    - data_np (numpy.ndarray): The input data as a numpy array.
    - all_spectra_df (pandas.DataFrame): The dataframe containing all spectra data.
    - db_distance_dict (dict): The dictionary containing the database distance information.
    - plotted_metadata (list of str, optional): The column name to be shown in the scatter plot. Defaults to "filename".
    - metadata_df (pandas.DataFrame, optional): The dataframe containing metadata information. Defaults to None.
    - db_search_columns (list, optional): The list of columns to be used for displaying database search result metadata. Defaults to None.
    - cluster_method (str, optional): The clustering method to be used for clustering the data. Defaults to "ward".
    - coloring_threshold (float, optional): The threshold for coloring the dendrogram. Defaults to None.
    - cutoff (float, optional): The cutoff line for the dendrogram. Defaults to None.
    - show_annotations (bool, optional): Whether to show annotations on the dendrogram. Defaults to True.

    Returns:
    - dendro (plotly.graph_objs._figure.Figure): The generated dendrogram as a Plotly figure.
    """
    # General Metadata Prep
    if metadata_df is not None:
        original_all_spectra_cols = all_spectra_df.columns
        all_spectra_df = all_spectra_df.merge(metadata_df, how="left", left_on="filename", right_on="Filename", suffixes=("", "_metadata"))
    
    # Prepare text metadata
    if metadata_df is not None and st.session_state["metadata_label"] != "None":
        metadata_column = st.session_state["metadata_label"]
        if metadata_column not in metadata_df.columns:
            st.error("Metadata file does not have the specified column")
        else:
            has_metadata = ~all_spectra_df[metadata_column].isna()
            all_spectra_df.loc[has_metadata, metadata_column] = all_spectra_df.loc[has_metadata, metadata_column].astype(str) + ' - ' # Prepend dash only if we have metadata
            all_spectra_df.loc[:, metadata_column] = all_spectra_df.loc[:, metadata_column].fillna("")
            db_result_mask = (all_spectra_df["db_search_result"] == False)
            all_spectra_df.loc[db_result_mask, "label"] = all_spectra_df.loc[db_result_mask][metadata_column].astype(str) + all_spectra_df.loc[db_result_mask]['label'].astype(str)
    
    # if metadata_df is not None and label_column != 'None':
    if metadata_df is not None and len(plotted_metadata) > 0:
        # Attempt to fall back to lowercase filename if uppercase filename is not present
        if 'Filename' not in metadata_df.columns and 'filename' in metadata_df.columns:
            metadata_df['Filename'] = metadata_df['filename']
        # Raise an error if there is not filename column
        if 'Filename' not in metadata_df.columns and 'filename' not in metadata_df.columns:
            st.error("Metadata file does not have a 'Filename' column")
        
        # If the label column is in the original dataframe, a suffix is added
        plotted_metadata = [metadata_column if metadata_column not in original_all_spectra_cols else metadata_column+"_metadata" for metadata_column in plotted_metadata]
        
    # Add metadata for db search results
    if db_label_column != "No Database Search Results" and sum(all_spectra_df["db_search_result"]) > 0:
        # all_spectra_df.loc[all_spectra_df["db_search_result"] == True, db_metadata_column].fillna("No Metadata", inplace=True)
        # all_spectra_df.loc[all_spectra_df["db_search_result"] == True, "label"] = 'DB Result - ' + all_spectra_df.loc[all_spectra_df["db_search_result"] == True][db_label_column].astype(str)
        if db_search_columns != 'None':
            all_spectra_df.loc[all_spectra_df["db_search_result"] == True, "label"] = 'DB Result - ' + all_spectra_df.loc[all_spectra_df["db_search_result"] == True][db_search_columns].astype(str) + ' - ' + all_spectra_df.loc[all_spectra_df["db_search_result"] == True]['db_strain_name'].astype(str)
        else:
            all_spectra_df.loc[all_spectra_df["db_search_result"] == True, "label"] = 'DB Result - ' + all_spectra_df.loc[all_spectra_df["db_search_result"] == True]['db_strain_name'].astype(str)
        
    # all_spectra_df["label"] = all_spectra_df["label"].astype(str) + " - " + all_spectra_df["filename"].astype(str)
    all_labels_list = all_spectra_df["label"].to_list()
    
    # Reset index to use as unique identifier
    all_spectra_df.reset_index(drop=True, inplace=True)

    # We only have (and only need to input) database_id when there are database search results
    if 'database_id' in all_spectra_df.columns:
        all_spectra_input = all_spectra_df[['filename','db_search_result','database_id']]
    else:
        all_spectra_input = all_spectra_df[['filename','db_search_result']]
        
    selected_distance_fun = st.session_state['distance_measure']
        
    # Creating Dendrogram  
    dendro = ff.create_dendrogram(np_data_wrapper(data_np, all_spectra_input, db_distance_dict),
                                  orientation='left',
                                  labels=all_spectra_df.index.values, # We will use the labels as a unique identifier
                                  distfun=get_dist_function_wrapper(selected_distance_fun),
                                  linkagefun=lambda x: linkage(x, method=cluster_method),
                                  color_threshold=coloring_threshold)
    dendrogram_width = 800
    dendrogram_height = max(15*len(all_labels_list), 350)
    
    if cutoff is not None:
        dendro.add_vline(x=cutoff, line_width=1, line_color='grey')
    
    # Add labels for each intersection
    if show_annotations:
        for dd in dendro.data:
            # Get middle two x's and y's
            # (it's actually drawing rectangles and you can think of these as the top right and bottom right corners)
            x = (dd.x[1] + dd.x[2]) / 2
            y = (dd.y[1] + dd.y[2]) / 2
            # Add a text element:
            dendro.add_trace(go.Scatter
                                (x=[x-0.005],
                                y=[y],
                                text=f'{x:.2f}',
                                mode='text',
                                showlegend=False,
                                textposition='middle left',
                                textfont=dict(size=10),
                                hoverinfo='none',
                                xaxis='x',
                                yaxis='y'
                                )
                            )
    # Remove Gridlines from Dendrogram
    dendro.update_xaxes(showgrid=False)
    dendro.update_yaxes(showgrid=False)
    
    # Get y values for the axis labels
    y_values            = dendro['layout']['yaxis']['tickvals']
    y_axis_identifiers  = dendro['layout']['yaxis']['ticktext']
    y_labels            = [all_labels_list[int(identifier)] for identifier in y_axis_identifiers]
    
    # Add a scatter to show metadata
    if metadata_df is not None and plotted_metadata != []:
        num_cols = len(plotted_metadata) + 1
        
        # Calculate ideal column widths based on the number of unique values.
        column_widths = []
        for col in plotted_metadata:
            # Use the unique count if not a numeric type
            if metadata_df[col].dtype == 'object':
                num_unique = len(all_spectra_df[col].unique())
                column_widths.append(max(0.02 * num_unique, 0.15))    # Max width is 0.3 for any given column
            else:
                column_widths.append(0.15)                          # Default width for numeric types
            
        column_widths.append(1 - sum(column_widths))            # Add the remaining width for the dendrogram
        fig = plotly.subplots.make_subplots(rows=1, cols=num_cols,
                                            shared_yaxes=True,
                                            column_widths=column_widths,
                                            horizontal_spacing=0.0)
        fig.update_layout(width=dendrogram_width, height=dendrogram_height, margin=dict(l=0, r=0, b=0, t=0, pad=0))
        
        col_counter = 1
        for col_name in plotted_metadata:
            # Reorder metadata array to be consistent with histogram axis
            consistently_ordered_metadata = all_spectra_df[col_name].loc[y_axis_identifiers]
        
            # Create the scatter on the new axis
            metadata_scatter = go.Scatter(x=consistently_ordered_metadata, y=y_values, mode='markers')
            # Show all x ticks
            fig.update_xaxes(tickvals=consistently_ordered_metadata,
                             ticktext=consistently_ordered_metadata,
                             row=1,
                             col=col_counter,
                             tickangle=90,
                             ticks="outside",
                             showgrid=True)
            fig.add_trace(metadata_scatter, row=1, col=col_counter)
            
            # ylim must be set for each axis, otherwise we get blank space
            fig.update_yaxes(tickvals=y_values, ticktext=y_labels, range=[min(y_values)-10, max(y_values)+25], row=1, col=col_counter, showgrid=True,)
            
            # Add title
            wrapped_title = col_name.replace(" ", "<br>") # TODO: Better wrapping
            fig.add_annotation(xref="x domain",yref="y domain",x=0.5, y=1, showarrow=False,
                   text=f"<b>{wrapped_title}</b>", row=1, col=col_counter)
            
            col_counter+=1
            
        # print(dir(fig), flush=True)
        # Add a border around the scatter plots
        rectangles_to_add = []
        for x,y in zip(range(1,num_cols), range(1, num_cols)):
            # print(x,y, flush=True)
            if x == 1:
                x = ""
            if y == 1:
                y = ""
            rectangles_to_add.append(go.layout.Shape(
                    type="rect",
                    xref=f"x{x} domain",
                    yref=f"y domain",
                    x0= 0.0,
                    y0 = 0,
                    x1= 1.,
                    y1= 1,
                    line={'width':1, 'color':"rgb(250, 250, 250)"})
                    )
        
        fig.update_layout(shapes=rectangles_to_add)
        
        # Add the dendrogram to the figure
        for trace in dendro.data:
            fig.add_trace(trace, row=1, col=col_counter)
        
        # Remove gridlines from dendrogram (again)
        fig.update_yaxes(showgrid=False, row=1, col=col_counter)
        
        # Remove legend from dendrogram
        fig.update_layout(showlegend=False)
        
        # Label y axis
        fig.update_yaxes(tickvals=y_values, ticktext=y_labels, row=1, col=1, autorange=False, ticksuffix=" ", ticks="outside")
        # # Set ylim
        #     fig.update_yaxes(range=[min(y_values)-10, max(y_values)+10], row=1, col=col_counter)
        
        return fig


    dendro.update_layout(width=dendrogram_width, height=dendrogram_height, margin=dict(l=0, r=0, b=0, t=0, pad=0))
    # Set ylim
    dendro.update_yaxes(range=[min(y_values)-5, max(y_values)+5], tickvals=y_values, ticktext=y_labels, autorange=False, ticksuffix=" ", ticks="outside")
    return dendro


def collect_database_search_results(task):
    """
    Collect the database search results from the IDBAC Database. If the database search results are not available, then None is returned.

    Parameters:
    - task (str): The GNPS2 task ID.

    Returns:
    - database_search_results_df (pandas.DataFrame): The dataframe containing the database search results.
    - database_distance_table (pandas.DataFrame): The dataframe containing the database distance tabl (contains original fikle)
    """
    try:
        # Getting the database search results
        if task.startswith("DEV-"):
            database_search_results_url = f"http://ucr-lemon.duckdns.org:4000/resultfile?task={task[4:]}&file=nf_output/search/enriched_db_results.tsv"
        else:
            database_search_results_url = f"https://gnps2.org/resultfile?task={task}&file=nf_output/search/enriched_db_results.tsv"
        database_search_results_df = pd.read_csv(database_search_results_url, sep="\t")
    except:
        database_search_results_df = None
        
    try:
        # Get the DB-DB distances
        if task.startswith("DEV-"):
            database_distance_url = f"http://ucr-lemon.duckdns.org:4000/resultfile?task={task[4:]}&file=nf_output/search/db_db_distance.tsv"
        else:
            database_distance_url = f"https://gnps2.org/resultfile?task={task}&file=nf_output/search/db_db_distance.tsv"
        database_database_distance_table = pd.read_csv(database_distance_url, sep="\t")
    except Exception:
        database_database_distance_table = None
        
        if database_search_results_df is not None:
            if 'distance' not in database_search_results_df.columns:
                st.warning("This is an old GNPS task. Please clone it to use the interactive dashboard.")
                st.stop()
        
    return database_search_results_df, database_database_distance_table
        


    return database_search_results_df



def integrate_database_search_results(all_spectra_df:pd.DataFrame, database_search_results_df:pd.DataFrame, database_database_distances:pd.DataFrame, session_state, db_label_column="db_strain_name"):
    """
    Integrate the database search results into the original data. Adds unique database search results to the original data and returns a dictionary of database distances.
    Only the database_id column is considered for uniqueness.
    
    Parameters:
    - all_spectra_df (pandas.DataFrame): The dataframe containing all spectra data.
    - database_search_results_df (pandas.DataFrame): The dataframe containing the database search results.
    - session_state (dict): The session state containing the display parameters.
    - db_label_column (str, optional): The column name to be used for displaying database search result metadata. Defaults to "db_strain_name".
    
    Returns:
    - all_spectra_df (pandas.DataFrame): The dataframe containing all spectra data with database search results added.
    - database_distance_dict (dict): The dictionary containing the database distances.
    """
    db_taxonomy_filter   = session_state["db_taxonomy_filter"]
    distance_threshold = session_state["db_distance_threshold"]
    maximum_db_results   = session_state["max_db_results"]
    
    # If there are no database search results, mark everything as not a database search result and return
    if database_search_results_df is None:
        all_spectra_df["db_search_result"] = False
        return all_spectra_df, None
    
    # Apply DB Taxonomy Filter
    split_taxonomy = database_search_results_df['db_taxonomy'].str.split(";")
    trimmed_search_results_df = database_search_results_df.loc[[any([x in db_taxonomy_filter for x in y]) for y in split_taxonomy]]
    
    # Reduce visible taxonomy to only Genus, Family, Species
    trimmed_search_results_df['db_taxonomy'] = trimmed_search_results_df['db_taxonomy'].str.split(";").str[-3:].str.join(" - ")
    
    # Apply distance Filter
    trimmed_search_results_df = trimmed_search_results_df[trimmed_search_results_df["distance"] <= distance_threshold]
       
    # Apply Maximum DB Results Filter
    #best_ids = trimmed_search_results_df.sort_values(by="distance", ascending=False).database_id.drop_duplicates(keep="first")
    # Get maximum_db_results database_ids per each query_filename
    if maximum_db_results != -1:
        best_ids = []
        for query_filename in trimmed_search_results_df.query_filename.unique():
            query_results_for_filename = trimmed_search_results_df.loc[trimmed_search_results_df.query_filename == query_filename]
            best_ids_for_filename = query_results_for_filename.sort_values(by="distance", ascending=True).database_id
            best_ids_for_filename = best_ids_for_filename.iloc[:maximum_db_results] # Safe out of bounds
            best_ids.extend(best_ids_for_filename.values)
            
        best_ids = np.unique(best_ids)
        trimmed_search_results_df = trimmed_search_results_df.loc[trimmed_search_results_df["database_id"].isin(best_ids)]
        if database_database_distances is not None:        
            database_database_distances = database_database_distances[(database_database_distances["database_id_left"].isin(best_ids)) &
                                                                            (database_database_distances["database_id_right"].isin(best_ids))]
    
    # We will abuse filename because during display, we display "metadata - filename"
    trimmed_search_results_df["filename"] = trimmed_search_results_df[db_label_column].astype(str)
    
    all_spectra_df["db_search_result"] = False
    trimmed_search_results_df["db_search_result"] = True
    
    # Concatenate DB search results
    to_concat = trimmed_search_results_df.drop_duplicates(subset=["database_id"])   # Get unique database hits, assuming databsae_id is unique
    to_concat = to_concat.drop(columns=['query_filename','distance'])             # Remove distance info 
    all_spectra_df = pd.concat((all_spectra_df, to_concat), axis=0)
    
    # Build a distance dict for the database hits
    database_distance_dict = {}
    for index, row in trimmed_search_results_df.iterrows():
        if database_distance_dict.get(row['query_filename']) is None:
            database_distance_dict[row['query_filename']] = {row['filename']: row['distance']}
        else:
            database_distance_dict[row['query_filename']][row['filename']] = row['distance']
    if database_database_distances is not None:    
        # Add the DB-DB distances to the distance dictionary
        for index, row in database_database_distances.iterrows():    # This is known to be square, no need to flip the indices
            if database_distance_dict.get(row['database_id_left']) is None:
                database_distance_dict[row['database_id_left']] = {row['database_id_right']: row['distance']}
            else:
                database_distance_dict[row['database_id_left']][row['database_id_right']] = row['distance']
    
    return all_spectra_df, database_distance_dict

# Set Page Configuration
st.set_page_config(page_title="IDBac - Dendrogram", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

# Here we will add an input field for the GNPS2 task ID
url_parameters = st.query_params

if "task" in url_parameters:
    st.session_state["task_id"]  = url_parameters["task"]
elif "task_id" not in st.session_state:
    st.session_state["task_id"] = "7bcac5e463b146879b1f6d4058e1ef19"
    
# Add other items to session state if available
if "metadata_label" in url_parameters:
    st.session_state["metadata_label"] = url_parameters["metadata_label"]
if "metadata_scatter" in url_parameters:
    st.session_state["metadata_scatter"] = url_parameters["metadata_scatter"]
if "db_search_result_label" in url_parameters:
    st.session_state["db_search_result_label"] = url_parameters["db_search_result_label"]
if "db_distance_threshold" in url_parameters:
    st.session_state["db_distance_threshold"] = float(url_parameters["db_distance_threshold"])
if "max_db_results" in url_parameters:
    st.session_state["max_db_results"] = int(url_parameters["max_db_results"])
if "db_taxonomy_filter" in url_parameters:
    st.session_state["db_taxonomy_filter"] = url_parameters["db_taxonomy_filter"].split(",")
if "clustering_method" in url_parameters:
    st.session_state["clustering_method"] = url_parameters["clustering_method"]
if "coloring_threshold" in url_parameters:
    st.session_state["coloring_threshold"] = float(url_parameters["coloring_threshold"])
if "hide_isolates" in url_parameters:
    st.session_state["hidden_isolates"] = url_parameters["hide_isolates"].split(",")
if "cutoff" in url_parameters:
    if url_parameters["cutoff"] == 'None':
        st.session_state["cutoff"] = None
    else:
        st.session_state["cutoff"] = float(url_parameters["cutoff"])
if "show_annotations" in url_parameters:
    st.session_state["show_annotations"] = bool(url_parameters["show_annotations"])


st.session_state["task_id"] = st.text_input('GNPS2 Task ID', st.session_state["task_id"])
if st.session_state["task_id"] == '':
    st.error("Please input a valid GNPS2 Task ID")
st.write(st.session_state["task_id"])

task_id = st.session_state["task_id"]

# Now we will get all the relevant data from GNPS2 for plotting
if st.session_state["task_id"].startswith("DEV-"):
    dev_task_id = st.session_state['task_id'][4:]
    labels_url = f"http://ucr-lemon.duckdns.org:4000/resultfile?task={dev_task_id}&file=nf_output/output_histogram_data_directory/labels_spectra.tsv"
    numpy_url = f"http://ucr-lemon.duckdns.org:4000/resultfile?task={dev_task_id}&file=nf_output/output_histogram_data_directory/numerical_spectra.npy"
    params_url = f"http://ucr-lemon.duckdns.org:4000/resultfile?task={dev_task_id}&file=job_parameters.yaml"
    warnings_url = f"http://ucr-lemon.duckdns.org:4000/resultfile?task={dev_task_id}&file=nf_output/errors.csv"
    
else:
    labels_url = f"https://gnps2.org/resultfile?task={st.session_state['task_id']}&file=nf_output/output_histogram_data_directory/labels_spectra.tsv"
    numpy_url = f"https://gnps2.org/resultfile?task={st.session_state['task_id']}&file=nf_output/output_histogram_data_directory/numerical_spectra.npy"
    params_url = f"https://gnps2.org/resultfile?task={st.session_state['task_id']}&file=job_parameters.yaml"
    warnings_url = f"https://gnps2.org/resultfile?task={st.session_state['task_id']}&file=nf_output/errors.csv"
    
workflow_params = write_job_params(params_url)
write_warnings(warnings_url)

# If workflow parameters specfiy a similarity function, use it. Otherwise, default to cosine
if "distance" in workflow_params and workflow_params is not None:
    given_distance_measure = workflow_params.get('distance', 'cosine')
    assert given_distance_measure in {'presence', 'cosine', 'euclidean'}
    if given_distance_measure == 'cosine':
        st.session_state['distance_measure'] = cosine_distances
    elif given_distance_measure == 'euclidean':
        st.session_state['distance_measure'] = euclidean_distances
    elif given_distance_measure == 'presence':
        st.session_state['distance_measure'] = cosine_distances # TODO: Ensure array is binary
else:
    st.warning("**Warning:** Unable to find a distance function in the workflow parameters. This may be an old task. Please rerun it. \
               Defaulting to cosine similarity.")
    st.session_state['distance_measure'] = cosine_distances
# By request, no longer displaying labels url
if False:
    st.write(labels_url)

# read numpy from url into a numpy array
numpy_file = requests.get(numpy_url)
numpy_file.raise_for_status()
numpy_array = np.load(io.BytesIO(numpy_file.content))
st.session_state['query_spectra_numpy_data'] = numpy_array

# read pandas dataframe from url
all_spectra_df = pd.read_csv(labels_url, sep="\t")

# By request, no longer displaying dataframe table
if False:
    st.write(all_spectra_df) # Currently, we're not displaying db search results

# Collect the database search results
db_search_results, db_db_distance_table = collect_database_search_results(task_id)

if db_db_distance_table is None and db_search_results is not None:
    st.warning("""Database-database distances are not available for this task, perhaps this is an old task?  
                Please clone and rerun the task on GNPS2 for proper visualization. The distances between these examples
                will be represented as 1.0 to the dendrogram. """)

# Get displayable metadata columns for the database search results
if db_search_results is not None:
    # Remove database search result columns we don't want displayed
    invisible_cols = ['query_filename','distance','query_index','database_index','row_count','database_id','database_scan']
    db_search_columns = [x for x in db_search_results.columns if x not in invisible_cols]
    db_search_results['db_taxonomy'] = db_search_results['db_taxonomy'].fillna("No Taxonomy")
    db_taxonomies = db_search_results['db_taxonomy'].str.split(";").to_list()
    # Flatten
    db_taxonomies = [item for sublist in db_taxonomies for item in sublist]
    db_taxonomies = list(set(db_taxonomies))
else:
    db_search_columns = []

##### Create Session States #####
# Create a session state for the clustering method
if "clustering_method" not in st.session_state:
    st.session_state["clustering_method"] = "ward"
# Create a session state for the coloring threshold
if "coloring_threshold" not in st.session_state:
    st.session_state["coloring_threshold"] = 0.70
# Create a plot for the metadata text labels
if "metadata_label" not in st.session_state:
    st.session_state["metadata_label"] = "None"
# Create a session state for the metadata scatter    
if "metadata_scatter" not in st.session_state:
    st.session_state["metadata_scatter"] = []
# Create a session state for the db search result label
if "db_search_result_label" not in st.session_state and db_search_results is not None:
    st.session_state["db_search_result_label"] = []
elif "db_search_result_label" not in st.session_state and db_search_results is None:
    st.session_state["db_search_result_label"] = "No Database Search Results"
# Create a session state for the db distance threshold
if "db_distance_threshold" not in st.session_state:
    st.session_state["db_distance_threshold"] = 0.30
# Check if db_distance_threshold is greater than the workflow threshold
if float(st.session_state["db_distance_threshold"]) > float(workflow_params["database_search_threshold"]):
    st.session_state["db_distance_threshold"] = float(workflow_params["database_search_threshold"])
# Create a session state for the maximum number of database results shown
if "max_db_results" not in st.session_state:
    st.session_state["max_db_results"] = 1
# Create a session state to filter by db taxonomy
if "db_taxonomy_filter" not in st.session_state:
    st.session_state["db_taxonomy_filter"] = None
# Create a session state for alternate metadata
if "upload_metadata" not in st.session_state:
    st.session_state["upload_metadata"] = False
# Create a session state for hidden isolates
if "hidden_isolates" not in st.session_state:
    st.session_state["hidden_isolates"] = []
if "cutoff" not in st.session_state:
    st.session_state["cutoff"] = None
if "show_annotations" not in st.session_state:
    st.session_state["show_annotations"] = False

# Add checkbox for manual metadata upload
if st.checkbox("Upload Metadata", help="If left unchecked, the metadata associated with the task will be used."):
    st.session_state["upload_metadata"] = True
    # Add file uploader
    metadata_file = st.file_uploader("Upload Metadata File", type=["csv", "tsv", "txt", "xlsx"])
    if metadata_file is not None:
        if metadata_file.name.endswith(".txt"):
            metadata_df = pd.read_table(metadata_file, index_col=False)
        elif metadata_file.name.endswith(".csv"):
            metadata_df = pd.read_csv(metadata_file, sep=",", index_col=False)
        elif metadata_file.name.endswith(".tsv"):
            metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=False)
        elif metadata_file.name.endswith(".xlsx"):
            metadata_df = pd.read_excel(metadata_file, index_col=False)
        else:
            st.error("Please upload a .csv, .tsv, or .txt file")
    else:
        metadata_df = None
else:
    # Getting the metadata
    if st.session_state['task_id'].startswith("DEV-"):
        metadata_url = f"http://ucr-lemon.duckdns.org:4000/resultfile?task={st.session_state['task_id'][4:]}&file=nf_output/output_histogram_data_directory/metadata.tsv"
    else:
        metadata_url = "https://gnps2.org/resultfile?task={}&file=nf_output/output_histogram_data_directory/metadata.tsv".format(task_id)
    try:
        metadata_df = pd.read_csv(metadata_url, sep="\t", index_col=False)
    except:
        metadata_df = None 

##### Add Display Parameters #####
st.header("Dendrogram Display Options")

st.subheader("Clustering Settings")
# Add Clustering Method dropdown
clustering_options = ["ward", "single", "complete", "average", "weighted", "centroid", "median"]
st.session_state["clustering_method"] = st.selectbox("Clustering Method", clustering_options, index=0)
# Add coloring threshold slider
st.slider("Coloring Threshold", 0.0, 1.0, step=0.05, key='coloring_threshold',
          help="Colors all links to the left of the threshold with the same color as long as they're linked below the threshold.")

st.subheader("Metadata")

st.checkbox("Use GenBank IDs to Enrich Metadata", help="If checked, the metadata will be enriched with GenBank accession numbers.", key="enrich_with_genbank", value=False)

# Add Metadata dropdown for text
if metadata_df is None:
    # If there is no metadata, then we will disable the dropdown
    st.session_state["metadata_label"] = st.selectbox("Select a metadata category that will be displayed as text", ["No Metadata Available"], disabled=True)
else:
    # Enrich metadata with genebank accession
    if st.session_state["enrich_with_genbank"]:
        metadata_df = enrich_genbank_metadata(metadata_df)
    
    columns_available = ["None"] + list(metadata_df.columns)
    # Remove filename and scan from the metadata
    columns_available =[x for x in columns_available if x.lower().strip() not in ['filename', 'scan/coordinate', 'small molecule file name']]
    st.session_state["metadata_label"] = st.selectbox("Select a metadata category that will be displayed as text", columns_available)

# Add Metadata dropdown for scatter plots
if metadata_df is None:
    # If there is no metadata, then we will disable the dropdown
    st.session_state["metadata_scatter"] = st.multiselect("Select a metadata category that will be plotted", ["No Metadata Available"], default="No Metadata Available", disabled=True, max_selections=5)
else:
    columns_available = list(metadata_df.columns)
    # Remove forbidden columns
    columns_available =[x for x in columns_available if x.lower().strip() not in ['filename', 'scan/coordinate', 'genbank accession', 'ncbi taxid', 'ms collected by', 'isolate collected by', 'sample collected by', 'pi', '16s sequence']]
    st.session_state["metadata_scatter"]  = st.multiselect("Select a metadata category that will be plotted", columns_available, default=[], max_selections=5)

if db_search_results is None:
    # Write a message saying there are no db search results
    text = "No database search results found for this task."
    st.write(f":grey[{text}]")

else:
    st.subheader("Database Search Result Filters")
    
    # Add DB Search Result dropdown
    db_search_columns_for_selection = ['None'] + [x for x in db_search_columns.copy() if x != 'db_strain_name']
    st.session_state["db_search_result_label"] = st.selectbox("Select a metadata category that will be displayed next to database hits", db_search_columns_for_selection)
    
    
    # Add DB distance threshold slider
    st.session_state["db_distance_threshold"] = st.slider("Maximum Database Distance Threshold", 0.0, float(workflow_params.get("database_search_threshold", 1.0)), st.session_state["db_distance_threshold"], 0.05)
    # Create a box for the maximum number of database results shown
    st.session_state["max_db_results"] = st.number_input("Maximum Number of Database Results Shown", min_value=-1, max_value=None, value=st.session_state["max_db_results"], help="The maximum number of unique database isolates shown, highest distance is prefered. Enter -1 to show all database results.")  
    # Create a 'select all' box for the db taxonomy filter
    
    col1, col2 = st.columns([0.84, 0.16])
    with col1:
        st.write("Select Displayed Database Taxonomies")
    
    with col2:
        st.checkbox("Select All", value=True, key="select_all_db_taxonomies")
    if st.session_state["select_all_db_taxonomies"] is True:
        st.session_state["db_taxonomy_filter"] = db_taxonomies
        # Add disabled multiselect to make this less jarring
        st.multiselect("Select Displayed Database Taxonomies", db_taxonomies, disabled=True, label_visibility="collapsed", placeholder ="Select the taxonomies to display in the dendrogram")
    else:
        # Add multiselect with update button
        st.session_state["db_taxonomy_filter"] = st.multiselect("Select Displayed Database Taxonomies", db_taxonomies, label_visibility="collapsed", placeholder ="Select the taxonomies to display in the dendrogram")

st.subheader("General")
# Add a selectbox that hides isolates
col1, col2 = st.columns([0.84, 0.16])
with col1:
    st.write("Select Isolates to be Hidden from the Dendrogram")
with col2:
    st.checkbox("Hide All", value=False, key="hide_all_isolates")
    
if st.session_state["hide_all_isolates"] is True:
    # st.info("Hiding all isolates is currently disabled. Please select isolates manually.")
    if True:
        st.session_state["hidden_isolates"] = all_spectra_df["filename"].unique()
    # Add disabled multiselect to make this less jarring
    st.multiselect("Select Isolates to be Hidden from the Dendrogram", all_spectra_df["filename"].unique(), disabled=True, label_visibility="collapsed", placeholder="Select isolates to hide from the dendrogram")
else:
    # Add multiselect with update button
    st.session_state["hidden_isolates"] = st.multiselect("Select Isolates to be Hidden from the Dendrogram", all_spectra_df["filename"].unique(), label_visibility="collapsed", placeholder="Select isolates to hide from the dendrogram")

# Add option to add a cutoff line
st.session_state["cutoff"] = st.number_input("Add Cutoff Line", min_value=0.0, max_value=1.0, value=None, help="Add a vertical line to the dendrogram at the specified distance.")
# Add option to show annotations
st.session_state["show_annotations"] = st.checkbox("Display Dendrogram Distances", value=bool(st.session_state["show_annotations"]), help="The values listed represent dendrogram distance. \
                                                                                                       To obtain similarity scores, use the 'Database Search Summary' tab within the workflow output.")

st.session_state['query_only_spectra_df'] = all_spectra_df

# Process the db search results (it's done in this order to allow for db_search parameters)
all_spectra_df, db_distance_dict = integrate_database_search_results(all_spectra_df, db_search_results, db_db_distance_table, st.session_state)

# Remove selected ones from all_spectra_df (believe it or not, we want to remove this after integrating the database search results. This will allow users to hide the queries)
all_spectra_df = all_spectra_df[~all_spectra_df["filename"].isin(st.session_state["hidden_isolates"])]
if len(all_spectra_df) == 0:
    # If there are no spectra to display, then we will stop the script
    st.error("There are no spectra to display. Please select different options.")
    st.stop()
    
# Add any remaining variables to the session state if needed
st.session_state["metadata_df"] = metadata_df

# Creating the dendrogram
dendro = create_dendrogram(numpy_array,
                           all_spectra_df,
                           db_distance_dict,
                           plotted_metadata=st.session_state["metadata_scatter"],
                           db_label_column=st.session_state["db_search_result_label"],
                           metadata_df=metadata_df,
                           db_search_columns=st.session_state["db_search_result_label"] ,
                           cluster_method=st.session_state["clustering_method"],
                           coloring_threshold=st.session_state["coloring_threshold"],
                           cutoff=st.session_state["cutoff"],
                           show_annotations=st.session_state["show_annotations"])

st.plotly_chart(dendro, use_container_width=True)

# Mirror Plot Options
draw_mirror_plot(all_spectra_df)

# Protein Heatmap
draw_protein_heatmap(all_spectra_df)

# Create a shareable link to this page
st.write("Shareable Link: ")
link = f"https://analysis.idbac.org/?task={st.session_state['task_id']}&\
        metadata_label={st.session_state['metadata_label']}&\
        metadata_scatter={st.session_state['metadata_scatter']}&\
        db_search_result_label={st.session_state['db_search_result_label']}&\
        db_distance_threshold={st.session_state['db_distance_threshold']}&\
        max_db_results={st.session_state['max_db_results']}&\
        clustering_method={st.session_state['clustering_method']}&\
        coloring_threshold={st.session_state['coloring_threshold']}&\
        hide_isolates={','.join(st.session_state['hidden_isolates'])}&\
        cutoff={st.session_state['cutoff']}&\
        show_annotations={st.session_state['show_annotations']}"
st.code(link)

# Add documentation
st.header("Additional Information")
# Add a bulleted list of more information
st.markdown("""
            #### Metadata
            * Metadata and input spectra match on the "filename" column, it must be included in both files.
            * The metadata file must be a .csv, .tsv, .txt, or .xlsx file.
            #### Visualization
            * Flat lines at x=0, are a result of perfect database search results.
            * Coloring: All descendant links below an arbitrary cluster node will be colored the same color as that cluster node if that node is the 
            first one below the cut threshold. Links between clustering nodes greater than the cut threshold are colored blue. See this 
            [link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html) for more details.
            """)