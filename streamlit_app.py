import streamlit as st
import pandas as pd
import requests
import numpy as np
import io
import plotly.figure_factory as ff
# Now lets do pairwise cosine similarity
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import time

import plotly.graph_objects as go

import numpy as np

class np_data_wrapper():
    def __init__(self, data_np, spectrum_data_df, db_similarity_dict):
        """
        A wrapper around a numpy array that contains metadata and database similarity information.
        
        Parameters:
        - data_np (numpy.ndarray): The input data as a numpy array where each row represents binary binned peaks.
        - spectrum_data_df (pandas.DataFrame): The dataframe containing columns ['filename','db_search_result']. 
            'filename' denotes the name of the id of the spectrum. 'db_search_result' denotes whether the spectrum is a database search result.
        - db_similarity_dict (dict): The dictionary containing the database similarity information.
        """
        self.data_np = data_np
        self.spectrum_data_df = spectrum_data_df
        self.db_similarity_dict = db_similarity_dict

    def __getitem__(self, index):
        return self.data_np[index]
    
    def __getattr__(self, name):
        return getattr(self.data_np, name)

def get_dist_function_wrapper(distfun):
    """
    A function that returns a wrapper around the distance function that allows us to pass in a numpy array with metadata and a dictionary of database 
    similarity information. The goal here is we want to use precomputed distances when given a database search result, but want to compute distances
    between non-database search results.
    
    Parameters:
    - distfun (function): The distance function to be used for calculating distances between data points.
    
    Returns:
    - dist_function_wrapper (function): The wrapped distance function.
    """
    def dist_function_wrapper(wrapped_np_array):
        """
        A wrapper around the distance function that allows us to pass in a numpy array with metadata and a dictionary of database similarity information.
        
        Parameters:
        - wrapped_np_array (np_data_wrapper): The numpy array with metadata and database similarity information. Contains 
            a numpy array, a dataframe with columns ['filename','db_search_result'], and a dictionary of database similarity information.
        
        Returns:
        - distance_matrix (numpy.ndarray): The distance matrix.
        """
        data_np = wrapped_np_array.data_np
        spectrum_data_df = wrapped_np_array.spectrum_data_df
        db_similarity_dict = wrapped_np_array.db_similarity_dict
        
        # Select rows that are not databse search results and send to the distance function
        non_db_search_result_filenames = spectrum_data_df.filename[spectrum_data_df['db_search_result'] == False].tolist()
        non_db_search_result_indices   = spectrum_data_df.index[spectrum_data_df['db_search_result'] == False].tolist()
        
        # Note that this is only going to work if the database search results are in the bottom of the dataframe
        begins_at_zero = non_db_search_result_indices[0] == 0
        is_contiguous = non_db_search_result_indices == list(range(non_db_search_result_indices[0], non_db_search_result_indices[-1] + 1))
        if not begins_at_zero or not is_contiguous:
            raise ValueError("To compute distances, database search results should be at the bottom of the dataframe")
        
        num_inputs = len(non_db_search_result_indices)
        
        computed_distances = distfun(data_np[non_db_search_result_indices])
        
        # Add database search results
        db_search_result_filenames = spectrum_data_df.filename[spectrum_data_df['db_search_result'] == True].tolist()
        num_db_search_results = len(db_search_result_filenames)
        
        # Shortcut out to speed up computation
        if num_db_search_results == 0:
            return squareform(computed_distances)
        
        # In theory this should never happen, but it's a good sanity check
        if num_db_search_results + num_inputs != spectrum_data_df.shape[0]:
            raise Exception("Error in creating distance matrix")
        
        db_distance_matrix = np.ones((num_inputs, num_db_search_results))
        for i, filename in enumerate(non_db_search_result_filenames):
            db_sim_dict = db_similarity_dict.get(filename)
            if db_sim_dict is None:
                continue
            for j, db_filename in enumerate(db_search_result_filenames):
                this_sim = db_sim_dict.get(db_filename)
                if this_sim is not None:
                    # Deal with numerical precisin error due to subtractive cancellation
                    if this_sim > 0.999:
                        db_distance_matrix[i, j] = 0
                    else:
                        db_distance_matrix[i, j] = 1 - this_sim # 1-sim because we want distance
        
        # Create a matrix of zeros
        distance_matrix = np.zeros((num_inputs + num_db_search_results, num_inputs + num_db_search_results))
        distance_matrix[:num_inputs, :num_inputs] = computed_distances
        distance_matrix[:num_inputs, num_inputs:] = db_distance_matrix
        distance_matrix[num_inputs:, :num_inputs] = db_distance_matrix.T

        # The bottom right corner is all ones
        # assert np.max(distance_matrix) < 1.000001, f"Maximum distnace is {np.max(distance_matrix)}"
        return squareform(distance_matrix)

    return dist_function_wrapper

def create_dendrogram(data_np, all_spectra_df, db_similarity_dict, selected_distance_fun=cosine_distances, 
                      label_column="filename", 
                      db_label_column=None,
                      metadata_df=None, 
                      db_search_columns=None, 
                      cluster_method="ward", 
                      coloring_threshold=None, 
                      cutoff=None,
                      show_annotations=True):
    """
    Create a dendrogram using the given data and parameters.

    Parameters:
    - data_np (numpy.ndarray): The input data as a numpy array.
    - all_spectra_df (pandas.DataFrame): The dataframe containing all spectra data.
    - db_similarity_dict (dict): The dictionary containing the database similarity information.
    - selected_distance_fun (function, optional): The distance function to be used for calculating distances between data points. Defaults to numpy.cosine_distances.
    - label_column (str, optional): The column name to be used as labels for the dendrogram. Defaults to "filename".
    - metadata_df (pandas.DataFrame, optional): The dataframe containing metadata information. Defaults to None.
    - db_search_columns (list, optional): The list of columns to be used for displaying database search result metadata. Defaults to None.
    - cluster_method (str, optional): The clustering method to be used for clustering the data. Defaults to "ward".
    - coloring_threshold (float, optional): The threshold for coloring the dendrogram. Defaults to None.
    - cutoff (float, optional): The cutoff line for the dendrogram. Defaults to None.
    - show_annotations (bool, optional): Whether to show annotations on the dendrogram. Defaults to True.

    Returns:
    - dendro (plotly.graph_objs._figure.Figure): The generated dendrogram as a Plotly figure.
    """
    if metadata_df is not None and label_column != 'None':
        # Attempt to fall back to lowercase filename if uppercase filename is not present
        if 'Filename' not in metadata_df.columns and 'filename' in metadata_df.columns:
            metadata_df['Filename'] = metadata_df['filename']
        # Raise an error if there is not filename column
        if 'Filename' not in metadata_df.columns and 'filename' not in metadata_df.columns:
            st.error("Metadata file does not have a 'Filename' column")
        
        # If the label column is in the original dataframe, a suffix is added
        if label_column in all_spectra_df.columns:
            label_column = label_column + "_metadata"
            
        all_spectra_df = all_spectra_df.merge(metadata_df, how="left", left_on="filename", right_on="Filename", suffixes=("", "_metadata"))
        
        all_spectra_df.loc[:, label_column].fillna("No Metadata", inplace=True)
        all_spectra_df.loc[:, "label"] = all_spectra_df[label_column].fillna("No Metadata")
    else:
        all_spectra_df.loc[:, "label"] = "No Metadata"
        
    # Add metadata for db search results
    if db_label_column != "No Database Search Results":
        all_spectra_df.loc[all_spectra_df["db_search_result"] == True, db_label_column].fillna("No Metadata", inplace=True)
        all_spectra_df.loc[all_spectra_df["db_search_result"] == True, "label"] = 'DB Result - ' + all_spectra_df.loc[all_spectra_df["db_search_result"] == True][db_label_column].astype(str)
        
    all_spectra_df["label"] = all_spectra_df["label"].astype(str) + " - " + all_spectra_df["filename"].astype(str)
    all_labels_list = all_spectra_df["label"].to_list()

    # Creating Dendrogram  
    dendro = ff.create_dendrogram(np_data_wrapper(data_np, all_spectra_df[['filename','db_search_result']], db_similarity_dict),
                                  orientation='left',
                                  labels=all_labels_list,
                                  distfun=get_dist_function_wrapper(selected_distance_fun),
                                  linkagefun=lambda x: linkage(x, method=cluster_method),
                                  color_threshold=coloring_threshold)
    dendro.update_layout(width=800, height=max(15*len(all_labels_list), 350))
    
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

    return dendro


def collect_database_search_results(task):
    """
    Collect the database search results from the IDBAC Database. If the database search results are not available, then None is returned.

    Parameters:
    - task (str): The GNPS2 task ID.

    Returns:
    - database_search_results_df (pandas.DataFrame): The dataframe containing the database search results.
    - database_similarity_table (pandas.DataFrame): The dataframe containing the database similarity tabl (contains original fikle)
    """
    try:
        # Getting the database search results
        database_search_results_url = f"https://gnps2.org/resultfile?task={task}&file=nf_output/search/enriched_db_results.tsv"
        database_search_results_df = pd.read_csv(database_search_results_url, sep="\t")
    except:
        database_search_results_df = None

    return database_search_results_df

def build_database_result_USI(database_id:str, file_name:str):
    """
    Build a USI for a database search result given a database_id. Note, if the original
    task is missing/deteleted from GNPS2, this will not work.
    
    Parameters:
    - database_id (str): The database_id of the database search result.
    
    Returns:
    - usi (str): The USI of the database search result.
    """
    
    # User database id to get original task id
    # Example URL: https://idbac-kb.gnps2.org/api/spectrum?database_id=01HHBSS17717HA7VN5C167FYHC
    url = "https://idbac-kb.gnps2.org/api/spectrum?database_id={}".format(database_id)
    r = requests.get(url)
    retries = 3
    while r.status_code != 200 and retries > 0:
        r = requests.get(url)
        retries -= 1
    if r.status_code != 200:
        # Throw an exception for this because the database ids are supplied internally
        raise ValueError("Database ID not found")
    result_dictionary = r.json()
    task = result_dictionary["task"]
    file_name = result_dictionary["Filename"]
       
    return_usi = f"mzspec:GNPS2:TASK-{task}-nf_output/merged/{file_name}:scan:1"
    
    # Test the USI, so we can return an error message on the page
    r = requests.get(f"https://metabolomics-usi.gnps2.org/json/?usi1={return_usi}")
    if r.status_code != 200:
        # Return None, signifying an error if the USI is not valid, this would imply that the original task is missing/deleted
        st.error("File Upload Task is Missing or Deleted from GNPS2")
        return None
    
    return return_usi

def get_USI(all_spectra_df: pd.DataFrame, filename: str, task:str):
    """
    Get the USI of a given filename.
    
    Parameters:
    - all_spectra_df (pandas.DataFrame): The dataframe containing all spectra data.
    - filename (str): The filename of the spectrum.
    - task (str): The IDBAc_analysis task number
    
    Returns:
    - usi (str): The USI of the spectrum.
    """
    if filename == 'None':
        return None
    
    db_result = False
    if filename.startswith("DB Result - "):
        filename = filename.replace("DB Result - ", "")
        db_result = True
        
    # Attempt to mitigate issues due to duplicate filenames
    row = all_spectra_df.loc[(all_spectra_df["filename"] == filename) & (all_spectra_df["db_search_result"] == db_result)]

    if db_result:
        # If it's a database search result, use the database_id to get the USI
        output_USI = build_database_result_USI(row["database_id"].iloc[0], row["filename"].iloc[0])
    else:
        # If it's a query, use the query job to get the USI
        output_USI = f"mzspec:GNPS2:TASK-{task}-nf_output/merged/{row['filename'].iloc[0]}:scan:1"
    return output_USI
    
def get_mirror_plot_url(usi1, usi2=None):
    if usi2 is None:
        url = f"https://metabolomics-usi.gnps2.org/dashinterface/?usi1={usi1}"
    else:
        url = f"https://metabolomics-usi.gnps2.org/dashinterface/?usi1={usi1}&usi2={usi2}"
    return url


def integrate_database_search_results(all_spectra_df: pd.DataFrame, database_search_results_df: pd.DataFrame, session_state, db_label_column="db_strain_name"):
    """
    Integrate the database search results into the original data. Adds unique database search results to the original data and returns a dictionary of database similarities.
    Only the database_id column is considered for uniqueness.
    
    Parameters:
    - all_spectra_df (pandas.DataFrame): The dataframe containing all spectra data.
    - database_search_results_df (pandas.DataFrame): The dataframe containing the database search results.
    - session_state (dict): The session state containing the display parameters.
    - db_label_column (str, optional): The column name to be used for displaying database search result metadata. Defaults to "db_strain_name".
    
    Returns:
    - all_spectra_df (pandas.DataFrame): The dataframe containing all spectra data with database search results added.
    - database_similarity_dict (dict): The dictionary containing the database similarities.
    """
    db_taxonomy_filter   = session_state["db_taxonomy_filter"]
    similarity_threshold = session_state["db_similarity_threshold"]
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
    
    # Apply Similarity Filter
    trimmed_search_results_df = trimmed_search_results_df[trimmed_search_results_df["similarity"] >= similarity_threshold]
       
    # Apply Maximum DB Results Filter
    best_ids = trimmed_search_results_df.sort_values(by="similarity", ascending=False).database_id.drop_duplicates(keep="first")
    # Get maximum_db_results database_ids
    if maximum_db_results != -1:
        best_ids = best_ids.iloc[:maximum_db_results]             # Safe out of bounds
        trimmed_search_results_df = trimmed_search_results_df[trimmed_search_results_df["database_id"].isin(best_ids["database_id"])]
    
    # We will abuse filename because during display, we display "metadata - filename"
    trimmed_search_results_df["filename"] = trimmed_search_results_df[db_label_column].astype(str)
    
    all_spectra_df["db_search_result"] = False
    trimmed_search_results_df["db_search_result"] = True
    
    # Concatenate DB search results
    to_concat = trimmed_search_results_df.drop_duplicates(subset=["database_id"])   # Get unique database hits, assuming databsae_id is unique
    to_concat = to_concat.drop(columns=['query_filename','similarity'])             # Remove similarity info 
    all_spectra_df = pd.concat((all_spectra_df, to_concat), axis=0)
    
    # Build a similarity dict for the database hits
    database_similarity_dict = {}
    for index, row in trimmed_search_results_df.iterrows():
        if database_similarity_dict.get(row['query_filename']) is None:
            database_similarity_dict[row['query_filename']] = {row['filename']: row['similarity']}
        else:
            database_similarity_dict[row['query_filename']][row['filename']] = row['similarity']
    
    return all_spectra_df, database_similarity_dict

# Here we will add an input field for the GNPS2 task ID
url_parameters = st.query_params

default_task = "0e744752fdd44faba37df671b9d1997c"
if "task" in url_parameters:
    default_task = url_parameters["task"]
# Add other items to session state if available
if "metadata_label" in url_parameters:
    st.session_state["metadata_label"] = url_parameters["metadata_label"]
if "db_search_result_label" in url_parameters:
    st.session_state["db_search_result_label"] = url_parameters["db_search_result_label"]
if "db_similarity_threshold" in url_parameters:
    st.session_state["db_similarity_threshold"] = float(url_parameters["db_similarity_threshold"])
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


task = st.text_input('GNPS2 Task ID', default_task)
if task == '':
    st.error("Please input a valid GNPS2 Task ID")
st.write(task)

# Now we will get all the relevant data from GNPS2 for plotting
labels_url = "https://gnps2.org/resultfile?task={}&file=nf_output/output_histogram_data_directory/labels_spectra.tsv".format(task)
numpy_url = "https://gnps2.org/resultfile?task={}&file=nf_output/output_histogram_data_directory/numerical_spectra.npy".format(task)

# By request, no longer displaying labels url
if False:
    st.write(labels_url)

# read numpy from url into a numpy array
numpy_file = requests.get(numpy_url)
numpy_file.raise_for_status()
numpy_array = np.load(io.BytesIO(numpy_file.content))

# read pandas dataframe from url
all_spectra_df = pd.read_csv(labels_url, sep="\t")

# By request, no longer displaying dataframe table
if False:
    st.write(all_spectra_df) # Currently, we're not displaying db search results

# Collect the database search results
db_search_results = collect_database_search_results(task)

# Get displayable metadata columns for the database search results
if db_search_results is not None:
    # Remove database search result columns we don't want displayed
    invisible_cols = ['query_filename','similarity','query_index','database_index','row_count','database_id','database_scan']
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
# Create a session state for the metadata label    
if "metadata_label" not in st.session_state:
    st.session_state["metadata_label"] = None
# Create a session state for the db search result label
if "db_search_result_label" not in st.session_state and db_search_results is not None:
    st.session_state["db_search_result_label"] = db_search_columns[0]
elif "db_search_result_label" not in st.session_state and db_search_results is None:
    st.session_state["db_search_result_label"] = "No Database Search Results"
# Create a session state for the db similarity threshold
if "db_similarity_threshold" not in st.session_state:
    st.session_state["db_similarity_threshold"] = 0.70
# Create a session state for the maximum number of database results shown
if "max_db_results" not in st.session_state:
    st.session_state["max_db_results"] = -1
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
    st.session_state["show_annotations"] = True

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
    metadata_url = "https://gnps2.org/resultfile?task={}&file=nf_output/output_histogram_data_directory/metadata.tsv".format(task)
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
# Add Metadata dropdown
if metadata_df is None:
    # If there is no metadata, then we will disable the dropdown
    st.session_state["metadata_label"] = st.selectbox("Select a metadata category that will be displayed", ["No Metadata Available"], placeholder="No Metadata Available", disabled=True)
else:
    columns_avaiable = list(metadata_df.columns) + ['None']
    columns_avaialable =[x for x in columns_avaiable if x not in ['filename', 'scan']]
    st.session_state["metadata_label"]  = st.selectbox("Select a metadata category that will be displayed", columns_avaialable, placeholder=columns_avaialable[0])

if db_search_results is None:
    # Write a message saying there are no db search results
    text = "No database search results found for this task."
    st.write(f":grey[{text}]")

else:
    st.subheader("Database Search Result Filters")
    
    # Add DB Search Result dropdown
    st.session_state["db_search_result_label"] = st.selectbox("Database Search Result Column", db_search_columns, placeholder=db_search_columns[0])
    
    
    # Add DB similarity threshold slider
    st.session_state["db_similarity_threshold"] = st.slider("Database Similarity Threshold", 0.0, 1.0, st.session_state["db_similarity_threshold"], 0.05)
    # Create a box for the maximum number of database results shown
    st.session_state["max_db_results"] = st.number_input("Maximum Number of Database Results Shown", min_value=-1, max_value=None, value=-1, help="The maximum number of unique database isolates shown, highest similarity is prefered. Enter -1 to show all database results.")  
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
    st.info("Hiding all isolates is currently disabled. Please select isolates manually.")
    if False:
        st.session_state["hidden_isolates"] = all_spectra_df["filename"].unique()
    # Add disabled multiselect to make this less jarring
    st.multiselect("Select Isolates to be Hidden from the Dendrogram", all_spectra_df["filename"].unique(), disabled=True, label_visibility="collapsed", placeholder="Select isolates to hide from the dendrogram")
else:
    # Add multiselect with update button
    st.session_state["hidden_isolates"] = st.multiselect("Select Isolates to be Hidden from the Dendrogram", all_spectra_df["filename"].unique(), label_visibility="collapsed", placeholder="Select isolates to hide from the dendrogram")

# Remove selected ones from all_spectra_df
all_spectra_df = all_spectra_df[~all_spectra_df["filename"].isin(st.session_state["hidden_isolates"])]
# Add option to add a cutoff line
st.session_state["cutoff"] = st.number_input("Add Cutoff Line", min_value=0.0, max_value=1.0, value=None, help="Add a vertical line to the dendrogram at the specified distance.")
# Add option to show annotations
st.session_state["show_annotations"] = st.checkbox("Display Dendrogram Distances", value=True, help="Show distance annotations on the dendrogram.")

# Process the db search results (it's done in this order to allow for db_search parameters)
all_spectra_df, db_similarity_dict = integrate_database_search_results(all_spectra_df, db_search_results, st.session_state)

# Creating the dendrogram
dendro = create_dendrogram(numpy_array,
                           all_spectra_df,
                           db_similarity_dict,
                           label_column=st.session_state["metadata_label"],
                           db_label_column=st.session_state["db_search_result_label"],
                           metadata_df=metadata_df,
                           db_search_columns=db_search_columns,
                           cluster_method=st.session_state["clustering_method"],
                           coloring_threshold=st.session_state["coloring_threshold"],
                           cutoff=st.session_state["cutoff"],
                           show_annotations=st.session_state["show_annotations"])

st.plotly_chart(dendro, use_container_width=True)

# Add a dropdown allowing for mirror plots:
st.header("Plot Spectra")

def mirror_plot_format_function(df):
    output = []
    for row in df.to_dict(orient="records"):
        if row['db_search_result']:
            output.append(f"DB Result - {row['filename']}")
        else:   
            output.append(row['filename'])
            
    return output
all_options = mirror_plot_format_function(all_spectra_df)

# Select spectra one
st.selectbox("Spectra One", all_options, key='mirror_spectra_one', help="Select the first spectra to be plotted. Database search results are denoted by 'DB Result -'.")
# Select spectra two
st.selectbox("Spectra Two", ['None'] + all_options, key='mirror_spectra_two', help="Select the second spectra to be plotted. Database search results are denoted by 'DB Result -'.")
# Add a button to generate the mirror plot
spectra_one_USI = get_USI(all_spectra_df, st.session_state['mirror_spectra_one'], task)
spectra_two_USI = get_USI(all_spectra_df, st.session_state['mirror_spectra_two'], task)

# If a user is able to get click the buttone before the USI is generated, they may get the page with an old option
st.link_button(label="View Plot", url=get_mirror_plot_url(spectra_one_USI, spectra_two_USI))

# Create a shareable link to this page
st.write("Shareable Link: ")
if st.session_state['db_taxonomy_filter'] is None:
    link = f"https://analysis.idbac.org/?task={task}&metadata_label={st.session_state['metadata_label']}&db_search_result_label={st.session_state['db_search_result_label']}&db_similarity_threshold={st.session_state['db_similarity_threshold']}&max_db_results={st.session_state['max_db_results']}&clustering_method={st.session_state['clustering_method']}&coloring_threshold={st.session_state['coloring_threshold']}&hide_isolates={','.join(st.session_state['hidden_isolates'])}&cutoff={st.session_state['cutoff']}&show_annotations={st.session_state['show_annotations']}"
else:
    link = f"https://analysis.idbac.org/?task={task}&metadata_label={st.session_state['metadata_label']}&db_search_result_label={st.session_state['db_search_result_label']}&db_similarity_threshold={st.session_state['db_similarity_threshold']}&max_db_results={st.session_state['max_db_results']}&db_taxonomy_filter={','.join(st.session_state['db_taxonomy_filter'])}&clustering_method={st.session_state['clustering_method']}&coloring_threshold={st.session_state['coloring_threshold']}&hide_isolates={','.join(st.session_state['hidden_isolates'])}&cutoff={st.session_state['cutoff']}&show_annotations={st.session_state['show_annotations']}"
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