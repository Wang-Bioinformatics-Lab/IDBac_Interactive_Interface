import streamlit as st
import pandas as pd
import requests
import numpy as np
import io
import plotly.figure_factory as ff
# Now lets do pairwise cosine similarity
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

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
    similarity information.
    
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
        num_inputs = len(non_db_search_result_indices)
        
        computed_distances = distfun(data_np[non_db_search_result_indices])
        
        # Add database search results
        db_search_result_filenames = spectrum_data_df.filename[spectrum_data_df['db_search_result'] == True].tolist()
        num_db_search_results = len(db_search_result_filenames)
        
        # Shortcut out to speed up computation
        if num_db_search_results == 0:
            return computed_distances
        
        # In theory this should never happen, but it's a good sanity check
        if num_db_search_results + num_inputs != spectrum_data_df.shape[0]:
            raise Exception("Error in creating distance matrix")
        
        db_distance_matrix = np.zeros((num_inputs, num_db_search_results))
        for i, filename in enumerate(non_db_search_result_filenames):
            for j, db_filename in enumerate(db_search_result_filenames):
                db_sim_lst = db_similarity_dict.get(filename)
                if db_sim_lst is not None:
                    db_sim = db_sim_lst.get(db_filename, 0)
                    db_distance_matrix[i, j] = db_sim
                else:
                    db_distance_matrix[i, j] = 0
        
        # Create a matrix of zeros
        distance_matrix = np.zeros((num_inputs + num_db_search_results, num_inputs + num_db_search_results))
        distance_matrix[:num_inputs, :num_inputs] = computed_distances
        distance_matrix[:num_inputs, num_inputs:] = db_distance_matrix
        distance_matrix[num_inputs:, :num_inputs] = db_distance_matrix.T
        # The bottom right corner is all zeros
        
        return distance_matrix

    return dist_function_wrapper

def create_dendrogram(data_np, all_spectra_df, db_similarity_dict, selected_distance_fun=cosine_distances, label_column="filename", db_label_column=None,metadata_df=None, db_search_columns=None):
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

    Returns:
    - dendro (plotly.graph_objs._figure.Figure): The generated dendrogram as a Plotly figure.
    """
    if metadata_df is not None:
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
        
        all_spectra_df[label_column].fillna("No Metadata", inplace=True)
        all_spectra_df["label"] = all_spectra_df[label_column].fillna("No Metadata")
    else:
        all_spectra_df["label"] = "No Metadata"
        
    # Add metadata for db search results
    if db_label_column != "No Database Search Results":
        all_spectra_df.loc[all_spectra_df["db_search_result"] == True, db_label_column].fillna("No Metadata", inplace=True)
        all_spectra_df.loc[all_spectra_df["db_search_result"] == True, "label"] = all_spectra_df.loc[all_spectra_df["db_search_result"] == True, db_label_column]
        
    all_spectra_df["label"] = all_spectra_df["label"].astype(str) + " - " + all_spectra_df["filename"].astype(str)
    all_labels_list = all_spectra_df["label"].to_list()

    # Creating Dendrogram
    dendro = ff.create_dendrogram(np_data_wrapper(data_np, all_spectra_df[['filename','db_search_result']], db_similarity_dict), orientation='left', labels=all_labels_list, distfun=get_dist_function_wrapper(selected_distance_fun))
    dendro.update_layout(width=800, height=max(15*len(all_labels_list), 350))

    return dendro

def collect_database_search_results(task):
    """
    Collect the database search results from the IDBAC Database. If the database search results are not available, then None is returned.

    Parameters:
    - task (str): The GNPS2 task ID.

    Returns:
    - database_search_results_df (pandas.DataFrame): The dataframe containing the database search results.
    """
    try:
        # Getting the database search results
        database_search_results_url = "https://gnps2.org/resultfile?task={}&file=nf_output/search/enriched_db_results.tsv".format(task)
        database_search_results_df = pd.read_csv(database_search_results_url, sep="\t")
    except:
        database_search_results_df = None
    return database_search_results_df


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
    
    # Apply Similarity Filter
    trimmed_search_results_df = trimmed_search_results_df[trimmed_search_results_df["similarity"] >= similarity_threshold]
       
    # Apply Maximum DB Results Filter
    trimmed_search_results_df = trimmed_search_results_df.sort_values(by="similarity", ascending=False)
    if maximum_db_results != -1:
        trimmed_search_results_df = trimmed_search_results_df.iloc[:maximum_db_results]             # Safe out of bounds
    
    # We will abuse filename because during display, we display "metadata - filename"
    trimmed_search_results_df["filename"] = trimmed_search_results_df[db_label_column].astype(str)
    
    all_spectra_df["db_search_result"] = False
    trimmed_search_results_df["db_search_result"] = True
    
    # Concatenate DB search results
    trimmed_search_results_df = trimmed_search_results_df.drop_duplicates(subset=["database_id"])   # Get unique database hits, assuming databsae_id is unique
    to_concat = trimmed_search_results_df.drop(columns=['query_filename','similarity'])             # Remove similarity info 
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
url_parameters = st.experimental_get_query_params()

default_task = "0e744752fdd44faba37df671b9d1997c"
if "task" in url_parameters:
    default_task = url_parameters["task"][0]
# Add other items to session state if available
if "metadata_label" in url_parameters:
    st.session_state["metadata_label"] = url_parameters["metadata_label"][0]
if "db_search_result_label" in url_parameters:
    st.session_state["db_search_result_label"] = url_parameters["db_search_result_label"][0]
if "db_similarity_threshold" in url_parameters:
    st.session_state["db_similarity_threshold"] = float(url_parameters["db_similarity_threshold"][0])
if "max_db_results" in url_parameters:
    st.session_state["max_db_results"] = int(url_parameters["max_db_results"][0])
if "db_taxonomy_filter" in url_parameters:
    st.session_state["db_taxonomy_filter"] = url_parameters["db_taxonomy_filter"][0].split(",")


task = st.text_input('GNPS2 Task ID', default_task)
if task == '':
    st.error("Please input a valid GNPS2 Task ID")
st.write(task)

# Now we will get all the relevant data from GNPS2 for plotting
labels_url = "https://gnps2.org/resultfile?task={}&file=nf_output/output_histogram_data_directory/labels_spectra.tsv".format(task)
numpy_url = "https://gnps2.org/resultfile?task={}&file=nf_output/output_histogram_data_directory/numerical_spectra.npy".format(task)

st.write(labels_url)

# read numpy from url into a numpy array
numpy_file = requests.get(numpy_url)
numpy_file.raise_for_status()
numpy_array = np.load(io.BytesIO(numpy_file.content))

# read pandas dataframe from url
all_spectra_df = pd.read_csv(labels_url, sep="\t")

st.write(all_spectra_df) # Currently, we're not displaying db search results

# Collect the database search results
db_search_results = collect_database_search_results(task)

# Get displayable metadata columns for the database search results
if db_search_results is not None:
    # Remove database search result columns we don't want displayed
    invisible_cols = ['query_filename','similarity','query_index','database_index','row_count']
    db_search_columns = [x for x in db_search_results.columns if x not in invisible_cols]
    db_taxonomies = db_search_results['db_taxonomy'].str.split(";").to_list()
    # Flatten
    db_taxonomies = [item for sublist in db_taxonomies for item in sublist]
    db_taxonomies = list(set(db_taxonomies))
else:
    db_search_columns = []

# Getting the metadata
metadata_url = "https://gnps2.org/resultfile?task={}&file=nf_output/output_histogram_data_directory/metadata.tsv".format(task)
try:
    metadata_df = pd.read_csv(metadata_url, sep="\t", index_col=False)
except:
    metadata_df = None

##### Create Session States #####
# Create a session state for the metadata label    
if "metadata_label" not in st.session_state:
    st.session_state["metadata_label"] = "filename"
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

##### Add Display Parameters #####
st.subheader("Dendrogram Display Options")
# Add Metadata dropdown
if metadata_df is None:
    # If there is no metadata, then we will disable the dropdown
    st.session_state["metadata_label"] = st.selectbox("Metadata Column", ["No Metadata Available"], placeholder="No Metadata Available", disabled=True)
else:
    st.session_state["metadata_label"]  = st.selectbox("Metadata Column", metadata_df.columns, placeholder=metadata_df.columns[0])

if db_search_results is None:
    # Write a message saying there are no db search results
    text = "No database search results found for this task."
    st.write(f":grey[{text}]")

else:
    # Add DB Search Result dropdown
    st.session_state["db_search_result_label"] = st.selectbox("Database Search Result Column", db_search_columns, placeholder=db_search_columns[0])
    # Add DB similarity threshold slider
    st.session_state["db_similarity_threshold"] = st.slider("Database Similarity Threshold", 0.0, 1.0, 0.70, 0.05)
    # Create a box for the maximum number of database results shown
    st.session_state["max_db_results"] = st.number_input("Maximum Number of Database Results Shown", min_value=-1, max_value=None, value=-1, help="Enter -1 to show all database results.")
    # Create a 'select all' box for the db taxonomy filter
    if st.checkbox("Select All DB Taxonomies", value=True):
        st.session_state["db_taxonomy_filter"] = db_taxonomies
        # Add disabled multiselect to make this less jarring
        st.multiselect("DB Taxonomy Filter", db_taxonomies, disabled=True)
    else:
        # st.session_state["db_taxonomy_filter"] = st.multiselect("DB Taxonomy Filter", db_taxonomies)
        # Add multiselect with update button
        st.session_state["db_taxonomy_filter"] = st.multiselect("DB Taxonomy Filter", db_taxonomies)

# Process the db search results (it's done in this order to allow for db_search parameters)
all_spectra_df, db_similarity_dict = integrate_database_search_results(all_spectra_df, db_search_results, st.session_state)

# Creating the dendrogram
dendro = create_dendrogram(numpy_array, 
                           all_spectra_df, 
                           db_similarity_dict, 
                           label_column=st.session_state["metadata_label"], 
                           db_label_column=st.session_state["db_search_result_label"], 
                           metadata_df=metadata_df, 
                           db_search_columns=db_search_columns)

st.plotly_chart(dendro, use_container_width=True)

# Create a shareable link to this page
st.write("Shareable Link: ")
if st.session_state['db_taxonomy_filter'] is None:
    link = f"https://analysis.idbac.org/?task={task}&metadata_label={st.session_state['metadata_label']}&db_search_result_label={st.session_state['db_search_result_label']}&db_similarity_threshold={st.session_state['db_similarity_threshold']}&max_db_results={st.session_state['max_db_results']}"
else:
    link = f"https://analysis.idbac.org/?task={task}&metadata_label={st.session_state['metadata_label']}&db_search_result_label={st.session_state['db_search_result_label']}&db_similarity_threshold={st.session_state['db_similarity_threshold']}&max_db_results={st.session_state['max_db_results']}&db_taxonomy_filter={','.join(st.session_state['db_taxonomy_filter'])}"
st.code(link)