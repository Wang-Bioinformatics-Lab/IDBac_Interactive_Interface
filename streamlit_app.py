import streamlit as st
import pandas as pd
import requests
import numpy as np
import io
import plotly.figure_factory as ff
# Now lets do pairwise cosine similarity
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

def create_dendrogram(data_np, all_spectra_df, selected_distance_fun=cosine_distances, label_column="filename", metadata_df=None):
    print("refreshing dendrogram")
    print("Label Column", label_column, flush=True)
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

        all_spectra_df["label"] = all_spectra_df[label_column].fillna("No Metadata")
    else:
        print("NO METADATA")
        all_spectra_df["label"] = "No Metadata"

    all_spectra_df["label"] = all_spectra_df["label"].astype(str) + " - " + all_spectra_df["filename"].astype(str)
    all_labels_list = all_spectra_df["label"].to_list()

    # Creating Dendrogram
    dendro = ff.create_dendrogram(data_np, orientation='left', labels=all_labels_list, distfun=selected_distance_fun)
    dendro.update_layout(width=800, height=max(15*len(all_labels_list), 350))

    return dendro

# Here we will add an input field for the GNPS2 task ID
url_parameters = st.experimental_get_query_params()

default_task = "0e744752fdd44faba37df671b9d1997c"
if "task" in url_parameters:
    default_task = url_parameters["task"][0]


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

st.write(all_spectra_df)


# Getting the metadata
metadata_url = "https://gnps2.org/resultfile?task={}&file=nf_output/output_histogram_data_directory/metadata.tsv".format(task)
try:
    metadata_df = pd.read_csv(metadata_url, sep="\t", index_col=False)
except:
    metadata_df = None


# Create a session state for the metadata label    
if "metadata_label" not in st.session_state:
    st.session_state["metadata_label"] = "filename"

# Add Metadata dropdown
if metadata_df is None:
    # If there is no metadata, then we will disable the dropdown
    st.session_state["metadata_label"] = st.selectbox("Metadata Column", ["No Metadata Available"], placeholder="No Metadata Available", disabled=True)
else:
    st.session_state["metadata_label"]  = st.selectbox("Metadata Column", metadata_df.columns, placeholder=metadata_df.columns[0])

# Creating the dendrogram
dendro = create_dendrogram(numpy_array, all_spectra_df, label_column=st.session_state["metadata_label"] , metadata_df=metadata_df)

st.plotly_chart(dendro, use_container_width=True)
