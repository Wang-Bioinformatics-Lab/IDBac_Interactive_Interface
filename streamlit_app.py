import streamlit as st
import pandas as pd
import requests
import numpy as np
import io
import plotly.figure_factory as ff
# Now lets do pairwise cosine similarity
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

def create_dendrogram(data_np, all_spectra_df, selected_distance_fun=cosine_distances, label_column="filename", metadata_df=None):

    if metadata_df is not None:
        all_spectra_df = all_spectra_df.merge(metadata_df, how="left", left_on="filename", right_on="Filename")
        all_spectra_df["label"] = all_spectra_df[label_column].fillna("No Metadata")
    else:
        all_spectra_df["label"] = "No Metadata"

    all_spectra_df["label"] = all_spectra_df["label"] + " - " + all_spectra_df["filename"]
    all_labels_list = all_spectra_df["label"].to_list()

    # Creating Dendrogram
    dendro = ff.create_dendrogram(data_np, orientation='left', labels=all_labels_list, distfun=selected_distance_fun)
    dendro.update_layout(width=800, height=max(15*len(all_labels_list), 350))

    return dendro

# Here we will add an input field for the GNPS2 task ID
task = st.text_input('GNPS2 Task ID', '0e744752fdd44faba37df671b9d1997c')
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

# Creating the dendrogram
dendro = create_dendrogram(numpy_array, all_spectra_df, metadata_df=None)

st.plotly_chart(dendro, use_container_width=True)
