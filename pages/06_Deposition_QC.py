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
import tenacity 
import requests
import json
import os
import base64
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
from urllib.parse import quote
from urllib.parse import urlencode

#####
# A note abote streamlit session states:
# All session states related to this page begin with "dqc_" to reduce the 
# chance of collisions with other pages. 
#####

# Set Page Configuration
st.set_page_config( 
                    page_title="IDBac - Deposition QC", 
                    page_icon="assets/idbac_logo_square.png", 
                    layout="wide",
                    initial_sidebar_state="auto",
                    menu_items=None
                )
custom_css()

st.title("Deposition QC")
st.markdown("""
    This page is for performing quality control on IDBac Deposition Dry Runs. If you are not looking to perform quality control for IDBac Knowledgebase depositions, this page is non-functional.\
        
    Below, pending knowledgebase contributions are shown in a table with their associated metadata. Spectra are shown as they will be processed by IDBac. They should be checked for the following:
    * Minimal noise
    * Minimal baseline slope (high intensity at lower m/z values)
    * A significant number of peaks (at minimum 5-7 peaks) 
            
    \
    
    To help you with this task, numerical QC scores are provided: "Total QC Score", "Peaks Score", "Noise Score", and "Resolving Power Score". The best score in any category is 100.
    * "Total QC Score": A weighted average of "Peaks Score", "Noise Score", and "Resolving Power Score".
    * "Peaks Score": A score based on the number of peaks in the spectrum.
    * "Noise Score": A score estimating the amount of noise in the baseline of a spectrum.
    * "Resolving Power Score": A score estimating the average resolving power of the spectrum based on the m/z and full-width half maximum value of the top 10 peaks.
    * "Status": A categorical score of "Green", "Yellow", or "Red" based on the "Total QC Score".

    \

    These metrics are inspired by the methodology introduced [here](https://www.nature.com/articles/s41597-025-04504-z), but are not identical.
            
    \
            
    You can view additional spectra by changing the page number on the side bar to the left. Below this table, you can find per-scan metrics and links to visualize individual replicate spectra.

""")

# get dqc_task_id from URL
url_task_id = st.query_params.get("task_id", None)

# Initialize session state only if not already set
if 'dqc_task_id' not in st.session_state:
    st.session_state['dqc_task_id'] = url_task_id

# Input box that can override the value
dqc_task_id_input = st.text_input("GNPS2 Task ID", value=st.session_state['dqc_task_id'] or "", help="Enter your GNPS2 Task ID here to select a different task. If you're coming from GNPS2, this is automatically populated.")

# Update session state if user types something new
if dqc_task_id_input != st.session_state['dqc_task_id']:
    st.session_state['dqc_task_id'] = dqc_task_id_input
# Show the task ID
st.write(f"Task Loaded: {st.session_state['dqc_task_id']}")

if st.session_state.get('dqc_task_id') is None:
    st.warning("Please enter a task ID to view the metadata.")
    st.stop()

# Get Metadata
dbc_task_id = st.session_state['dqc_task_id']
if st.session_state['dqc_task_id'].startswith("DEV-"):
    base_url = "http://dev.gnps2.org:4000"
    dbc_task_id = st.session_state['dqc_task_id'].replace("DEV-", "")
elif st.session_state['dqc_task_id'].startswith("BETA-"):
    base_url = "https://beta.gnps2.org"
    dbc_task_id = st.session_state['dqc_task_id'].replace("BETA-", "")
else:
    base_url = "https://gnps2.org"

metadata_url = f"{base_url}/resultfile?task={dbc_task_id}&file=metadata_converted/converted_metadata.tsv"
try:
    metadata_df = pd.read_csv(metadata_url, sep="\t", index_col=False)
except Exception as e:
    print("Error loading metadata from URL:", metadata_url, flush=True)
    st.error("Error: Unable to load metadata. Please check the task ID.")
    st.write(f"Error: {e}")
    st.stop()

qc_url = f"{base_url}/resultfile?task={dbc_task_id}&file=nf_output/qc/combined_output.tsv"
qc_df = None
try:
    qc_df = pd.read_csv(qc_url, sep=",", index_col=False)
except Exception as e:
    print("Error loading QC data from URL:", qc_url, flush=True)
    st.error("Error: Unable to load QC data. Is this an old task?")
    st.write(f"Error: {e}")

# Get all spectra
filenames = metadata_df["Filename"].unique()

def get_USI(filename: str, task:str):
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
    output_USI = f"mzspec:GNPS2:TASK-{task}-nf_output/merged/{filename}:scan:1"
    return output_USI

def image_to_base64(img):
    """Convert a PIL image to base64 string."""
    with BytesIO() as buffer:
        from PIL import Image
        img = Image.open(BytesIO(img))
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(5), reraise=True)
def fetch_with_retries(url):
    return requests.get(url)

@st.cache_resource(ttl="1d", max_entries=400, show_spinner=True)
def request_image(usi: str):
    """
    Request an image from the USI.
    
    Parameters:
    - usi (str): The USI of the spectrum.
    
    Returns:
    - image (bytes): The image data.
    """
    if usi is None:
        return None

    query = urlencode({"usi1": usi,
                       "width":8.0,
                       "height":6.0,})
    url = f"https://metabolomics-usi.gnps2.org/png/?{query}"

    response = fetch_with_retries(url)
    if response.status_code == 200:
        return image_to_base64(response.content)
    else:
        print("Returning None for image due to error:", response.status_code, flush=True)
        print("URL:", url, flush=True)
        raise ValueError(f"Error fetching image for USI {usi}: {response.status_code}")
    
def request_img_cache_wrapper(usi: str):
    """
    Wrapper function to request an image with caching.
    
    Parameters:
    - usi (str): The USI of the spectrum.
    
    Returns:
    - image (bytes): The image data.
    """
    try:
        return request_image(usi)
    except Exception as e:
        return None

# Add USI to metadata_df
metadata_df["USI"] = metadata_df.apply(lambda x: get_USI(x["Filename"], dbc_task_id), axis=1)

def aggregate_qc_data(df):
    """For each "Filename", get the lowest "Total QC Score" (note value "Error" is lower than zero). Then,
    get the ["Status", "Peaks Score", "Noise Score", "Baseline Score", "Resolving Power Score"]
    associated with that value. 
    """

    aggregated_df = df.groupby("Filename").apply(lambda group: group.loc[group["Total QC Score"].replace("Error", -1).astype(float).idxmin()]).reset_index(drop=True)
    
    aggregated_df.rename(columns={
        "scan": "Scan with Worst QC Score",
        "Total QC Score": "Worst Total QC Score",
    }, inplace=True)

    return aggregated_df

if qc_df is not None:
    # Drop "Baseline Score" since it's not being used in the total equation
    if "Baseline Score" in qc_df.columns:
        qc_df.drop(columns=["Baseline Score"], inplace=True)
    qc_df.rename(columns={"original_filename": "Filename"}, inplace=True)

    aggregated_qc_df = aggregate_qc_data(qc_df)
    qc_and_metadata_df = aggregated_qc_df.merge(metadata_df, on="Filename", how="right")
else:
    qc_and_metadata_df = metadata_df.copy()


# Display pagenated dataframe
def display_dataframe(df):
    page_size = 10  # Number of rows per page
    total_rows = len(df)
    total_pages = (total_rows // page_size) + int(total_rows % page_size != 0)

    # Initialize page in session state if it doesn't exist
    if "page" not in st.session_state:
        st.session_state.page = 1

    page_input = st.sidebar.text_input("Go to page (1 to {})".format(total_pages), str(st.session_state.page))
    if page_input:
        try:
            page_number = int(page_input)
            if 1 <= page_number <= total_pages:
                st.session_state.page = page_number
        except ValueError:
            st.sidebar.warning("Please enter a valid page number.")

    # Get the current page's data
    page = st.session_state.page
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    relevant_df = df.iloc[start_idx:end_idx].copy()

    # Add base64 image strings
    relevant_df["Image"] = relevant_df["USI"].apply(
        lambda x: request_img_cache_wrapper(x) if x is not None else None
    )
    
    # Format image strings as `data:image/png;base64,...`
    relevant_df["Image"] = relevant_df["Image"].apply(
        lambda x: f"data:image/png;base64,{x}" if x is not None else None
    )

    # Reorder columns to make "Image" the first column
    columns = ["Image"] + [col for col in relevant_df.columns if col != "Image"]
    relevant_df = relevant_df[columns]

    column_config = {
        "Image": st.column_config.ImageColumn("Spectrum", width='large', pinned=True),
        "Filename": st.column_config.TextColumn("Filename", pinned=True),
    }

    st.data_editor(
        relevant_df,
        column_config=column_config,
        column_order=[x for x in relevant_df.columns if x not in {'16S Sequence', }], # Hide the 16S Sequence column
        use_container_width=True,
        hide_index=True,
        disabled=True,                  # Disable editing
        row_height=320,
        height=320*page_size+40,               # Adjust height for 10 rows
    )

    # Display page range
    st.sidebar.write(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")


# Display the metadata dataframe
display_dataframe(qc_and_metadata_df)

# Give the option to select a filename and view the per-scan QC results
if qc_df is not None:
    st.subheader("Per-Scan QC Results")
    qc_df['View Raw Scan'] = qc_df.apply(
        lambda row: f"https://metabolomics-usi.gnps2.org/dashinterface/?usi1=mzspec:GNPS2:TASK-{dbc_task_id}-input_spectra_folder/{row['Filename']}:scan:{row['scan']}",
        axis=1
    )
    try:
        selected_filename = st.selectbox(
            "Select a filename to view per-scan QC results:",
            options=qc_df["Filename"].unique()
        )
        if selected_filename:
            selected_qc_data = qc_df[qc_df["Filename"] == selected_filename]
            st.write(f"QC results for {selected_filename}:")
            st.dataframe(
                selected_qc_data,
                column_config={
                    "View Raw Scan": st.column_config.LinkColumn("View Raw Scan")
                },
                use_container_width=True,
                hide_index=True
            )
    except Exception as e:
        print("Error displaying per-scan QC results:", e, flush=True)
        st.error("Error: Unable to display per-scan QC results. Is the QC data properly formatted?")