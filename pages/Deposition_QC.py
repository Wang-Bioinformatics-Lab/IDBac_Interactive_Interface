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
from pages.Plot_Spectra import get_peaks, get_peaks_from_db_result


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
st.write("This page is for performing quality control on IDBac Deposition Dry Runs. Switch between pages of the table via the sidebar.")

# get dqc_task_id from URL
url_task_id = st.query_params.get("task_id", None)

# Initialize session state only if not already set
if 'dqc_task_id' not in st.session_state:
    st.session_state['dqc_task_id'] = url_task_id

# Input box that can override the value
dqc_task_id_input = st.text_input("Task ID", value=st.session_state['dqc_task_id'] or "")

# Update session state if user types something new
if dqc_task_id_input != st.session_state['dqc_task_id']:
    st.session_state['dqc_task_id'] = dqc_task_id_input
# Show the task ID
st.write(f"Task ID: {st.session_state['dqc_task_id']}")

if st.session_state.get('dqc_task_id') is None:
    st.warning("Please enter a task ID to view the metadata.")
    st.stop()

# Get Metadata
if st.session_state['dqc_task_id'].startswith("DEV-"):
        base_url = "http://ucr-lemon.duckdns.org:4000"
elif st.session_state['dqc_task_id'].startswith("BETA-"):
    base_url = "https://beta.gnps2.org"
else:
    base_url = "https://gnps2.org"
try:
    metadata_url = f"{base_url}/resultfile?task={st.session_state['dqc_task_id']}&file=metadata_converted/converted_metadata.tsv"
    metadata_df = pd.read_csv(metadata_url, sep="\t", index_col=False)
except Exception as e:
    st.error("Error: Unable to load metadata. Please check the task ID.")
    st.write(f"Error: {e}")
    st.stop()

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
metadata_df["USI"] = metadata_df.apply(lambda x: get_USI(x["Filename"], st.session_state['dqc_task_id']), axis=1)

# Display pagenated dataframe
def display_dataframe(df,):
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
display_dataframe(metadata_df)
