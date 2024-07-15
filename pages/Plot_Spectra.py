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
from utils import custom_css, format_proteins_as_strings

# import StreamlitAPIException
from streamlit.errors import StreamlitAPIException

#####
# A note abote streamlit session states:
# All session states related to this page begin with "mp_" to reduce the 
# chance of collisions with other pages. 
#####

# Set Page Configuration
st.set_page_config(page_title="IDBac - Plot Spectra", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
custom_css()

st.info("Welcome to the IDBac Plot Spectra Page! Here you can view and compare spectra from your IDBac analysis. \
        For the time being, database spectra have exact peaks, workflow spectra have binned peaks.")

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
    # Example URL: https://idbac.org/api/spectrum?database_id=01HHBSS17717HA7VN5C167FYHC
    url = "https://idbac.org/api/spectrum?database_id={}".format(database_id)
    r = requests.get(url, timeout=60)
    retries = 3
    while r.status_code != 200 and retries > 0:
        r = requests.get(url, timeout=60)
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

def get_peaks_from_USI(usi:str):
    """ Get the peaks from a USI.

    Parameters:
    - usi (str): The USI of the spectrum.

    Returns:
    - peaks (list): The peaks of the spectrum.
    """

    url = f"https://metabolomics-usi.gnps2.org/json/?usi1={usi}"

    r = requests.get(url, timeout=60)
    retries = 3
    while r.status_code != 200 and retries > 0:
        r = requests.get(url, timeout=60)
        retries -= 1
    if r.status_code != 200:
        raise ValueError("USI not found")

    result_dictionary = r.json()
    peaks = result_dictionary["peaks"]

    return peaks

def get_peaks(all_spectra_df: pd.DataFrame, filename: str, task:str):
    """
    Get the peaks of a given filename.
    
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
        peaks = get_peaks_from_db_result(row["database_id"].iloc[0])
    else:
        # If it's a query, use the query job to get the USI
        USI = f"mzspec:GNPS2:TASK-{task}-nf_output/merged/{row['filename'].iloc[0]}:scan:1"

        peaks = get_peaks_from_USI(USI)

    # Discretize peaks to two decimal points and add intensities for the same m/z
    peaks_dict = {}
    for peak in peaks:
        mz = round(peak[0], 2)
        if mz not in peaks_dict:
            peaks_dict[mz] = peak[1]
        else:
            peaks_dict[mz] += peak[1]
    peaks = [[mz, intensity] for mz, intensity in peaks_dict.items()]

    # Normalize intensities
    max_intensity = max([peak[1] for peak in peaks])
    peaks = [[peak[0], peak[1] / max_intensity] for peak in sorted(peaks)]

    return peaks

def get_peaks_from_db_result(database_id:str):
    """ Get the peaks from a database search result.

    Parameters:
    - database_id (str): The database_id of the database search result.

    Returns:
    - peaks (list): The peaks of the database search result.
    """
    url = f"https://idbac.org/api/spectrum?database_id={database_id}"

    r = requests.get(url, timeout=60)
    retries = 3
    while r.status_code != 200 and retries > 0:
        r = requests.get(url, timeout=60)
        retries -= 1
    if r.status_code != 200:
        # Throw an exception for this because the database ids are supplied internally
        raise ValueError("Database ID not found")

    result_dictionary = r.json()
    peaks = result_dictionary["spectrum"]

    return peaks

import plotly.graph_objects as go

def stick_plot(peaks_a, peaks_b=None):
    """Create a stick plot for two spectra with the bottom spectra mirrored, if specified.
    
    Parameters:
    - peaks_a (list): The peaks of the first spectrum.
    - peaks_b (list): The peaks of the second spectrum.
    """
    fig = go.Figure(
        layout=dict(
            xaxis=dict(
                title="m/z"
            ),
            yaxis=dict(
                title="Intensity"
            ),
            # Set width, height
            width=650,
            height=500
        )
    )
    
    for peak in peaks_a:
        fig.add_trace(go.Scatter(
            x=[peak[0], peak[0]],
            y=[0, peak[1]],
            mode='lines',
            line=dict(color='blue'),
			name="" # Hide "Trace 0"
        ))
    if peaks_b is not None:
        # If peaks_b is provided, plot two stick plots
        for peak in peaks_b:
            fig.add_trace(go.Scatter(
                x=[peak[0], peak[0]],
                y=[0, -peak[1]],
                mode='lines',
                line=dict(color='blue'),
			name="" # Hide "Trace 0"
            ))
    
    # Get the current y-axis tick values and tick text
    yaxis = fig.layout.yaxis
    tickvals = yaxis.tickvals
    ticktext = yaxis.ticktext
    
    # If tickvals or ticktext is None, generate them from the data
    if tickvals is None:
        tickvals = list(set(y for trace in fig.data for y in trace.y))
        tickvals.sort()
    if ticktext is None:
        ticktext = [str(val) for val in tickvals]
    
    # Update the tick text for negative values
    # updated_ticktext = [
    #     f'{abs(float(val)):.2f}' if val < 0 else f'{float(text):.2f}'
    #     for val, text in zip(tickvals, ticktext)
    # ]

    # Redo ticks to they're every 0.1
    tickvals = np.arange(0, 1.1, 0.1)
    # Add negative values if needed
    if peaks_b is not None:
        tickvals = np.concatenate([np.arange(-1.0, 0.0, 0.1), tickvals])
    updated_ticktext = [f'{abs(val):.1f}' for val in tickvals]


    
    # Update the y-axis with new tick text
    fig.update_layout(
        yaxis=dict(
            tickvals=tickvals,
            ticktext=updated_ticktext,
            title_font_color='black',
            tickfont=dict(
                          color='black'
                   ),
            showgrid=True
        ),
        showlegend=False,
        xaxis=dict(visible=True,
                   title_font_color='black',
                   tickfont=dict(
                          color='black'
                   )),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    st.plotly_chart(fig)

def draw_mirror_plot(all_spectra_df):
    # Add a dropdown allowing for mirror plots:
    st.header("Plot Spectra")

    all_options = format_proteins_as_strings(all_spectra_df)

    # Select spectra one
    st.selectbox("Spectra One", all_options, key='mirror_spectra_one', help="Select the first spectra to be plotted. Database search results are denoted by 'DB Result -'.")
    # Select spectra two
    st.selectbox("Spectra Two", ['None'] + all_options, key='mirror_spectra_two', help="Select the second spectra to be plotted. Database search results are denoted by 'DB Result -'.")
    # Add a button to generate the mirror plot
    spectra_one_USI = get_USI(all_spectra_df, st.session_state['mirror_spectra_one'], st.session_state["task_id"])
    spectra_two_USI = get_USI(all_spectra_df, st.session_state['mirror_spectra_two'], st.session_state["task_id"])
    
    
    # For Local Plot
    peaks_a = get_peaks(all_spectra_df, st.session_state['mirror_spectra_one'], st.session_state["task_id"])
    peaks_b = None
    if st.session_state['mirror_spectra_two'] != 'None':
        peaks_b = get_peaks(all_spectra_df, st.session_state['mirror_spectra_two'], st.session_state["task_id"])

    # Test stick plot
    stick_plot(peaks_a, peaks_b)

    # For Metabolomics Resolver
    def _get_mirror_plot_url(usi1, usi2=None):
        if usi2 is None:
            url = f"https://metabolomics-usi.gnps2.org/dashinterface/?usi1={usi1}"
        else:
            url = f"https://metabolomics-usi.gnps2.org/dashinterface/?usi1={usi1}&usi2={usi2}"
        return url

    # If a user is able to get click the buttone before the USI is generated, they may get the page with an old option
    st.link_button(label="View Binned Peaks In Spectrum Resolver", url=_get_mirror_plot_url(spectra_one_USI, spectra_two_USI))

draw_mirror_plot(st.session_state['spectra_df'])