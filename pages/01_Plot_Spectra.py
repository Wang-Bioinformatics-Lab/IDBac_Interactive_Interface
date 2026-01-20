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
from streamlit.errors import StreamlitAPIException
from collections import defaultdict
import plotly.graph_objects as go
from urllib.parse import quote
import logging

# TODO:
# [ ] Get processed spectra from DB, rather than raw

# Set the number of decimal places to round to. Should be consistent with binning setting in workflow
DECIMAL_PLACES=0

#####
# A note abote streamlit session states:
# All session states related to this page begin with "mp_" to reduce the 
# chance of collisions with other pages. 
#####

# Set Page Configuration
st.set_page_config(page_title="IDBac - Plot Spectra", page_icon="assets/idbac_logo_square.png", layout="centered", initial_sidebar_state="auto", menu_items=None)
custom_css()

st.title("Spectra Visualization Page")
st.markdown("""
    This page allows you to visualize protein MS spectra stick and mirror plots. Additionally, raw spectra can be overlaid on the stick plots for comparison. \
            
    To get started, select the spectra you want to visualize, choose whether to include raw spectra, and adjust the plot settings as needed.
    """)

@st.cache_data(ttl=60*5, max_entries=20, show_spinner=True,)
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
    if filename.startswith("KB Result - "):
        filename = filename.replace("KB Result - ", "")
        db_result = True
        
    # Attempt to mitigate issues due to duplicate filenames
    row = all_spectra_df.loc[(all_spectra_df["filename"] == filename) & (all_spectra_df["db_search_result"] == db_result)]

    if db_result:
        # If it's a database search result, use the database_id to get the USI
        raise ValueError("Knowledgebase search results are not supported for this function.")
        output_USI = build_database_result_USI(row["database_id"].iloc[0], row["filename"].iloc[0])
    else:
        # If it's a query, use the query job to get the USI
        output_USI = f"mzspec:GNPS2:TASK-{task}-nf_output/merged/{row['filename'].iloc[0]}:scan:1"
    return output_USI

# def build_database_result_USI(database_id:str, file_name:str):
#     """
#     Build a USI for a database search result given a database_id. Note, if the original
#     task is missing/deteleted from GNPS2, this will not work.
    
#     Parameters:
#     - database_id (str): The database_id of the database search result.
    
#     Returns:
#     - usi (str): The USI of the database search result.
#     """
    
#     # User database id to get original task id
#     # Example URL: https://idbac.org/api/spectrum?database_id=01HHBSS17717HA7VN5C167FYHC
#     url = "https://idbac.org/api/spectrum?database_id={}".format(database_id)
#     r = requests.get(url, timeout=60)
#     retries = 3
#     while r.status_code != 200 and retries > 0:
#         r = requests.get(url, timeout=60)
#         retries -= 1
#     if r.status_code != 200:
#         # Throw an exception for this because the database ids are supplied internally
#         raise ValueError("Database ID not found")
#     result_dictionary = r.json()
#     task = result_dictionary["task"]
#     file_name = result_dictionary["Filename"]
       
#     return_usi = f"mzspec:GNPS2:TASK-{task}-nf_output/merged/{file_name}:scan:1"     ######################## You can't do this
    
#     # Test the USI, so we can return an error message on the page
#     r = requests.get(f"https://metabolomics-usi.gnps2.org/json/?usi1={return_usi}")
#     if r.status_code != 200:
#         # Return None, signifying an error if the USI is not valid, this would imply that the original task is missing/deleted
#         st.error("File Upload Task is Missing or Deleted from GNPS2")
#         return None
    
#     return return_usi

@st.cache_data(ttl=60*5, max_entries=20, show_spinner=True,)
def get_peaks_from_USI(usi:str):
    """ Get the peaks from a USI.

    Parameters:
    - usi (str): The USI of the spectrum.

    Returns:
    - peaks (list): The peaks of the spectrum.
    """

    # Escape reserved characters in USI
    try:
        portion_prior_to_task = usi.split("TASK-", 1)[0]
        task_id_and_beyond = usi.split("TASK-", 1)[1]
        task_id_and_path = task_id_and_beyond.split(':scan:', 1)[0]
        task_id_and_path = quote(task_id_and_path)
        scan = task_id_and_beyond.split(':scan:', 1)[1]
        quoted_usi = f"{portion_prior_to_task}TASK-{task_id_and_path}:scan:{scan}"
    except Exception as e:
        st.error(f"Not a valid USI format: '{usi}'. Error: {str(e)}")
        
    urls = [
        f"https://metabolomics-usi.gnps2.org/json/?usi1={quoted_usi}",
        f"https://de.metabolomics-usi.gnps2.org/json/?usi1={quoted_usi}"
    ]

    for url in urls:
        try:
            logging.warning(f"Fetching peaks from USI: {usi} using URL: {url}")
            r = requests.get(url, timeout=10)
            retries = 3
            while r.status_code != 200 and retries > 0:
                r = requests.get(url, timeout=10)
                retries -= 1
            if r.status_code == 200:
                result_dictionary = r.json()
                peaks = result_dictionary["peaks"]
                return peaks
        except requests.exceptions.RequestException:
            continue
    st.error("GNPS2 USI not found, you may need to rerun the analysis workflow.")
    st.stop()

def get_peaks(all_spectra_df: pd.DataFrame, filename: str, task:str, mass_range: tuple):
    """
    Get the peaks of a given filename.
    
    Parameters:
    - all_spectra_df (pandas.DataFrame): The dataframe containing all spectra data.
    - filename (str): The filename of the spectrum.
    - task (str): The IDBAc_analysis task number
    - mass_range (tuple): The mass range for filtering peaks (min_mz, max_mz)
    
    Returns:
    - usi (str): The USI of the spectrum.
    """
    if filename == 'None':
        return None
    
    db_result = False
    if filename.startswith("KB Result - "):
        filename = filename.replace("KB Result - ", "")
        db_result = True
        
    # Attempt to mitigate issues due to duplicate filenames
    row = all_spectra_df.loc[(all_spectra_df["filename"] == filename) & (all_spectra_df["db_search_result"] == db_result)]

    if db_result:
        # If it's a database search result, use the database_id to get the USI
        peaks = get_peaks_from_db_result(row["database_id"].iloc[0])
    else:
        # If it's a query, use the query job to get the USI
        _task = task.strip("BETA-").strip("DEV-")
        USI = f"mzspec:GNPS2:TASK-{_task}-nf_output/search/query_spectra/{row['filename'].iloc[0]}:scan:1"

        peaks = get_peaks_from_USI(USI)

    # Discretize peaks to two decimal points and add intensities for the same m/z
    peaks_dict = {}
    for peak in peaks:
        mz = round(peak[0], DECIMAL_PLACES)
        if mz not in peaks_dict:
            peaks_dict[mz] = peak[1]
        else:
            peaks_dict[mz] += peak[1]
    peaks = [[mz, intensity] for mz, intensity in peaks_dict.items()]

    # Normalize intensities
    max_intensity = max([peak[1] for peak in peaks])
    peaks = [[peak[0], peak[1] / max_intensity] for peak in sorted(peaks)]

    # Filter peaks based on the selected mass range
    peaks = [[peak[0], peak[1]] for peak in peaks if mass_range[0] <= peak[0] <= mass_range[1]]

    return peaks

def get_raw_peaks(filename: str, task:str, mass_range: tuple):
    """ Get the raw peaks of a given filename.
    
    Parameters:
    - filename (str): The filename of the spectrum.
    - task (str): The IDBAc_analysis task number
    - mass_range (tuple): The mass range for filtering peaks (min_mz, max_mz)
    
    Returns:
    - peaks (list): The raw peaks of the spectrum.
    """
    peaks=None
    is_db_result = False
    if filename.startswith("KB Result - "):
        filename = filename.replace("KB Result - ", "")
        is_db_result = True

    if is_db_result:
        st.warning("Raw peaks for knowledgebase search results are not supported.")

    else:
        if task.startswith("BETA-") or task.startswith("DEV-"):
            st.warning("Raw peaks for beta/dev tasks are not supported.")

        # For query jobs, we can get the raw peaks from the USI
        usi = f"mzspec:GNPS2:TASK-{task.strip('BETA-').strip('DEV-')}-nf_output/raw_merged_for_plotting/{filename}:scan:1"
        peaks = get_peaks_from_USI(usi)


    if peaks:
        # Normalize to base peak
        max_intensity = max([peak[1] for peak in peaks])
        peaks = [[peak[0], peak[1] / max_intensity] for peak in sorted(peaks)]  # This is going to be so big, it may be legitimately better to cast to numpy and back

        # Filter peaks based on the selected mass range
        peaks = [[peak[0], peak[1]] for peak in peaks if mass_range[0] <= peak[0] <= mass_range[1]]

    return peaks

def bin_peaks(peaks, bin_size):
    
    output_dict = defaultdict(float)
    for peak in peaks:
        k = int(peak[0] / bin_size)
        output_dict[k] += peak[1]

    output =  [[k * bin_size, v] for k, v in output_dict.items()]
    output = sorted(output, key=lambda x: x[0])
    return output

def get_peaks_from_db_result(database_id:str):
    """ Get the peaks from a database search result. This function will return the peaks 
    after processing (e.g., baseline correction, merging, and binning). 

    Parameters:
    - database_id (str): The database_id of the database search result.

    Returns:
    - peaks (list): The peaks of the database search result.
    """
    url = f"https://idbac.org/api/spectrum/filtered?database_id={database_id}"

    r = requests.get(url, timeout=60)
    retries = 3
    while r.status_code != 200 and retries > 0:
        r = requests.get(url, timeout=60)
        retries -= 1
    if r.status_code != 200:
        # Throw an exception for this because the database ids are supplied internally
        raise ValueError("Database ID not found")

    result_dictionary = r.json()
    peaks = result_dictionary["peaks"]   # Gives unbinned, unnormalized peaks unmerged by scan
    peaks = [[p['mz'], p['i']] for p in peaks]

    peaks = bin_peaks(peaks, st.session_state['workflow_params']['bin_size'])

    return peaks

def stick_plot(peaks_a, peaks_b=None, raw_peaks_a=None, raw_peaks_b=None, title=None, font_size_multiplier=1.0, gridlines='None'):
    """Create a stick plot for two spectra with the bottom spectra mirrored, if specified.
    
    Parameters:
    - peaks_a (list): The peaks of the first spectrum.
    - peaks_b (list): The peaks of the second spectrum.
    - raw_peaks_a (list): The raw peaks of the first spectrum (optional).
    - raw_peaks_b (list): The raw peaks of the second spectrum (optional).
    - title (str): The title of the plot.
    - font_size_multiplier (float): A multiplier for the font size of the plot.
    - gridlines (str): The type of gridlines to display ('None', 'Horizontal', 'Grid').
    """
    
    base_font_size = 14 * font_size_multiplier
    tick_font_size = 12 * font_size_multiplier
    title_font_size = 18 * font_size_multiplier

    assert gridlines in ['None', 'Horizontal', 'Grid'], "Gridlines must be one of 'None', 'Horizontal', or 'Grid'."

    fig = go.Figure(
        layout=dict(
            xaxis=dict(
                title="m/z",
                title_font=dict(size=base_font_size, color='black'),
                tickfont=dict(size=tick_font_size, color='black'),
                showgrid=(gridlines in ['Grid']),
            ),
            yaxis=dict(
                title="Normalized Intensity",
                title_font=dict(size=base_font_size, color='black'),
                tickfont=dict(size=tick_font_size, color='black'),
                showgrid=(gridlines in ['Grid', 'Horizontal']),
            ),
            title=dict(
                text=title if title else "Spectra Plot",
                font=dict(size=title_font_size, color='black'),
                x=0.5,  # Center the title
                xanchor='center'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            # Set width, height
            width=650,
            height=500
        )
    )
    
    color_dict = {}
    if peaks_b is not None:
        # Color peaks green if they match, blue if not
        # Get intersection
        peaks_a_mz = set([peak[0] for peak in peaks_a])
        peaks_b_mz = set([peak[0] for peak in peaks_b])
        intersection = peaks_a_mz.intersection(peaks_b_mz)
        for mz in intersection:
            color_dict[mz] = "green"
    
    for peak in peaks_a:
        fig.add_trace(go.Scatter(
            x=[peak[0], peak[0]],
            y=[0, peak[1]],
            mode='lines',
            line=dict(color=color_dict.get(peak[0], 'blue')),
			name="" # Hide "Trace 0"
        ))
    if peaks_b is not None:
        # If peaks_b is provided, plot two stick plots
        for peak in peaks_b:
            fig.add_trace(go.Scatter(
                x=[peak[0], peak[0]],
                y=[0, -peak[1]],
                mode='lines',
                line=dict(color=color_dict.get(peak[0], 'blue')),
			name="" # Hide "Trace 0"
            ))

    # Overlay a line plot of the raw spectrum on top of the stick plot
    if raw_peaks_a is not None:
        # Plot the raw spectrum for peaks_a
        fig.add_trace(go.Scatter(
            x=[peak[0] for peak in raw_peaks_a],
            y=[peak[1] for peak in raw_peaks_a],
            mode='lines',
            line=dict(color='black', width=1),
            name="Raw Spectrum A"
        ))
    if raw_peaks_b is not None:
        # Plot the raw spectrum for peaks_b
        fig.add_trace(go.Scatter(
            x=[peak[0] for peak in raw_peaks_b],
            y=[-peak[1] for peak in raw_peaks_b],
            mode='lines',
            line=dict(color='black', width=1),
            name="Raw Spectrum B"
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
    
    # Compute lower and upper bound for range
    lower_range = max(min([peak[0] for peak in peaks_a]) - 100,0)
    upper_range = max([peak[0] for peak in peaks_a]) + 100

    if peaks_b is not None:
        lower_range = min(lower_range, min([peak[0] for peak in peaks_b]) - 100)
        upper_range = max(upper_range, max([peak[0] for peak in peaks_b]) + 100)

    # Update the y-axis with new tick text
    fig.update_layout(
        title=dict(
        text=title,
        x=0.5,  # Center the title
        xanchor='center',
        font=dict(color='black')  # Set the title color to black
        ),
        margin=dict(l=5, r=5, t=50, b=5),
        yaxis=dict(
            tickvals=tickvals,
            ticktext=updated_ticktext,
            title_font_color='black',
            tickfont=dict(
                          color='black'
                   ),
        ),
        showlegend=False,
        xaxis=dict(visible=True,
                   title_font_color='black',
                   tickfont=dict(
                          color='black'
                   ),
                   range=[lower_range, upper_range]),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    # Apply font size multiplier
    fig.update_layout(
        font=dict(
            size=12 * font_size_multiplier,  # Default font size is 12, adjust as needed
            color='black'
        )
    )

    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'IDBac_Protein_Spectra_Plot',
            'scale':5 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }


    st.plotly_chart(fig, config=config)

def draw_mirror_plot(all_spectra_df):
    # Add a dropdown allowing for mirror plots:
    all_options = format_proteins_as_strings(all_spectra_df)

    # Select spectrum one
    st.selectbox("Spectrum One", all_options, key='mirror_spectra_one', help="Select the first spectrum to be plotted. Knowledgebase search results are denoted by 'KB Result -'.")
    # Select spectrum two
    st.selectbox("Spectrum Two", ['None'] + all_options, key='mirror_spectra_two', help="Select the second spectrum to be plotted. Knowledgebase search results are denoted by 'KB Result -'.")

    st.slider(
        "Select Mass Range (m/z):",
        min_value=2000,
        max_value=30000,
        value=(2000, 30000),
        step=10,
        key="mp_mass_range",
        help="Adjust the mass range for spectra visualization and cosine calculation."
    )

    # Add a checkbox to include the raw spectra in the plot
    st.checkbox("Include Raw Spectra", key='include_raw_spectra', value=False,
                help="If checked, the raw spectra will be included in the plot.")
    # Gridlines (horizontal, horizontal & vertical, or none)
    st.selectbox("Gridlines", ['None', 'Horizontal', 'Grid'], key='gridlines')

    # Font size multiplier
    st.slider("Font Size Multiplier", min_value=0.5, max_value=15.0, value=1.0, step=0.1, key='font_size_multiplier',
              help="Adjust the font size of the plot. This is a multiplier, so a value of 1.0 will use the default font size, while a value of 2.0 will double the font size.")
    
    
    # For Local Plot
    if st.session_state['mirror_spectra_two'] == 'None':
        default_title = f"{st.session_state['mirror_spectra_one']}"
    else:
        default_title = f"{st.session_state['mirror_spectra_one']} vs {st.session_state['mirror_spectra_two']}"
    plot_title = st.text_input("Set Title:", value=default_title)

    peaks_a = get_peaks(all_spectra_df, st.session_state['mirror_spectra_one'], st.session_state["task_id"], st.session_state["mp_mass_range"])
    peaks_b = None
    if st.session_state['mirror_spectra_two'] != 'None':
        peaks_b = get_peaks(all_spectra_df, st.session_state['mirror_spectra_two'], st.session_state["task_id"], st.session_state["mp_mass_range"])

    raw_peaks_a = None
    raw_peaks_b = None
    if st.session_state['include_raw_spectra']:
        raw_peaks_a = get_raw_peaks(st.session_state['mirror_spectra_one'], st.session_state["task_id"], st.session_state["mp_mass_range"])
        if st.session_state['mirror_spectra_two'] != 'None':
            raw_peaks_b = get_raw_peaks(st.session_state['mirror_spectra_two'], st.session_state["task_id"], st.session_state["mp_mass_range"])

    stick_plot(peaks_a,
               peaks_b,
               raw_peaks_a,
               raw_peaks_b,
               title=plot_title,
               font_size_multiplier=st.session_state['font_size_multiplier'],
               gridlines=st.session_state['gridlines'])
    
    if peaks_b is not None:
        # Create dictionaries from the peaks
        peaks_a_dict = {peak[0]: peak[1] for peak in peaks_a}
        peaks_b_dict = {peak[0]: peak[1] for peak in peaks_b}

        # Get the union of the two sets of m/z values
        all_mz = set(peaks_a_dict.keys()).union(set(peaks_b_dict.keys()))

        # Create vectors for both sets of peaks, filling with 0 where necessary
        peaks_a_vector = np.array([peaks_a_dict.get(mz, 0) for mz in all_mz])
        peaks_b_vector = np.array([peaks_b_dict.get(mz, 0) for mz in all_mz])

        # Calculate the norms
        norm_a = np.linalg.norm(peaks_a_vector)
        norm_b = np.linalg.norm(peaks_b_vector)

        # Check if any norm is zero to avoid division by zero
        if norm_a == 0 or norm_b == 0:
            cosine_similarity = 0.0
        else:
            # Calculate cosine similarity
            cosine_similarity = np.dot(peaks_a_vector, peaks_b_vector) / (norm_a * norm_b)

        # Presence/Absence cosine similarity
        peaks_a_pa_vector = np.array([1 if peaks_a_dict.get(mz, 0) > 0 else 0 for mz in all_mz])
        peaks_b_pa_vector = np.array([1 if peaks_b_dict.get(mz, 0) > 0 else 0 for mz in all_mz])
        norm_a_pa = np.linalg.norm(peaks_a_pa_vector)
        norm_b_pa = np.linalg.norm(peaks_b_pa_vector)

        if norm_a_pa == 0 or norm_b_pa == 0:
            cosine_similarity_pa = 0.0
        else:
            cosine_similarity_pa = np.dot(peaks_a_pa_vector, peaks_b_pa_vector) / (norm_a_pa * norm_b_pa)

        st.write(f"""
                    Cosine **Distance**: {1 - cosine_similarity:.2f} \n
                    Presence/Absence Cosine **Distance**: {1 - cosine_similarity_pa:.2f} \n
                    Cosine **Similarity**: {cosine_similarity:.2f} \n
                    Presence/Absence Cosine **Similarity**: {cosine_similarity_pa:.2f} \n
                    """)
        
        st.write("""
                 Note: Cosine **distance** ranges from 0 to 1, where 0 indicates identical spectra and 1 indicates no similarity. \
                 Cosine **similarity** ranges from 0 to 1, where 1 indicates identical spectra and 0 indicates no similarity. \
                 Cosine **distance** more closely reflects dendrogram distances, while cosine **similarity** is more commonly used in database search settings. \
                 """)

    # Print the number of matched peaks and the total number of peaks
    if False:
        if peaks_b is not None:
            peaks_a_mz = set([peak[0] for peak in peaks_a])
            peaks_b_mz = set([peak[0] for peak in peaks_b])
            # Remove peaks lower than 3k # DEBUG
            peaks_a_mz = {mz for mz in peaks_a_mz if mz >= 3000}
            peaks_b_mz = {mz for mz in peaks_b_mz if mz >= 3000}

            intersection = peaks_a_mz.intersection(peaks_b_mz)
            st.write(f"Matched Peaks: {len(intersection)}")
            st.write(f"Total Peaks in Spectra One: {len(peaks_a_mz)}")
            st.write(f"Total Peaks in Spectra Two: {len(peaks_b_mz)}")
            st.write(f"Unique m/z's Peaks in Both Spectra: {len(set(peaks_a_mz).union(set(peaks_b_mz)))}")
        else:
            st.write(f"Total Peaks in Spectra One: {len(peaks_a)}")
    
if st.session_state.get('spectra_df') is not None and \
    len(st.session_state['spectra_df']) > 0:
        draw_mirror_plot(st.session_state['spectra_df'])
else:
    st.warning("No protein spectra data found. Please check that this task contains protein spectra data.")