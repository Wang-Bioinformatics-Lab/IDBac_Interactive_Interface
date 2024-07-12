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
from utils import custom_css

# import StreamlitAPIException
from streamlit.errors import StreamlitAPIException

#####
# A note abote streamlit session states:
# All session states related to this page begin with "mp_" to reduce the 
# chance of collisions with other pages. 
#####

# Set Page Configuration
st.set_page_config(page_title="IDBac - Plot Spectra", page_icon=None, layout="wide", initial_sidebar_state="collapsed", menu_items=None)
custom_css()


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
    
    
    def _get_mirror_plot_url(usi1, usi2=None):
        if usi2 is None:
            url = f"https://metabolomics-usi.gnps2.org/dashinterface/?usi1={usi1}"
        else:
            url = f"https://metabolomics-usi.gnps2.org/dashinterface/?usi1={usi1}&usi2={usi2}"
        return url

    # If a user is able to get click the buttone before the USI is generated, they may get the page with an old option
    st.link_button(label="View Plot", url=_get_mirror_plot_url(spectra_one_USI, spectra_two_USI))

draw_mirror_plot(st.session_state['spectra_df'])