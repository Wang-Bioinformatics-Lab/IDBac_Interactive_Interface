import streamlit as st
import streamlit.components.v1 as components
from pyvis import network as net
import pandas as pd
import json

# Set Page Configuration
st.set_page_config(page_title="IDBac - Small Molecule Association", page_icon=None, layout="wide", initial_sidebar_state="collapsed", menu_items=None)

st.info("Welcome to IDBac's Small Molecule Association Page! This page is currently in development.")

def generate_network(height=1000, width=600):
    # TODO: Right now we don't integrate all_spectra_df which means there could be nodes that aren't truely in the network
    if st.session_state.get("metadata_df") is None:
        st.error("Please upload a metadata file first.")
        st.stop()
    
    df = st.session_state["metadata_df"]
    if 'Small molecule file name' not in df.columns:
        st.error("Please upload a metadata file with a 'Small molecule file name' column.")
        st.stop()
    
    graph = net.Network(height=f'{height}px', width='100%')
    
    # Add nodes from df['Filename'] and df['Small molecule file name']
    all_filenames = df.loc[~ df['Filename'].isna()].Filename.tolist()
    all_small_molecule_filenames = df.loc[~ df['Small molecule file name'].isna()]['Small molecule file name'].tolist()
    for filename in all_filenames:
        graph.add_node(filename, title=filename)
    for small_molecule_filename in all_small_molecule_filenames:
        graph.add_node(small_molecule_filename, title=small_molecule_filename)
    
    # Add edges from df['Filename] to df['Small molecule file name]
    for index, row in df.iterrows():
        if not pd.isna(row['Filename']) and not pd.isna(row['Small molecule file name']):
            graph.add_edge(row['Filename'], row['Small molecule file name'])
    
    # Generate HTML code for the graph
    html_graph = graph.generate_html()
    
    # html_graph = update_graph_html(html_graph)
    # print(html_graph)
    
    components.html(html_graph, height=height)

generate_network()