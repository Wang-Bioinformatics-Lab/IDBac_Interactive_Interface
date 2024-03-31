import streamlit as st
import streamlit.components.v1 as components
from pyvis import network as net
import pandas as pd
import numpy as np
import json
import requests
import plotly

# Set Page Configuration
st.set_page_config(page_title="IDBac - Small Molecule Association", page_icon=None, layout="wide", initial_sidebar_state="collapsed", menu_items=None)

st.info("Welcome to IDBac's Small Molecule Association Page! This page is currently in development.")

def get_small_molecule_dict():
    if st.session_state['task_id'].startswith("DEV-"):
        url = f"http://ucr-lemon.duckdns.org:4000/resultfile?task={st.session_state['task_id'][4:]}&file=nf_output/small_molecule/summary.json"
    else:
        url = f"https://gnps2.org/resultfile?task={st.session_state['task_id']}&file=nf_output/small_molecule/summary.json"
    
    response = requests.get(url, timeout=(120,120))
    if response.status_code != 200:
        st.error(f"Error loading small molecule summary file: {response.status_code}")
        st.stop()
    
    response_dict = json.loads(response.content)
    
    output_dict = {}
    
    all_filenames = [d['filename'] for d in response_dict]
    
    for d in response_dict:
        if output_dict.get(d['filename']) is None:
            output_dict[d['filename']] = [d]
        else: 
            output_dict[d['filename']].append(d)
            
    # For each filename, combine the m/z and intensity arrays
    for k, v in output_dict.items():
        mz_intensity_dict = {}
        for d in v:
            mz_array = d['m/z array']
            intensity_array = d['intensity array']
            
            for mz, intensity in zip(mz_array, intensity_array):
                if mz in mz_intensity_dict:
                    mz_intensity_dict[mz].append(intensity)
                else:
                    mz_intensity_dict[mz] = [intensity]
            
        for mz, intensities in mz_intensity_dict.items():
            mz_intensity_dict[mz] = sum(intensities) / len(intensities)
                
        mz_array = sorted(list(mz_intensity_dict.keys()))
        intensity_array = [mz_intensity_dict[mz] for mz in mz_array]
        
        output_dict[k] = {
            'm/z array': mz_array,
            'intensity array': intensity_array
        }
    
    return output_dict

def filter_small_molecule_dict(small_molecule_dict):
    
    output = {}
    
    for k, d in small_molecule_dict.items():
        mz_array = d['m/z array']
        intensity_array = d['intensity array']
        
        # Get indices where intensity is above threshold
        indices = [i for i, intensity in enumerate(intensity_array) if intensity > st.session_state.get("sm_relative_intensity_threshold", 0.1)]
        
        # Filter mz_array and intensity_array
        mz_array = [mz_array[i] for i in indices]
        intensity_array = [intensity_array[i] for i in indices]
        
        d['m/z array'] = mz_array
        d['intensity array'] = intensity_array
    
        output[k] = d
    return output

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
    # for small_molecule_filename in all_small_molecule_filenames:
    #     graph.add_node(small_molecule_filename, title=small_molecule_filename)
    
    small_mol_dict = filter_small_molecule_dict(get_small_molecule_dict())
    all_mzs = [small_mol_dict.get('m/z array', []) for small_mol_dict in small_mol_dict.values()]
    all_mzs = [mz for sublist in all_mzs for mz in sublist] # Flatten
    all_mzs = np.unique(all_mzs)
    for mz in all_mzs:
        graph.add_node(str(mz), title=f'{mz} m/z', color='#ff7f0e')
    
    # Add edges from df['Filename] to m/z's associated with df['Small molecule file name']
    for index, row in df.iterrows():
        if not pd.isna(row['Filename']) and not pd.isna(row['Small molecule file name']):
            for mz in small_mol_dict[row['Small molecule file name']]['m/z array']:
                graph.add_edge(row['Filename'], str(mz))
    
    # Generate HTML code for the graph
    html_graph = graph.generate_html()
    
    # html_graph = update_graph_html(html_graph)
    # print(html_graph)
    
    components.html(html_graph, height=height)
    
def make_heatmap():
    """
    Make a heatmap that shows which m/z values are associated with each protein file with which intensity
    """
       
    mapping = st.session_state["metadata_df"]
       
    small_mol_dict = filter_small_molecule_dict(get_small_molecule_dict())
    
    all_mzs = [small_mol_dict.get('m/z array', []) for small_mol_dict in small_mol_dict.values()]
    all_mzs = [mz for sublist in all_mzs for mz in sublist] # Flatten
    all_mzs = np.sort(np.unique(all_mzs))
    
    heatmap = np.zeros((len(st.session_state["sm_selected_isolates"]), len(all_mzs)))
    
    df = pd.DataFrame(heatmap, columns=all_mzs, index=st.session_state["sm_selected_isolates"])
    
    small_mol_dict = get_small_molecule_dict()
    
    for filename in st.session_state["sm_selected_isolates"]:
        relevant_mapping = mapping[mapping['Filename'] == filename]
        all_small_molecule_filenames = relevant_mapping['Small molecule file name'].tolist()
        
        for small_molecule_filename in all_small_molecule_filenames:
            mz_array = small_mol_dict[small_molecule_filename]['m/z array']
            intensity_array = small_mol_dict[small_molecule_filename]['intensity array']
            
            for mz, intensity in zip(mz_array, intensity_array):
                if intensity > st.session_state.get("sm_relative_intensity_threshold", 0.1):
                    df.at[filename, mz] = intensity
            
    # Remove cols with all zeros
    df = df.loc[:, (df != 0).any(axis=0)]
    
    if len(df) != 0:
        st.markdown("Common m/z values between selected isolates and their intensities")
        # Draw Heatmap
        fig = plotly.express.imshow(df.values,
                                    aspect ='equal', 
                                    width=1600, 
                                    # height=1600,
                                    color_continuous_scale='Bluered')
        # Update axis text (we do this here otherwise spacing is not even)
        fig.update_layout(
            xaxis=dict(title="m/z", ticktext=[str(x) for x in df.columns], tickvals=list(range(len(df.columns)))),
            yaxis=dict(title="Isolate", ticktext=list(df.index.values), tickvals=list(range(len(df.index)))),
            margin=dict(t=5, pad=2),
        )
        
        # Add text to color bar to indicate intensity
        fig.update_layout(coloraxis_colorbar=dict(title="Relative Intensity"))
        
        st.plotly_chart(fig)
        
        # Draw Table
        df
        
    

# Add a slider for the relative intensity threshold
st.slider("Relative Intensity Threshold", min_value=0.05, max_value=1.0, value=0.15, step=0.01, key="sm_relative_intensity_threshold")

generate_network()


st.header("Small Molecule Table")
st.multiselect("Select Isolates", st.session_state["metadata_df"]['Filename'].unique(), key='sm_selected_isolates')

make_heatmap()