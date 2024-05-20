import streamlit as st
import numpy as np
import pandas as pd
import plotly
from plotly import figure_factory as ff
import requests

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
    r = requests.get(url)
    retries = 3
    while r.status_code != 200 and retries > 0:
        r = requests.get(url)
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

def format_protins_as_strings(df):
    output = []
    for row in df.to_dict(orient="records"):
        if row['db_search_result']:
            output.append(f"DB Result - {row['filename']}")
        else:   
            output.append(row['filename'])
            
    return output

def draw_mirror_plot(all_spectra_df):
    # Add a dropdown allowing for mirror plots:
    st.header("Plot Spectra")

    all_options = format_protins_as_strings(all_spectra_df)

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

def draw_protein_heatmap(all_spectra_df):
    st.subheader("Protein Spectra m/z Heatmap")
    
    # Options
    all_options = format_protins_as_strings(all_spectra_df)
    selected_proteins = st.multiselect("Select proteins to display", all_options)
    min_freq = st.slider("Minimum m/z Frequency", min_value=0.0, max_value=1.0, step=0.01, value=0.75,
                         help="The minimum number of times an m/z value must be present \
                               in the selected proteins to be displayed.")
    min_intensity = st.slider("Minimum Relative Intensity", min_value=0.0, max_value=1.0, step=0.01, value=0.75,
                              help="The minimum relative intensity value to display.")
    
    
    # Remove "DB Result - " from the selected proteins
    selected_proteins = [x.replace("DB Result - ", "") for x in selected_proteins]
       
    # Set index to filename
    all_spectra_df = all_spectra_df.set_index("filename")
    all_spectra_df = all_spectra_df.loc[selected_proteins, :]
    
    bin_columns = [col for col in all_spectra_df.columns if col.startswith("BIN_")]
    bin_columns = sorted(bin_columns, key=lambda x: int(x.split("_")[-1]))
    all_spectra_df = all_spectra_df.loc[:, bin_columns]
    # Normalize Intensity (Normalize Across Row)
    all_spectra_df = all_spectra_df.div(all_spectra_df.max(axis=1), axis=0)
    # Set zeros to nan
    all_spectra_df = all_spectra_df.replace(0, np.nan)
    # Set all values less than min_intensity to nan
    all_spectra_df = all_spectra_df.where(all_spectra_df > min_intensity)
    # Filter bins by frequency
    min_count = min_freq * len(selected_proteins)
    bin_columns = [col for col in bin_columns if all_spectra_df[col].notna().sum() >= min_count]
    all_spectra_df = all_spectra_df.loc[:, bin_columns]
    
    def _convert_bin_to_mz(bin_name):
        bin = int(bin_name.split("_")[-1])
        bin_size = 10.0
        
        return f"[{bin * bin_size}, {(bin + 1) * bin_size})"
    
    # Remove rows with all nan
    all_spectra_df = all_spectra_df.dropna(how='all', axis='columns')
    
    if len(all_spectra_df.columns) != 0:
        # Note: We transpose the dataframe so that the proteins are on the x-axis
        st.markdown("Common m/z values between selected proteins and their relative intensities.")
        # Draw Heatmap
        dynamic_height = max(500, len(all_spectra_df.columns) * 24) # Dyanmic height based on number of m/z values
        
        # If we're suppled a dendrogram, use it to reorder the heatmap
        x = None
        if False:   # The below code is copied from the small molecule heatmap, but it doesn't work for proteins.
                    # I've left it here in case we want to try to get it working in the future.
           
            # Remove any rows where the filename is not currently selected
            all_filenames = st.session_state['query_only_spectra_df'].filename.values
            all_data      = st.session_state['query_spectra_numpy_data']
            
            # Get the indices of the selected proteins
            selected_indices = [i for i, filename in enumerate(all_filenames) if filename in st.session_state["sma_selected_proteins"]]
            # Get the data for the selected proteins
            numpy_data = all_data[selected_indices]
            
            # Unfortunately, we have to recalculate the dendrogram, because things may cluster differently 
            # depending on the selected proteins.
            # Note though, that we share parameters with the above dendrogram.
            dendro = ff.create_dendrogram(numpy_data,
                                orientation='bottom',
                                labels=st.session_state["sma_selected_proteins"],
                                distfun=st.session_state['distance_measure'],
                                linkagefun=lambda x: linkage(x, method=st.session_state["sma_clustering_method"],),
                                color_threshold=st.session_state["sma_coloring_threshold"])
            
            # Reorder the dataframe based on the dendrogram
            reordered_df = all_spectra_df.reindex(index=dendro.layout.xaxis.ticktext)
            reordered_df = reordered_df.reindex(columns=dendro.layout.yaxis.ticktext)
            all_spectra_df = reordered_df
            # Also us the X values from the dendrogram
            x = dendro.layout.xaxis.tickvals
        
        heatmap = plotly.express.imshow(all_spectra_df.T.values,    # Transpose so m/zs are rows
                                        x=x,
                                        aspect ='auto', 
                                        width=1500, 
                                        height=dynamic_height,
                                        color_continuous_scale='Bluered',)
        # Update axis text (we do this here otherwise spacing is not even)
        heatmap.update_layout(
            xaxis=dict(title="Protein", ticktext=list(all_spectra_df.index.values), tickvals=list(range(len(all_spectra_df.index))), side='top'),
            yaxis=dict(title="m/z", ticktext=[_convert_bin_to_mz(x) for x in all_spectra_df.columns], tickvals=list(range(len(all_spectra_df.columns)))),
            margin=dict(t=5, pad=0),
        )
        
        heatmap.update_coloraxes(cmin=0.0, cmax=1.0, cmid=0.5)
        
        if False: # The below code is copied from the small molecule heatmap, but it doesn't work for proteins.
            #  I've left it here in case we want to try to get it working in the future.
            
            dendrogram_height = 200
            dendrogram_height_as_percent = dendrogram_height / (dynamic_height + dendrogram_height)
            
            fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                                shared_xaxes=True,
                                                vertical_spacing=0.02,
                                                row_heights=[dendrogram_height_as_percent, 1-dendrogram_height_as_percent])
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0), width=1500, height=dynamic_height + dendrogram_height)
        
            for trace in dendro.data:
                fig.add_trace(trace, row=1, col=1)
            
            # Add x-axis labels from dendrogram
            print(dendro.layout.xaxis.ticktext, flush=True)
            fig.update_xaxes(ticktext=dendro.layout.xaxis.ticktext, tickvals=dendro.layout.xaxis.tickvals, row=1, col=1)
            fig.update_xaxes(ticktext=dendro.layout.xaxis.ticktext, tickvals=dendro.layout.xaxis.tickvals, row=2, col=1)
            # Add y labels to dendrogram
            fig.update_yaxes(ticktext=dendro.layout.yaxis.ticktext, tickvals=dendro.layout.yaxis.tickvals, row=1, col=1, title="Dendrogram Distance")
            # Add y labels to heatmap
            fig.update_yaxes(ticktext=heatmap.layout.yaxis.ticktext, tickvals=heatmap.layout.yaxis.tickvals, row=2, col=1,title="m/z")
            
            for trace in heatmap.data:
                fig.add_trace(trace, row=2, col=1)
            
        else:
            fig = heatmap
            
        fig.update_layout(showlegend=False,
                    coloraxis_colorbar=dict(title="Relative Intensity", 
                                            len=min(500, dynamic_height), 
                                            lenmode="pixels", 
                                            y=0.75)
                                        )
        fig.update_coloraxes(cmin=0.0, cmax=1.0, cmid=0.5,colorscale='Bluered')
        
        st.plotly_chart(fig,use_container_width=True)