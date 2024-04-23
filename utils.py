import streamlit as st
import requests
import yaml
import pandas as pd
import io

def write_job_params(param_url:str):
    r = requests.get(param_url,timeout=(60,60))
    r.raise_for_status()
    
    yaml_content = yaml.safe_load(r.text)
        
    if yaml_content.get('input_metadata_file') != 'NO_FILE':
        metadata_provided = 'Yes'
    else:
        metadata_provided = 'No'
    
    if yaml_content.get('database_search_mass_range_lower') is not None and \
        yaml_content.get('database_search_mass_range_upper') is not None:
        protein_mass_range = f"({yaml_content.get('database_search_mass_range_lower')}, {yaml_content.get('database_search_mass_range_upper')})"
    else:
        protein_mass_range = 'Unknown Parameter'
    
    st.info(f"""
                **Workflow Parameters:** \n
                **Description:** {yaml_content.get('description', '')}  
                **Merge Replicates:** {yaml_content.get('merge_replicates', 'Unknown Parameter')}  
                **Distance Measure:** {yaml_content.get('distance', 'Unkown Parameter')}  
                **Database Search Threshold:** {yaml_content.get('database_search_threshold', 'Unkown Parameter')}  
                **Protein Database Search Mass Range:** {protein_mass_range}  
                **Metadata File Provided:** {metadata_provided}
                """)

    return yaml_content
    
def write_warnings(param_url:str)->None:
    r = requests.get(param_url,timeout=(60,60))
    if r.status_code == 500:
        st.warning(f"**Warning:** This is an old task and warnings are not available. Please re-run the task if you want to gather warnings.")
        return None
    r.raise_for_status()
    
    warnings_df = pd.read_csv(io.StringIO(r.text), index_col=0)
    
    if len(warnings_df) > 0:
        # For backwards compatability of tasks
        if 'Error' in warnings_df.columns:
            warnings_df['Warning'] = warnings_df['Error']
            warnings_df.drop(columns=['Error'], inplace=True)
            # Drop index
        
        with st.expander(f":warning: View {len(warnings_df)} Input File Warnings"):
            # Show the table
            st.write(warnings_df)

    return None