import streamlit as st
import requests
import yaml

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
    