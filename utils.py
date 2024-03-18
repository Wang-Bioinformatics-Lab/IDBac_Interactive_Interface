import streamlit as st
import requests
import yaml

def write_job_params(param_url:str):
    r = requests.get(param_url,timeout=(60,60))
    r.raise_for_status()
    
    yaml_content = yaml.safe_load(r.text)
    
    print(yaml_content)
    
    """
    {'OMETAFLOW_SERVER': 'http://ometaflow-launchserver:4000',
    'OMETALIBRARY_SERVER': 'http://ometalibrary-web:5000/library',
    'OMETAMASST_SERVER': 'http://ometamasst-web:5000/masst',
    'OMETATASK': '17546de01d44431a91793363c567815d',
    'OMETAUSER': 'mstro016',
    'create_time': '2024-03-08 10:16:24 PST-0800',
    'database_search_threshold': '0.7',
    'description': 'Trial anlysis 2024-09-30',
    'input_metadata_file': 'NO_FILE',
    'input_spectra_folder': '/data/nf_data/server/nf_tasks/17546de01d44431a91793363c567815d/input_spectra',
    'merge_replicates': 'Yes',
    'outdir': '/data/nf_data/server/nf_tasks/17546de01d44431a91793363c567815d',
    'similarity': 'presence',
    'task': '17546de01d44431a91793363c567815d',
    'workflow_version': 'SERVER:0.1.2;WORKFLOW:0.5.1',
    'workflowname': 'idbac_analysis_workflow'}
    """
    
    if yaml_content.get('input_metadata_file') != 'NO_FILE':
        metadata_provided = 'Yes'
    else:
        metadata_provided = 'No'
    
    st.info(f"""
                **Workflow Paramters:** \n
                **Description:** {yaml_content.get('description', '')}  
                **Merge Replicates:** {yaml_content.get('merge_replicates', 'Unknown Parameter')}  
                **Distance Measure:** {yaml_content.get('distance', 'Unkown Parameter')}  
                **Database Search Threshold:** {yaml_content.get('database_search_threshold', 'Unkown Parameter')}  
                **Metadata File Provided:** {metadata_provided}
                """)

    return yaml_content
    