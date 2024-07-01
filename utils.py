import streamlit as st
import requests
import yaml
import pandas as pd
import io
from xml.etree import ElementTree
import time

def write_job_params(task_id:str):
    if task_id.startswith("DEV-"):
        params_url = f"http://ucr-lemon.duckdns.org:4000/resultfile?task={dev_task_id}&file=job_parameters.yaml"
        merge_params_url = f"http://ucr-lemon.duckdns.org:4000/resultfile?task={dev_task_id}&file=nf_output/merge_parameters.txt"
    else:
        params_url = f"https://gnps2.org/resultfile?task={task_id}&file=job_parameters.yaml"
        merge_params_url = f"https://gnps2.org/resultfile?task={task_id}&file=nf_output/merge_parameters.txt"

    r = requests.get(params_url,timeout=(60,60))
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

    # Get bin size from merge
    r = requests.get(merge_params_url,timeout=(60,60))
    if r.status_code == 500:
        # This is an old file, the bin size is 10.0
        bin_size = 10.0
    else:
        r.raise_for_status()
        merge_params = yaml.safe_load(r.text)
        bin_size = merge_params.get('bin_size', None)
        if bin_size is None:
            st.error("Bin size not found in merge parameters. Please re-run the task.")
            st.stop()
    yaml_content['bin_size'] = bin_size
    
    st.info(f"""
                **Workflow Parameters:** \n
                **Description:** {yaml_content.get('description', '')}  
                **Merge Replicates:** {yaml_content.get('merge_replicates', 'Unknown Parameter')}  
                **Distance Measure:** {yaml_content.get('distance', 'Unkown Parameter')}  
                **Database Search Threshold:** {yaml_content.get('database_search_threshold', 'Unkown Parameter')}  
                **Protein Database Search Mass Range:** {protein_mass_range}  
                **Metadata File Provided:** {metadata_provided}
                **Heatmap Bin Size:** {bin_size}
                """)

    return yaml_content
    
def write_warnings(param_url:str)->None:
    r = requests.get(param_url,timeout=(60,60))
    if r.status_code == 500:
        st.warning(f"**Warning:** This is an old task and warnings are not available. Please re-run the task if you want to gather warnings.")
        return None
    
    # No warnings
    if r.text == '':
        return None
    
    warnings_df = pd.read_csv(io.StringIO(r.text), index_col=0)
    if 'Error_Level' in warnings_df.columns:    # For backwards compatability
        errors_df   = warnings_df.loc[warnings_df['Error_Level'] == 'critical']
        warnings_df = warnings_df.loc[warnings_df['Error_Level'] != 'critical']
    else:
        errors_df = pd.DataFrame()
    
    if len(warnings_df) > 0:
        # For backwards compatability of tasks
        if 'Error' in warnings_df.columns:
            warnings_df['Warning'] = warnings_df['Error']
            warnings_df.drop(columns=['Error'], inplace=True)
            # Drop index
        
        with st.expander(f":warning: View {len(warnings_df)} Input File Warnings"):
            # Show the table
            st.write(warnings_df)

    if len(errors_df) > 0:
        with st.expander(f":exclamation: View {len(errors_df)} Input File Errors"):
            st.write(errors_df)

    return None

import requests
from xml.etree import ElementTree

def get_genbank_metadata(genbank_accession: str) -> dict:
    """
    Fetches metadata and taxonomy for a given GenBank accession number using NCBI's E-utilities API.

    Parameters:
    genbank_accession (str): The GenBank accession number.

    Returns:
    dict: A dictionary containing metadata and taxonomy for the accession.
    """
    # Function to fetch metadata from NCBI
    def _fetch_metadata(accession, db):
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        params = {
            "db": db,
            "term": accession,
            "retmode": "json"
        }
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            raise Exception(f"Error fetching data from NCBI: {response.status_code}")
        return response.json()

    # Function to fetch taxonomy from NCBI
    def _fetch_taxonomy(taxid):
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
        params = {
            "db": "taxonomy",
            "id": taxid,
            "retmode": "json"
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(f"Error fetching taxonomy from NCBI: {response.status_code}")
        return response.json()
    
    # Function to fetch assembly information from NCBI
    def _fetch_assembly(assembly_id):
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
        params = {
            "db": "assembly",
            "id": assembly_id,
            "retmode": "json"
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(f"Error fetching taxonomy from NCBI: {response.status_code}")
        return response.json()

    # Fetch metadata
    try:
        genbank_id = None
        metadata_json = _fetch_metadata(genbank_accession, 'assembly')  # Changed db to 'assembly'
        metadata = {}
        num_results = int(metadata_json['esearchresult'].get('count', 0))
        if num_results > 1:
            st.warning(f"Multiple results found for GenBank accession {genbank_accession}. Using the first one.")
        elif num_results == 1:
            idlist = metadata_json['esearchresult'].get('idlist', [])
            if len(idlist) > 1:
                st.warning(f"Multiple taxonomy ids found for GenBank accession {genbank_accession}. Using the first one.")
            elif len(idlist) == 1:
                genbank_id = idlist[0]
        # at this point taxid will be None if it was not identified            
            
        # Fetch taxid if genbank_id is found
        if genbank_id:
            taxid = None
            # Overwrite metadata_json with assembly information
            metadata_json = _fetch_assembly(genbank_id)
            if 'error' in metadata_json['result'][genbank_id]:
                raise Exception(f"Error fetching assembly information for GenBank accession {genbank_accession}: {metadata_json['result'][genbank_id]['error']}")
            taxid = metadata_json['result'][genbank_id].get('taxid', None)

        # Fetch taxonomy if taxid is found
        if taxid:
            taxonomy_json = None
            taxonomy_json = _fetch_taxonomy(taxid)
            taxonomy_json = taxonomy_json['result'][taxid]
            if 'error' in taxonomy_json:
                st.warning(f"Error fetching taxonomy for GenBank accession {genbank_accession}: {taxonomy_json['error']}")
                taxonomy_json = None

        # Filter the metadata to only what we want to integrate:
        if taxonomy_json:
            output_dict =  {
                'Genbank accession': genbank_accession,
                'Genbank-Division': taxonomy_json.get('division', 'Unknown'),
                'Genbank-Scientific Name': taxonomy_json.get('scientificname', 'Unknown'),
                'Genbank-Genus': taxonomy_json.get('genus', 'Unknown'),
                'Genbank-Species': taxonomy_json.get('species', 'Unknown'),
                'Genbank-Subspecies': taxonomy_json.get('subsp', 'Unknown'),
                'Genbank-GenBank Division': taxonomy_json.get('genbankdivision', 'Unknown'),
            }
            
            for key in output_dict:
                if output_dict[key] == '':
                    output_dict[key] = 'None'
            
            return output_dict, 1
    except Exception as e:
        return {
            'Genbank accession': genbank_accession,
            'Genbank-Division': 'Unknown',
            'Genbank-Scientific Name': 'Unknown',
            'Genbank-Genus': 'Unknown',
            'Genbank-Species': 'Unknown',
            'Genbank-Subspecies': 'Unknown',
            'Genbank-GenBank Division': 'Unknown',
        }, 0
        
@st.cache_data(max_entries=1000)
def enrich_genbank_metadata(df:pd.DataFrame)->pd.DataFrame:
    """Enriches a DataFrame with metadata from GenBank accessions based on the 'Genbank accession' column.
    returns a DataFrame with the metadata added as new columns. If no metadata was found for all accessions,
    the original DataFrame is returned unchanged.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to enrich.
    
    Returns:
    pd.DataFrame: The enriched DataFrame.
    """
    
    # Get the unique genbank accessions
    genbank_accessions = df['Genbank accession'].dropna().unique()
    
    # Get the metadata for each genbank accession
    result = []
    for accession in genbank_accessions:
        result.append(get_genbank_metadata(accession))
        time.sleep(0.1)    # To avoid rate limiting
    
    # Check that they are not all empty
    metadata = [r[0] for r in result if r[1] == 1]
    success  = [r[1] for r in result]
    
    if sum(success) == 0:
        return df
    
    # Create a DataFrame from the metadata
    metadata_df = pd.DataFrame(metadata)
    
    # Merge the metadata with the original DataFrame
    df = df.merge(metadata_df, left_on='Genbank accession', right_on='Genbank accession')
    
    return df

def custom_css():
    # Some custom CSS to allow for markdown labels on buttons
    st.markdown(
        """
        <style>
        .button-label {
            margin-bottom: 5px;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def metadata_validation(metadata_table:pd.DataFrame, spectrum_df:pd.DataFrame):
    """Validates the metadata table against the spectrum file. This function checks the following:
    1. There are no duplicate values in the "Filename" column. If there are, the fucntion will raise an error and halt execution.
    2. That the values in the "Filename" column of the metadata_table and spectrum_df match. If they do not,
        the function will display a warning.
    Any changes to this function should also be reflected in the IDBac workflow code.
    """

    # Check for duplicates in the metadata table
    duplicated_rows = metadata_table[metadata_table['Filename'].duplicated(keep=False)]
    if not duplicated_rows.empty:
        st.error("The metadata table contains duplicate values in the 'Filename' column. Please remove duplicates and try again.")
        st.write("Duplicated Rows:", duplicated_rows)
        st.stop()

    # Check that the values in the metadata table and spectrum_df match
    metadata_filenames = set(metadata_table['Filename'])
    spectrum_filenames = set(spectrum_df['filename'])
    
    filenames_in_metadata_not_in_spectrum = list(metadata_filenames - spectrum_filenames)
    filenames_in_spectrum_not_in_metadata = list(spectrum_filenames - metadata_filenames)
    
    if len(filenames_in_metadata_not_in_spectrum) > 0:
        with st.expander(":warning: Filenames in metadata table not in spectrum file:"):
            st.write(filenames_in_metadata_not_in_spectrum)
    
    if len(filenames_in_spectrum_not_in_metadata) > 0:
        with st.expander(":warning: Filenames in spectrum file not in metadata table:"):
            st.write(filenames_in_spectrum_not_in_metadata)