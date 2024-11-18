import streamlit as st
import numpy as np
import pandas as pd
import plotly
import plotly.figure_factory as ff
from utils import custom_css
from SPARQLWrapper import SPARQLWrapper, JSON
from utils import get_genbank_metadata, _convert_bin_to_mz_tuple

import os
import sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# from Protein_Heatmap import _convert_bin_to_mz_tuple


# import StreamlitAPIException
from streamlit.errors import StreamlitAPIException

#####
# A note abote streamlit session states:
# All session states related to this page begin with "pq_" to reduce the
# chance of collisions with other pages.
#####

# Set Page Configuration
st.set_page_config(page_title="IDBac - Protein Query", page_icon="assets/idbac_logo_square.png", layout="centered", initial_sidebar_state="collapsed", menu_items=None)
custom_css()


st.info("This page is currently in Beta, and under development. All results are subject to change.")

def remove_adducts_to_mass_bins(bins:np.array, hydrogen=True, sodium=False)->np.array:
    """ Extend the mass bins with the mass of hydrogen, sodium and proton

    Args:
        bins (np.array): Ranges currently included in the query (n x 2)
        hydrogen (bool, optional): Add the mass of hydrogen
        sodium (bool, optional): Add the mass of sodium

    Returns:
        np.array: Extended mass bins
    """
    H_mass = 1.007276
    Na_mass = 22.989769

    new_output = []
    if hydrogen:
        new_output.extend([[lb-H_mass, ub-H_mass] for lb, ub in bins])
    if sodium:
        new_output.extend([[lb-Na_mass, ub-Na_mass] for lb, ub in bins])


    return np.round(np.array(new_output), 0)   # Uniprot doesn't have a mass precision higher than 1, so we should just round

def merge_overlapping_bins(bins:np.array)->np.array:
    """Merges bins with overlapping endpoints to reduce query size.
    
    Args:
        bins (np.array): Ranges currently included in the query (n x 2)

    Returns:
        np.array: Merged mass bins
    """
    if len(bins) == 0:
        return bins
    bins = bins[np.argsort(bins[:, 0])]
    merged_bins = [bins[0]]
    for i in range(1, len(bins)):
        if bins[i][0] <= merged_bins[-1][1]:
            merged_bins[-1][1] = bins[i][1]
        else:
            merged_bins.append(bins[i])
    return np.array(merged_bins)

#@st.cache # TODO
def build_and_execute_query(taxid, mass_bins):
    """Builds and executes a SPARQL query to retrieve proteins from Uniprot with a given taxid and mass range.

    Args:
        taxid (int): NCBI TaxID
        mass_bins (np.array): Mass bins to query

    Returns:
        dict: SPARQL results
    """

    sparql_endpoint = "https://sparql.uniprot.org/sparql"
    sparql = SPARQLWrapper(sparql_endpoint)

    # Create query 
    mass_filter = f"""FILTER ( {" || ".join([f"(?mass >={m[0]} && ?mass <= {m[1]})" for m in mass_bins])} 
                            )"""

    query = f"""
    PREFIX up: <http://purl.uniprot.org/core/>
    PREFIX taxon: <http://purl.uniprot.org/taxonomy/>

    SELECT ?protein ?proteinLabel ?mass
    WHERE {{
    # Protein entity with organism filter
    ?protein a up:Protein ;
            up:organism taxon:{int(taxid)} ;
            rdfs:label ?proteinLabel .

    # Link to Chain_Annotation and retrieve mass
    ?protein up:annotation ?annotation .
    ?annotation a up:Chain_Annotation ;
                up:mass ?mass .

    # Optional filter for mass range (if needed)
    # FILTER (?mass >= 2000 && ?mass <= 100000)
    {mass_filter}
    }}
    LIMIT 100
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert(), query

# Collect data from session state
all_spectra_df = st.session_state.get('spectra_df', None)
if all_spectra_df is None:
    raise StreamlitAPIException("No data found (spectra_df). Please select a task on the main page.")

metadata_df = st.session_state.get('metadata_df', None).copy(deep=True)

if metadata_df is None:
    raise StreamlitAPIException("No data found (metadata_df). Please select a task on the main page.")

bin_size = st.session_state['workflow_params'].get('bin_size', None)
if bin_size is None:
    raise StreamlitAPIException("No bin_size found. Is this an old task?")

st.selectbox("Select a Spectrum", all_spectra_df['filename'].unique(), key="pq_protein_selection")

def __convert_bin_to_mz_tuple(bin):
    return _convert_bin_to_mz_tuple(bin, bin_size)

if (str(st.session_state['pq_protein_selection']) != metadata_df['Filename']).all():
    st.warning(f"Filename `{st.session_state['pq_protein_selection']}` not found in metadata sheet.")
    st.stop()

# Collect strain-specific spectra
genbank_accession = metadata_df.loc[metadata_df['Filename'] == st.session_state['pq_protein_selection'], 'Genbank accession'].values[0]
ncbi_taxid = metadata_df.loc[metadata_df['Filename'] == st.session_state['pq_protein_selection'], 'NCBI taxid'].values[0]
binned_spectrum = all_spectra_df.loc[all_spectra_df['filename'] == metadata_df.iloc[0:1]['Filename'].item()]
masses = binned_spectrum[[x for x in binned_spectrum.columns if "BIN_" in str(x)]]
mass_bins = list(map(__convert_bin_to_mz_tuple, masses.columns.values))
mass_intensities = masses.values[0]
# Select non-zero intensity bins
non_zero_mask = mass_intensities > 0
mass_bins = np.array(mass_bins)[non_zero_mask]
mass_intensities = mass_intensities[non_zero_mask]
# Subtract adducts to mass bins
query_bins = mass_bins.copy()
query_bins = remove_adducts_to_mass_bins(query_bins, hydrogen=True, sodium=True)
query_bins = merge_overlapping_bins(query_bins)


st.markdown(f"Metadata GenBank Accession: `{genbank_accession}`")
st.markdown(f"Metadata NCBI TaxID: `{ncbi_taxid}`")

# Verify that we have some form of taxonomic information
genbank_exists = True
if str(genbank_accession).lower() in ['nan', 'none', 'n/a', 'na', ''] or not genbank_accession:
    genbank_exists = False

ncbi_taxid_exists = True
if str(ncbi_taxid).lower() in ['nan', 'none', 'n/a', 'na', ''] or not ncbi_taxid:
    ncbi_taxid_exists = False

# Translate genbank to ncbitaxid if needed
if not ncbi_taxid_exists and genbank_exists:
    t, success = get_genbank_metadata(genbank_accession)

    if success == 0:
        st.error("Failed to convert genbank accession to an NCBI TaxID.")
        st.stop()


    new_tax_id = t.get('TaxID From Genbank', None)
    if new_tax_id:
        # Horizontal line
        st.markdown(f"---")
        st.markdown(f"**Computed:**")
        st.markdown(f"Converted GenBank Accession to NCBI TaxID: `{new_tax_id}`")
        ncbi_taxid = new_tax_id
        ncbi_taxid_exists = True

if not genbank_exists and not ncbi_taxid_exists:
    st.warning("No GenBank Accession or NCBI TaxID found for this protein. Please provide one to continue.")
    st.stop()

results, query = build_and_execute_query(ncbi_taxid, query_bins)

bindings = results['results']['bindings']

data = []
for binding in bindings:
    row = {
        'mass': float(binding.get('mass', {}).get('value', 0)),
        'proteinLabel': binding.get('proteinLabel', {}).get('value', None),
        'protein': binding.get('protein', {}).get('value', None),
    }
    data.append(row)

df = pd.DataFrame(data)
# Display results
df

# Show query:
st.write("SPARQL Query:")
st.code(query, language="sparql")