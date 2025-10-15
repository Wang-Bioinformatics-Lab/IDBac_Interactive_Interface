import streamlit as st
import requests
import json
import numpy as np
from typing import Dict, List, Tuple
from pandas import DataFrame

def get_small_molecule_dict():
    if st.session_state['task_id'].startswith("DEV-"):
        base_url = "http://ucr-lemon.duckdns.org:4000"
        task_id = st.session_state['task_id'].replace("DEV-", "")
    elif st.session_state['task_id'].startswith('BETA-'):
        base_url = "https://beta.gnps2.org"
        task_id = st.session_state['task_id'].replace("BETA-", "")
    else:
        base_url = "https://gnps2.org"
        task_id = st.session_state['task_id']
    url = f"{base_url}/resultfile?task={task_id}&file=nf_output/small_molecule/summary.json"
    
    response = requests.get(url, timeout=(120,120))
    if response.status_code != 200:
        st.error(f"Error loading small molecule summary file: {response.status_code}")
        st.stop()
    
    response_dict = json.loads(response.content)
    
    output_dict = {}
    
    all_filenames = [d['filename'] for d in response_dict]
    
    # Map scans to their filenames
    for scan in response_dict:
        if output_dict.get(scan['filename']) is None:
            output_dict[scan['filename']] = [scan]
        else: 
            output_dict[scan['filename']].append(scan)
            
    # For each filename, combine the m/z and intensity arrays
    for filename, scan_list in output_dict.items():
        mz_intensity_dict = {}
        mz_frequency_dict = {}  # For each m/z, what percent of scans is it in?
        for scan in scan_list:
            mz_array        = scan['m/z array']
            intensity_array = scan['intensity array']
            
            for mz, intensity in zip(mz_array, intensity_array):
                _mz = np.round(float(mz), 0)
                if _mz in mz_intensity_dict:
                    mz_intensity_dict[_mz].append(float(intensity))
                else:
                    mz_intensity_dict[_mz] = [float(intensity)]
            
        for mz, intensities in mz_intensity_dict.items():
            mz_intensity_dict[mz] = sum(intensities) / len(intensities)
            mz_frequency_dict[mz] = len(intensities) / len(scan_list)
                
        mz_array           = sorted(list(mz_intensity_dict.keys()))
        intensity_array    = [mz_intensity_dict[mz] for mz in mz_array]
        mz_frequency_array = [mz_frequency_dict[mz] for mz in mz_array]
        
        output_dict[filename] = {
            'm/z array': mz_array,
            'intensity array': intensity_array,
            'frequency array': mz_frequency_array
        }
    
    return output_dict

def filter_small_molecule_dict(
        small_molecule_dict,
        relative_intensity_threshold=0.1,
        replicate_frequency_threshold=0.7,
        parsed_selected_mzs=[],
        sma_mz_tolerance=0.1,
    )->Dict[str, Dict[str, List[float]]]:
    """ Applies intensity and frequency filters to the small molecule dictionary. Note that the replicate frequency is applied
    independently of the intensity filter.

    Parameters:
    small_molecule_dict (dict): A dictionary of small molecule data

    Returns:
    dict: A filtered dictionary of small molecule data {filename: {m/z array: [float], intensity array: [float], frequency array: [float]}}
    """
    
    output = {}
    
    for k, d in small_molecule_dict.items():
        mz_array        = [float(x) for x in d['m/z array']]
        intensity_array = [float(x) for x in d['intensity array']]
        frequency_array = [float(x) for x in d['frequency array']]
        
        # Get indices where intensity is above threshold
        indices = [i for i, (intensity, frequency) in enumerate(zip(intensity_array, frequency_array)) if intensity > relative_intensity_threshold and frequency > replicate_frequency_threshold]
        # Get indices where m/z is within tolerance
        if len(parsed_selected_mzs) > 0:
            mz_filtered_indices = set()
            
            for i, mz in enumerate(mz_array):
                for filt in parsed_selected_mzs:
                    if isinstance(filt, float):
                        if abs(mz - filt) <= sma_mz_tolerance:
                            mz_filtered_indices.add(i)
                    else:
                        start, end = filt
                        if start <= mz <= end:
                            mz_filtered_indices.add(i)
                            
            indices = list(set(indices).intersection(mz_filtered_indices))
        
        # Filter mz_array and intensity_array
        mz_array = [mz_array[i] for i in indices]
        intensity_array = [intensity_array[i] for i in indices]
        
        d['m/z array'] = mz_array
        d['intensity array'] = intensity_array
    
        output[k] = d
    return output

def load_small_molecule_dict_as_dataframe(small_molecule_dict, bin_size=1.0):
    data = []
    for filename, data_dict in small_molecule_dict.items():
        row = {'filename': filename, 'db_search_result': False}
        mz_array = data_dict['m/z array']
        for mz in mz_array:
            bin_key = f'BIN_{int(mz // bin_size * bin_size)}_{int(mz // bin_size * bin_size + bin_size)}'
            row[bin_key] = row.get(bin_key, 0) + 1
        data.append(row)
    
    df = DataFrame(data)
    # Fill BIN_ columns with 0 where NaN
    df.fillna(0, inplace=True)
    return df