import streamlit as st
from streamlit.components.v1 import html
from utils import write_job_params, write_warnings, enrich_genbank_metadata, metadata_validation, custom_css

import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(page_title="IDBac - Dendrogram", page_icon="assets/idbac_logo_square.png", layout="wide", initial_sidebar_state="auto", menu_items=None)
# Add tracking
html('<script async defer data-website-id="4611e28d-c0ff-469d-a2f9-a0b54c0c8ee0" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)
custom_css()

# Show the table: st.session_state['spectra_df']
st.title("Search Results")
st.markdown("""
    This page shows your knowledgebase search results in table form.
    """)


data = st.session_state['db_search_results']
non_bin_cols = [x for x in data.columns if not x.startswith("BIN_")]
data = data[non_bin_cols]

# Drop useless columns:
useless_cols = ['query_index', 'database_index', 'row_count', 'database_scan', 'db_taxonomy']
data = data.drop(columns=[x for x in useless_cols if x in data.columns])

# Rename columns replacing "_" with " ". Capitalize first letter of each word. Replace "Db" with "KB"
columns = data.columns
columns = {
    col: col.replace("_", " ").title().replace("Db ", "KB ").replace(" Id", "ID").replace("Ncbi", "NCBI") for col in columns
}
data = data.rename(columns=columns)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df
data = filter_dataframe(data)
st.dataframe(data, height=800)