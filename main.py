import streamlit as st
import os
from database import init_connection, release_connection, get_schemas, get_tables, get_columns, load_data_from_db
from data_processing import get_unique_seasons_modified, get_unique_teams, filter_season, filter_matchday
from pages.main_page import main_page
from pages.players_analysis import players_analysis_page
from pages.team_analysis import team_analysis_page
from pages.match_analysis import match_analysis_page

# Set page configuration
st.set_page_config(layout="wide", page_title="La Liga Stats Analysis")

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'main'

# Initialize schema state
if 'selected_schema' not in st.session_state:
    st.session_state['selected_schema'] = 'team_season'

if 'selected_table' not in st.session_state:
    st.session_state['selected_table'] = None

# Get database connection
conn, conn_error = init_connection()

if conn_error:
    st.error(f"Database connection error: {conn_error}")
    st.info("Some functionality may be limited.")

# Sidebar for data selection
st.sidebar.title("Data Selection")

# Schema selection
schemas = get_schemas(conn) if conn else []
selected_schema = st.sidebar.selectbox(
    "Select schema:",
    schemas,
    index=schemas.index(st.session_state['selected_schema']) if st.session_state['selected_schema'] in schemas else 0
)
st.session_state['selected_schema'] = selected_schema

# Table selection
tables = get_tables(conn, selected_schema) if conn else []
selected_table = st.sidebar.selectbox("Select table:", tables)
st.session_state['selected_table'] = selected_table

# Load data
if conn and selected_schema and selected_table:
    query = f"SELECT * FROM {selected_schema}.{selected_table};"
    df_database, error = load_data_from_db(query, conn)
    
    if error:
        st.error(f"Error loading data: {error}")
        df_database = None
    else:
        # Preprocess data to handle types and NULL values
        from data_processing import preprocess_dataframe
        df_database = preprocess_dataframe(df_database)
else:
    df_database = None

# Navigation
st.sidebar.title("Navigation")
navigation = st.sidebar.radio(
    "Go to:",
    ["Main Dashboard", "Team Analysis", "Player Analysis", "Match Analysis"]
)

# Map navigation to session state
nav_map = {
    "Main Dashboard": "main",
    "Team Analysis": "team_analysis",
    "Player Analysis": "players_analysis",
    "Match Analysis": "match_analysis"
}

if navigation in nav_map:
    st.session_state['current_page'] = nav_map[navigation]

# Render the appropriate page
if df_database is not None:
    if st.session_state['current_page'] == 'main':
        main_page(df_database, conn)
    elif st.session_state['current_page'] == 'team_analysis':
        team_analysis_page(df_database, conn)
    elif st.session_state['current_page'] == 'players_analysis':
        players_analysis_page(df_database, conn)
    elif st.session_state['current_page'] == 'match_analysis':
        match_analysis_page(df_database, conn)
else:
    st.error("No data available. Please check database connection and selections.")

# Clean up resources
if conn:
    release_connection(conn)
