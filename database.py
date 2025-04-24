import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv()

# Get database credentials from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', 5432)
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Build SQLAlchemy connection string
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

@st.cache_resource
def init_engine():
    """Create and cache SQLAlchemy engine"""
    try:
        engine = create_engine(DB_URL)
        return engine, None
    except Exception as e:
        error_msg = f"Error connecting to database: {e}"
        st.error(error_msg)
        return None, error_msg

@st.cache_data
def load_data_from_db(query, _engine):
    """Execute query and return results with caching

    Args:
        query (str): SQL query to execute
        _engine (sqlalchemy.engine.Engine): SQLAlchemy engine

    Returns:
        tuple: (DataFrame with results or None, error message or None)
    """
    try:
        df = pd.read_sql_query(text(query), _engine)
        return df, None
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        return None, error_msg

def get_schemas(_engine):
    """Get list of schemas from database"""
    query = "SELECT schema_name FROM information_schema.schemata"
    df_schemas, error = load_data_from_db(query, _engine)
    if error:
        return []
    return df_schemas['schema_name'].tolist()

def get_tables(_engine, schema):
    """Get list of tables in a schema"""
    query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
    df_tables, error = load_data_from_db(query, _engine)
    if error:
        return []
    return df_tables['table_name'].tolist()

def get_columns(_engine, selected_schema, selected_table):
    """Get column information for a specific table"""
    query = f"""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = '{selected_schema}' 
    AND table_name = '{selected_table}';
    """
    return load_data_from_db(query, _engine)[0]

def release_engine(engine):
    """Dispose the SQLAlchemy engine when no longer needed"""
    if engine is not None:
        try:
            engine.dispose()
        except Exception as e:
            print(f"Error disposing engine: {str(e)}")
    else:
        print("Warning: Attempted to dispose a None engine")
