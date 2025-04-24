import os
import streamlit as st
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database credentials from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Cache the database connection using st.cache_resource
# This is appropriate for connection objects which cannot be serialized
@st.cache_resource
def init_connection():
    """Create and cache database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn, None  # Return connection and no error
    except Exception as e:
        error_msg = f"Error connecting to database: {e}"
        st.error(error_msg)
        return None, error_msg  # Return no connection and error message

# Notice the underscore before conn parameter - this tells Streamlit not to hash this argument
@st.cache_data
def load_data_from_db(query, _conn):
    """Execute query and return results with caching
    
    Args:
        query (str): SQL query to execute
        _conn (psycopg2.extensions.connection): Database connection (underscore prefix prevents hashing)
    
    Returns:
        tuple: (DataFrame with results or None, error message or None)
    """
    try:
        df = pd.read_sql_query(query, _conn)
        return df, None
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        return None, error_msg

def get_schemas(_conn):
    """Get list of schemas from database"""
    query = "SELECT schema_name FROM information_schema.schemata"
    df_schemas, error = load_data_from_db(query, _conn)
    if error:
        return []
    return df_schemas['schema_name'].tolist()

def get_tables(_conn, schema):
    """Get list of tables in a schema"""
    query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
    df_tables, error = load_data_from_db(query, _conn)
    if error:
        return []
    return df_tables['table_name'].tolist()

def get_columns(_conn, selected_schema, selected_table):
    """Get column information for a specific table"""
    query = f"""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = '{selected_schema}' 
    AND table_name = '{selected_table}';
    """
    return load_data_from_db(query, _conn)[0]

def release_connection(conn):
    """Close the database connection when no longer needed"""
    if conn is not None:
        try:
            conn.close()
        except Exception as e:
            print(f"Error closing connection: {str(e)}")
    else:
        print("Warning: Attempted to close a None connection")
