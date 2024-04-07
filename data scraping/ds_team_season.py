from soccerdata import FBref
import psycopg2
from psycopg2.extras import execute_batch, execute_values
import pandas as pd
import os
import json
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

def flatten_columns(df):
    """
    Flatten a DataFrame's multi-level column headers into a single level,
    ensuring key columns like 'league', 'season', and 'team' are preserved.

    Parameters:
    - df (DataFrame): DataFrame with multi-level columns.

    Returns:
    - DataFrame with flattened column headers.
    """
    # Define key columns that should be preserved
    key_columns = ['league', 'season', 'team']
    
    # Check if the DataFrame has multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten the columns, preserving key columns and combining others with an underscore
        new_columns = []
        for col in df.columns:
            # If the first level is one of the key columns, use it as is
            if col[0].lower() in key_columns:
                new_columns.append(col[0])
            else:
                # Combine levels with an underscore for other columns
                new_col_name = '_'.join(filter(None, map(str, col))).strip()
                new_columns.append(new_col_name)
        df.columns = new_columns
    return df

def load_cached_data(cache_file):
    """
    Load cached data from a JSON file.

    Parameters:
    - cache_file (str): Path to the cache file.

    Returns:
    - dict: Cached data if file is found, otherwise an empty dictionary.
    """
    try:
        with open(cache_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_cached_data(cache_file, data):
    """
    Save data to a cache file in JSON format.

    Parameters:
    - cache_file (str): Path to the cache file.
    - data (dict): Data to be cached.
    """
    with open(cache_file, 'w') as file:
        json.dump(data, file)

def download_all_team_season_stats(leagues, seasons, opponent_stats=False):
    """
    Downloads all types of team season stats for specified leagues and seasons,
    and flattens the multi-level column headers.

    Parameters:
    - leagues (str or iterable): IDs of leagues to include.
    - seasons (str, int, or list): Seasons to include.
    - opponent_stats (bool): If True, will retrieve opponent stats for each stat type.

    Returns:
    - dict: A dictionary of pd.DataFrame(s) with the team season stats for each stat type.
    """
    # Initialize FBref with specified leagues and seasons
    fbref = FBref(leagues=leagues, seasons=seasons)
    
    stat_types = ['standard', 'keeper', 'keeper_adv', 'shooting', 'passing', 'passing_types',
                  'goal_shot_creation', 'defense', 'possession', 'playing_time', 'misc']
    
    all_stats = {}
    for stat_type in stat_types:
        try:
            print(f"Downloading {stat_type} stats for leagues: {leagues}, seasons: {seasons}...")
            # Correctly call read_team_season_stats without the 'season' argument
            stats = fbref.read_team_season_stats(stat_type=stat_type, opponent_stats=opponent_stats)
            # Flatten the columns immediately after download
            flattened_stats = flatten_columns(stats)
            all_stats[stat_type] = flattened_stats
        except ValueError as e:
            print(f"Invalid stat_type '{stat_type}' provided: {e}")
        except Exception as e:
            print(f"Failed to download {stat_type} stats: {e}")
    
    return all_stats


def generate_seasons(start_year, end_year):
    """
    Generates a list of season strings from start_year to end_year.
    
    Parameters:
    - start_year: The starting year of the first season (as an integer).
    - end_year: The ending year of the last season (as an integer).
    
    Returns:
    - A list of strings, each representing a season in the format "YY-YY".
    """
    seasons = []
    for year in range(start_year, end_year):
        # Format the current year and the next year as two-digit strings
        start_str = str(year)[-2:]
        end_str = str(year + 1)[-2:]
        # Combine them to form the season string
        season = f"{start_str}-{end_str}"
        seasons.append(season)
    return seasons

def add_timestamp_column(df, timestamp):
    """
    Adds a new column with the provided timestamp to the DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame to which the timestamp column will be added.
    - timestamp (datetime): The timestamp to add as a new column.

    Returns:
    - DataFrame with the new timestamp column.
    """
    df['script_run_time'] = timestamp
    return df

def connect_to_db():
    """
    Connects to a PostgreSQL database using environment variables and returns the connection and cursor.
    
    Returns:
    - conn: Database connection object
    - cursor: Database cursor object
    """
    # Retrieve database connection details from environment variables
    host = os.getenv('DB_HOST')
    dbname = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    
    # Establish the database connection
    conn = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password
    )
    return conn, conn.cursor()

def create_schema(schema_name, conn):
    """
    Create a new schema in the database if it does not exist.

    Parameters:
    - schema_name (str): Name of the schema to create.
    - conn (connection): Database connection object.
    """
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")
        conn.commit()
        print(f"Schema '{schema_name}' created successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to create schema '{schema_name}': {e}")
    finally:
        cursor.close()


def upload_df_to_postgres(df, file_name, schema_name, conn):
    """
    Uploads a DataFrame to a PostgreSQL table.
    
    Parameters:
    - df: DataFrame to upload.
    - file_name: Name of the CSV file to derive the table name from.
    - schema_name: Name of the schema in the database.
    - conn_details: Dictionary containing connection details (host, dbname, user, password).
    """
    table_name = os.path.splitext(file_name)[0]  # Remove file extension to get table name
    cursor = conn.cursor()
    
    # Reset the index to convert MultiIndex columns to regular columns
    df = df.reset_index()
    df.columns = df.columns.str.replace('%', 'percent')

    for col in df.columns:
        # Convert pandas NA and NaT to None, which psycopg2 interprets as SQL NULL
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].where(df[col].notna(), None)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(object).where(df[col].notna(), None)

    try:
        # Drop the existing table (if it exists)
        drop_table_query = f"DROP TABLE IF EXISTS {schema_name}.{table_name}"
        cursor.execute(drop_table_query)
        
        # Create a new table with the desired schema
        columns_with_types = []
        for col in df.columns:
            # Escape double quotes in column names
            col_name = col.replace('"', '""')
            # Wrap the column name in double quotes
            col_name = f'"{col_name}"'
            # Determine the data type of the column
            if pd.api.types.is_integer_dtype(df[col]):
                col_type = 'INTEGER'
            elif pd.api.types.is_float_dtype(df[col]):
                col_type = 'REAL'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_type = 'TIMESTAMP'
            else:
                col_type = 'TEXT'
            columns_with_types.append(f'{col_name} {col_type}')
        
        create_table_query = f"CREATE TABLE {schema_name}.{table_name} ({', '.join(columns_with_types)})"
        cursor.execute(create_table_query)
        
        # Construct the insert query with proper escaping of special characters
        cols = ','.join(['"' + col.replace('"', '""') + '"' for col in df.columns])

        insert_query = f'INSERT INTO {schema_name}.{table_name} ({cols}) VALUES %s'
        
        # Convert DataFrame to a list of tuples
        data = [tuple(x) for x in df.to_numpy()]
        
        # Use execute_values to efficiently insert data
        execute_values(cursor, insert_query, data)
        conn.commit()
        print(f"Data uploaded successfully to table {schema_name}.{table_name}.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to upload data to table {schema_name}.{table_name}: {e}")
    finally:
        cursor.close()

def main():
    """
    Main function to execute the script functionality.
    It downloads team season stats for specified leagues and seasons,
    connects to a PostgreSQL database, creates a schema, and uploads the data.
    """
    leagues = "ESP-La Liga"
    seasons = generate_seasons(2022, 2024)
    all_stats = download_all_team_season_stats(leagues, seasons)

    conn, cursor = connect_to_db()
    create_schema("team_season", conn)

    current_datetime = datetime.now()

    for stat_type, stats in all_stats.items():
        # Add the current date and time as a new column to the DataFrame
        stats_with_timestamp = add_timestamp_column(stats, current_datetime)
        print(f"\nStats Type: {stat_type}")
        print("DataFrame columns:", stats.columns)
        upload_df_to_postgres(stats, f"{stat_type}_stats", "team_season", conn)

if __name__ == "__main__":
    main()

