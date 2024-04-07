from soccerdata import FBref
import psycopg2
from psycopg2.extras import execute_batch, execute_values
import pandas as pd
import os

def flatten_columns(df):
    """
    Flatten a DataFrame's multi-level column headers into a single level,
    ensuring key columns like 'league', 'season', and 'team' are preserved.
    
    Parameters:
    - df: DataFrame with multi-level columns.
    
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

def download_all_team_season_stats(leagues, seasons, opponent_stats=False):
    """
    Downloads all types of team season stats for specified leagues and seasons,
    and flattens the multi-level column headers.
    
    Parameters:
    - leagues: string or iterable, IDs of leagues to include.
    - seasons: string, int, or list, Seasons to include.
    - opponent_stats: bool, If True, will retrieve opponent stats for each stat type.
    
    Returns:
    - A dictionary of pd.DataFrame(s) with the team season stats for each stat type.
    """
    # Initialize FBref with specified leagues and seasons
    fbref = FBref(leagues=leagues, seasons=seasons)
    
    stat_types = ['standard', 'keeper', 'keeper_adv', 'shooting', 'passing', 'passing_types',
                  'goal_shot_creation', 'defense', 'possession', 'playing_time', 'misc']
    
    all_stats = {}
    for stat_type in stat_types:
        try:
            print(f"Downloading {stat_type} stats for leagues: {leagues}, seasons: {seasons}...")
            stats = fbref.read_team_season_stats(stat_type=stat_type, opponent_stats=opponent_stats)
            # Flatten the columns immediately after download
            flattened_stats = flatten_columns(stats)
            all_stats[stat_type] = flattened_stats
        except ValueError as e:
            print(f"Invalid stat_type '{stat_type}' provided: {e}")
        except Exception as e:
            print(f"Failed to download {stat_type} stats: {e}")
    
    return all_stats

# Example usage
leagues = "ESP-La Liga"  # Or a specific league ID or list of league IDs
seasons = ["23-24"]  # Can be a single season or a list of seasons
all_stats = download_all_team_season_stats(leagues, seasons)
# for stat_type, stats in all_stats.items():
    # print(f"\nStats Type: {stat_type}")
    # print(stats.head())
    # stats.to_csv(f"{stat_type}_stats.csv") # Save to CSV if needed

def connect_to_db(host='psql01.mikr.us', dbname='db_m185', user='m185', password='5854_9a960a'):
    """
    Connects to a PostgreSQL database and returns the connection and cursor.
    
    Parameters:
    - host: Database host address
    - dbname: Name of the database
    - user: Username for the database
    - password: Password for the database user
    
    Returns:
    - conn: Database connection object
    - cursor: Database cursor object
    """
    conn = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password
    )
    return conn, conn.cursor()

def create_schema(schema_name, conn_details):
    conn, cursor = connect_to_db(**conn_details)
    try:
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")
        conn.commit()
        print(f"Schema '{schema_name}' created successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to create schema '{schema_name}': {e}")
    finally:
        cursor.close()
        conn.close()

def upload_df_to_postgres(df, file_name, schema_name, conn_details):
    """
    Uploads a DataFrame to a PostgreSQL table.
    
    Parameters:
    - df: DataFrame to upload.
    - file_name: Name of the CSV file to derive the table name from.
    - schema_name: Name of the schema in the database.
    - conn_details: Dictionary containing connection details (host, dbname, user, password).
    """
    table_name = os.path.splitext(file_name)[0]  # Remove file extension to get table name
    conn, cursor = connect_to_db(**conn_details)
    
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
        conn.close()

conn_details = {
    'host': os.getenv('DB_HOST'),
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

create_schema("team_season", conn_details)

for stat_type, stats in all_stats.items():
    print(f"\nStats Type: {stat_type}")
    print("DataFrame columns:", stats.columns)
    file_name = f"{stat_type}_stats.csv"
    upload_df_to_postgres(stats, file_name, "team_season", conn_details)
