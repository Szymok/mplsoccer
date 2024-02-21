from soccerdata import FBref
from typing import List, Dict
import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd

def flatten_columns(df):
    """
    Flatten a DataFrame's multi-level column headers into a single level,
    ensuring key columns are preserved.
    
    Parameters:
    - df: DataFrame with multi-level columns.
    
    Returns:
    - DataFrame with flattened column headers.
    """
    key_columns = ['league', 'season', 'team', 'player', 'nation', 'pos', 'age', 'born']  # Extend as necessary
    
    if isinstance(df.columns, pd.MultiIndex):
        new_columns = []
        for col in df.columns:
            if col[0].lower() in key_columns:
                new_columns.append(col[0])
            else:
                new_col_name = '_'.join(filter(None, map(str, col))).strip()
                new_columns.append(new_col_name)
        df.columns = new_columns
    return df

def download_all_player_season_stats(leagues, seasons, stat_types=None):
    """
    Downloads various player season stats for specified leagues and seasons.
    
    Parameters:
    - leagues: string or iterable, IDs of leagues to include.
    - seasons: string, int, or list, Seasons to include.
    - stat_types: list of str, Types of stats to retrieve. If None, defaults to a predefined list.
    
    Returns:
    - A dictionary of pd.DataFrame(s) with the player season stats for each stat type.
    """
    if stat_types is None:
        stat_types = ['standard', 'shooting', 'passing', 'passing_types', 'goal_shot_creation',
                      'defense', 'possession', 'playing_time', 'misc', 'keeper', 'keeper_adv']
    
    # Initialize FBref with specified leagues and seasons
    fbref = FBref(leagues=leagues, seasons=seasons)
    
    all_stats = {}
    for stat_type in stat_types:
        try:
            print(f"Downloading player {stat_type} stats for leagues: {leagues}, seasons: {seasons}...")
            stats = fbref.read_player_season_stats(stat_type=stat_type)
            flattened_stats = flatten_columns(stats)
            all_stats[stat_type] = flattened_stats
        except TypeError as e:
            print(f"Invalid stat_type '{stat_type}' provided: {e}")
        except Exception as e:
            print(f"Failed to download player {stat_type} stats: {e}")
    
    return all_stats

# Example usage
# leagues = "ESP-La Liga"
# seasons = ["23-24"]
# player_stats = download_all_player_season_stats(leagues, seasons)

# for stat_type, stats in player_stats.items():
#     print(f"\nStats Type: {stat_type}")
#     print(stats.head())

def connect_to_db(host='localhost', dbname='soccer_stats', user='username', password='password'):
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

def upload_df_to_postgres(df, table_name, conn_details):
    """
    Uploads a DataFrame to a PostgreSQL table.
    
    Parameters:
    - df: DataFrame to upload.
    - table_name: Name of the target table in the database.
    - conn_details: Dictionary containing connection details (host, dbname, user, password).
    """
    conn, cursor = connect_to_db(**conn_details)
    cols = ','.join(list(df.columns))
    values_placeholder = ','.join(['%s'] * len(df.columns))
    insert_query = f'INSERT INTO {table_name} ({cols}) VALUES ({values_placeholder})'
    
    try:
        execute_batch(cursor, insert_query, df.values)
        conn.commit()
        print(f"Data uploaded successfully to table {table_name}.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to upload data to table {table_name}: {e}")
    finally:
        cursor.close()
        conn.close()
