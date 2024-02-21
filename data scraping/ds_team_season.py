from soccerdata import FBref
import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd

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

# The rest of your code remains the same...

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
for stat_type, stats in all_stats.items():
    print(f"\nStats Type: {stat_type}")
    print(stats.head())
    # stats.to_csv(f"{stat_type}_stats.csv") # Save to CSV if needed

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
        execute_batch(cursor, insert_query, df.values.tolist())
        conn.commit()
        print(f"Data uploaded successfully to table {table_name}.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to upload data to table {table_name}: {e}")
    finally:
        cursor.close()
        conn.close()
