from soccerdata import FBref
import psycopg2
from psycopg2.extras import execute_batch, execute_values
import pandas as pd
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests_toolbelt.sessions import BaseUrlSession

# Load environment variables from .env file
load_dotenv()

def flatten_columns(df):
    key_columns = ['date', 'league', 'season', 'team', 'player', 'position', 'minutes']
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

def create_tor_session():
    # Use Tor network for proxies
    proxies = {
        'http': 'socks5h://127.0.0.1:9050',
        'https': 'socks5h://127.0.0.1:9050'
    }

    # Create a session
    session = BaseUrlSession(base_url='https://fbref.com/')
    session.proxies.update(proxies)

    # Add retry strategy to handle request failures gracefully
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    return session

def download_all_player_match_stats(leagues, seasons, stat_types=None, match_id=None, force_cache=True):
    if stat_types is None:
        stat_types = ['summary', 'keepers', 'passing', 'passing_types', 'defense', 'possession', 'misc']
    
    # Create Tor session
    session = create_tor_session()

    fbref = FBref(leagues=leagues, seasons=seasons)

    # Override the _session attribute of the FBref object
    fbref._session = session

    all_stats = {}
    for stat_type in stat_types:
        try:
            print(f"Downloading player match {stat_type} stats for leagues: {leagues}, seasons: {seasons}...")
            stats = fbref.read_player_match_stats(stat_type=stat_type, match_id=match_id, force_cache=force_cache)
            flattened_stats = flatten_columns(stats)
            all_stats[stat_type] = flattened_stats
        except ValueError as e:
            print(f"Invalid match_id or no games found: {e}")
        except TypeError as e:
            print(f"Invalid stat_type '{stat_type}' provided: {e}")
        except Exception as e:
            print(f"Failed to download player match {stat_type} stats: {e}")
    
    return all_stats

def generate_seasons(start_year, end_year):
    seasons = []
    for year in range(start_year, end_year):
        start_str = str(year)[-2:]
        end_str = str(year + 1)[-2:]
        season = f"{start_str}-{end_str}"
        seasons.append(season)
    return seasons

def add_timestamp_column(df, timestamp):
    df['script_run_time'] = timestamp
    return df

def connect_to_db():
    host = os.getenv('DB_HOST')
    dbname = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    
    conn = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password
    )
    return conn, conn.cursor()

def create_schema(schema_name, conn):
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

def create_table_if_not_exists(df, table_name, schema_name, cursor):
    columns_with_types = []
    for col in df.columns:
        col_name = col.replace('"', '""')
        col_name = f'"{col_name}"'
        if pd.api.types.is_integer_dtype(df[col]):
            col_type = 'INTEGER'
        elif pd.api.types.is_float_dtype(df[col]):
            col_type = 'REAL'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = 'TIMESTAMP'
        else:
            col_type = 'TEXT'
        columns_with_types.append(f'{col_name} {col_type}')
    
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
        {', '.join(columns_with_types)}
    )
    """
    cursor.execute(create_table_query)

def upload_df_to_postgres(df, file_name, schema_name, conn, season):
    table_name = os.path.splitext(file_name)[0]
    cursor = conn.cursor()
    
    df = df.reset_index()
    df.columns = df.columns.str.replace('%', 'percent')

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].where(df[col].notna(), None)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(object).where(df[col].notna(), None)

    try:
        create_table_if_not_exists(df, table_name, schema_name, cursor)

        cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s", (schema_name, table_name))
        existing_columns = [row[0] for row in cursor.fetchall()]
        
        for col in df.columns:
            if col not in existing_columns:
                col_name = col.replace('"', '""')
                col_name = f'"{col_name}"'
                if pd.api.types.is_integer_dtype(df[col]):
                    col_type = 'INTEGER'
                elif pd.api.types.is_float_dtype(df[col]):
                    col_type = 'REAL'
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    col_type = 'TIMESTAMP'
                else:
                    col_type = 'TEXT'
                alter_table_query = f'ALTER TABLE {schema_name}.{table_name} ADD COLUMN {col_name} {col_type}'
                cursor.execute(alter_table_query)

        delete_query = f'DELETE FROM {schema_name}.{table_name} WHERE "season" = %s'
        cursor.execute(delete_query, (season,))

        cols = ','.join(['"' + col.replace('"', '""') + '"' for col in df.columns])
        insert_query = f'INSERT INTO {schema_name}.{table_name} ({cols}) VALUES %s'
        
        data = [tuple(x) for x in df.to_numpy()]
        
        execute_values(cursor, insert_query, data)
        conn.commit()
        print(f"Data for season {season} uploaded successfully to table {schema_name}.{table_name}.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to upload data to table {schema_name}.{table_name}: {e}")
    finally:
        cursor.close()

def main():
    leagues = "ESP-La Liga"
    seasons = generate_seasons(2016, 2017)
    all_stats = download_all_player_match_stats(leagues, seasons)

    conn, cursor = connect_to_db()
    create_schema("player_match", conn)

    current_datetime = datetime.now()

    for stat_type, stats in all_stats.items():
        stats_with_timestamp = add_timestamp_column(stats, current_datetime)
        print(f"\nStats Type: {stat_type}")
        print("DataFrame columns:", stats.columns)
        upload_df_to_postgres(stats, f"{stat_type}_stats", "player_match", conn, "16-17")

if __name__ == "__main__":
    main()
