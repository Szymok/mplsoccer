import pandas as pd
import numpy as np
import streamlit as st

def preprocess_dataframe(df):
    """Convert text columns to appropriate numeric types and handle NULL values"""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Identify text columns that should remain as text
    text_columns = ['league', 'season', 'team', 'url', 'script_run_time']
    
    # Convert all non-text columns to numeric, coercing errors to NaN
    for col in df_processed.columns:
        if col not in text_columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Replace 'NULL' strings with NaN
    df_processed.replace('NULL', np.nan, inplace=True)
    
    return df_processed


@st.cache_data(ttl=3600)
def convert_columns_to_numeric(df_data, columns):
    """Convert specified columns to numeric values"""
    df = df_data.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def get_unique_seasons_modified(df_data):
    """Converts season format from a 4-digit number (e.g., '1718') to '2017/2018' and returns unique seasons."""
    seasons_modified = []
    unique_seasons = df_data['season'].unique()
    
    for season in unique_seasons:
        season_str = str(season)
        if len(season_str) == 4 and season_str.isdigit():
            start_year = int(season_str[:2])
            end_year = int(season_str[2:])
            
            # Determine the century for the start and end years
            start_year += 1900 if start_year >= 90 else 2000
            end_year += 1900 if end_year >= 90 else 2000
            
            # Format the season string as '2017/2018'
            season_formatted = f"{start_year}/{end_year}"
            seasons_modified.append(season_formatted)
    
    return sorted(seasons_modified)

def filter_season(df_data, start_season, end_season):
    """Filter dataframe by season range"""
    # Create a copy to avoid modifying the original dataframe
    df = df_data.copy()
    
    # Convert all seasons in df to the readable format
    df['season_readable'] = df['season'].apply(lambda x: f"20{str(x)[:2]}/20{str(x)[2:]}")
    
    # Find unique readable seasons
    unique_seasons_readable = np.unique(df['season_readable']).tolist()
    
    # Check if start and end season are valid
    if start_season not in unique_seasons_readable:
        return pd.DataFrame(), f"Start season {start_season} is not available."
    
    if end_season not in unique_seasons_readable:
        return pd.DataFrame(), f"End season {end_season} is not available."
        
    # Find the index of the start and end seasons
    start_index = unique_seasons_readable.index(start_season)
    end_index = unique_seasons_readable.index(end_season) + 1
    
    # Select seasons within the range
    seasons_selected_readable = unique_seasons_readable[start_index:end_index]
    
    # Filter the DataFrame
    df_filtered_season = df[df['season_readable'].isin(seasons_selected_readable)]
    
    # Drop the temporary 'season_readable' column
    df_filtered_season.drop(columns=['season_readable'], inplace=True)
    
    return df_filtered_season, None

def filter_matchday(df_data, selected_matchdays):
    """Filter dataframe by matchday"""
    df = df_data.copy()
    matchdays_list = []

    for matchday in selected_matchdays:
        if isinstance(matchday, str):
            matchweek_number = matchday.split()[1]
        elif isinstance(matchday, int):
            matchweek_number = str(matchday)
        
        formatted_matchday = f"Matchweek {matchweek_number}"
        matchdays_list.append(formatted_matchday)

    df_filtered_matchday = df[df['round'].isin(matchdays_list)]
    return df_filtered_matchday

def get_unique_matchdays(df_data):
    """Get unique matchdays from dataframe"""
    if 'round' in df_data.columns:
        unique_matchdays = df_data['round'].unique().tolist()
        matchday_numbers = [int(matchday.split()[1]) for matchday in unique_matchdays]
        sorted_matchday_numbers = sorted(matchday_numbers)
        sorted_unique_matchdays = [f"Matchweek {num}" for num in sorted_matchday_numbers]
        return sorted_unique_matchdays
    else:
        return []

def get_unique_teams(df_data):
    """Get unique teams from dataframe"""
    return np.unique(df_data['team']).tolist()

def filter_teams(df_data, all_teams_selected, selected_teams=None):
    """Filter dataframe by selected teams"""
    if all_teams_selected == 'Select teams manually (choose below)' and selected_teams:
        return df_data[df_data['team'].isin(selected_teams)]
    return df_data

def stack_team_dataframe(df_data):
    """Stack and sort team dataframe for readability"""
    # Get all column names from the DataFrame
    column_names = df_data.columns.tolist()
    
    # Create a copy of the dataframe with all columns
    df_filtered = df_data[column_names].copy()

    # Sort by season and team if these columns exist
    if 'season' in df_filtered.columns and 'team' in df_filtered.columns:
        df_sorted = df_filtered.sort_values(['season', 'team'], ascending=[True, True])
    else:
        df_sorted = df_filtered

    return df_sorted

def group_measure_by_attribute(df_data, aspect, attribute, measure):
    """Group data by selected attribute and calculate measure"""
    df = df_data.copy()
    
    # Convert attribute to numeric
    df[attribute] = pd.to_numeric(df[attribute], errors='coerce')
    df.dropna(subset=[attribute], inplace=True)
    
    # Get numeric columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Group and aggregate based on measure
    if measure == 'Absolute':
        df_return = df.groupby([aspect]).sum(numeric_only=True)
    elif measure == 'Mean':
        df_return = df.groupby([aspect]).mean(numeric_only=True)
    elif measure == 'Median':
        df_return = df.groupby([aspect]).median(numeric_only=True)
    elif measure == 'Maximum':
        df_return = df.groupby([aspect]).max(numeric_only=True)
    elif measure == 'Minimum':
        df_return = df.groupby([aspect]).min(numeric_only=True)

    df_return['aspect'] = df_return.index
    if aspect == 'team':
        df_return = df_return.sort_values(by=[attribute], ascending=False)
        
    return df_return

def find_match_game_id(df_data, min_max, attribute, what):
    """Find match with minimum or maximum value for a given attribute"""
    # Create a copy to avoid modifying the original
    df_find = df_data.copy()
    
    # Convert attribute to numeric
    df_find[attribute] = pd.to_numeric(df_find[attribute], errors='coerce')
    df_find.dropna(subset=[attribute], inplace=True)
    
    # Group by relevant columns based on what we're looking for
    if what == "by both teams":
        df_grouped = df_find.groupby(['season', 'team'], as_index=False).agg({attribute: 'sum'})
    else:
        df_grouped = df_find.groupby(['season', 'team'], as_index=False).agg({attribute: 'sum'})
    
    # Get index of min/max value
    if min_max == 'Minimum':
        index = df_grouped[attribute].idxmin()
    elif min_max == 'Maximum':
        index = df_grouped[attribute].idxmax()
    
    # Get relevant information
    season = df_grouped.at[index, 'season']
    value = df_grouped.at[index, attribute]
    team = df_grouped.at[index, 'team']
    
    return [season, value, team]

def validate_input(df, start_season, end_season, teams=None):
    """Validate user input selections"""
    # Validate seasons
    unique_seasons = get_unique_seasons_modified(df)
    
    if start_season not in unique_seasons:
        return False, f"Start season '{start_season}' is not available in the data"
    
    if end_season not in unique_seasons:
        return False, f"End season '{end_season}' is not available in the data"
    
    # Validate season range
    start_idx = unique_seasons.index(start_season)
    end_idx = unique_seasons.index(end_season)
    
    if start_idx > end_idx:
        return False, "Start season cannot be after end season"
    
    # Validate teams if provided
    if teams:
        unique_teams = get_unique_teams(df)
        invalid_teams = [team for team in teams if team not in unique_teams]
        
        if invalid_teams:
            return False, f"Invalid team(s): {', '.join(invalid_teams)}"
    
    return True, None
