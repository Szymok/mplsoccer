from soccerdata import FBref
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
    key_columns = ['date', 'league', 'season', 'team', 'opponent', 'result', 'venue']  # Extend based on your data
    
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

def download_all_team_match_stats(leagues, seasons, stat_types=None, opponent_stats=False, team=None):
    """
    Downloads various team match stats for specified leagues, seasons, and optionally for specific teams.
    
    Parameters:
    - leagues: string or iterable, IDs of leagues to include.
    - seasons: string, int, or list, Seasons to include.
    - stat_types: list of str, Types of stats to retrieve. If None, defaults to a predefined list.
    - opponent_stats: bool, If True, will retrieve opponent stats.
    - team: str or list of str, optional, Team(s) to retrieve. If None, retrieves all teams.
    
    Returns:
    - A dictionary of pd.DataFrame(s) with the team match stats for each stat type.
    """
    if stat_types is None:
        stat_types = ['schedule', 'keeper', 'shooting', 'passing', 'passing_types', 
                      'goal_shot_creation', 'defense', 'possession', 'misc']
    
    # Initialize FBref with specified leagues and seasons
    fbref = FBref(leagues=leagues, seasons=seasons)
    
    all_stats = {}
    for stat_type in stat_types:
        try:
            print(f"Downloading team match {stat_type} stats for leagues: {leagues}, seasons: {seasons}...")
            stats = fbref.read_team_match_stats(stat_type=stat_type, opponent_stats=opponent_stats, team=team)
            flattened_stats = flatten_columns(stats)
            all_stats[stat_type] = flattened_stats
        except ValueError as e:
            print(f"Invalid stat_type '{stat_type}' provided: {e}")
        except Exception as e:
            print(f"Failed to download team match {stat_type} stats: {e}")
    
    return all_stats

# Example usage
leagues = "ESP-La Liga"
seasons = ["23-24"]
team_match_stats = download_all_team_match_stats(leagues, seasons, opponent_stats=True, team=["FC Barcelona"])

for stat_type, stats in team_match_stats.items():
    print(f"\nStats Type: {stat_type}")
    stats.reset_index(drop=True, inplace=True)
    print(stats.head())
