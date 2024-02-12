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
    key_columns = ['date', 'league', 'season', 'team', 'player', 'position', 'minutes']  # Extend based on your data
    
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

def download_all_player_match_stats(leagues, seasons, stat_types=None, match_id=None, force_cache=False):
    """
    Downloads various player match stats for specified leagues, seasons, and optionally for specific matches.
    
    Parameters:
    - leagues: string or iterable, IDs of leagues to include.
    - seasons: string, int, or list, Seasons to include.
    - stat_types: list of str, Types of stats to retrieve. If None, defaults to a predefined list.
    - match_id: int or list of int, optional, Match ID(s) to retrieve stats for. If None, retrieves for all matches.
    - force_cache: bool, If True, forces the use of cached data.
    
    Returns:
    - A dictionary of pd.DataFrame(s) with the player match stats for each stat type.
    """
    if stat_types is None:
        stat_types = ['summary', 'keepers', 'passing', 'passing_types', 'defense', 'possession', 'misc']
    
    # Initialize FBref with specified leagues and seasons
    fbref = FBref(leagues=leagues, seasons=seasons)
    
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

# Example usage
leagues = "ESP-La Liga"
seasons = ["23-24"]
player_match_stats = download_all_player_match_stats(leagues, seasons)

for stat_type, stats in player_match_stats.items():
    print(f"\nStats Type: {stat_type}")
    stats.reset_index(drop=True, inplace=True)
    print(stats.head())
