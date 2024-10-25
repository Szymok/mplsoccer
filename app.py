import streamlit as st
import numpy as np
import pandas as pd
import psycopg2
import pickle
import time
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns
import os

st.set_page_config(layout="wide")

host = os.getenv('DB_HOST')
dbname = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')

# Establishing database connection using psycopg2
def create_db_connection():
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host
    )
    return conn

# Load the data from PostgreSQL database
def load_data_from_db(query, conn):
    df = pd.read_sql_query(query, conn)
    return df

# Convert relevant columns to numeric
def convert_columns_to_numeric(df_data, columns):
    for column in columns:
        if column in df_data.columns:
            df_data[column] = pd.to_numeric(df_data[column], errors='coerce')
    return df_data

def get_schemas(conn):
    query = "SELECT schema_name FROM information_schema.schemata"
    df_schemas = pd.read_sql_query(query, conn)
    return df_schemas['schema_name'].tolist()

def get_tables(conn, schema):
    query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
    df_tables = pd.read_sql_query(query, conn)
    return df_tables['table_name'].tolist()

def get_columns(conn, selected_schema, selected_table):
    query = f"""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = '{selected_schema}' 
    AND table_name = '{selected_table}';
    """
    return pd.read_sql(query, conn)

# Initialize the database connection
conn = create_db_connection()

# Fetch available schemas
schemas = get_schemas(conn)

# Assuming you have a variable or input where the user selects the schema
selected_schema = st.selectbox("Select the schema:", ["team_season", "team_match", "player_season", "player_match"])

# Function to filter data by season
def filter_season(df, start_season, end_season):
    return df[(df['season'] >= start_season) & (df['season'] <= end_season)]

# Function to filter data by matchday (only relevant for player_match or team_match schemas)
def filter_matchday(df, matchdays):
    return df[df['round'].isin(matchdays)]

# Fetch available tables based on the selected schema
tables = get_tables(conn, selected_schema)

# Check that tables is indeed a list or array-like structure
if isinstance(tables, list) or isinstance(tables, tuple):
    selected_table = st.sidebar.selectbox('Select Table', tables)
else:
    st.error("Failed to retrieve tables. Please check your database connection.")

# Fetch columns of the selected table
columns_info = get_columns(conn, selected_schema, selected_table)
# Check that columns_info is a DataFrame
if isinstance(columns_info, pd.DataFrame) and not columns_info.empty:
    column_names = columns_info['column_name'].tolist()
else:
    st.error("Failed to retrieve columns for the selected table.")
    st.stop()  # Stop the execution if there's an error
column_names = columns_info['column_name'].tolist()

def some_function_using_columns(df):
    # Example usage of dynamic columns instead of hard-coded ones
    for column in column_names:
        if column in df.columns:
            # Perform operations on the column, like checking or plotting
            st.write(f"Processing {column}")
        else:
            st.warning(f"Column '{column}' not found in the data.")

# Construct the SQL query dynamically
query = f"SELECT * FROM {selected_schema}.{selected_table};"

# Load the data from the selected schema and table
df_database = load_data_from_db(query, conn)
# List of columns to convert to numeric
numeric_columns = column_names  
# Convert the relevant columns to numeric
# df_database = convert_columns_to_numeric(df_database, numeric_columns)
current_columns = column_names
types = ['Mean', 'Absolute', 'Median', 'Maximum', 'Minimum']
color_dict = {
    'Alavés': '#1E90FF',  # Light blue
    'Almería': '#B22222',  # Firebrick red
    'Athletic Club': '#A50034',  # Red
    'Atlético Madrid': '#C8102E',  # Red and white
    'Barcelona': '#A500A1',  # Red and blue 
    'Betis': '#5C9A24',  # Green
    'Cádiz': '#FFD700',  # Gold or yellow
    'Celta Vigo': '#0080FF',  # Sky blue
    'Eibar': '#A00000',  # Dark red
    'Elche': '#005236',  # Dark green
    'Espanyol': '#0066CC',  # Blue
    'Getafe': '#A500A1',  # Purple 
    'Girona': '#FF4500',  # Orange
    'Granada': '#E50000',  # Dark red
    'Huesca': '#8A2BE2',  # BlueViolet
    'La Coruña': '#003DA5',  # Blue
    'Las Palmas': '#FDD835',  # Yellow
    'Leganés': '#008000',  # Green
    'Levante': '#FF0000',  # Red
    'Málaga': '#003DA5',  # Dark blue
    'Mallorca': '#A61C24',  # Red
    'Osasuna': '#D50032',  # Dark red
    'Rayo Vallecano': '#FF0000',  # Red
    'Real Betis': '#008000',  # Green
    'Real Madrid': '#FFFFFF',  # White
    'Real Sociedad': '#003DA5',  # Blue
    'Sevilla': '#FF0000',  # Red
    'Sporting Gijón': '#FF6347',  # Tomato
    'Valencia': '#FF8C00',  # Dark orange
    'Valladolid': '#580F4D',  # Dark purple
    'Villarreal': '#FDD835',  # Yellow
}

# label_attr_dict = {
#     "Goals Scored": "Standard_Gls", 
#     "Shots": "Standard_Sh", 
#     "Shots on Target": "Standard_SoT",
#     "Shots on Target Percentage": "Standard_SoTpercent",
#     "Goals per Shot": "Standard_G/Sh",
#     "Goals per Shot on Target": "Standard_G/SoT",
#     "Average Shot Distance": "Standard_Dist",
#     "Free Kicks": "Standard_FK",
#     "Penalty Kicks": "Standard_PK",
#     "Penalty Kicks Attempted": "Standard_PKatt",
#     "Expected Goals": "Expected_xG",
#     "Non-Penalty Expected Goals": "Expected_npxG",
#     "Non-Penalty Expected Goals per Shot": "Expected_npxG/Sh",
#     "Goals minus Expected Goals": "Expected_G-xG",
#     "Non-Penalty Goals minus Expected Goals": "Expected_np:G-xG"
# }

# label_attr_dict_teams = {
#     "Goals Scored": "Standard_Gls",
#     "Shots": "Standard_Sh",
#     "Shots on Target": "Standard_SoT",
#     "Shots on Target Percentage": "Standard_SoTpercent",
#     "Goals per Shot": "Standard_G/Sh",
#     "Goals per Shot on Target": "Standard_G/SoT",
#     "Average Shot Distance": "Standard_Dist",
#     "Free Kicks": "Standard_FK",
#     "Penalty Kicks": "Standard_PK",
#     "Penalty Kicks Attempted": "Standard_PKatt",
#     "Expected Goals": "Expected_xG",
#     "Non-Penalty Expected Goals": "Expected_npxG",
#     "Non-Penalty Expected Goals per Shot": "Expected_npxG/Sh",
#     "Goals minus Expected Goals": "Expected_G-xG",
#     "Non-Penalty Goals minus Expected Goals": "Expected_np:G-xG"
# }

# label_attr_dict_correlation = {
#     "Goals Scored": "delta_Standard_Gls",
#     "Shots": "delta_Standard_Sh",
#     "Shots on Target": "delta_Standard_SoT",
#     "Shots on Target Percentage": "delta_Standard_SoTpercent",
#     "Goals per Shot": "delta_Standard_G/Sh",
#     "Goals per Shot on Target": "delta_Standard_G/SoT",
#     "Average Shot Distance": "delta_Standard_Dist",
#     "Free Kicks": "delta_Standard_FK",
#     "Penalty Kicks": "delta_Standard_PK",
#     "Penalty Kicks Attempted": "delta_Standard_PKatt",
#     "Expected Goals": "delta_Expected_xG",
#     "Non-Penalty Expected Goals": "delta_Expected_npxG",
#     "Non-Penalty Expected Goals per Shot": "delta_Expected_npxG/Sh",
#     "Goals minus Expected Goals": "delta_Expected_G-xG",
#     "Non-Penalty Goals minus Expected Goals": "delta_Expected_np:G-xG"
# }

# label_fact_dict = {
#     "goals scored": "Standard_Gls",
#     "shots": "Standard_Sh",
#     "shots on target": "Standard_SoT",
#     "shots on target percentage": "Standard_SoTpercent",
#     "goals per shot": "Standard_G/Sh",
#     "goals per shot on target": "Standard_G/SoT",
#     "average shot distance": "Standard_Dist",
#     "free kicks": "Standard_FK",
#     "penalty kicks": "Standard_PK",
#     "penalty kicks attempted": "Standard_PKatt",
#     "expected goals": "Expected_xG",
#     "non-penalty expected goals": "Expected_npxG",
#     "non-penalty expected goals per shot": "Expected_npxG/Sh",
#     "goals minus expected goals": "Expected_G-xG",
#     "non-penalty goals minus expected goals": "Expected_np:G-xG"
# }
# Helper methods
def get_unique_seasons_modified(df_data):
    """
    Converts season format from a 4-digit number (e.g., '1718') to '2017/2018' and returns the unique seasons.
    """
    seasons_modified = []
    unique_seasons = df_data['season'].unique()  # Assuming 'season' is a column in your DataFrame

    for season in unique_seasons:
        season_str = str(season)  # Convert to string if it's an integer
        if len(season_str) == 4 and season_str.isdigit():  # Ensure it has 4 digits
            start_year = int(season_str[:2])
            end_year = int(season_str[2:])
            
            # Determine the century for the start and end years
            start_year += 1900 if start_year >= 90 else 2000
            end_year += 1900 if end_year >= 90 else 2000
            
            # Format the season string as '2017/2018'
            season_formatted = f"{start_year}/{end_year}"
            seasons_modified.append(season_formatted)
        else:
            st.warning(f"Skipping invalid season value: {season}")
    
    return seasons_modified

def get_unique_matchdays(df_data):
    '''
    Returns the unique matchdays from the dataframe
    '''
    if 'round' in df_data.columns:
        # Extract matchday from the 'round' column
        df_data['matchday'] = df_data['round'].str.extract('(\d+)').astype(int)
        unique_matchdays = np.unique(df_data['matchday']).tolist()
        return unique_matchdays
    else:
        # If 'round' column is not present, return an empty list or handle accordingly
        return []
        
def get_unique_teams(df_data):
    '''
    Returns the unique teams from the dataframe
    '''
    unique_teams = np.unique(df_data.team).tolist()
    return unique_teams

def filter_season(df_data, start_season, end_season):
    # Convert the start and end seasons from '2000/2001' format to '0001' format
    start_season_compact = start_season.replace('/', '')[2:]
    end_season_compact = end_season.replace('/', '')[2:]
    
    # Convert the season column to a more readable format '2000/2001'
    df_data['season_readable'] = df_data['season'].apply(lambda x: f"20{str(x)[:2]}/20{str(x)[2:]}")
    
    # Find the unique seasons in the readable format
    unique_seasons_readable = np.unique(df_data['season_readable']).tolist()
    
    # Find the index of the start and end seasons in the unique seasons list
    start_index = unique_seasons_readable.index(start_season)
    end_index = unique_seasons_readable.index(end_season) + 1
    
    # Select the seasons within the start and end index
    seasons_selected_readable = unique_seasons_readable[start_index:end_index]
    
    # Filter the DataFrame based on the selected seasons
    df_filtered_season = df_data[df_data['season_readable'].isin(seasons_selected_readable)]
    
    # Drop the temporary 'season_readable' column
    df_filtered_season = df_filtered_season.drop(columns=['season_readable'])
    
    return df_filtered_season

def filter_matchday(df_data, selected_matchdays):
    '''
    Filters the DataFrame based on the selected matchday (round) values.
    '''
    matchdays_list = []

    for matchday in selected_matchdays:
        # Format the matchday to match the string format in the DataFrame
        if isinstance(matchday, str):
            matchweek_number = matchday.split()[1]  # Extract the numeric part
        elif isinstance(matchday, int):
            matchweek_number = str(matchday)  # Convert to string for comparison
        
        # Construct the matchweek string
        formatted_matchday = f"Matchweek {matchweek_number}"
        matchdays_list.append(formatted_matchday)

    # Debugging print statements
    print("Matchdays List:", matchdays_list)  # Check matchdays_list content
    print("Unique rounds in DataFrame:", df_data['round'].unique())  # Unique rounds

    # Filter based on matchdays_list using string match
    df_filtered_matchday = df_data[df_data['round'].isin(matchdays_list)]
    
    # Debug statement to check filtered DataFrame
    print("Filtered DataFrame:", df_filtered_matchday)  # Check what the filtered DataFrame contains
    
    return df_filtered_matchday
    
def filter_teams(df_data):
    df_filtered_team = pd.DataFrame()
    if all_teams_selected == 'Select teams manally (choose below)':
        df_filtered_team = df_data[df_data['team'].isin(selected_teams)]
        return df_filtered_team
    return df_data

def stack_team_dataframe(df_data):
    # Get all column names from the DataFrame
    column_names = df_data.columns.tolist()
    
    # Filter the dataframe to include only the columns you're interested in
    # Since we're now including all columns, there's no need to filter them out
    df_filtered = df_data[column_names].copy()

    # Assuming you might need to sort or perform other operations on the filtered data
    # For example, sorting by 'season' and 'team' for readability
    # Ensure 'season' and 'team' are in your DataFrame to avoid KeyError
    if 'season' in df_filtered.columns and 'team' in df_filtered.columns:
        df_sorted = df_filtered.sort_values(['season', 'team'], ascending=[True, True])
    else:
        # If 'season' or 'team' columns are not present, return the filtered DataFrame as is
        df_sorted = df_filtered

    return df_sorted

def group_measure_by_attribute(aspect, attribute, measure):
    df_data = df_data_filtered.copy()  # Work on a copy to avoid modifying the original DataFrame
    
    # Convert the selected attribute to numeric
    df_data[attribute] = pd.to_numeric(df_data[attribute], errors='coerce')  # Convert to numeric, coercing errors to NaNs
    df_data.dropna(subset=[attribute], inplace=True)  # Drop rows where the attribute is NaN after conversion

    # Ensure we handle only numeric columns aggregated
    numerical_cols = df_data.select_dtypes(include=['number']).columns.tolist()  # Get numeric column names

    if measure == 'Absolute':
        df_return = df_data.groupby([aspect]).sum(numeric_only=True)  # Use only numeric columns for sum
    elif measure == 'Mean':
        df_return = df_data.groupby([aspect]).mean(numeric_only=True)  # Use only numeric columns for mean
    elif measure == 'Median':
        df_return = df_data.groupby([aspect]).median(numeric_only=True)
    elif measure == 'Maximum':
        df_return = df_data.groupby([aspect]).max(numeric_only=True)
    elif measure == 'Minimum':
        df_return = df_data.groupby([aspect]).min(numeric_only=True)

    df_return['aspect'] = df_return.index  # Add aspect column for easier handling later
    if aspect == 'team':
        df_return = df_return.sort_values(by=[attribute], ascending=False)  # Sort by the aggregated column

    return df_return

########################
### ANALYSIS METHODS ###
########################

def plot_x_per_season(attr, measure, df_data):
    rc = {
        'figure.figsize': (8, 4.5),
        'axes.facecolor': '#0e1117',
        'axes.edgecolor': '#0e1117',
        'axes.labelcolor': 'white',
        'figure.facecolor': '#0e1117',
        'patch.edgecolor': '#0e1117',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'grey',
        'font.size': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    }
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()

    # Here you would group and calculate your desired stats.
    df_plot = group_measure_by_attribute("season", attr, measure)

    # Bar plot
    sns.barplot(x='aspect', y=attr, data=df_plot.reset_index(), color='#b80606', ax=ax)

    ax.set(xlabel='Season', ylabel=attr)

    # Annotate bars with the correct formatting
    for p in ax.patches:
        # Use format to control precision for small values
        value = p.get_height()
        
        if value < 1:
            ax.annotate(f'{value:.2f}', (p.get_x() + p.get_width() / 2., value), 
                        ha='center', va='bottom', fontsize=10, color='white', weight='bold')
        else:
            ax.annotate(f'{int(value)}', (p.get_x() + p.get_width() / 2., value),
                        ha='center', va='bottom', fontsize=10, color='white', weight='bold')

    st.pyplot(fig)

def plot_x_per_matchday(attr, measure, df_data):
    rc = {
        'figure.figsize': (8, 4.5),
        'axes.facecolor': '#0e1117',
        'axes.edgecolor': '#0e1117',
        'axes.labelcolor': 'white',
        'figure.facecolor': '#0e1117',
        'patch.edgecolor': '#0e1117',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'grey',
        'font.size': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    }
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()

    # Ensure the selected attribute is numeric and handle any NaNs
    df_data[attr] = pd.to_numeric(df_data[attr], errors='coerce')
    df_filtered = df_data.dropna(subset=[attr])  # Drop NaN values

    # Initialize df_plot
    df_plot = pd.DataFrame()

    # Group by `round` and calculate based on the selected measure
    if measure == 'Maximum':
        df_plot = df_filtered.groupby(['round'])[attr].max().reset_index()
    elif measure == 'Minimum':
        df_plot = df_filtered.groupby(['round'])[attr].min().reset_index()
    elif measure == 'Mean':
        df_plot = df_filtered.groupby(['round'])[attr].mean().reset_index()
    elif measure == 'Median':
        df_plot = df_filtered.groupby(['round'])[attr].median().reset_index()
    elif measure == 'Absolute':
        df_plot = df_filtered.groupby(['round'])[attr].sum().reset_index()  # Raw sum for Absolute
    else:
        # Handle unexpected measure values
        st.error("Unknown measure selected.")
        return  # Exit the function if an unknown measure is used
    
    # Check if df_plot is empty before plotting
    if df_plot.empty:
        st.warning("No data available for the selected measure and round.")
        return

    # Create a bar plot
    ax = sns.barplot(x='round', y=attr, data=df_plot, color='#b80606')
    ax.set(xlabel='Round', ylabel=attr)

    # Annotate bars with the actual values
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='white')
    
    st.pyplot(fig)  # Display the plot

def plot_x_per_team(attr, measure, df_data_filtered):  # Added df_data_filtered as a parameter
    rc = {'figure.figsize': (8, 4.5),
          'axes.facecolor': '#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size': 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}

    plt.rcParams.update(rc)
    fig, ax = plt.subplots()

    # Assume attr directly corresponds to a column name in df_data_filtered
    if attr not in df_data_filtered.columns:
        st.error(f"Attribute '{attr}' not found in data.")
        return

    # Convert relevant columns to numeric
    df_data_filtered[attr] = pd.to_numeric(df_data_filtered[attr], errors='coerce')

    df_plot = group_measure_by_attribute("team", attr, measure)
    
    if specific_team_colors:  # assumed available in context
        ax = sns.barplot(x="aspect", y=attr, data=df_plot.reset_index(), palette=color_dict)
    else:
        ax = sns.barplot(x="aspect", y=attr, data=df_plot.reset_index(), color="#b80606")
        
    y_str = measure + " " + attr + " per Game"
    if measure == "Absolute":
        y_str = measure + " " + attr
    if measure in ["Minimum", "Maximum"]:
        y_str = measure + " " + attr + " in a Game"
        
    ax.set(xlabel="Team", ylabel=y_str)
    plt.xticks(rotation=66, horizontalalignment="right")

    if measure == "Mean" or attr in ["distance", "pass_ratio", "possession", "tackle_ratio"]:
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 18),
                        rotation=90,
                        textcoords='offset points')
    else:
        for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 18),
                        rotation=90,
                        textcoords='offset points')

    st.pyplot(fig)

def plt_attribute_correlation(aspect1, aspect2, df_data_filtered, corr_type):
    rc = {
        'figure.figsize': (5, 5),
        'axes.facecolor': '#0e1117',
        'axes.edgecolor': '#0e1117',
        'axes.labelcolor': 'white',
        'figure.facecolor': '#0e1117',
        'patch.edgecolor': '#0e1117',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'grey',
        'font.size': 8,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    }
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()

    # Convert both attributes to numeric
    df_data_filtered[aspect1] = pd.to_numeric(df_data_filtered[aspect1], errors='coerce')
    df_data_filtered[aspect2] = pd.to_numeric(df_data_filtered[aspect2], errors='coerce')

    # Drop rows with NaN values for either attribute
    df_filtered = df_data_filtered.dropna(subset=[aspect1, aspect2])

    # Check if df_filtered is empty
    if df_filtered.empty:
        st.warning("No data available for correlating the selected attributes.")
        return

    # Create scatter or regplot based on corr_type
    if corr_type == "Regression Plot (Recommended)":
        ax = sns.regplot(x=aspect1, y=aspect2, x_jitter=.1, data=df_filtered, color='#f21111',
                         scatter_kws={"color": "#f21111"}, line_kws={"color": "#c2dbfc"})
    elif corr_type == "Standard Scatter Plot":
        ax = sns.scatterplot(x=aspect1, y=aspect2, data=df_filtered, color='#f21111')

    ax.set(xlabel=aspect1, ylabel=aspect2)
    st.pyplot(fig)
    
def find_match_game_id(min_max, attribute, what, df_data_filtered):
    # Ensure the selected attribute is numeric and handle any NaNs
    df_data_filtered[attribute] = pd.to_numeric(df_data_filtered[attribute], errors='coerce')
    df_data_filtered.dropna(subset=[attribute], inplace=True)  # Drop rows where the attribute is NaN

    # Group by game and season and summarize the attribute of interest
    if what == "by both teams":
        df_grouped = df_data_filtered.groupby(['game', 'season'], as_index=False).agg({attribute: 'sum'})
    else:
        # For 'by a team', we're aggregating for each individual team in the game
        df_grouped = df_data_filtered.groupby(['game', 'season', 'team'], as_index=False).agg({attribute: 'sum'})
    
    # Determine index based on min/max selection
    if min_max == 'Minimum':
        index = df_grouped[attribute].idxmin()
    elif min_max == 'Maximum':
        index = df_grouped[attribute].idxmax()
    
    # Retrieve the relevant information
    game_info = df_grouped.at[index, 'game']
    season = df_grouped.at[index, 'season']
    value = df_grouped.at[index, attribute]
    
    # If it is by-a-team, retrieve the corresponding team
    team = df_grouped.at[index, 'team'] if what == "by a team" else "N/A"

    return_game_info_value_team = [game_info, season, value, team]
    return return_game_info_value_team

def build_matchfacts_return_string(return_game_id_value_team, min_max, attribute, what):
    game_id = return_game_id_value_team[0]
    df_match_result = df_data_filtered.loc[df_data_filtered['game'] == game_id]  # Update based on your game identifier
    
    season = df_match_result.iloc[0]['season'].replace('-', '/')
    matchday = str(df_match_result.iloc[0]['matchday'])
    home_team = df_match_result.iloc[0]['team']
    away_team = df_match_result.iloc[1]['team']
    goals_home = str(df_match_result.iloc[0]['goals'])
    goals_away = str(df_match_result.iloc[1]['goals'])
    
    string1 = f'On matchday {matchday} of the season {season}, {home_team} played against {away_team}.'
    string2 = ''
    
    if goals_home > goals_away:
        string2 = f"The match resulted in a {goals_home}:{goals_away} ({df_match_result.iloc[0]['ht_goals']}:{df_match_result.iloc[1]['ht_goals']}) win for {home_team}."
    elif goals_home < goals_away:
        string2 = f"The match resulted in a {goals_home}:{goals_away} ({df_match_result.iloc[0]['ht_goals']}:{df_match_result.iloc[1]['ht_goals']}) loss for {home_team}."
    else:
        string2 = f"The match resulted in a {goals_home}:{goals_away} ({df_match_result.iloc[0]['ht_goals']}:{df_match_result.iloc[1]['ht_goals']}) draw."

    value = str(abs(round(return_game_id_value_team[1], 2)))
    team = str(return_game_id_value_team[2])
    string3 = ""
    string4 = ""

    if what == "difference between teams":
        # Ensure you're aggregating on numeric columns only
        numeric_columns = df_match_result.select_dtypes(include='number')  # Only consider numeric columns
        
        if not numeric_columns.empty:
            difference = numeric_columns.sum(axis=1)  # Adjust based on your logic
            string3 = f" Over the course of the match, a difference of {value} {attribute} was recorded between the teams."
            string4 = f" This is the {min_max.lower()} difference for two teams in the currently selected data."
    
    elif what == "by both teams":
        # Similar logic to aggregate numbers only
        numeric_columns = df_match_result.select_dtypes(include='number')  # Only consider numeric columns

        if not numeric_columns.empty:
            total = numeric_columns.sum(axis=1).iloc[0]  # Adjust this if you need to sum for all rows
            string3 = f" Over the course of the match, both teams recorded {total} {attribute} together."
            string4 = f" This is the {min_max.lower()} value for two teams in the currently selected data."
    
    elif what == "by a team":
        string3 = f" Over the course of the match, {team} recorded {value} {attribute}."
        string4 = f" This is the {min_max.lower()} value for a team in the currently selected data."

    answer = string1 + string2 + string3 + string4
    st.markdown(answer)
    return df_match_result

######################
### INITIALIZATION ###
######################

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Real Madrid Stats Analysis')
with row0_2:
    st.text('')
    st.subheader('App by [SkSzymon](https://www.twitter.com/SkSzymon)')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown("Hello there! Have you ever spent your weekend watching the LaLiga matches and had your friends complain about how 'players definitely used to run more' ? However, you did not want to start an argument because you did not have any stats at hand? Well, this interactive application containing LaLiga data from season ... to ... allows you to discover just that! If you're on a mobile device, I would recommend switching over to landscape for viewing ease.")
    st.markdown("You can find the source code in the [GitHub repository]()")
    
#################
### SELECTION ###
#################
    
df_stacked = stack_team_dataframe(df_database)

st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')

### SEASON RANGE ###

if selected_schema in ['team_match', 'team_season']:
    st.sidebar.markdown('**First select the data range you want to analyze:** 👇')
    
    # Get unique seasons from the database and ensure they are sorted
    unique_seasons = get_unique_seasons_modified(df_database)
    unique_seasons.sort()  # Ensure the seasons are sorted

    # Sidebar for season selection
    start_season, end_season = st.sidebar.select_slider(
        'Select the season range you want to include',
        options=unique_seasons,
        value=(unique_seasons[0], unique_seasons[-1])  # Default to the entire range
    )

    # At this point, start_season and end_season are already defined by select_slider
# Filter data based on selected seasons
df_data_filtered_season = filter_season(df_database, start_season, end_season)

# Conditional logic based on the schema selected
if selected_schema in ["player_match", "team_match"]:
    # Only for schemas that include matchdays
    selected_matchdays = st.multiselect("Select matchdays to include", options=df_data_filtered_season['round'].unique())
    
    if selected_matchdays:  # Ensure that matchdays are selected
        df_data_filtered = filter_matchday(df_data_filtered_season, selected_matchdays)
    else:
        st.warning("Please select at least one matchday.")
        df_data_filtered = df_data_filtered_season
else:
    # For schemas where matchdays are not relevant
    df_data_filtered = df_data_filtered_season

# Continue with the rest of your Streamlit app logic
st.write(df_data_filtered)

# Get unique seasons
unique_seasons = get_unique_seasons_modified(df_database)

### MATCHDAY RANGE ###
# Check if the selected schema is 'team_match'
if selected_schema == 'team_match':
    # Get unique matchdays
    unique_matchdays = get_unique_matchdays(df_data_filtered_season)
    
    # Sidebar for matchday selection
    selected_matchdays = st.sidebar.select_slider(
        'Select the matchweek range you want to include',
        options=unique_matchdays,
        value=(min(unique_matchdays), max(unique_matchdays))
    )
    
    # Filter data based on selected matchdays
    df_data_filtered_matchday = filter_matchday(df_data_filtered_season, selected_matchdays)
else:
    df_data_filtered_matchday = df_data_filtered_season  # If not 'team_match', use the season-filtered data


### TEAMS SELECTION ###
unique_teams = get_unique_teams(df_stacked)
all_teams_selected = st.sidebar.selectbox('Do you want to only include specific teams? If the answer is yes, please check the box below and then select the team(s) in the new field.', ['Include all available teams', 'Select teams manually (choose below)'])

if all_teams_selected == 'Select teams manually (choose below)':
    selected_teams = st.sidebar.multiselect('Select and deselect the teams you would like to include in the analysis. You can clear the current selection by clicking the corresponding x-button on the right.', unique_teams, default = unique_teams)

# Adjust filtering based on schema
if selected_schema in ["player_match", "team_match"]:
    df_data_filtered = filter_matchday(df_data_filtered_season, selected_matchdays)
else:
    df_data_filtered = df_data_filtered_season

### SEE DATA ###
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader('Currently selected data:')

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
if selected_schema == 'team_match':
    with row2_1:
        unique_games_in_df = df_data_filtered['game'].nunique()
        str_games = "🏟️ " + str(unique_games_in_df) + " Matches (Distinct)"
        st.markdown(str_games)
else:
    with row2_1:
        # Ensure the 'players_used' column is numeric
        df_data_filtered['players_used'] = pd.to_numeric(df_data_filtered['players_used'], errors='coerce')
        
        # Calculate the total players used
        total_players_used = df_data_filtered['players_used'].sum()  # Sum of players used
        
        # Format the result as an integer
        str_players_used = "👥 " + str(int(total_players_used)) + " Players Used"
        st.markdown(str_players_used)
with row2_2:
    unique_teams_in_df = len(np.unique(df_data_filtered.team).tolist())
    t = ' Teams'
    if(unique_teams_in_df==1):
        t = ' Team'
    str_teams = "🏃‍♂️ " + str(unique_teams_in_df) + t
    st.markdown(str_teams)

with row2_3:
    # Add count of selected rows
    num_selected_rows = df_data_filtered.shape[0]  # Number of selected rows in the DataFrame
    str_selected_rows = "🔢 " + str(num_selected_rows) + " Selected Rows"
    st.markdown(str_selected_rows)

with row2_4:
    # Count distinct seasons in df_data_filtered
    unique_seasons_in_df = df_data_filtered['season'].nunique()  # Assuming the season column is named 'season'
    str_seasons = "📅 " + str(unique_seasons_in_df) + " Distinct Seasons"
    st.markdown(str_seasons)
row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
with row3_1:
    st.markdown("")
    see_data = st.expander('You can click here to see the raw data first 👉')
    with see_data:
        st.dataframe(data=df_data_filtered.reset_index(drop=True))
st.text('')

##########################
### ANALYSIS SELECTION ###
##########################

def find_match_game_id(min_max, attribute, what, df_data_filtered):
    # Ensure the selected attribute is numeric and handle any NaNs
    df_data_filtered[attribute] = pd.to_numeric(df_data_filtered[attribute], errors='coerce')
    df_data_filtered.dropna(subset=[attribute], inplace=True)  # Drop NaN values resulting from conversion
    
    if what == "by both teams":
        # Group by game and season and summarize the attribute of interest
        df_grouped = df_data_filtered.groupby(['game', 'season'], as_index=False).agg({attribute: 'sum'})
    else:
        # For 'by a team', we're aggregating for each individual team in the game
        df_grouped = df_data_filtered.groupby(['game', 'season', 'team'], as_index=False).agg({attribute: 'sum'})
    
    # Determine index based on min/max selection
    if min_max == 'Minimum':
        index = df_grouped[attribute].idxmin()
    elif min_max == 'Maximum':
        index = df_grouped[attribute].idxmax()
    
    # Retrieve the relevant information
    game_info = df_grouped.at[index, 'game']
    season = df_grouped.at[index, 'season']
    value = df_grouped.at[index, attribute]
    
    # If it is by-a-team, retrieve the corresponding team
    team = df_grouped.at[index, 'team'] if what == "by a team" else "N/A"

    return_game_info_value_team = [game_info, season, value, team]
    return return_game_info_value_team

def plot_top_5_stat(df_data, attribute, order='max'):
    # Convert the selected attribute to numeric, if not already done
    df_data[attribute] = pd.to_numeric(df_data[attribute], errors='coerce')
    
    # Drop NaN values resulted from the conversion
    df_data = df_data.dropna(subset=[attribute])
    
    # Get top 5 matches based on the specified order
    if order == 'max':
        top_matches = df_data.nlargest(5, attribute)  # Get top 5 by maximum value
    else:
        top_matches = df_data.nsmallest(5, attribute)  # Get top 5 by minimum value

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='game', y=attribute, data=top_matches, palette='viridis')
    plt.title(f'Top 5 Matches for {attribute} ({order.capitalize()})')
    plt.xlabel('Match')
    plt.ylabel(attribute)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

### DATA EXPLORATION ###
row12_spacer1, row12_1, row12_spacer2 = st.columns((.2, 7.1, .2))
with row12_1:
    st.subheader('Match Finder')
    st.markdown('Show the (or a) match with the...')

if all_teams_selected == 'Include all available teams':
    # Define all columns in one go to avoid missing definitions
    row13_spacer1, row13_1, row13_spacer2, row13_2, row13_spacer3, row13_3, row13_spacer4 = st.columns((.2, 2.3, .2, 2.3, .2, 2.3, .2))
    
    with row13_1:
        show_me_hi_lo = st.selectbox('', ['Maximum', 'Minimum'], key='hi_lo')

    with row13_2:
        show_me_aspect = st.selectbox('', df_data_filtered.columns.tolist(), key='what')

    with row13_3:  # This must be defined above
        show_me_what = st.selectbox('', ['by a team', 'by both teams', 'difference between teams'], key='one_both_diff')

    row14_spacer1, row14_1, row14_spacer2 = st.columns((.2, 7.1, .2))
    with row14_1:
        return_game_info_value_team = find_match_game_id(show_me_hi_lo, show_me_aspect, show_me_what, df_data_filtered)
        game_info, season, value, team = return_game_info_value_team        # Display the result
        st.markdown(f"Selected Match: {game_info}, {show_me_aspect}: {value} by Team: {team}")
        # Create a chart for the top 5 matches based on the selected statistic
        plot_top_5_stat(df_data_filtered, show_me_aspect, show_me_hi_lo.lower())  # Pass order as 'min' or 'max'    

    row15_spacer1, row15_1, row15_2, row15_3, row15_4, row15_spacer2  = st.columns((0.5, 1.5, 1.5, 1, 2, 0.5))
    with row15_1:
        st.subheader(" ‎")
    # with row15_2:
    #     st.subheader(str(df_match_result.iloc[0]['team']))
    # with row15_3:
    #     end_result = str(df_match_result.iloc[0]['goals']) + " : " + str(df_match_result.iloc[1]['goals'])
    #     ht_result = " ‎ ‎( " + str(df_match_result.iloc[0]['ht_goals']) + " : " + str(df_match_result.iloc[1]['ht_goals']) + " )"
    #     st.subheader(end_result + " " + ht_result)  
    # with row15_4:
    #     st.subheader(str(df_match_result.iloc[1]['team']))
else:
    row17_spacer1, row17_1, row17_spacer2 = st.columns((.2, 7.1, .2))
    with row17_1:
        st.warning('Unfortunately this analysis is only available if all teams are included')

### TEAM ###
row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
with row4_1:
    st.subheader('Analysis per Team')
row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row5_1:
    st.markdown('Investigate a variety of stats for each team. Which team scores the most goals per game? How does your team compare in terms of distance ran per game?')
    plot_x_per_team_selected = st.selectbox('Which attribute do you want to analyze?', df_data_filtered.columns.tolist(), key='attribute_team')  # Use DataFrame columns
    plot_x_per_team_type = st.selectbox('Which measure do you want to analyze?', types, key='measure_team')
    specific_team_colors = st.checkbox('Use team specific color scheme')

with row5_2:
    if all_teams_selected != 'Select teams manually (choose below)' or selected_teams:
        plot_x_per_team(plot_x_per_team_selected, plot_x_per_team_type, df_data_filtered)  # Pass df_data_filtered
    else:
        st.warning('Please select at least one team')

### SEASON ###
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader('Analysis per Season')
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row7_1:
    st.markdown('Investigate developments and trends. Which season had teams score the most goals? Has the amount of passes per game changed?')
    plot_x_per_season_selected = st.selectbox('Which attribute do you want to analyze?', df_data_filtered.columns.tolist(), key='attribute_season')  # Use DataFrame columns
    plot_x_per_season_type = st.selectbox('Which measure do you want to analyze?', types, key='measure_season')

with row7_2:
    if all_teams_selected != 'Select teams manually (choose below)' or selected_teams:
        plot_x_per_season(plot_x_per_season_selected, plot_x_per_season_type, df_data_filtered)  # Pass df_data_filtered
    else:
        st.warning('Please select at least one team')

### MATCHDAY ###
row8_spacer1, row8_1, row8_spacer2 = st.columns((.2, 7.1, .2))
with row8_1:
    st.subheader('Analysis per Matchday')
row9_spacer1, row9_1, row9_spacer2, row9_2, row9_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row9_1:
    st.markdown('Investigate stats over the course of a season. At what point in the season do teams score the most goals? Do teams run less toward the end of the season?')
    plot_x_per_matchday_selected = st.selectbox('Which aspect do you want to analyze?', df_data_filtered.columns.tolist(), key='attribute_matchday')  # Use DataFrame columns
    plot_x_per_matchday_type = st.selectbox('Which measure do you want to analyze?', types, key='measure_matchday')

with row9_2:
    if all_teams_selected != 'Select teams manually (choose below)' or selected_teams:
        plot_x_per_matchday(plot_x_per_matchday_selected, plot_x_per_matchday_type, df_data_filtered)  # Pass df_data_filtered
    else:
        st.warning('Please select at least one team')

### CORRELATION ###
corr_plot_types = ['Regression Plot (Recommended)', 'Standard Scatter Plot']
row10_spacer1, row10_1, row10_spacer2 = st.columns((.2, 7.1, .2))
with row10_1:
    st.subheader('Correlation Analysis')
row11_spacer1, row11_1, row11_spacer2, row11_2, row11_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row11_1:
    st.markdown('Investigate the correlation between two attributes. Do teams that run more also score more goals?')
    corr_type = st.selectbox('Which plot type do you want to use?', corr_plot_types)
    y_axis_aspect2 = st.selectbox('Which attribute do you want on the y-axis?', df_data_filtered.columns.tolist())  # Use DataFrame columns
    x_axis_aspect1 = st.selectbox('Which attribute do you want on the x-axis?', df_data_filtered.columns.tolist())  # Use DataFrame columns

with row11_2:
    if all_teams_selected != 'Select teams manually (choose below)' or selected_teams:
        plt_attribute_correlation(x_axis_aspect1, y_axis_aspect2, df_data_filtered, corr_type)  # Pass df_data_filtered and corr_type
    else:
        st.warning('Please select at least one team')

for variable in dir():
    if variable[0:2] != '__':
        del globals()[variable]
del variable