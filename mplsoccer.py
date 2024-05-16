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

def get_schemas(conn):
    query = "SELECT schema_name FROM information_schema.schemata"
    df_schemas = pd.read_sql_query(query, conn)
    return df_schemas['schema_name'].tolist()

def get_tables(conn, schema):
    query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
    df_tables = pd.read_sql_query(query, conn)
    return df_tables['table_name'].tolist()

# Initialize the database connection
conn = create_db_connection()

# Fetch available schemas
schemas = get_schemas(conn)

# Sidebar for schema selection
selected_schema = st.sidebar.selectbox('Select Schema', schemas)

# Fetch available tables based on the selected schema
tables = get_tables(conn, selected_schema)

# Sidebar for table selection
selected_table = st.sidebar.selectbox('Select Table', tables)

# Construct the SQL query dynamically
query = f"SELECT * FROM {selected_schema}.{selected_table};"

# Load the data from the selected schema and table
df_database = load_data_from_db(query, conn)
types = ['Mean', 'Absolute', 'Median', 'Maximum', 'Minimum']
label_attr_dict = {}
label_attr_dict_teams = {}
color_dict = {}
label_attr_dict_correlation = {}
label_fact_dict = {}

# Helper methods
def get_unique_seasons_modified(df_data):
    '''
    Converts season format from '0001' to '2000/2001' and returns the unique seasons.
    '''
    unique_seasons = np.unique(df_data.season).tolist()
    seasons_modified = []
    for season in unique_seasons:
        # Extract the start and end years from the season string
        start_year = int(season[:2]) + 2000  # Convert the first two characters to an integer and add 2000
        end_year = int(season[2:]) + 2000  # Convert the last two characters to an integer and add 2000
        # Format the season string as '2000/2001'
        season_str = f"{start_year}/{end_year}"
        seasons_modified.append(season_str)
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
    df_data['season_readable'] = df_data['season'].apply(lambda x: f"20{x[:2]}/20{x[2:]}")
    
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
    Filters the DataFrame based on the selected matchday range.
    '''
    matchdays_list = list(range(selected_matchdays[0], selected_matchdays[1] + 1))
    df_filtered_matchday = df_data[df_data['matchday'].isin(matchdays_list)]
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
    df_data = df_data_filtered
    df_return = pd.DataFrame()
    if(measure == 'Absolute'):
        if(attribute == 'pass_ratio' or attribute == 'tackle_ratio' or attribute == 'possession'):
            measure = 'Mean'
        else:
            df_return = df_data.groupby([aspect]).sum()

        
    if(measure == 'Mean'):
        df_return = df_data.groupby([aspect]).mean()

    if(measure == 'Median'):
        df_return = df_data.groupby([aspect]).median()

    if(measure == 'Maximum'):
        df_return = df_data.groupby([aspect]).max()

    if(measure == 'Minimum'):
        df_return = df_data.groupby([aspect]).min()

    df_return['aspect'] = df_return.index
    if aspect == 'team':
        df_return = df_return.sort_values(by=[attribute], ascending=False)
    return df_return

########################
### ANALYSIS METHODS ###
########################

def plot_x_per_season(attr, measure):
    rc = {
        'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 12,
          'axes.labelsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12
    }
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ### Goals
    attribute = label_attr_dict[attr]
    df_plot = pd.DataFrame()
    df_plot = group_measure_by_attribute('season', attribute, measure)
    ax = sns.barplot(x='aspect', y=attribute, data=df_plot, color='#b80606')
    y_str = measure + ' ' + attr + ' ' + ' per Team'
    if measure == 'Absolute':
        y_str = measure + ' ' + attr
    if measure == 'Minimum' or measure == 'Maximum':
        y_str = measure + ' ' + attr + ' ' + ' by a Team'

    ax.set(xlabel = 'Season', ylabel = y_str)
    if measure == 'Mean' or attribute in ['distance', 'pass_ratio', 'possesion', 'tackle_ratio']:
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center',
                        va = 'center',
                        xytext = (0, 15),
                        textcoords = 'offset points')
    else:
        for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center',
                        va = 'center',
                        xytext = (0, 15),
                        textcoords = 'offset points')
    st.pyploy(fig)

def plot_x_per_matchday(attr, measure):
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ### Goals
    attribute = label_attr_dict[attr]
    df_plot = pd.DataFrame()
    df_plot = group_measure_by_attribute("matchday",attribute,measure)
    ax = sns.barplot(x="aspect", y=attribute, data=df_plot.reset_index(), color = "#b80606")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)+1))
    y_str = measure + " " + attr + " per Team"
    if measure == "Absolute":
        y_str = measure + " " + attr
    if measure == "Minimum" or measure == "Maximum":
        y_str = measure + " " + attr + " by a Team"
        
    ax.set(xlabel = "Matchday", ylabel = y_str)
    if measure == "Mean" or attribute in ["distance","pass_ratio","possession","tackle_ratio"]:
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    else:
        for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    st.pyplot(fig)

def plot_x_per_team(attr,measure): #total #against, #conceived
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
    
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ### Goals
    attribute = label_attr_dict_teams[attr]
    df_plot = pd.DataFrame()
    df_plot = group_measure_by_attribute("team",attribute,measure)
    if specific_team_colors:
        ax = sns.barplot(x="aspect", y=attribute, data=df_plot.reset_index(), palette = color_dict)
    else:
        ax = sns.barplot(x="aspect", y=attribute, data=df_plot.reset_index(), color = "#b80606")
    y_str = measure + " " + attr + " " + "per Game"
    if measure == "Absolute":
        y_str = measure + " " + attr
    if measure == "Minimum" or measure == "Maximum":
        y_str = measure + " " + attr + "in a Game"
    ax.set(xlabel = "Team", ylabel = y_str)
    plt.xticks(rotation=66,horizontalalignment="right")
    if measure == "Mean" or attribute in ["distance","pass_ratio","possession","tackle_ratio"]:
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    else:
        for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    st.pyplot(fig)

def plt_attribute_correlation(aspect1, aspect2):
    df_plot = df_data_filtered
    rc = {'figure.figsize':(5,5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    asp1 = label_attr_dict_correlation[aspect1]
    asp2 = label_attr_dict_correlation[aspect2]
    if(corr_type=="Regression Plot (Recommended)"):
        ax = sns.regplot(x=asp1, y=asp2, x_jitter=.1, data=df_plot, color = '#f21111',scatter_kws={"color": "#f21111"},line_kws={"color": "#c2dbfc"})
    if(corr_type=="Standard Scatter Plot"):
        ax = sns.scatterplot(x=asp1, y=asp2, data=df_plot, color = '#f21111')
    #if(corr_type=="Violin Plot (High Computation)"):
    #    ax = sns.violinplot(x=asp1, y=asp2, data=df_plot, color = '#f21111')
    ax.set(xlabel = aspect1, ylabel = aspect2)
    st.pyplot(fig, ax)

def find_match_game_id(min_max, attribute, what):
    df_find = df_data_filtered
    search_attribute = label_fact_dict[attribute]
    if(what == 'difference between teams'):
        search_attribute = 'delta_' + label_fact_dict[attribute]
        df_find[search_attribute] = df_find[search_attribute].abs()
    if(what == 'by both teams'):
        df_find = df_data_filtered.groupby(['game_id'], as_index=False).sum()
    column = df_find[search_attribute]
    index = 0
    if(min_max == 'Minimum'):
        index = column.idxmin()
    if(min_max == 'Maximum'):
        index = column.idxmax()
    game_id = df_find.at[index, 'game_id']
    value = df_find.at[index, search_attribute]
    team = ''
    if(what != 'by both teams'):
        team = df_find.at[index, 'team']
    return_game_id_value_team = [game_id,value,team]
    return return_game_id_value_team

def build_matchfacts_return_string(return_game_id_value_team, min_max, attribute, what):
    game_id = return_game_id_value_team[0]
    df_match_result = df_data_filtered.loc[df_data_filtered['game_id'] == game_id]
    season = df_match_result.iloc[0]['season'].replace('-','/')
    matchday = str(df_match_result.iloc[0]['matchday'])
    home_team = df_match_result.iloc[0]['team']
    away_team = df_match_result.iloc[1]['team']
    goals_home = str(df_match_result.iloc[0]['goals'])
    goals_away = str(df_match_result.iloc[1]['goals'])
    string1 = 'On matchday ' + matchday + ' of the season ' + season + ' ' + home_team + ' played against ' + away_team + '.'
    string2 = ''
    if(goals_home>goals_away):
        string2 = "The match resulted in a " + goals_home + ":" + goals_away + " (" + str(df_match_result.iloc[0]['ht_goals']) + ":" + str(df_match_result.iloc[1]['ht_goals']) +") win for " + home_team + "."
    if(goals_home<goals_away):
        string2 = "The match resulted in a " + goals_home + ":" + goals_away + " (" + str(df_match_result.iloc[0]['ht_goals']) + ":" + str(df_match_result.iloc[1]['ht_goals']) +") loss for " + home_team + "."
    if(goals_home==goals_away):
        string2 = "The match resulted in a " + goals_home + ":" + goals_away + " (" + str(df_match_result.iloc[0]['ht_goals']) + ":" + str(df_match_result.iloc[1]['ht_goals']) +") draw. "
    string3 = ""
    value = str(abs(round(return_game_id_value_team[1],2)))
    team = str(return_game_id_value_team[2])
    if(what == "difference between teams"):
        string3 = " Over the course of the match, a difference of " + value + " " + attribute + " was recorded between the teams."
        string4 = " This is the " + min_max.lower() + " difference for two teams in the currently selected data."
    if(what == "by both teams"):
        string3 = " Over the course of the match, both teams recorded " + value + " " + attribute + " together."
        string4 = " This is the " + min_max.lower() +" value for two teams in the currently selected data."
    if(what == "by a team"):
        string3 = " Over the course of the match, " + team + " recorded " + value + " " + attribute + "."
        string4 = " This is the " + min_max.lower() +" value for a team in the currently selected data."
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

st.sidebar.markdown('**First select the data range you want to analyze:** 👇')
unique_seasons = get_unique_seasons_modified(df_database)
# Sidebar for season selection
start_season, end_season = st.sidebar.select_slider(
    'Select the season range you want to include',
    options=unique_seasons,
    value=(unique_seasons[0], unique_seasons[-1])
)
# Filter data based on selected seasons
df_data_filtered_season = filter_season(df_database, start_season, end_season)
# Get unique seasons and matchdays
unique_seasons = get_unique_seasons_modified(df_database)
unique_matchdays = get_unique_matchdays(df_database)

### MATCHDAY RANGE ###
# unique_matchdays = get_unique_matchdays(df_data_filtered_season)
# Sidebar for matchday selection
# Check if the selected schema is 'team_season'
if selected_schema == 'team_match':
    # Get unique matchdays
    unique_matchdays = get_unique_matchdays(df_data_filtered_season)
    
    # Sidebar for matchday selection
    selected_matchdays = st.sidebar.select_slider(
        'Select the matchday range you want to include',
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
df_data_filtered = filter_matchday(df_data_filtered_season, selected_matchdays)

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
with row2_2:
    unique_teams_in_df = len(np.unique(df_data_filtered.team).tolist())
    t = ' Teams'
    if(unique_teams_in_df==1):
        t = ' Team'
    str_teams = "🏃‍♂️ " + str(unique_teams_in_df) + t
    st.markdown(str_teams)
if selected_schema == 'team_match':
    with row2_3:
        # Ensure the 'GF' column is numeric
        df_data_filtered['GF'] = pd.to_numeric(df_data_filtered['GF'], errors='coerce')
        
        # Sum the goals and divide by 2
        total_goals_in_df = df_data_filtered['GF'].sum() / 2  # Divide by 2 because of home and away rows for each game
        
        # Format the result as an integer
        str_goals = "🥅 " + str(int(total_goals_in_df)) + " Goals"
        st.markdown(str_goals)
# with row2_4:
#     total_shots_in_df = df_data_filtered['shots_on_goal'].sum()
#     str_shots = "👟⚽ " + str(total_shots_in_df) + " Shots"
#     st.markdown(str_shots)

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

### DATA EXPLORATION ###
row12_spacer1, row12_1, row12_spacer2 = st.columns((.2, 7.1, .2))
with row12_1:
    st.subheader('Match Finder')
    st.markdown('Show the (or a) match with the...')
if all_teams_selected == 'Include all available teams':
    row13_spacer1, row13_1, row13_spacer2, row13_2, row13_spacer3, row13_3, row13_spacer4 = st.columns((.2, 2.3, .2, 2.3, .2, 2.3, .2))
    with row13_1:
        show_me_hi_lo = st.selectbox('', ['Maximum', 'Minimum'], key='hi_lo')
    with row13_2:
        show_me_aspect = st.selectbox('', list(label_fact_dict.keys()), key='what')
    with row13_3:
        show_me_what = st.selectbox('', ['by a team', 'by both teams', 'difference between teams'], key='one_both_diff')
    row14_spacer1, row14_1, row14_spacer2 = st.columns((.2, 7.1, .2))
    # with row14_1:
    #     return_game_id_value_team = find_match_game_id(show_me_hi_lo,show_me_aspect,show_me_what)
    #     df_match_result = build_matchfacts_return_string(return_game_id_value_team,show_me_hi_lo,show_me_aspect,show_me_what)     
    row15_spacer1, row15_1, row15_2, row15_3, row15_4, row15_spacer2  = st.columns((0.5, 1.5, 1.5, 1, 2, 0.5))
    with row15_1:
        st.subheader(" ‎")
    # with row15_2:
    #     st.subheader(str(df_match_result.iloc[0]['team']))
    # with row15_3:
    #     end_result = str(df_match_result.iloc[0]['goals']) + " : " +str(df_match_result.iloc[1]['goals'])
    #     ht_result = " ‎ ‎( " + str(df_match_result.iloc[0]['ht_goals']) + " : " +str(df_match_result.iloc[1]['ht_goals']) + " )"
    #     st.subheader(end_result + " " + ht_result)  
    # with row15_4:
    #     st.subheader(str(df_match_result.iloc[1]['team']))
else:
    row17_spacer1, row17_1, row17_spacer2 = st.columns((.2, 7.1, .2))
    with row17_1:
        st.warning('Unfortunately this analysis is only available if all teams are included')

if all_teams_selected == 'Include all available teams':
    row16_spacer1, row16_1, row16_2, row16_3, row16_4, row16_spacer2  = st.columns((0.5, 1.5, 1.5, 1, 2, 0.5))
    with row16_1:
        st.markdown("👟 Shots on Goal")
        st.markdown("🏃‍♂️ Distance (in km)")
        st.markdown("🔁 Passes")
        st.markdown("🤹‍♂️ Possession")
        st.markdown("🤕 Fouls")
        st.markdown("🚫 Offside")
        st.markdown("📐 Corners")
    # with row16_2:
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[0]['shots_on_goal']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[0]['distance']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎"+str(df_match_result.iloc[0]['total_passes']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎ ‎‎"+str(df_match_result.iloc[0]['possession']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[0]['fouls']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[0]['offside']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[0]['corners']))
    # with row16_4:
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[1]['shots_on_goal']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[1]['distance']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎"+str(df_match_result.iloc[1]['total_passes']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎"+str(df_match_result.iloc[1]['possession']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[1]['fouls']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[1]['offside']))
    #     st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[1]['corners']))

### TEAM ###
row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
with row4_1:
    st.subheader('Analysis per Team')
row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row5_1:
    st.markdown('Investigate a variety of stats for each team. Which team scores the most goals per game? How deas your team compare in terms of distance ran per game?')
    plot_x_per_team_selected = st.selectbox('Which attribute do you want to analyze?', list(label_attr_dict_teams.keys()), key='attribute_team')
    plot_x_per_team_type = st.selectbox('Which measure do you want to analyze?', types, key = 'measure_team')
    specific_team_colors = st.checkbox('Use team specific color scheme')
with row5_2:
    if all_teams_selected != 'Select teams manually (choose below)' or selected_teams:
        plot_x_per_team(plot_x_per_team_selected, plot_x_per_team_type)
    else:
        st.warning('Please select at least one team')

### SEASON ###
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader('Analysis per Season')
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row7_1:
    st.markdown('Investigate developments and trends. Which season had teams score the most goals? Has the amount of passes per games changed?')
    plot_x_per_season_selected = st.selectbox('Which attribute do you want to analyze?', list(label_attr_dict.keys()), key='attribute_season')
    plot_x_per_season_type = st.selectbox('Which measure do you want to analyze?', types, key = 'measure_season')
with row7_2:
    if all_teams_selected != 'Select teams manually (choose below)' or selected_teams:
        plot_x_per_season(plot_x_per_season_selected, plot_x_per_season_type)
    else:
        st.warning('Please select at least one team')

### MATCHDAY ###
row8_spacer1, row8_1, row8_spacer2 = st.columns((.2, 7.1, .2))
with row8_1:
    st.subheader('Analysis per Matchday')
row9_spacer1, row9_1, row9_spacer2, row9_2, row9_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row9_1:
    st.markdown('Investigate stats over the course of a season. At what point in the season do teams score the most goals? Do teams run less toward the end of the season?')
    plot_x_per_matchday_selected = st.selectbox('Which aspect do you want to analyze?', list(label_attr_dict.keys()), key='attribute_matchday')
    plot_x_per_matchday_type = st.selectbox('Which measure do you want to analyze?', types, key = 'measure_matchday')
with row9_2:
    if all_teams_selected != 'Select teams manually (choose below)' or selected_teams:
        plot_x_per_matchday(plot_x_per_matchday_selected, plot_x_per_matchday_type)
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
    y_axis_aspect2 = st.selectbox('Which attribute do you want on the y-axis?', list(label_attr_dict_correlation.keys()))
    x_axis_aspect1 = st.selectbox('Which attribute do you want on the x-axis?', list(label_attr_dict_correlation.keys()))
with row11_2:
    if all_teams_selected != 'Select teams manually (choose below)' or selected_teams:
        plt_attribute_correlation(x_axis_aspect1, y_axis_aspect2)
    else:
        st.warning('Please select at least one team')

for variable in dir():
    if variable[0:2] != '__':
        del globals()[variable]
del variable