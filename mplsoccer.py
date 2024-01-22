import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns

st.set_page_config(layout="wide")

# Load the data
df_database = pd.read_csv('')
types = ['Mean', 'Absolute', 'Median', 'Maximum', 'Minimum']
label_attr_dict = {}
label_attr_dict_teams = {}
color_dict = {}
label_attr_dict_correlation = {}
label_fact_dict = {}

# Helper methods
def get_unique_seasons_modified(df_data):
    '''
    Returns the unique seasons from the dataframe
    '''
    unique_seasons = np.unique(df_data.season).tolist()
    seasons_modified = []
    for s, season in enumerate(unique_seasons):
        if s==0:
            season = " " + season
        if s==len(unique_seasons)-1:
            season = season + " "
        seasons_modified.append(season.replace("-","/"))
    return seasons_modified

def get_unique_matchdays(df_data):
    '''
    Returns the minimum and maximum
    '''
    unique_matchdays = np.unique(df_data.matchday).tolist()
    return unique_matchdays

def get_unique_teams(df_data):
    '''
    Returns the unique teams from the dataframe
    '''
    unique_teams = np.unique(df_data.team).tolist()
    return unique_teams

def filter_season(df_data):
    df_filtered_season = pd.DataFrame()
    seasons = np.unique(df_data.season).tolist() # Season list 13-14
    start_raw = start_season.replace('/','-').replace(' ','')
    end_raw = end_season.replace('/','-').replace(' ','')
    start_index = seasons.index(start_raw)
    end_index = seasons.index(end_raw)+1
    seasons_selected = seasons[start_index:end_index]
    df_filtered_season = df_data[df_data['season'].isin(seasons_selected)]
    return df_filtered_season

def filter_matchday(df_data):
    df_filtered_matchday = pd.DataFrame()
    matchdays_list = list(range(selected_matchdays[0], selected_matchdays[1]+1))
    df_filtered_matchday = df_Data[df_data['matchday'].isin(matchdays_list)]
    return df_filtered_matchday
    
def filter_teams(df_data):
    df_filtered_team = pd.DataFrame()
    if all_teams_selected == 'Select teams manally (choose below)':
        df_filtered_team = df_data[df_data['team'].isin(selected_teams)]
        return df_filtered_team
    return df_data

def stack_home_away_dataframe(df_data):
    df_data['game_id'] = df_data.index + 1
    delta_names = ['goals', 'ht_goals', 'shots_on_goal', 'distance', 'total_passes', 'pass_ratio', 'possession', 'tackle_ratio', 'fouls', 'offside', 'corners']
    for column in delta_names:
        h_delta_column = 'h_delta_' + column
        a_delta_column = 'a_delta_' + column
        h_column = 'h_' + column
        a_column = 'a_' + column
        df_data[h_delta_column] = df_data[h_column]-df_data[a_column]
        df_data[a_delta_column] = df_data[a_column]-df_data[h_column]

    column_names = ['distance','total_passes','success_passes','failed_passes','pass_ratio','possession','tackle_ratio','offside','corners','delta_goals','delta_ht_goals','delta_shots_on_goal','delta_distance','delta_total_passes','delta_pass_ratio','delta_possession','delta_tackle_ratio','delta_fouls','delta_offside','delta_corners']
    h_column_names = ['game_id','season','matchday','h_team','h_goals','a_goals','h_ht_goals','a_ht_goals','h_shots_on_goal','a_shots_on_goal','h_fouls','a_fouls']
    a_column_names = ['game_id','season','matchday','a_team','a_goals','h_goals','a_ht_goals','h_ht_goals','a_shots_on_goal','h_shots_on_goal','a_fouls','h_fouls']
    column_names_new = ['game_id','season','matchday','location','team','goals','goals_received','ht_goals','ht_goals_received','shots_on_goal','shots_on_goal_received','fouls','got_fouled','distance','total_passes','success_passes','failed_passes','pass_ratio','possession','tackle_ratio','offside','corners','delta_goals','delta_ht_goals','delta_shots_on_goal','delta_distance','delta_total_passes','delta_pass_ratio','delta_possession','delta_tackle_ratio','delta_fouls','delta_offside','delta_corners']
    for column in column_names: 
        h_column_names.append('h_' + column)
        a_column_names.append('a_' + column)
    df_home = df_data.filter(h_column_names)
    df_away = df_data.filter(a_column_names)
    df_home.insert(3, 'location', 'h')
    df_away.insert(3, 'location', 'a')
    df_home.columns = column_names_new
    df_away.columns = column_names_new
    df_total = df_home.append(df_Away, ignore_index=True).sort_values(['game_id', 'season', 'matchday'], ascending=[True, True, True])
    df_total_sorted = df_total[['game_id','season','matchday','location','team','goals','goals_received','delta_goals','ht_goals','ht_goals_received','delta_ht_goals','shots_on_goal','shots_on_goal_received','delta_shots_on_goal','distance','delta_distance','total_passes','delta_total_passes','success_passes','failed_passes','pass_ratio','delta_pass_ratio','possession','delta_possession','tackle_ratio','delta_tackle_ratio','fouls','got_fouled','delta_fouls','offside','delta_offside','corners','delta_corners']]
    return df_total_sorted

def