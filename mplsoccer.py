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
    