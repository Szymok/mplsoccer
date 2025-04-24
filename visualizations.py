import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from data_processing import group_measure_by_attribute

# Define team colors for visualizations
team_colors = {
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

def set_plot_style():
    """Set standard plot style for consistency"""
    return {
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

def plot_team_analysis(df_data, attribute, measure, specific_team_colors=False):
    """Plot team analysis visualization"""
    plt.rcParams.update(set_plot_style())
    fig, ax = plt.subplots()
    
    # Group data
    df_plot = group_measure_by_attribute(df_data, "team", attribute, measure)
    
    # Create bar plot with appropriate color scheme
    if specific_team_colors:
        ax = sns.barplot(x="aspect", y=attribute, data=df_plot.reset_index(), palette=team_colors)
    else:
        ax = sns.barplot(x="aspect", y=attribute, data=df_plot.reset_index(), color="#b80606")
    
    # Set label based on measure type
    y_str = measure + " " + attribute + " per Game"
    if measure == "Absolute":
        y_str = measure + " " + attribute
    if measure in ["Minimum", "Maximum"]:
        y_str = measure + " " + attribute + " in a Game"
    
    ax.set(xlabel="Team", ylabel=y_str)
    plt.xticks(rotation=66, horizontalalignment="right")
    
    # Add value annotations to bars
    if measure == "Mean" or attribute in ["distance", "pass_ratio", "possession", "tackle_ratio"]:
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 18),
                        rotation=90, textcoords='offset points')
    else:
        for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 18),
                        rotation=90, textcoords='offset points')
    
    return fig

def plot_season_analysis(df_data, attribute, measure):
    """Plot season analysis visualization"""
    plt.rcParams.update(set_plot_style())
    fig, ax = plt.subplots()
    
    # Group data
    df_plot = group_measure_by_attribute(df_data, "season", attribute, measure)
    
    # Create bar plot
    ax = sns.barplot(x="aspect", y=attribute, data=df_plot.reset_index(), color="#b80606")
    
    # Set labels
    ax.set(xlabel='Season', ylabel=attribute)
    plt.xticks(rotation=66, horizontalalignment="center")
    
    # Add value annotations to bars
    for p in ax.patches:
        value = p.get_height()
        
        if value < 1:
            ax.annotate(f'{value:.2f}', (p.get_x() + p.get_width() / 2., value), 
                        ha='center', va='bottom', fontsize=10, color='white', weight='bold')
        else:
            ax.annotate(f'{int(value)}', (p.get_x() + p.get_width() / 2., value),
                        ha='center', va='bottom', fontsize=10, color='white', weight='bold')
    
    return fig

def plot_matchday_analysis(df_data, attribute, measure, selected_matchdays):
    """Plot matchday analysis visualization"""
    plt.rcParams.update(set_plot_style())
    fig, ax = plt.subplots()
    
    # Ensure df_data is not empty
    if df_data.empty:
        return fig
    
    # Convert attribute to numeric
    df_data[attribute] = pd.to_numeric(df_data[attribute], errors='coerce')
    df_filtered = df_data.dropna(subset=[attribute])
    
    if df_filtered.empty:
        return fig
    
    # Extract matchday number for sorting
    df_filtered['round_number'] = df_filtered['round'].apply(lambda x: int(x.split()[1]))
    df_filtered.sort_values('round_number', inplace=True)
    
    # Calculate statistics for plotting based on measure
    if measure == 'Maximum':
        df_plot = df_filtered.groupby(['round_number'])[attribute].max().reset_index()
    elif measure == 'Minimum':
        df_plot = df_filtered.groupby(['round_number'])[attribute].min().reset_index()
    elif measure == 'Mean':
        df_plot = df_filtered.groupby(['round_number'])[attribute].mean().reset_index()
    elif measure == 'Absolute':
        df_plot = df_filtered.groupby(['round_number'])[attribute].sum().reset_index()
    elif measure == 'Median':
        df_plot = df_filtered.groupby(['round_number'])[attribute].median().reset_index()
    
    # Plot the data
    ax = sns.barplot(x='round_number', y=attribute, data=df_plot, color='#b80606')
    ax.set(xlabel='Matchweek', ylabel=attribute)
    plt.xticks(rotation=45, ha='right')
    
    # Add value annotations
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f' if measure in ['Mean', 'Median'] else 'd'),
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='white')
    
    return fig

def plot_correlation_analysis(df_data, x_attribute, y_attribute, corr_type):
    """Plot correlation analysis visualization"""
    plt.rcParams.update(set_plot_style())
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Create a copy of the dataframe to avoid modifying the original
    df_temp = df_data.copy()
    
    # Convert attributes to numeric
    df_temp[x_attribute] = pd.to_numeric(df_temp[x_attribute], errors='coerce')
    df_temp[y_attribute] = pd.to_numeric(df_temp[y_attribute], errors='coerce')
    
    # Drop rows with NaN values
    df_filtered = df_temp.dropna(subset=[x_attribute, y_attribute])
    
    # Check if dataframe is empty
    if df_filtered.empty:
        return fig
    
    # Check for zero variance which causes division errors
    x_var = df_filtered[x_attribute].var()
    y_var = df_filtered[y_attribute].var()
    
    # Return empty plot if either variable has zero variance
    if x_var <= 1e-10 or y_var <= 1e-10:
        ax.text(0.5, 0.5, f"Cannot create plot: One or both variables have near-zero variance",
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Check for extreme values that might cause numerical issues
    if df_filtered[x_attribute].abs().max() > 1e10 or df_filtered[y_attribute].abs().max() > 1e10:
        # Normalize data if values are extreme
        df_filtered[x_attribute] = (df_filtered[x_attribute] - df_filtered[x_attribute].mean()) / df_filtered[x_attribute].std()
        df_filtered[y_attribute] = (df_filtered[y_attribute] - df_filtered[y_attribute].mean()) / df_filtered[y_attribute].std()
    
    # Create plot based on corr_type
    if corr_type == "Regression Plot (Recommended)":
        try:
            ax = sns.regplot(x=x_attribute, y=y_attribute, x_jitter=.1, data=df_filtered, color='#f21111',
                            scatter_kws={"color": "#f21111"}, line_kws={"color": "#c2dbfc"})
        except Exception as e:
            # Fall back to scatterplot if regression fails
            ax = sns.scatterplot(x=x_attribute, y=y_attribute, data=df_filtered, color='#f21111')
            ax.text(0.5, 0.1, f"Regression calculation failed: {str(e)}", 
                   ha='center', transform=ax.transAxes, fontsize=9)
    elif corr_type == "Standard Scatter Plot":
        ax = sns.scatterplot(x=x_attribute, y=y_attribute, data=df_filtered, color='#f21111')
    
    ax.set(xlabel=x_attribute, ylabel=y_attribute)
    return fig

def create_interactive_team_chart(df_data, attribute, teams, measure):
    """Create an interactive bar chart for team comparison using Plotly"""
    # Filter data for selected teams
    df_filtered = df_data[df_data['team'].isin(teams)].copy() if teams else df_data.copy()
    
    # Convert attribute to numeric
    df_filtered[attribute] = pd.to_numeric(df_filtered[attribute], errors='coerce')
    df_filtered = df_filtered.dropna(subset=[attribute])
    
    if df_filtered.empty:
        return go.Figure()
    
    # Group data based on measure
    if measure == 'Absolute':
        df_grouped = df_filtered.groupby('team')[attribute].sum().reset_index()
        title = f'Total {attribute} by Team'
    elif measure == 'Mean':
        df_grouped = df_filtered.groupby('team')[attribute].mean().reset_index()
        title = f'Average {attribute} by Team'
    elif measure == 'Median':
        df_grouped = df_filtered.groupby('team')[attribute].median().reset_index()
        title = f'Median {attribute} by Team'
    elif measure == 'Maximum':
        df_grouped = df_filtered.groupby('team')[attribute].max().reset_index()
        title = f'Maximum {attribute} by Team'
    elif measure == 'Minimum':
        df_grouped = df_filtered.groupby('team')[attribute].min().reset_index()
        title = f'Minimum {attribute} by Team'
    
    # Sort values for better visualization
    df_grouped = df_grouped.sort_values(by=attribute, ascending=False)
    
    # Create custom color map based on team colors
    colors = [team_colors.get(team, '#b80606') for team in df_grouped['team']]
    
    # Create the interactive bar chart
    fig = px.bar(
        df_grouped,
        x='team',
        y=attribute,
        color='team',
        color_discrete_sequence=colors,
        title=title,
        labels={'team': 'Team', attribute: attribute.replace('_', ' ').title()}
    )
    
    # Customize layout
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(14, 17, 23, 0.8)',
        paper_bgcolor='rgba(14, 17, 23, 0.8)',
        font=dict(color='white'),
        xaxis_title='Team',
        yaxis_title=attribute.replace('_', ' ').title(),
        showlegend=False
    )
    
    # Add value annotations on bars
    fig.update_traces(
        texttemplate='%{y:.1f}' if measure == 'Mean' else '%{y:,.0f}',
        textposition='outside',
        textfont_color='white'
    )
    
    # Update x-axis for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_player_radar_chart(first_player, second_player, player_stats=None):
    """
    Create a radar chart comparing two players
    
    Parameters:
    first_player (str): First player name
    second_player (str): Second player name
    player_stats (dict): Optional dictionary of player statistics
    
    Returns:
    plotly.graph_objects.Figure: Radar chart
    """
    # If no player stats provided, use dummy data
    if not player_stats:
        # Categories for comparison
        categories = ['Goals', 'Assists', 'Pass Accuracy', 'Distance', 'Tackles', 'Shots']
        
        # Sample data for two players
        player1_values = [12, 8, 85, 320, 45, 65]
        player2_values = [8, 14, 78, 290, 52, 48]
    else:
        categories = list(player_stats[first_player].keys())
        player1_values = list(player_stats[first_player].values())
        player2_values = list(player_stats[second_player].values())
    
    # Create radar chart
    fig = go.Figure()
    
    # Add trace for first player
    fig.add_trace(go.Scatterpolar(
        r=player1_values,
        theta=categories,
        fill='toself',
        name=first_player
    ))
    
    # Add trace for second player
    fig.add_trace(go.Scatterpolar(
        r=player2_values,
        theta=categories,
        fill='toself',
        name=second_player
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(player1_values), max(player2_values)) * 1.1]
            )
        ),
        showlegend=True,
        title=f"{first_player} vs {second_player}",
        template='plotly_dark',
        plot_bgcolor='rgba(14, 17, 23, 0.8)',
        paper_bgcolor='rgba(14, 17, 23, 0.8)',
        font=dict(color='white')
    )
    
    return fig

def create_interactive_correlation_plot(df_data, x_attribute, y_attribute):
    """Create an interactive scatter plot for correlation analysis using Plotly"""
    # Create a copy of the dataframe
    df_filtered = df_data.copy()
    
    # Convert attributes to numeric
    df_filtered[x_attribute] = pd.to_numeric(df_filtered[x_attribute], errors='coerce')
    df_filtered[y_attribute] = pd.to_numeric(df_filtered[y_attribute], errors='coerce')
    
    # Drop rows with NaN values
    df_filtered = df_filtered.dropna(subset=[x_attribute, y_attribute])
    
    if df_filtered.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Check for zero variance which causes division errors
    x_var = df_filtered[x_attribute].var()
    y_var = df_filtered[y_attribute].var()
    
    if x_var <= 1e-10 or y_var <= 1e-10:
        # Return empty figure with message about variance
        fig = go.Figure()
        fig.add_annotation(
            text=f"Cannot create plot: One or both variables have near-zero variance",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Check for extreme values that might cause numerical issues
    needs_scaling = False
    if df_filtered[x_attribute].abs().max() > 1e10 or df_filtered[y_attribute].abs().max() > 1e10:
        # Create copies of columns for scaling
        df_filtered[f"{x_attribute}_orig"] = df_filtered[x_attribute].copy()
        df_filtered[f"{y_attribute}_orig"] = df_filtered[y_attribute].copy()
        
        # Scale the data
        df_filtered[x_attribute] = (df_filtered[x_attribute] - df_filtered[x_attribute].mean()) / df_filtered[x_attribute].std()
        df_filtered[y_attribute] = (df_filtered[y_attribute] - df_filtered[y_attribute].mean()) / df_filtered[y_attribute].std()
        needs_scaling = True
    
    # Determine if we can use OLS trendline safely
    use_ols = True
    # Check if there are enough data points for regression
    if len(df_filtered) < 3:
        use_ols = False
    
    # Create interactive scatter plot
    try:
        if use_ols:
            fig = px.scatter(
                df_filtered, 
                x=x_attribute, 
                y=y_attribute, 
                color='team',
                hover_data=['team', 'season'],
                title=f'{y_attribute} vs {x_attribute}',
                trendline='ols'  # Add trend line
            )
        else:
            fig = px.scatter(
                df_filtered, 
                x=x_attribute, 
                y=y_attribute, 
                color='team',
                hover_data=['team', 'season'],
                title=f'{y_attribute} vs {x_attribute}'
            )
    except Exception as e:
        # Fall back to basic scatter plot if trendline fails
        fig = px.scatter(
            df_filtered, 
            x=x_attribute, 
            y=y_attribute, 
            color='team',
            hover_data=['team', 'season'],
            title=f'{y_attribute} vs {x_attribute} (Trendline calculation failed)'
        )
    
    # Add note if data was scaled
    if needs_scaling:
        fig.add_annotation(
            text="Note: Data was scaled due to extreme values",
            xref="paper", yref="paper",
            x=0.5, y=1.05, showarrow=False
        )
    
    # Customize layout
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(14, 17, 23, 0.8)',
        paper_bgcolor='rgba(14, 17, 23, 0.8)',
        font=dict(color='white')
    )
    
    return fig
