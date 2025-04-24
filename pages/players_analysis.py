import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_processing import (
    get_unique_seasons_modified, 
    get_unique_teams,
    filter_season
)
from visualizations import create_player_radar_chart

def players_analysis_page(df_database, conn, schema_info=None):
    st.title('Players Analysis')
    
    # Sidebar filters for player analysis
    st.sidebar.header("Player Filters")
    
    # Season filter
    unique_seasons = get_unique_seasons_modified(df_database)
    start_season, end_season = st.sidebar.select_slider(
        'Select season range',
        options=unique_seasons,
        value=(unique_seasons[0], unique_seasons[-1])
    )
    
    # Filter data by season
    df_filtered_season, error = filter_season(df_database, start_season, end_season)
    
    if error:
        st.error(error)
        return
    
    # Team filter
    teams = get_unique_teams(df_filtered_season)
    selected_teams = st.sidebar.multiselect('Select teams', teams, default=teams[:5] if len(teams) > 5 else teams)
    
    # Filter by selected teams
    if selected_teams:
        df_filtered = df_filtered_season[df_filtered_season['team'].isin(selected_teams)]
    else:
        df_filtered = df_filtered_season
        st.warning("No teams selected. Showing data for all teams.")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Player Stats", "Player Comparison", "Performance Trends", "Player Rankings"])
    
    with tab1:
        st.header("Individual Player Statistics")
        
        # Team selection
        selected_team = st.selectbox("Select Team", selected_teams if selected_teams else teams)
        
        # Player selection for the team
        if 'player' in df_filtered.columns:
            team_players = df_filtered[df_filtered['team'] == selected_team]['player'].unique().tolist()
            
            if team_players:
                selected_player = st.selectbox("Select Player", team_players)
                
                # Filter data for selected player
                player_data = df_filtered[(df_filtered['team'] == selected_team) & (df_filtered['player'] == selected_player)]
                
                if not player_data.empty:
                    # Display player stats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Try to extract key metrics, handle errors gracefully
                    try:
                        with col1:
                            if 'goals' in player_data.columns:
                                goals = player_data['goals'].astype(float).sum()
                                st.metric("Goals", f"{goals:.0f}")
                        
                        with col2:
                            if 'assists' in player_data.columns:
                                assists = player_data['assists'].astype(float).sum()
                                st.metric("Assists", f"{assists:.0f}")
                        
                        with col3:
                            if 'minutes' in player_data.columns:
                                minutes = player_data['minutes'].astype(float).sum()
                                st.metric("Minutes Played", f"{minutes:.0f}")
                        
                        with col4:
                            if 'games' in player_data.columns:
                                games = player_data['games'].astype(float).sum()
                                st.metric("Games", f"{games:.0f}")
                    
                    except Exception as e:
                        st.error(f"Error calculating player metrics: {str(e)}")
                    
                    # Player stats over time
                    st.subheader(f"{selected_player}'s Performance")
                    
                    # If we have time-series data like multiple matches
                    if 'date' in player_data.columns and player_data.shape[0] > 1:
                        try:
                            # Convert date to datetime if it's not already
                            player_data['date'] = pd.to_datetime(player_data['date'])
                            
                            # Sort by date
                            player_data = player_data.sort_values('date')
                            
                            # Select stat to visualize
                            available_stats = [col for col in player_data.columns if col not in ['player', 'team', 'date', 'season']]
                            selected_stat = st.selectbox("Select statistic to visualize", available_stats)
                            
                            # Create time series plot
                            fig = px.line(
                                player_data,
                                x='date',
                                y=selected_stat,
                                title=f"{selected_player}'s {selected_stat} Over Time",
                                markers=True
                            )
                            
                            # Customize layout
                            fig.update_layout(
                                template='plotly_dark',
                                plot_bgcolor='rgba(14, 17, 23, 0.8)',
                                paper_bgcolor='rgba(14, 17, 23, 0.8)',
                                font=dict(color='white')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating time series plot: {str(e)}")
                    
                    # Display player data table
                    with st.expander("View Player Data"):
                        st.dataframe(player_data)
                else:
                    st.warning(f"No data available for {selected_player}")
            else:
                st.warning(f"No players found for {selected_team}")
        else:
            st.error("Player data not available in the selected table. Please select a table with player information.")
    
    with tab2:
        st.header("Player Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            first_team = st.selectbox("First Team", selected_teams if selected_teams else teams, key='first_team')
            
            # Get players for first team
            if 'player' in df_filtered.columns:
                first_team_players = df_filtered[df_filtered['team'] == first_team]['player'].unique().tolist()
                
                if first_team_players:
                    first_player = st.selectbox("First Player", first_team_players)
                else:
                    st.warning(f"No players found for {first_team}")
                    first_player = None
            else:
                st.error("Player data not available")
                first_player = None
        
        with col2:
            second_team = st.selectbox("Second Team", selected_teams if selected_teams else teams, key='second_team')
            
            # Get players for second team
            if 'player' in df_filtered.columns:
                second_team_players = df_filtered[df_filtered['team'] == second_team]['player'].unique().tolist()
                
                if second_team_players:
                    second_player = st.selectbox("Second Player", second_team_players)
                else:
                    st.warning(f"No players found for {second_team}")
                    second_player = None
            else:
                st.error("Player data not available")
                second_player = None
        
        # Display radar chart if both players are selected
        if first_player and second_player:
            st.subheader(f"Comparing {first_player} vs {second_player}")
            
            # Create radar chart
            try:
                # Extract player stats for comparison
                # In a real implementation, you would query and aggregate actual player stats
                # For now, using placeholder function that generates dummy data
                
                fig = create_player_radar_chart(first_player, second_player)
                st.plotly_chart(fig, use_container_width=True)
                
                # Side-by-side comparison table
                player1_data = df_filtered[df_filtered['player'] == first_player]
                player2_data = df_filtered[df_filtered['player'] == second_player]
                
                # Create comparison dataframe
                if not player1_data.empty and not player2_data.empty:
                    # Select numeric columns for comparison
                    numeric_cols = player1_data.select_dtypes(include=['number']).columns.tolist()
                    
                    # Calculate aggregates
                    player1_stats = player1_data[numeric_cols].sum().to_dict()
                    player2_stats = player2_data[numeric_cols].sum().to_dict()
                    
                    # Create comparison dataframe
                    comparison_data = {
                        'Metric': numeric_cols,
                        first_player: [player1_stats.get(col, 0) for col in numeric_cols],
                        second_player: [player2_stats.get(col, 0) for col in numeric_cols]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display comparison table
                    st.dataframe(comparison_df)
            except Exception as e:
                st.error(f"Error creating player comparison: {str(e)}")
    
    with tab3:
        st.header("Performance Trends")
        
        if 'player' in df_filtered.columns:
            # Team selection for trends
            trend_team = st.selectbox("Select Team", selected_teams if selected_teams else teams, key='trend_team')
            
            # Get players for the selected team
            team_players = df_filtered[df_filtered['team'] == trend_team]['player'].unique().tolist()
            
            if team_players:
                trend_player = st.selectbox("Select Player", team_players, key='trend_player')
                
                # Filter data for selected player
                player_trend_data = df_filtered[(df_filtered['team'] == trend_team) & (df_filtered['player'] == trend_player)]
                
                if not player_trend_data.empty:
                    # Select statistic for trend analysis
                    numeric_cols = player_trend_data.select_dtypes(include=['number']).columns.tolist()
                    selected_stat = st.selectbox("Select Statistic", numeric_cols)
                    
                    # Group by season to see trends over seasons
                    if 'season' in player_trend_data.columns:
                        try:
                            # Format season for readability
                            player_trend_data['season_readable'] = player_trend_data['season'].apply(
                                lambda x: f"20{str(x)[:2]}/20{str(x)[2:]}"
                            )
                            
                            # Group by season and calculate stat total or average
                            agg_method = st.radio("Aggregation Method", ["Total", "Average"])
                            
                            if agg_method == "Total":
                                season_stats = player_trend_data.groupby('season_readable')[selected_stat].sum().reset_index()
                            else:
                                season_stats = player_trend_data.groupby('season_readable')[selected_stat].mean().reset_index()
                            
                            # Sort by season
                            season_stats = season_stats.sort_values('season_readable')
                            
                            # Create trend chart
                            fig = px.line(
                                season_stats,
                                x='season_readable',
                                y=selected_stat,
                                title=f"{trend_player}'s {selected_stat} by Season ({agg_method})",
                                markers=True,
                                line_shape='linear'
                            )
                            
                            # Customize layout
                            fig.update_layout(
                                template='plotly_dark',
                                plot_bgcolor='rgba(14, 17, 23, 0.8)',
                                paper_bgcolor='rgba(14, 17, 23, 0.8)',
                                font=dict(color='white'),
                                xaxis_title="Season",
                                yaxis_title=selected_stat
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating trend chart: {str(e)}")
                else:
                    st.warning(f"No data available for {trend_player}")
            else:
                st.warning(f"No players found for {trend_team}")
        else:
            st.error("Player data not available in the selected table")
    
    with tab4:
        st.header("Player Rankings")
        
        if 'player' in df_filtered.columns:
            # Select season for rankings
            ranking_season = st.selectbox("Season", unique_seasons, key='ranking_season')
            
            # Filter data for selected season
            season_data, error = filter_season(df_filtered, ranking_season, ranking_season)
            
            if error:
                st.error(error)
            elif not season_data.empty:
                # Select statistic for ranking
                numeric_cols = season_data.select_dtypes(include=['number']).columns.tolist()
                rank_stat = st.selectbox("Ranking By", numeric_cols)
                
                # Minimum games/minutes filter for qualification
                if 'games' in season_data.columns:
                    min_games = st.slider("Minimum Games Played", 1, int(season_data['games'].max()), 5)
                    season_data = season_data[season_data['games'] >= min_games]
                
                # Group by player and calculate total or average
                ranking_method = st.radio("Ranking Method", ["Total", "Per Game"])
                
                try:
                    if ranking_method == "Total":
                        player_rankings = season_data.groupby(['player', 'team'])[rank_stat].sum().reset_index()
                    else:
                        # Calculate per game stats
                        if 'games' in season_data.columns:
                            # Group by player first
                            player_totals = season_data.groupby(['player', 'team']).agg({
                                rank_stat: 'sum',
                                'games': 'sum'
                            }).reset_index()
                            
                            # Calculate per game
                            player_rankings = player_totals.copy()
                            player_rankings[rank_stat] = player_rankings[rank_stat] / player_rankings['games']
                        else:
                            st.warning("Games column not available, using totals instead")
                            player_rankings = season_data.groupby(['player', 'team'])[rank_stat].sum().reset_index()
                    
                    # Sort by the selected stat in descending order
                    player_rankings = player_rankings.sort_values(rank_stat, ascending=False)
                    
                    # Add rank column
                    player_rankings['Rank'] = range(1, len(player_rankings) + 1)
                    
                    # Reorder columns to show rank first
                    columns_order = ['Rank', 'player', 'team', rank_stat]
                    if ranking_method == "Per Game" and 'games' in player_rankings.columns:
                        columns_order.append('games')
                    
                    player_rankings = player_rankings[columns_order]
                    
                    # Rename columns for display
                    player_rankings = player_rankings.rename(columns={
                        'player': 'Player',
                        'team': 'Team',
                        rank_stat: f"{rank_stat.title()}" + (" Per Game" if ranking_method == "Per Game" else ""),
                        'games': 'Games Played'
                    })
                    
                    # Display rankings
                    st.subheader(f"Top Players by {rank_stat.title()} ({ranking_season})")
                    st.dataframe(player_rankings, use_container_width=True)
                    
                    # Visualize top players
                    top_n = st.slider("Show Top N Players", 5, 30, 10)
                    top_players = player_rankings.head(top_n)
                    
                    fig = px.bar(
                        top_players,
                        x='Player',
                        y=f"{rank_stat.title()}" + (" Per Game" if ranking_method == "Per Game" else ""),
                        color='Team',
                        title=f"Top {top_n} Players by {rank_stat.title()}" + (" Per Game" if ranking_method == "Per Game" else ""),
                        text='Rank'
                    )
                    
                    fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='rgba(14, 17, 23, 0.8)',
                        paper_bgcolor='rgba(14, 17, 23, 0.8)',
                        font=dict(color='white'),
                        xaxis_title="Player",
                        xaxis={'categoryorder':'total descending'},
                        yaxis_title=f"{rank_stat.title()}" + (" Per Game" if ranking_method == "Per Game" else "")
                    )
                    
                    # Update bar text position
                    fig.update_traces(
                        textposition='inside',
                        texttemplate='#%{text}'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating player rankings: {str(e)}")
            else:
                st.warning(f"No data available for season {ranking_season}")
        else:
            st.error("Player data not available in the selected table")
    
    # Navigation button
    if st.button("Back to Main Page"):
        st.session_state['current_page'] = 'main'
