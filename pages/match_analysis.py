import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import load_data_from_db 
from data_processing import (
    get_unique_seasons_modified, 
    get_unique_teams,
    get_unique_matchdays,
    filter_season,
    filter_matchday
)

def match_analysis_page(df_database, conn, schema_info=None):
    st.title('Match Analysis')
    st.markdown("Match-by-match statistics and insights")
    
    # Only proceed if we have match data
    if 'selected_schema' in st.session_state and st.session_state['selected_schema'] in ['team_match', 'player_match']:
        # Sidebar filters
        st.sidebar.header("Match Filters")
        
        # Season filter
        unique_seasons = get_unique_seasons_modified(df_database)
        selected_season = st.sidebar.selectbox(
            'Select season',
            unique_seasons,
            index=len(unique_seasons)-1 if unique_seasons else 0
        )
        
        # Filter data by selected season
        df_filtered_season, error = filter_season(df_database, selected_season, selected_season)
        
        if error:
            st.error(error)
            return
            
        # Get unique matchdays for the selected season
        unique_matchdays = get_unique_matchdays(df_filtered_season)
        
        # Matchday selection
        selected_matchday = st.sidebar.selectbox(
            'Select matchday',
            unique_matchdays,
            index=0
        )
        
        # Filter data for selected matchday
        df_filtered = filter_matchday(df_filtered_season, [selected_matchday])
        
        # Team filter
        teams = get_unique_teams(df_filtered)
        selected_team = st.sidebar.selectbox('Select team', teams)
        
        # Filter for selected team
        df_team = df_filtered[df_filtered['team'] == selected_team]
        
        # Find opponent
        if not df_team.empty and 'opponent' in df_team.columns:
            opponent = df_team['opponent'].iloc[0]
            df_opponent = df_filtered[df_filtered['team'] == opponent]
        else:
            st.warning("No opponent data available")
            opponent = None
            df_opponent = pd.DataFrame()
        
        # Create tabs for different match analyses
        tab1, tab2, tab3 = st.tabs(["Match Overview", "Team Comparison", "Player Performance"])
        
        with tab1:
            st.header("Match Overview")
            
            if not df_team.empty:
                # Match info
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    st.subheader(selected_team)
                    if 'goals' in df_team.columns:
                        goals_home = df_team['goals'].iloc[0]
                        st.metric("Goals", goals_home)
                
                with col2:
                    # Display match date
                    if 'date' in df_team.columns:
                        match_date = df_team['date'].iloc[0]
                        st.markdown(f"**Date:** {match_date}")
                    
                    # Display venue
                    if 'venue' in df_team.columns:
                        venue = df_team['venue'].iloc[0]
                        st.markdown(f"**Venue:** {venue}")
                    
                    # Display result
                    if 'result' in df_team.columns:
                        result = df_team['result'].iloc[0]
                        st.markdown(f"**Result:** {result}")
                
                with col3:
                    st.subheader(opponent if opponent else "Unknown")
                    if not df_opponent.empty and 'goals' in df_opponent.columns:
                        goals_away = df_opponent['goals'].iloc[0]
                        st.metric("Goals", goals_away)
                
                # Match statistics comparison
                st.subheader("Match Statistics")
                
                # Create two columns for side-by-side stats
                stat_col1, stat_col2 = st.columns(2)
                
                # List of statistics to display (adjust based on your data)
                match_stats = [
                    'shots', 'shots_on_target', 'possession', 'passes', 
                    'pass_accuracy', 'fouls', 'yellow_cards', 'red_cards',
                    'corners', 'offsides'
                ]
                
                # Display available statistics
                for stat in match_stats:
                    if stat in df_team.columns:
                        # Get stat values
                        team_stat = df_team[stat].iloc[0] if not df_team.empty else "N/A"
                        opponent_stat = df_opponent[stat].iloc[0] if not df_opponent.empty else "N/A"
                        
                        # Display in columns
                        with stat_col1:
                            st.metric(f"{stat.replace('_', ' ').title()}", team_stat)
                        
                        with stat_col2:
                            st.metric(f"{stat.replace('_', ' ').title()}", opponent_stat)
            else:
                st.warning("No match data available for the selected criteria")
        
        with tab2:
            st.header("Team Comparison")
            
            if not df_team.empty and not df_opponent.empty:
                # Select metrics for comparison
                exclude_columns = ['league', 'team', 'game', 'date', 'round', 'day', 'venue', 'result', 'opponent']
                all_columns = df_team.columns.tolist()
                metric_columns = [col for col in all_columns if col not in exclude_columns]
                
                selected_metrics = st.multiselect(
                    "Select metrics to compare",
                    metric_columns,
                    default=metric_columns[:5] if len(metric_columns) >= 5 else metric_columns
                )
                
                if selected_metrics:
                    # Create comparison dataframe
                    comparison_data = {
                        'Metric': selected_metrics,
                        selected_team: [df_team[metric].iloc[0] if metric in df_team.columns else 0 for metric in selected_metrics],
                        opponent: [df_opponent[metric].iloc[0] if metric in df_opponent.columns else 0 for metric in selected_metrics]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display comparison table
                    st.dataframe(comparison_df)
                    
                    # Create bar chart for comparison
                    chart_data = pd.melt(
                        comparison_df,
                        id_vars=['Metric'],
                        value_vars=[selected_team, opponent],
                        var_name='Team',
                        value_name='Value'
                    )
                    
                    # Use plotly for interactive chart
                    fig = px.bar(
                        chart_data,
                        x='Metric',
                        y='Value',
                        color='Team',
                        barmode='group',
                        title="Team Comparison",
                        labels={'Value': 'Value', 'Metric': 'Metric'}
                    )
                    
                    fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='rgba(14, 17, 23, 0.8)',
                        paper_bgcolor='rgba(14, 17, 23, 0.8)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one metric for comparison")
            else:
                st.warning("Insufficient data for team comparison")
        
        with tab3:
            st.header("Player Performance")
            
            # Check if we have player data
            if 'selected_schema' in st.session_state and st.session_state['selected_schema'] == 'player_match':
                # Fetch player data for the match
                player_query = f"""
                SELECT * FROM player_match
                WHERE game = '{df_team['game'].iloc[0]}'
                AND team = '{selected_team}'
                """
                
                try:
                    player_data, player_error = load_data_from_db(player_query, conn)
                    
                    if player_error:
                        st.error(player_error)
                    elif not player_data.empty:
                        # Display player stats
                        st.subheader(f"{selected_team} Player Statistics")
                        
                        # Sort by minutes played
                        if 'minutes' in player_data.columns:
                            player_data = player_data.sort_values('minutes', ascending=False)
                        
                        # Display player data table
                        st.dataframe(player_data)
                        
                        # Select a player for detailed view
                        if 'player' in player_data.columns:
                            selected_player = st.selectbox(
                                "Select player for detailed statistics",
                                player_data['player'].unique()
                            )
                            
                            # Display detailed player stats
                            player_detail = player_data[player_data['player'] == selected_player]
                            
                            if not player_detail.empty:
                                st.subheader(f"{selected_player} - Detailed Statistics")
                                
                                # Create metrics grid
                                metrics = [
                                    ('Minutes', 'minutes'),
                                    ('Goals', 'goals'),
                                    ('Assists', 'assists'),
                                    ('Shots', 'shots'),
                                    ('Passes', 'passes'),
                                    ('Key Passes', 'key_passes'),
                                    ('Tackles', 'tackles'),
                                    ('Interceptions', 'interceptions')
                                ]
                                
                                # Create metrics in columns
                                cols = st.columns(4)
                                
                                for i, (label, col) in enumerate(metrics):
                                    if col in player_detail.columns:
                                        with cols[i % 4]:
                                            value = player_detail[col].iloc[0]
                                            st.metric(label, value)
                            else:
                                st.warning(f"No detailed statistics available for {selected_player}")
                    else:
                        st.warning("No player data available for this match")
                        
                except Exception as e:
                    st.error(f"Error fetching player data: {str(e)}")
            else:
                st.info("Player performance analysis is only available when using the player_match schema. Please change your schema selection to view player data.")
    else:
        st.warning("Match analysis is only available for match data. Please select the team_match or player_match schema.")
    
    # Navigation button
    if st.button("Back to Main Page"):
        st.session_state['current_page'] = 'main'
