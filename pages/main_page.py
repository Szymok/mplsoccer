import streamlit as st
import pandas as pd
import numpy as np
from data_processing import (
    get_unique_seasons_modified, 
    get_unique_teams, 
    filter_season, 
    filter_matchday, 
    get_unique_matchdays,
    filter_teams,
    stack_team_dataframe,
    find_match_game_id
)
from visualizations import (
    plot_team_analysis, 
    plot_season_analysis, 
    plot_matchday_analysis, 
    plot_correlation_analysis,
    create_interactive_team_chart,
    create_interactive_correlation_plot
)

def main_page(df_database, conn, schema_info=None):
    st.title('La Liga Stats Analysis')
    st.subheader('App by [SkSzymon](https://www.twitter.com/SkSzymon)')
    
    st.markdown("Hello there! This interactive application allows you to discover LaLiga statistics across seasons. If you're on a mobile device, I would recommend switching over to landscape for viewing ease.")
    st.markdown("You can find the source code in the [GitHub repository]()")
    
    # Stack dataframe for better organization
    df_stacked = stack_team_dataframe(df_database)
    
    # Sidebar filters
    if 'selected_schema' in st.session_state and st.session_state['selected_schema'] in ['team_season', 'team_match']:
        st.sidebar.markdown('**First select the data range you want to analyze:** 👇')
        
        # Get unique seasons
        unique_seasons = get_unique_seasons_modified(df_database)
        
        # Season selection
        start_season, end_season = st.sidebar.select_slider(
            'Select the season range you want to include',
            options=unique_seasons,
            value=(unique_seasons[0], unique_seasons[-1])
        )
        
        # Filter data by season
        df_data_filtered_season, season_error = filter_season(df_database, start_season, end_season)
        
        if season_error:
            st.error(season_error)
            return
        
        # Matchday selection for match data
        if st.session_state['selected_schema'] in ["player_match", "team_match"]:
            st.sidebar.markdown('**Select the matchweek range you want to analyze:** 👇')
            
            # Get unique matchdays
            unique_matchdays = get_unique_matchdays(df_data_filtered_season)
            
            if unique_matchdays:
                # Matchday selection
                start_matchweek, end_matchweek = st.sidebar.select_slider(
                    'Select the matchweek range you want to include',
                    options=unique_matchdays,
                    value=(unique_matchdays[0], unique_matchdays[-1])
                )
                
                # Create list of selected matchweeks
                selected_start = int(start_matchweek.split()[1])
                selected_end = int(end_matchweek.split()[1])
                full_selected_matchweeks = [f"Matchweek {i}" for i in range(selected_start, selected_end + 1)]
                
                # Filter by matchday
                df_data_filtered = filter_matchday(df_data_filtered_season, full_selected_matchweeks)
            else:
                df_data_filtered = df_data_filtered_season
                st.warning("No matchday data available in the selected seasons.")
        else:
            df_data_filtered = df_data_filtered_season
        
        # Team selection
        unique_teams = get_unique_teams(df_stacked)
        all_teams_selected = st.sidebar.selectbox(
            'Do you want to only include specific teams?',
            ['Include all available teams', 'Select teams manually (choose below)']
        )
        
        selected_teams = None
        if all_teams_selected == 'Select teams manually (choose below)':
            selected_teams = st.sidebar.multiselect(
                'Select teams to include in the analysis',
                unique_teams,
                default=unique_teams
            )
            
            # Filter by teams
            df_data_filtered = filter_teams(df_data_filtered, all_teams_selected, selected_teams)
        
        # Show data summary
        st.subheader('Currently selected data:')
        
        # Create metric rows
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state['selected_schema'] == 'team_match':
                unique_games = df_data_filtered['game'].nunique()
                st.metric("Matches", f"{unique_games}")
            else:
                # Calculate total players used
                if 'players_used' in df_data_filtered.columns:
                    total_players = pd.to_numeric(df_data_filtered['players_used'], errors='coerce').sum()
                    st.metric("Players Used", f"{int(total_players)}")
        
        with col2:
            unique_teams_count = len(np.unique(df_data_filtered.team).tolist())
            team_label = 'Teams' if unique_teams_count != 1 else 'Team'
            st.metric(team_label, f"{unique_teams_count}")
        
        with col3:
            row_count = df_data_filtered.shape[0]
            st.metric("Rows", f"{row_count}")
        
        with col4:
            season_count = df_data_filtered['season'].nunique()
            season_label = 'Seasons' if season_count != 1 else 'Season'
            st.metric(season_label, f"{season_count}")
        
        # Data preview expander
        with st.expander('Click here to see the raw data 👉'):
            st.dataframe(df_data_filtered.reset_index(drop=True))
        
        # Define which columns to exclude from analysis selectors
        exclude_columns = ['league', 'team', 'game', 'date', 'round', 'day', 'venue', 'result', 'opponent']
        all_columns = df_data_filtered.columns.tolist()
        filtered_columns = [col for col in all_columns if col not in exclude_columns]
        
        # Match Finder
        st.subheader('Match Finder')
        st.markdown('Show the match with the...')
        
        if all_teams_selected == 'Include all available teams':
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_me_hi_lo = st.selectbox('', ['Maximum', 'Minimum'], key='hi_lo')
            
            with col2:
                show_me_aspect = st.selectbox('Select attribute to analyze:', filtered_columns, key='what')
            
            with col3:
                show_me_what = st.selectbox('', ['by a team', 'by both teams', 'difference between teams'], key='one_both_diff')
            
            # Find match with min/max stat
            try:
                match_info = find_match_game_id(df_data_filtered, show_me_hi_lo, show_me_aspect, show_me_what)
                season, value, team = match_info
                
                # Display match information
                st.info(f"The {show_me_hi_lo.lower()} {show_me_aspect} {show_me_what} was {value:.2f} by {team} in season {season}.")
                
                # Add visualizations or more details about the match here
            except Exception as e:
                st.error(f"Error finding match: {str(e)}")
        else:
            st.warning('Match finder is only available when all teams are included')
        
        # Team Analysis
        st.subheader('Analysis per Team')
        
        team_col1, team_col2 = st.columns([1, 2])
        
        with team_col1:
            st.markdown('Investigate a variety of stats for each team. Which team scores the most goals per game?')
            
            plot_x_per_team_selected = st.selectbox('Which attribute do you want to analyze?', filtered_columns, key='attribute_team')
            plot_x_per_team_type = st.selectbox('Which measure do you want to analyze?', ['Mean', 'Absolute', 'Median', 'Maximum', 'Minimum'], key='measure_team')
            specific_team_colors = st.checkbox('Use team specific color scheme')
            use_interactive = st.checkbox('Use interactive chart', value=True)
        
        with team_col2:
            if all_teams_selected != 'Select teams manually (choose below)' or (selected_teams and len(selected_teams) > 0):
                try:
                    if use_interactive:
                        teams_to_plot = selected_teams if all_teams_selected == 'Select teams manually (choose below)' else get_unique_teams(df_data_filtered)
                        fig = create_interactive_team_chart(df_data_filtered, plot_x_per_team_selected, teams_to_plot, plot_x_per_team_type)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = plot_team_analysis(df_data_filtered, plot_x_per_team_selected, plot_x_per_team_type, specific_team_colors)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating team chart: {str(e)}")
            else:
                st.warning('Please select at least one team')
        
        # Season Analysis
        st.subheader('Analysis per Season')
        
        season_col1, season_col2 = st.columns([1, 2])
        
        with season_col1:
            st.markdown('Investigate developments and trends. Which season had teams score the most goals?')
            
            plot_x_per_season_selected = st.selectbox('Which attribute do you want to analyze?', filtered_columns, key='attribute_season')
            plot_x_per_season_type = st.selectbox('Which measure do you want to analyze?', ['Mean', 'Absolute', 'Median', 'Maximum', 'Minimum'], key='measure_season')
        
        with season_col2:
            if all_teams_selected != 'Select teams manually (choose below)' or (selected_teams and len(selected_teams) > 0):
                try:
                    fig = plot_season_analysis(df_data_filtered, plot_x_per_season_selected, plot_x_per_season_type)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating season chart: {str(e)}")
            else:
                st.warning('Please select at least one team')
        
        # Matchday Analysis - only for match data
        if st.session_state['selected_schema'] != 'team_season':
            st.subheader('Analysis per Matchday')
            
            matchday_col1, matchday_col2 = st.columns([1, 2])
            
            with matchday_col1:
                st.markdown('Investigate stats over the course of a season. At what point in the season do teams score the most goals?')
                
                plot_x_per_matchday_selected = st.selectbox('Which attribute do you want to analyze?', filtered_columns, key='attribute_matchday')
                plot_x_per_matchday_type = st.selectbox('Which measure do you want to analyze?', ['Mean', 'Absolute', 'Median', 'Maximum', 'Minimum'], key='measure_matchday')
            
            with matchday_col2:
                if all_teams_selected != 'Select teams manually (choose below)' or (selected_teams and len(selected_teams) > 0):
                    try:
                        fig = plot_matchday_analysis(df_data_filtered, plot_x_per_matchday_selected, plot_x_per_matchday_type, full_selected_matchweeks)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error creating matchday chart: {str(e)}")
                else:
                    st.warning('Please select at least one team')
        
        # Correlation Analysis
        st.subheader('Correlation Analysis')
        
        corr_col1, corr_col2 = st.columns([1, 2])
        
        with corr_col1:
            st.markdown('Investigate the correlation between two attributes. Do teams that run more also score more goals?')
            
            corr_type = st.selectbox('Which plot type do you want to use?', ['Regression Plot (Recommended)', 'Standard Scatter Plot'])
            use_interactive_corr = st.checkbox('Use interactive correlation plot', value=True)
            y_axis_aspect = st.selectbox('Which attribute do you want on the y-axis?', filtered_columns, key='y_axis')
            x_axis_aspect = st.selectbox('Which attribute do you want on the x-axis?', filtered_columns, key='x_axis')
        
        with corr_col2:
            if all_teams_selected != 'Select teams manually (choose below)' or (selected_teams and len(selected_teams) > 0):
                try:
                    if use_interactive_corr:
                        fig = create_interactive_correlation_plot(df_data_filtered, x_axis_aspect, y_axis_aspect)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = plot_correlation_analysis(df_data_filtered, x_axis_aspect, y_axis_aspect, corr_type)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating correlation plot: {str(e)}")
            else:
                st.warning('Please select at least one team')
        
        # Data Export
        with st.expander("Export Data"):
            st.subheader("Export Current Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df_data_filtered.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="laliga_stats.csv",
                    mime="text/csv"
                )
            
            with col2:
                try:
                    import io
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_data_filtered.to_excel(writer, sheet_name='La Liga Data', index=False)
                    
                    buffer.seek(0)
                    st.download_button(
                        label="Download as Excel",
                        data=buffer,
                        file_name="laliga_stats.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                except Exception as e:
                    st.error(f"Excel export error: {str(e)}")
                    st.info("Install xlsxwriter package for Excel export: pip install xlsxwriter")
    else:
        st.warning("Please select a schema and table from the sidebar to begin analysis.")
    
    # Add navigation buttons to other pages
    st.subheader("Explore More Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Player Analysis"):
            st.session_state['current_page'] = 'players_analysis'
    
    with col2:
        if st.button("Team Analysis"):
            st.session_state['current_page'] = 'team_analysis'
    
    with col3:
        if st.button("Match Analysis"):
            st.session_state['current_page'] = 'match_analysis'
