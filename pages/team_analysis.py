import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data_processing import (
    get_unique_seasons_modified, 
    get_unique_teams,
    filter_season,
    filter_teams,
    group_measure_by_attribute
)
from visualizations import (
    create_interactive_team_chart,
    plot_team_analysis
)

def team_analysis_page(df_database, conn, schema_info=None):
    st.title('Team Analysis')
    st.markdown("Detailed team performance statistics and comparisons")
    
    # Sidebar filters
    st.sidebar.header("Team Filters")
    
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
    
    # Team selection
    teams = get_unique_teams(df_filtered_season)
    selected_teams = st.sidebar.multiselect(
        'Select teams to analyze',
        teams,
        default=teams[:5] if len(teams) > 5 else teams
    )
    
    # Filter by teams if selected
    if selected_teams:
        df_filtered = df_filtered_season[df_filtered_season['team'].isin(selected_teams)]
    else:
        df_filtered = df_filtered_season
        st.warning("No teams selected. Showing data for all teams.")
    
    # Create tabs for different team analyses
    tab1, tab2, tab3 = st.tabs(["Team Performance", "Team Comparison", "Season Trends"])
    
    with tab1:
        st.header("Team Performance Analysis")
        
        # Exclude non-metric columns
        exclude_columns = ['league', 'team', 'game', 'date', 'round', 'day', 'venue', 'result', 'opponent']
        all_columns = df_filtered.columns.tolist()
        filtered_columns = [col for col in all_columns if col not in exclude_columns]
        
        # Select performance metric
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select Metrics")
            
            selected_metric = st.selectbox(
                "Performance Metric", 
                filtered_columns,
                key="perf_metric"
            )
            
            measure_type = st.selectbox(
                "Aggregate By",
                ['Mean', 'Total', 'Maximum', 'Minimum'],
                key="perf_measure"
            )
            
            use_interactive = st.checkbox("Interactive Chart", value=True, key="perf_interactive")
            
            if len(selected_teams) > 1:
                sort_order = st.radio(
                    "Sort Teams By",
                    ["Performance (High to Low)", "Alphabetically"],
                    key="perf_sort"
                )
        
        with col2:
            try:
                if use_interactive:
                    # Convert measure type to expected value
                    measure_map = {'Total': 'Absolute', 'Mean': 'Mean', 'Maximum': 'Maximum', 'Minimum': 'Minimum'}
                    measure = measure_map.get(measure_type, 'Mean')
                    
                    # Create interactive chart
                    fig = create_interactive_team_chart(df_filtered, selected_metric, selected_teams, measure)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Use matplotlib chart
                    measure_map = {'Total': 'Absolute', 'Mean': 'Mean', 'Maximum': 'Maximum', 'Minimum': 'Minimum'}
                    measure = measure_map.get(measure_type, 'Mean')
                    
                    fig = plot_team_analysis(df_filtered, selected_metric, measure, True)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating performance chart: {str(e)}")
        
        # Team stats table
        with st.expander("View Detailed Team Statistics"):
            if not df_filtered.empty:
                try:
                    # Group data by team
                    numeric_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()
                    team_stats = df_filtered.groupby('team')[numeric_cols].agg(['mean', 'sum', 'max', 'min']).reset_index()
                    
                    # Display multi-level column stats
                    st.dataframe(team_stats)
                    
                    # Option to download stats
                    st.download_button(
                        "Download Team Stats",
                        data=team_stats.to_csv(index=False),
                        file_name="team_stats.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error creating team stats table: {str(e)}")
    
    with tab2:
        st.header("Team Comparison")
        
        # Select teams to compare
        if len(selected_teams) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                team1 = st.selectbox("First Team", selected_teams, key="compare_team1")
                
                # Team 1 color for charts
                team1_color = '#1f77b4'  # Default blue
            
            with col2:
                # Filter to exclude first team
                remaining_teams = [team for team in selected_teams if team != team1]
                team2 = st.selectbox("Second Team", remaining_teams, key="compare_team2")
                
                # Team 2 color for charts
                team2_color = '#ff7f0e'  # Default orange
            
            # Filter data for selected teams
            team1_data = df_filtered[df_filtered['team'] == team1]
            team2_data = df_filtered[df_filtered['team'] == team2]
            
            if not team1_data.empty and not team2_data.empty:
                # Select metrics to compare
                exclude_columns = ['league', 'team', 'game', 'date', 'round', 'day', 'venue', 'result', 'opponent']
                all_columns = df_filtered.columns.tolist()
                filtered_columns = [col for col in all_columns if col not in exclude_columns]
                
                selected_metrics = st.multiselect(
                    "Select metrics to compare",
                    filtered_columns,
                    default=filtered_columns[:3] if len(filtered_columns) >= 3 else filtered_columns
                )
                
                if selected_metrics:
                    try:
                        # Calculate team averages for selected metrics
                        team1_avg = team1_data[selected_metrics].mean()
                        team2_avg = team2_data[selected_metrics].mean()
                        
                        # Create comparison dataframe
                        comparison_data = pd.DataFrame({
                            'Metric': selected_metrics,
                            team1: team1_avg.values,
                            team2: team2_avg.values
                        })
                        
                        # Radar chart comparison
                        fig = px.line_polar(
                            comparison_data,
                            r=[val/max(team1_avg.max(), team2_avg.max()) for val in team1_avg.values] + 
                              [val/max(team1_avg.max(), team2_avg.max()) for val in team2_avg.values],
                            theta=selected_metrics + selected_metrics,
                            line_close=True,
                            color_discrete_sequence=[team1_color, team2_color],
                            labels={'r': 'Normalized Value', 'theta': 'Metric'},
                            range_r=[0, 1],
                            title=f"{team1} vs {team2} Comparison",
                        )
                        
                        # Update traces to add team names
                        fig.data[0].name = team1
                        fig.data[1].name = team2
                        
                        # Update layout
                        fig.update_layout(
                            template='plotly_dark',
                            plot_bgcolor='rgba(14, 17, 23, 0.8)',
                            paper_bgcolor='rgba(14, 17, 23, 0.8)',
                            font=dict(color='white'),
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display comparison table
                        st.subheader("Metric Comparison")
                        
                        # Calculate differences and percent differences
                        comparison_data['Difference'] = comparison_data[team1] - comparison_data[team2]
                        comparison_data['% Difference'] = (comparison_data['Difference'] / comparison_data[team2] * 100).round(2)
                        
                        # Show table
                        st.dataframe(comparison_data)
                    except Exception as e:
                        st.error(f"Error creating team comparison: {str(e)}")
                else:
                    st.warning("Please select at least one metric to compare")
            else:
                st.warning("No data available for one or both selected teams")
        else:
            st.warning("Please select at least two teams in the sidebar to enable comparison")
    
    with tab3:
        st.header("Season Trends")
        
        # Select team for trend analysis
        if selected_teams:
            trend_team = st.selectbox("Select Team", selected_teams, key="trend_team")
            
            # Filter data for selected team
            team_data = df_filtered[df_filtered['team'] == trend_team]
            
            if not team_data.empty:
                # Only proceed if we have season data
                if 'season' in team_data.columns:
                    try:
                        # Format season for readability
                        team_data['season_readable'] = team_data['season'].apply(
                            lambda x: f"20{str(x)[:2]}/20{str(x)[2:]}"
                        )
                        
                        # Select metrics for trend analysis
                        exclude_columns = ['league', 'team', 'game', 'date', 'round', 'day', 'venue', 'result', 'opponent', 'season', 'season_readable']
                        all_columns = team_data.columns.tolist()
                        metric_columns = [col for col in all_columns if col not in exclude_columns]
                        
                        selected_trend_metrics = st.multiselect(
                            "Select metrics to track over seasons",
                            metric_columns,
                            default=metric_columns[:3] if len(metric_columns) >= 3 else metric_columns
                        )
                        
                        if selected_trend_metrics:
                            # Group by season
                            agg_method = st.radio("Aggregation Method", ["Average", "Total"])
                            
                            if agg_method == "Average":
                                season_stats = team_data.groupby('season_readable')[selected_trend_metrics].mean().reset_index()
                            else:
                                season_stats = team_data.groupby('season_readable')[selected_trend_metrics].sum().reset_index()
                            
                            # Sort by season
                            season_stats = season_stats.sort_values('season_readable')
                            
                            # Convert to long format for plotting
                            df_long = pd.melt(
                                season_stats,
                                id_vars=['season_readable'],
                                value_vars=selected_trend_metrics,
                                var_name='Metric',
                                value_name='Value'
                            )
                            
                            # Create trend chart
                            fig = px.line(
                                df_long,
                                x='season_readable',
                                y='Value',
                                color='Metric',
                                title=f"{trend_team}'s Performance Trends Over Seasons ({agg_method})",
                                markers=True
                            )
                            
                            # Customize layout
                            fig.update_layout(
                                template='plotly_dark',
                                plot_bgcolor='rgba(14, 17, 23, 0.8)',
                                paper_bgcolor='rgba(14, 17, 23, 0.8)',
                                font=dict(color='white'),
                                xaxis_title="Season",
                                yaxis_title="Value",
                                legend_title="Metric"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Please select at least one metric to analyze")
                    except Exception as e:
                        st.error(f"Error creating trend analysis: {str(e)}")
                else:
                    st.warning("Season data not available for trend analysis")
            else:
                st.warning(f"No data available for {trend_team}")
        else:
            st.warning("Please select at least one team in the sidebar")
    
    # Add export functionality
    with st.expander("Export Data"):
        st.subheader("Export Current Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="team_analysis.csv",
                mime="text/csv"
            )
        
        with col2:
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_filtered.to_excel(writer, sheet_name='Team Analysis', index=False)
                
                buffer.seek(0)
                st.download_button(
                    label="Download as Excel",
                    data=buffer,
                    file_name="team_analysis.xlsx",
                    mime="application/vnd.ms-excel"
                )
            except Exception as e:
                st.error(f"Excel export error: {str(e)}")
                st.info("Install xlsxwriter package for Excel export: pip install xlsxwriter")
    
    # Navigation button
    if st.button("Back to Main Page"):
        st.session_state['current_page'] = 'main'
