# La Liga Analyzer

This repository conatains a Python Streamlit application for analyzing LaLiga data, as well as Jupyter Notebooks for scraping and cleaning the data.

## Deployed App
Click here to get to the deployed []()

## Current Data Scope
* 🏆 Season 2013/2014 to Season 2023/2024
* 🏟️ 2,142 Matches
* 🏃‍♂️ 25 Teams
* 🥅 6,363 Goals
* 👟⚽ 56,036 Shots

## Repository Structure
| Folder/Code | Content |
| ------------- | ------------- |
| .streamlit | Contains the confiq.toml to set certain design parameters |
| data | Contains the scraped (and cleaned) Bundesliga data in CSV format |
| mplsoccer.py | Contains the actual Streamlit application |
| data_preoprocessing.py | Jupyter Notebook used for data cleaning |
| data_scraping.py | Jupyter Notebook used for data scraping (URLs not included for legal reasons) |
| requirements.txt | Contains all requirements (necessary for Streamlit sharing) |