import pandas as pd
import os

def main():
    # Go up one directory, then into 'nba'
    csv_path = os.path.join("..", "nba", "game.csv")
    
    df = pd.read_csv(csv_path)
    
    # Filter for the 2017-2018 season
    df_2018 = df[df['season_id'] == 22017]
    
    # Keep only relevant columns for margin of victory
    df_2018 = df_2018[[
        'team_abbreviation_home', 'team_abbreviation_away',
        'pts_home', 'pts_away', 'game_date'
    ]]
    
    # Save in current directory as games_2018.csv
    df_2018.to_csv("games_2018.csv", index=False)
    print(f" Saved 2018 season data to games_2018.csv with {len(df_2018)} games.")

if __name__ == "__main__":
    main()
