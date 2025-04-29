import pandas as pd
import os

def main():
    # Ask user for the starting year of the season
    start_year = input("Enter the starting year of the season (e.g., 2017 for 2017–2018): ").strip()

    try:
        season_id = 20000 + int(start_year)
    except ValueError:
        print("Invalid input. Please enter a valid year like 2017.")
        return

    # Construct path to CSV
    csv_path = os.path.join("..", "nba", "game.csv")
    df = pd.read_csv(csv_path)

    # Filter for the chosen season
    df_season = df[df['season_id'] == season_id]

    if df_season.empty:
        print(f"No data found for season ID {season_id}.")
        return

    # Keep only relevant columns for margin of victory
    df_season = df_season[[
        'team_abbreviation_home', 'team_abbreviation_away',
        'pts_home', 'pts_away', 'game_date'
    ]]

    # Save filtered data
    filename = f"games_{start_year}_{int(start_year)+1}.csv"
    df_season.to_csv(filename, index=False)
    print(f" Saved {len(df_season)} games to {filename}")

if __name__ == "__main__":
    main()
