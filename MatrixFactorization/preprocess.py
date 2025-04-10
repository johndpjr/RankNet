import pandas as pd
import numpy as np
import json

def build_team_index(df):
    teams = sorted(set(df['team_abbreviation_home']) | set(df['team_abbreviation_away']))
    team_to_index = {team: idx for idx, team in enumerate(teams)}
    return team_to_index

def build_margin_matrix(df, team_to_index, clip_margin=None):
    n = len(team_to_index)
    matrix = np.full((n, n), np.nan)
    
    matchup_results = {}

    for _, row in df.iterrows():
        home = row['team_abbreviation_home']
        away = row['team_abbreviation_away']
        pts_home = row['pts_home']
        pts_away = row['pts_away']

        if pd.isna(pts_home) or pd.isna(pts_away):
            continue

        i = team_to_index[home]
        j = team_to_index[away]
        diff = pts_home - pts_away

        if clip_margin:
            diff = max(min(diff, clip_margin), -clip_margin)

        if (i, j) not in matchup_results:
            matchup_results[(i, j)] = []
        matchup_results[(i, j)].append(diff)

    for (i, j), margins in matchup_results.items():
        matrix[i][j] = np.mean(margins)

    return matrix

def main():
    start_year = input("Enter the starting year of the season (e.g., 2018 for 2018–2019): ").strip()
    
    try:
        end_year = int(start_year) + 1
        filename = f"games_{start_year}_{end_year}.csv"
    except ValueError:
        print(" Invalid year entered.")
        return

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f" File {filename} not found. Make sure to run filter.py first.")
        return

    team_to_index = build_team_index(df)
    matrix = build_margin_matrix(df, team_to_index, clip_margin=30)

    np.save("adj_matrix.npy", matrix)
    with open("team_index.json", "w") as f:
        json.dump(team_to_index, f)

    print(" Preprocessing complete!")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Teams: {list(team_to_index.keys())}")

if __name__ == "__main__":
    main()
