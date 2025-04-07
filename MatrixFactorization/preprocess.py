import pandas as pd
import numpy as np
import json

def build_team_index(df):
    teams = sorted(set(df['team_abbreviation_home']) | set(df['team_abbreviation_away']))
    team_to_index = {team: idx for idx, team in enumerate(teams)}
    return team_to_index

def build_adjacency_matrix(df, team_to_index):
    n = len(team_to_index)
    matrix = np.full((n, n), np.nan)  # start with NaNs (unknown results)

    for _, row in df.iterrows():
        home = row['team_abbreviation_home']
        away = row['team_abbreviation_away']
        result = row['wl_home']  # 'W' or 'L'

        i = team_to_index[home]
        j = team_to_index[away]

        if pd.isna(result):
            continue

        matrix[i][j] = 1 if result == 'W' else 0  # 1 = home win, 0 = away win

    return matrix

def main():
    # Load the filtered 2018 season data
    df = pd.read_csv("games_2018.csv")

    # Build team-to-index mapping
    team_to_index = build_team_index(df)

    # Create the adjacency matrix
    matrix = build_adjacency_matrix(df, team_to_index)

    # Save matrix and index
    np.save("adj_matrix.npy", matrix)
    with open("team_index.json", "w") as f:
        json.dump(team_to_index, f)

    print("✅ Preprocessing complete!")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Teams: {list(team_to_index.keys())}")

if __name__ == "__main__":
    main()
