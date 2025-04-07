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
    
    # Dictionary to store all results for each (home, away) matchup
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

        # Append margin to list of results for that matchup
        if (i, j) not in matchup_results:
            matchup_results[(i, j)] = []
        matchup_results[(i, j)].append(diff)

    # Now compute averages
    for (i, j), margins in matchup_results.items():
        matrix[i][j] = np.mean(margins)

    return matrix


def main():
    # Load the filtered 2018 season data
    df = pd.read_csv("games_2018.csv")

    # Build team-to-index mapping
    team_to_index = build_team_index(df)

    # Build matrix using margin of victory
    matrix = build_margin_matrix(df, team_to_index, clip_margin=30)  # clip huge blowouts

    # Save matrix and index
    np.save("adj_matrix.npy", matrix)
    with open("team_index.json", "w") as f:
        json.dump(team_to_index, f)

    print("Preprocessing complete!")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Teams: {list(team_to_index.keys())}")

if __name__ == "__main__":
    main()
