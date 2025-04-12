import pandas as pd
import numpy as np
import json

def build_team_index(df):
    teams = sorted(set(df['team_abbreviation_home']) | set(df['team_abbreviation_away']))
    team_to_index = {team: idx for idx, team in enumerate(teams)}
    return team_to_index

def extract_games_and_matrix(df, team_to_index, clip_margin=None, home_bias=5):
    games = []
    matrix = np.full((len(team_to_index), len(team_to_index)), np.nan)
    matchups = {}

    for _, row in df.iterrows():
        home = row['team_abbreviation_home']
        away = row['team_abbreviation_away']
        pts_home = row['pts_home']
        pts_away = row['pts_away']

        if pd.isna(pts_home) or pd.isna(pts_away):
            continue

        i = team_to_index[home]
        j = team_to_index[away]

        margin = pts_home - pts_away + home_bias
        if clip_margin is not None:
            margin = max(min(margin, clip_margin), -clip_margin)

        games.append({
            'home_idx': i,
            'away_idx': j,
            'margin': margin
        })

        if (i, j) not in matchups:
            matchups[(i, j)] = []
        matchups[(i, j)].append(margin)

    for (i, j), margins in matchups.items():
        matrix[i][j] = np.mean(margins)

    return matrix, team_to_index, games

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
    matrix, team_to_index, games = extract_games_and_matrix(df, team_to_index, clip_margin=30)

    np.save("adj_matrix.npy", matrix)
    with open("team_index.json", "w") as f:
        json.dump(team_to_index, f)
    pd.DataFrame(games).to_csv("games_flat.csv", index=False)

    print(" Preprocessing complete!")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Saved {len(games)} individual games for evaluation.")
    print(f"Teams: {list(team_to_index.keys())}")

if __name__ == "__main__":
    main()
