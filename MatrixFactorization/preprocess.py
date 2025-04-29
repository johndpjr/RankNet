import pandas as pd
import numpy as np
import json

def build_team_index(df):
    teams = sorted(set(df['team_abbreviation_home']) | set(df['team_abbreviation_away']))
    team_to_index = {team: idx for idx, team in enumerate(teams)}
    return team_to_index

def extract_games_individual(df, team_to_index, clip_margin=None, home_bias=0):
    games = []

    for _, row in df.iterrows():
        home = row['team_abbreviation_home']
        away = row['team_abbreviation_away']
        pts_home = row['pts_home']
        pts_away = row['pts_away']

        if pd.isna(pts_home) or pd.isna(pts_away):
            continue

        home_idx = team_to_index[home]
        away_idx = team_to_index[away]

        margin = pts_home - pts_away + home_bias
        if clip_margin is not None:
            margin = max(min(margin, clip_margin), -clip_margin)

        games.append({
            'home_idx': home_idx,
            'away_idx': away_idx,
            'margin': margin
        })

    return games, team_to_index

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
    games, team_to_index = extract_games_individual(df, team_to_index, clip_margin=30)

    with open("team_index.json", "w") as f:
        json.dump(team_to_index, f)

    flat_filename = f"games_{start_year}_{end_year}_flat.csv"
    home_win_rate = sum(g['margin'] > 0 for g in games) / len(games)
    print(f"[preprocess] home‑wins in margins: {home_win_rate:.4f}")
# sanity‑check – should be ≈ 0.5927
    pd.DataFrame(games).to_csv(flat_filename, index=False)

    print(" Preprocessing complete!")
    print(f"Saved {len(games)} individual games for training and evaluation.")
    print(f"Saved flat game data to {flat_filename}")
    print(f"Teams: {list(team_to_index.keys())}")

if __name__ == "__main__":
    main()
