import pandas as pd
import numpy as np
import json
import os

def build_team_index(all_dfs):
    teams = set()
    for df in all_dfs:
        teams |= set(df['team_abbreviation_home']) | set(df['team_abbreviation_away'])
    teams = sorted(teams)
    return {team: idx for idx, team in enumerate(teams)}

def extract_games(df, team_to_index, clip_margin=None, home_bias=5):
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

    return games

def main():
    target_year = input("Enter the starting year of the target season (e.g., 2018): ").strip()
    try:
        target_year = int(target_year)
    except ValueError:
        print("Invalid year input.")
        return

    current_filename = f"games_{target_year}_{target_year+1}.csv"
    prev_files = [
        (f"games_{target_year-3}_{target_year-2}.csv", target_year - 3),
        (f"games_{target_year-2}_{target_year-1}.csv", target_year - 2),
        (f"games_{target_year-1}_{target_year}.csv", target_year - 1)
    ]

    dfs = []
    for path, _ in prev_files:
        if not os.path.exists(path):
            print(f"Missing file: {path}. Run filter.py first.")
            return
        dfs.append(pd.read_csv(path))

    if not os.path.exists(current_filename):
        print(f"Missing file: {current_filename}")
        return
    current_df = pd.read_csv(current_filename)

    print("Loaded all required season files.")

    team_to_index = build_team_index(dfs + [current_df])

    training_games = []
    for (file, year) in prev_files:
        season_df = pd.read_csv(file)
        tagged_games = extract_games(season_df, team_to_index, clip_margin=30)
        for game in tagged_games:
            game['season'] = year
        training_games.extend(tagged_games)


    test_games = extract_games(current_df, team_to_index, clip_margin=30)

    # Save outputs
    with open("team_index.json", "w") as f:
        json.dump(team_to_index, f)

    pd.DataFrame(training_games).to_csv("games_training_flat.csv", index=False)
    pd.DataFrame(test_games).to_csv(f"games_{target_year}_{target_year+1}_flat.csv", index=False)

    print("Advanced preprocessing complete!")
    print(f"Training games saved to games_training_flat.csv")
    print(f"Test games saved to games_{target_year}_{target_year+1}_flat.csv")
    print(f"Team mapping saved to team_index.json")
    print(f"Teams: {list(team_to_index.keys())}")

if __name__ == "__main__":
    main()