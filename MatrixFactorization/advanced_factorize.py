import numpy as np
import pandas as pd
import json
import os

# Hyperparameters
LATENT_DIM = 5
LEARNING_RATE = 0.01
EPOCHS = 100
HOME_BIAS = 5

def load_data(start_year):
    with open("team_index.json", "r") as f:
        team_to_index = json.load(f)
    team_to_index = {k: int(v) for k, v in team_to_index.items()}

    raw_filename = f"games_{start_year}_{int(start_year)+1}.csv"
    try:
        raw_games = pd.read_csv(raw_filename)
        true_home_win_rate = (raw_games['pts_home'] > raw_games['pts_away']).mean()
        print(f"True NBA Home Win Rate (no bias): {true_home_win_rate:.4f}")
    except:
        print("Could not load raw CSV to compute true home win rate.")

    test_path = f"games_{start_year}_{int(start_year)+1}_flat.csv"
    train_path = "games_training_flat.csv"

    if not os.path.exists(test_path) or not os.path.exists(train_path):
        raise FileNotFoundError("Missing training or testing flat files. Did you run advanced_preprocessing.py?")

    test_games = pd.read_csv(test_path)
    training_games = pd.read_csv(train_path)
    return team_to_index, training_games, test_games

def train_latent_model(train_games, num_teams):
    team_vecs = np.random.normal(0, 0.1, size=(num_teams, LATENT_DIM))
    team_bias = np.zeros(num_teams)

    for epoch in range(EPOCHS):
        train_games = train_games.sample(frac=1).reset_index(drop=True)
        for _, row in train_games.iterrows():
            i, j, true_margin = int(row['home_idx']), int(row['away_idx']), row['margin']

            pred_margin = team_vecs[i] @ team_vecs[j] + team_bias[i] - team_bias[j] + HOME_BIAS
            error = pred_margin - true_margin

            grad_i = error * team_vecs[j]
            grad_j = error * team_vecs[i]

            team_vecs[i] -= LEARNING_RATE * grad_i
            team_vecs[j] -= LEARNING_RATE * grad_j

            team_bias[i] -= LEARNING_RATE * error
            team_bias[j] += LEARNING_RATE * error

    return team_vecs, team_bias

def evaluate(test_games, team_vecs, team_bias):
    correct = 0
    for _, row in test_games.iterrows():
        i, j, actual_margin = int(row['home_idx']), int(row['away_idx']), row['margin']
        predicted_margin = team_vecs[i] @ team_vecs[j] + team_bias[i] - team_bias[j] + HOME_BIAS
        if np.sign(predicted_margin) == np.sign(actual_margin):
            correct += 1

    accuracy = correct / len(test_games)
    return accuracy

def main():
    start_year = input("Enter the starting year of the season (e.g., 2018): ").strip()

    try:
        team_to_index, all_training_games, full_test_games = load_data(start_year)
    except Exception as e:
        print("Error loading data:", e)
        return

    num_teams = len(team_to_index)
    results = []

    for keep_ratio in np.linspace(0.1, 0.9, 9):
        shuffled = full_test_games.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(len(shuffled) * keep_ratio)
        seen_games = shuffled.iloc[:split]
        unseen_games = shuffled.iloc[split:]

        # Combine historical training data with observed games from the current season
        def apply_decay(df, weight):
            return pd.concat([df] * int(weight * 10), ignore_index=True)

        # Assuming you split your historical data per season beforehand
        df_2015 = all_training_games[all_training_games['season'] == 2015]
        df_2016 = all_training_games[all_training_games['season'] == 2016]
        df_2017 = all_training_games[all_training_games['season'] == 2017]

        combined_training = pd.concat([
            apply_decay(df_2015, 0.4),
            apply_decay(df_2016, 0.6),
            apply_decay(df_2017, 0.8),
            apply_decay(seen_games, 1.0)
        ], ignore_index=True)

        team_vecs, team_bias = train_latent_model(combined_training, num_teams)
        accuracy = evaluate(unseen_games, team_vecs, team_bias)

        print(f"Keep Ratio: {keep_ratio:.2f} | Accuracy: {accuracy:.4f}")
        results.append({
            "keep_ratio": round(keep_ratio, 2),
            "accuracy": round(accuracy, 4)
        })

    df = pd.DataFrame(results)
    filename = f"advanced_results_{start_year}_{int(start_year)+1}.csv"
    df.to_csv(filename, index=False)
    print(f"\nSaved results to {filename}")

if __name__ == "__main__":
    main()
