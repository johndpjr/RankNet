import numpy as np
from sklearn.decomposition import TruncatedSVD
import json
import pandas as pd
import os

def load_data(start_year):
    matrix = np.load("adj_matrix.npy")
    with open("team_index.json", "r") as f:
        team_to_index = json.load(f)
    team_to_index = {k: int(v) for k, v in team_to_index.items()}
    end_year = int(start_year) + 1
    games_path = f"games_{start_year}_{end_year}_flat.csv"
    if not os.path.exists(games_path):
        games_path = "games_flat.csv"
    games = pd.read_csv(games_path)
    return matrix, team_to_index, games

def apply_svd(matrix, rank=1):
    known_mask = ~np.isnan(matrix)
    filled_matrix = matrix.copy()
    mean_val = np.nanmean(matrix)
    filled_matrix[~known_mask] = mean_val

    svd = TruncatedSVD(n_components=rank)
    U = svd.fit_transform(filled_matrix)
    V = svd.components_
    reconstructed = np.dot(U, V)
    return reconstructed

def evaluate_by_game(games, reconstructed, keep_ratio, seed=42):
    np.random.seed(seed)
    games_shuffled = games.sample(frac=1, random_state=seed).reset_index(drop=True)

    total_games = len(games_shuffled)
    train_count = int(total_games * keep_ratio)
    test_games = games_shuffled.iloc[train_count:]

    correct = 0
    for _, row in test_games.iterrows():
        i, j = int(row['home_idx']), int(row['away_idx'])
        predicted_margin = reconstructed[i][j]
        actual_margin = row['margin']
        if np.sign(predicted_margin) == np.sign(actual_margin):
            correct += 1

    total_test = len(test_games)
    return total_test, total_test - correct, correct / total_test

def main():
    start_year = input("Enter the starting year of the season (e.g., 2018 for 2018–2019): ").strip()

    try:
        end_year = int(start_year) + 1
        season_name = f"{start_year}_{end_year}"
    except ValueError:
        print(" Invalid year entered.")
        return

    matrix, team_to_index, games = load_data(start_year)
    reconstructed = apply_svd(matrix, rank=1)

    results = []

    for keep_ratio in np.linspace(0.1, 0.9, 9):
        test_size, prediction_errors, accuracy = evaluate_by_game(games, reconstructed, keep_ratio)
        print(f"Keep Ratio: {keep_ratio:.2f} | Accuracy: {accuracy:.4f}")
        results.append({
            "keep_ratio": round(keep_ratio, 2),
            "test_size": int(test_size),
            "prediction_errors": int(prediction_errors),
            "accuracy": round(accuracy, 4)
        })

    df = pd.DataFrame(results)
    filename = f"results_{season_name}.csv"
    df.to_csv(filename, index=False)
    print(f"\n Saved results to {filename}")

if __name__ == "__main__":
    main()
