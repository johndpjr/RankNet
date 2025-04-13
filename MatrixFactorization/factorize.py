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

    end_year = int(start_year) + 1
    games_path = f"games_{start_year}_{end_year}_flat.csv"
    if not os.path.exists(games_path):
        raise FileNotFoundError(f"Could not find game data at {games_path}")

    games = pd.read_csv(games_path)
    return team_to_index, games


def train_latent_model(train_games, num_teams):
    # Initialize latent vectors and biases
    team_vecs = np.random.normal(0, 0.1, size=(num_teams, LATENT_DIM))
    team_bias = np.zeros(num_teams)

    for epoch in range(EPOCHS):
        train_games = train_games.sample(frac=1).reset_index(drop=True)  # <- FIXED SHUFFLE
        for _, row in train_games.iterrows():
            i, j, true_margin = int(row['home_idx']), int(row['away_idx']), row['margin']

            pred_margin = team_vecs[i] @ team_vecs[j] + team_bias[i] - team_bias[j] + HOME_BIAS
            error = pred_margin - true_margin

            # Gradient updates
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
        team_to_index, games = load_data(start_year)
    except Exception as e:
        print("Error loading data:", e)
        return

    num_teams = len(team_to_index)
    results = []

    for keep_ratio in np.linspace(0.1, 0.9, 9):
        shuffled = games.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(len(shuffled) * keep_ratio)
        train_games = shuffled.iloc[:split]
        test_games = shuffled.iloc[split:]

        team_vecs, team_bias = train_latent_model(train_games, num_teams)
        accuracy = evaluate(test_games, team_vecs, team_bias)

        print(f"Keep Ratio: {keep_ratio:.2f} | Accuracy: {accuracy:.4f}")
        results.append({
            "keep_ratio": round(keep_ratio, 2),
            "accuracy": round(accuracy, 4)
        })

    # Save results
    df = pd.DataFrame(results)
    filename = f"results_{start_year}_{int(start_year)+1}.csv"
    df.to_csv(filename, index=False)
    print(f"\nSaved results to {filename}")

    # Averaged one-game test
    print("\n--- Averaged One-Game Training Test (20 trials) ---")
    accuracies = []
    trials = 20
    for i in range(trials):
        train_games = games.sample(n=1)
        test_games = games.drop(train_games.index)

        team_vecs, team_bias = train_latent_model(train_games.copy(), num_teams)
        acc = evaluate(test_games, team_vecs, team_bias)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"One-game (avg of {trials}) | Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    baseline = (games['margin'] > 0).mean()
    print(f"Naive baseline (predict home team wins): {baseline:.4f}")


if __name__ == "__main__":
    main()