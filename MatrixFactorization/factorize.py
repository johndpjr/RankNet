import numpy as np
from sklearn.decomposition import TruncatedSVD
import json
import pandas as pd
import os

def load_data():
    matrix = np.load("adj_matrix.npy")
    with open("team_index.json", "r") as f:
        team_to_index = json.load(f)
    return matrix, team_to_index

def mask_known_entries(matrix, keep_ratio=0.75, seed=42):
    np.random.seed(seed)
    known_mask = ~np.isnan(matrix)
    indices = np.array(np.where(known_mask)).T
    np.random.shuffle(indices)

    total = len(indices)
    keep_count = int(total * keep_ratio)
    test_indices = indices[keep_count:]

    train_matrix = matrix.copy()
    for i, j in test_indices:
        train_matrix[i, j] = np.nan

    test_mask = np.full(matrix.shape, False)
    for i, j in test_indices:
        test_mask[i, j] = True

    return train_matrix, test_mask

def apply_svd(matrix, rank=10):
    known_mask = ~np.isnan(matrix)
    filled_matrix = matrix.copy()
    mean_val = np.nanmean(matrix)
    filled_matrix[~known_mask] = mean_val

    svd = TruncatedSVD(n_components=rank)
    U = svd.fit_transform(filled_matrix)
    V = svd.components_
    reconstructed = np.dot(U, V)
    return reconstructed

def evaluate_on_test(reconstructed, actual, test_mask):
    pred_sign = np.sign(reconstructed)
    true_sign = np.sign(actual)

    mismatches = (pred_sign != true_sign) & test_mask
    total_errors = np.sum(mismatches)
    total_test = np.sum(test_mask)
    accuracy = (total_test - total_errors) / total_test
    return total_errors, total_test, accuracy

def main():
    # Detect season name from the CSV file used in preprocess step
    season_name = "2017_2018"  # You can later automate this with CLI args or filenames

    matrix, _ = load_data()

    results = []

    for keep_ratio in np.linspace(0.1, 0.9, 9):
        train_matrix, test_mask = mask_known_entries(matrix, keep_ratio=keep_ratio)
        reconstructed = apply_svd(train_matrix, rank=30)
        total_errors, total_test, accuracy = evaluate_on_test(reconstructed, matrix, test_mask)

        if total_test == 0:
            print(f"Keep Ratio: {keep_ratio:.2f} | Skipped (no test data)")
            continue

        print(f"Keep Ratio: {keep_ratio:.2f} | Accuracy: {accuracy:.4f}")
        results.append({
            "keep_ratio": round(keep_ratio, 2),
            "test_size": int(total_test),
            "prediction_errors": int(total_errors),
            "accuracy": round(accuracy, 4)
    })

    # Save results to CSV
    df = pd.DataFrame(results)
    filename = f"results_{season_name}.csv"
    df.to_csv(filename, index=False)
    print(f"\n Saved results to {filename}")

if __name__ == "__main__":
    main()
