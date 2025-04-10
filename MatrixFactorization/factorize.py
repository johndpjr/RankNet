import numpy as np
from sklearn.decomposition import TruncatedSVD
import json
import pandas as pd

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

def apply_svd(matrix, rank=30):
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
    start_year = input("Enter the starting year of the season (e.g., 2018 for 2018–2019): ").strip()

    try:
        end_year = int(start_year) + 1
        season_name = f"{start_year}_{end_year}"
    except ValueError:
        print(" Invalid year entered.")
        return

    matrix, _ = load_data()

    results = []
    ranks_to_test = [1, 10, 20, 30]

    for rank in ranks_to_test:
        print(f"\n Testing SVD rank: {rank}")
        for keep_ratio in np.linspace(0.1, 0.9, 9):
            train_matrix, test_mask = mask_known_entries(matrix, keep_ratio=keep_ratio)
            reconstructed = apply_svd(train_matrix, rank=rank)
            total_errors, total_test, accuracy = evaluate_on_test(reconstructed, matrix, test_mask)

            if total_test == 0:
                print(f"  Keep Ratio: {keep_ratio:.2f} | Skipped (no test data)")
                continue

            print(f"  Keep Ratio: {keep_ratio:.2f} | Accuracy: {accuracy:.4f}")
            results.append({
                "rank": rank,
                "keep_ratio": round(keep_ratio, 2),
                "test_size": int(total_test),
                "prediction_errors": int(total_errors),
                "accuracy": round(accuracy, 4)
            })

    df = pd.DataFrame(results)
    filename = f"results_{season_name}_multi_rank.csv"
    df.to_csv(filename, index=False)
    print(f"\n Saved results to {filename}")


if __name__ == "__main__":
    main()
