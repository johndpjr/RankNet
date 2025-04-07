import numpy as np
from sklearn.decomposition import TruncatedSVD
import json

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
    keep_indices = indices[:keep_count]
    test_indices = indices[keep_count:]

    # Create a new matrix for training
    train_matrix = matrix.copy()
    for i, j in test_indices:
        train_matrix[i, j] = np.nan  # hide test values

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

def evaluate_on_test(reconstructed, actual, test_mask, threshold=0.5):
    predictions = (reconstructed >= threshold).astype(int)
    actual_binary = np.where(actual >= 0.5, 1, 0)

    mismatches = (predictions != actual_binary) & test_mask
    total_errors = np.sum(mismatches)
    total_test = np.sum(test_mask)

    accuracy = (total_test - total_errors) / total_test
    return total_errors, total_test, accuracy

def main():
    matrix, _ = load_data()

    # Try this at different values: 0.25, 0.5, 0.75
    train_matrix, test_mask = mask_known_entries(matrix, keep_ratio=0.9)

    reconstructed = apply_svd(train_matrix, rank=10)

    total_errors, total_test, accuracy = evaluate_on_test(
        reconstructed, matrix, test_mask
    )

    print("🎯 Evaluation on Held-Out Data:")
    print(f"Tested matchups: {total_test}")
    print(f"Prediction errors: {total_errors}")
    print(f"Accuracy on unseen games: {accuracy:.4f}")

if __name__ == "__main__":
    main()
