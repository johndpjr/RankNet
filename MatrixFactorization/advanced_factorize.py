import numpy as np, pandas as pd, json, os
from pathlib import Path

# ---------- hyper‑parameters ----------
LATENT_DIM = 5
LR          = 0.01
EPOCHS      = 100
CLIP        = 5.0          # gradient / error clip
# --------------------------------------

def load_data(start_year: int):
    test_path  = Path(f"games_{start_year}_{start_year+1}_flat.csv")
    train_path = Path("games_training_flat.csv")
    if not test_path.exists() or not train_path.exists():
        raise FileNotFoundError("Run advanced_preprocessing.py first.")

    idx   = {k: int(v) for k, v in json.load(open("team_index.json")).items()}
    test  = pd.read_csv(test_path)
    train = pd.read_csv(train_path)

    raw   = pd.read_csv(f"games_{start_year}_{start_year+1}.csv")
    print("True NBA home‑win rate :", (raw.pts_home > raw.pts_away).mean())
    print("Label test home‑wins   :", (test.margin  > 0).mean())
    return idx, train, test


def train_latent_model(df: pd.DataFrame, n_teams: int):
    U = np.random.normal(0, 0.1, (n_teams, LATENT_DIM))
    b = np.zeros(n_teams)

    for _ in range(EPOCHS):
        for r in df.sample(frac=1).itertuples():
            i, j, y = int(r.home_idx), int(r.away_idx), r.margin

            dot  = np.dot(U[i], U[j])
            err  = dot + b[i] - b[j] - y
            err_c = np.clip(err, -CLIP, CLIP)

            grad_Ui = err_c * U[j]
            grad_Uj = err_c * U[i]

            U[i] -= LR * np.clip(grad_Ui, -CLIP, CLIP)
            U[j] -= LR * np.clip(grad_Uj, -CLIP, CLIP)
            b[i] -= LR * err_c
            b[j] += LR * err_c

    return U, b


def evaluate(df, U, b):
    i = df.home_idx.values
    j = df.away_idx.values
    preds = np.sign((U[i] * U[j]).sum(1) + b[i] - b[j])
    return (preds == np.sign(df.margin.values)).mean()


def main():
    start_year = int(input("Target season start year (e.g. 2019): ").strip())
    idx, train_all, test_full = load_data(start_year)
    n = len(idx)

    # previous three seasons in ascending order
    seasons = sorted(train_all.season.unique())
    decay_w = {seasons[0]: 0.4, seasons[1]: 0.6, seasons[2]: 0.8}

    def apply_decay(df, w):
        k = max(1, int(w * 10))            # at least one copy
        return pd.concat([df] * k, ignore_index=True)

    results = []
    for keep in np.linspace(0.1, 0.9, 9):
        shuffled = test_full.sample(frac=1)      # new shuffle each loop
        split    = int(len(shuffled) * keep)
        seen, unseen = shuffled[:split], shuffled[split:]

        combined = pd.concat(
            [apply_decay(train_all[train_all.season == s], w)
             for s, w in decay_w.items()] + [seen],
            ignore_index=True
        )

        U, b = train_latent_model(combined, n)
        acc  = evaluate(unseen, U, b)
        results.append({"keep_ratio": round(keep, 2),
                        "accuracy":   round(acc, 4)})
        print(f"Keep {keep:0.2f} | Accuracy {acc:.4f}")

    pd.DataFrame(results).to_csv(
        f"advanced_results_{start_year}_{start_year+1}.csv", index=False
    )
    print("Saved results file.")


if __name__ == "__main__":
    main()
