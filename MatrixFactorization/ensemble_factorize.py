# ensemble_factorize.py
import numpy as np, pandas as pd, json, os

# ---------------- parameters ----------------
LATENT_DIM, LR, EPOCHS = 5, 0.01, 100
CLIP                   = 5.0        # gradient / err clip
ALPHA, BETA, GAMMA     = 1.0, 1.0, 1.0   # weights for PR / off / def
HOME_BIAS              = 10         # keep unbiased
# -------------------------------------------

def load_data(year):
    flat = f"games_{year}_{int(year)+1}_flat_ensemble.csv"
    if not os.path.exists(flat): raise FileNotFoundError("Run ensemble_preprocessing.py first.")
    idx  = {k:int(v) for k,v in json.load(open("team_index.json")).items()}
    games= pd.read_csv(flat)

    raw  = pd.read_csv(f"games_{year}_{int(year)+1}.csv")
    print("True NBA home‑win rate :", (raw.pts_home>raw.pts_away).mean())
    print("Label home‑wins        :", (games.margin>0).mean())
    return idx, games

# ---------- prediction helper ----------
def aux_term(r):
    # treat missing values as 0
    pr  = ALPHA * ( (r.home_pr  - r.away_pr)  if not np.isnan(r.home_pr)  else 0 )
    off = BETA  * ( (r.home_off - r.away_off) if not np.isnan(r.home_off) else 0 )
    dfs = GAMMA * ( (r.away_def - r.home_def) if not np.isnan(r.home_def) else 0 )
    return pr + off + dfs

# ---------- training ----------
def train(df, n):
    U = np.random.normal(0, 0.1, (n, LATENT_DIM))
    b = np.zeros(n)

    for _ in range(EPOCHS):
        for r in df.sample(frac=1).itertuples():
            i,j,y = int(r.home_idx), int(r.away_idx), r.margin
            y_hat = U[i]@U[j] + b[i]-b[j] + aux_term(r)  # add signals in training too
            err   = np.clip(y_hat - y, -CLIP, CLIP)

            grad_i = err * U[j]
            grad_j = err * U[i]
            U[i] -= LR * np.clip(grad_i, -CLIP, CLIP)
            U[j] -= LR * np.clip(grad_j, -CLIP, CLIP)
            b[i] -= LR * err
            b[j] += LR * err
    return U, b

# ---------- evaluation ----------
def accuracy(df,U,b):
    preds = []
    for r in df.itertuples():
        i,j=int(r.home_idx),int(r.away_idx)
        y_hat = U[i]@U[j] + b[i]-b[j] + aux_term(r)
        preds.append(np.sign(y_hat))
    return (np.array(preds) == np.sign(df.margin.values)).mean()

# ---------- driver ----------
def main():
    year = input("Target season start year (e.g. 2022): ").strip()
    idx, games = load_data(year)
    n = len(idx)

    results=[]
    for keep in np.linspace(0.1,0.9,9):
        shuf  = games.sample(frac=1)
        split = int(len(shuf)*keep)
        train_df, test_df = shuf[:split], shuf[split:]
        U,b   = train(train_df, n)
        acc   = accuracy(test_df, U, b)
        print(f"Keep {keep:0.2f} | Acc {acc:.4f}")
        results.append({"keep_ratio":round(keep,2),"accuracy":round(acc,4)})

    pd.DataFrame(results).to_csv(
        f"ensemble_results_{year}_{int(year)+1}.csv", index=False)
    print("Saved ensemble results.")

if __name__ == "__main__":
    main()
