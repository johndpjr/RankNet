# ensemble_preprocessing.py
import pandas as pd, json, numpy as np, os

# ---------- helpers ----------
def build_index(df):
    teams = sorted(set(df.team_abbreviation_home) | set(df.team_abbreviation_away))
    return {t: i for i, t in enumerate(teams)}

def extract(df, idx, clip=30):
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.pts_home) or pd.isna(r.pts_away):
            continue
        m = r.pts_home - r.pts_away                 # <-- NO bias added
        m = max(min(m, clip), -clip)
        rows.append({"home_idx": idx[r.team_abbreviation_home],
                     "away_idx": idx[r.team_abbreviation_away],
                     "margin":   m,
                     "home_team": r.team_abbreviation_home,
                     "away_team": r.team_abbreviation_away})
    return rows

# ---------- main ----------
def main():
    start = input("Enter start year for target season (e.g. 2022): ").strip()
    end   = str(int(start)+1)
    csv   = f"games_{start}_{end}.csv"
    if not os.path.exists(csv):
        print("Run filter.py first."); return

    df  = pd.read_csv(csv)
    idx = build_index(df)
    games = pd.DataFrame(extract(df, idx))

    # ---------- merge external scores ----------
    pr  = pd.read_csv("pagerank_2021.csv")
    od  = pd.read_csv("offdef_2021.csv")

    games = games.merge(pr.rename(columns={"Team":"home_team",
                                           "PageRank_Score":"home_pr"}),
                        on="home_team", how="left") \
                 .merge(pr.rename(columns={"Team":"away_team",
                                           "PageRank_Score":"away_pr"}),
                        on="away_team", how="left") \
                 .merge(od.rename(columns={"Team":"home_team",
                                           "Offense_Score":"home_off",
                                           "Defense_Score":"home_def"}),
                        on="home_team", how="left") \
                 .merge(od.rename(columns={"Team":"away_team",
                                           "Offense_Score":"away_off",
                                           "Defense_Score":"away_def"}),
                        on="away_team", how="left")

    # ---------- save ----------
    games.to_csv(f"games_{start}_{end}_flat_ensemble.csv", index=False)
    json.dump(idx, open("team_index.json", "w"))
    print(f"Ensemble preprocessing complete!  {len(games)} games written.")
    print("Label home‑wins:", (games.margin > 0).mean())

if __name__ == "__main__":
    main()
