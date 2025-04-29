import pandas as pd, json, os
from pathlib import Path

def build_team_index(dfs):
    teams = set()
    for df in dfs:
        teams |= set(df.team_abbreviation_home) | set(df.team_abbreviation_away)
    return {t: i for i, t in enumerate(sorted(teams))}


def extract_games(df, idx, clip=30):
    out = []
    for _, r in df.iterrows():
        if pd.isna(r.pts_home) or pd.isna(r.pts_away):
            continue
        m = r.pts_home - r.pts_away          # **NO home‑bias added**
        m = max(min(m, clip), -clip)
        out.append({"home_idx": idx[r.team_abbreviation_home],
                    "away_idx": idx[r.team_abbreviation_away],
                    "margin":   m})
    return out


def main():
    target_year = int(input("Target season start year (e.g. 2019): ").strip())
    files = [(f"games_{y}_{y+1}.csv", y) for y in (target_year-3,
                                                   target_year-2,
                                                   target_year-1,
                                                   target_year)]
    missing = [f for f, _ in files if not os.path.exists(f)]
    if missing:
        print("Missing files:", *missing, "\nRun filter.py for each first."); return

    dfs = [pd.read_csv(f) for f, _ in files]
    idx = build_team_index(dfs)

    training, test = [], []
    for (file, season), df in zip(files, dfs):
        part = extract_games(df, idx)
        for g in part:
            g["season"] = season
        (training if season < target_year else test).extend(part)

    Path(".").mkdir(exist_ok=True)    # current dir
    json.dump(idx, open("team_index.json", "w"))
    pd.DataFrame(training).to_csv("games_training_flat.csv", index=False)
    pd.DataFrame(test).to_csv(f"games_{target_year}_{target_year+1}_flat.csv",
                              index=False)

    print("Advanced preprocessing complete!")
    print("  training home‑wins:", (pd.DataFrame(training).margin > 0).mean())
    print("  test     home‑wins:", (pd.DataFrame(test).margin > 0).mean())
    print("  team_index.json written with", len(idx), "teams.")


if __name__ == "__main__":
    main()
