from collections import defaultdict

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


df = pd.read_csv("./nba/game.csv")
df.sort_values(by=["game_date"], inplace=True)

cond = df["team_abbreviation_home"] == "HOU"
cols = ["game_date", "matchup_home", "wl_home"]

grouped = df.groupby("season_id")
HOME = "team_abbreviation_home"
AWAY = "team_abbreviation_away"
for id_, season in grouped:
    if id_ != 22022: continue
    print(f"Season {id_} contained {len(season)} games")
    teams = set(season[HOME]) | set(season[AWAY])
    DG = nx.DiGraph()
    DG.add_nodes_from(teams)
    mapping = defaultdict(int)
    for index, game in season.iterrows():
        home_team = game[HOME]
        away_team = game[AWAY]
        pair = (home_team, away_team) if home_team < away_team else (away_team, home_team)
        pts_diff = game["plus_minus_home"] if home_team == pair[0] else game["plus_minus_away"]
        mapping[pair] += pts_diff
    i = 0
    for pair, diff in mapping.items():
        team_a, team_b = pair
        i += 1
        if diff < 0:
            # Team A lost against Team B: directed edge from A to B (loser -> winner)
            DG.add_edge(team_a, team_b, weight=abs(diff))
        else:
            # Team B lost against Team A: directed edge from B to A (loser -> winner)
            DG.add_edge(team_b, team_a, weight=diff)
    # print(f"Processed {i} edges")
    # print(dict(mapping))
    options = {
        "node_color": "none",
        "edgecolors": "black",
    }
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(DG, k=2)
    nx.draw_networkx(DG, pos, **options)
    # labels = nx.get_edge_attributes(DG, "weight")
    # nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels)
    rank = nx.pagerank(DG)
    print("teams scores")
    for team, score in sorted(rank.items(), key=lambda item: item[1], reverse=True):
        print(f"{team}: {score}")
    plt.axis("off")
    plt.show()
    break
