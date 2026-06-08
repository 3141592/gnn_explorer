import os
import binascii
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

sec_teams = {
    "Alabama", "Arkansas", "Auburn", "Florida", "Georgia", "Kentucky",
    "LSU", "Mississippi St.", "Missouri", "Ole Miss", "Oklahoma",
    "South Carolina", "Tennessee", "Texas", "Texas A&M", "Vanderbilt"
}

games = []
line_number = 0
with open('scores.txt', 'r') as f:
    for line in f:
        line_number += 1
        try:
            if line.startswith("Week") or line.startswith("Home"):
                continue
            if line == "\n":
                continue

            split = line.split("\t")
            game = [(split[0], split[1], split[2])]
            games.append(game)

        except Exception as error:
            print(error)

# Clean up games list
games_final = []
for game in games:
    try:
        team1 = game[0][0].split("(")[0].strip()
        team2 = game[0][2].split("(")[0].strip()
        score1 = game[0][1].split("-")[0].strip()
        score2 = game[0][1].split("-")[1].strip()
        team1 = team1.split("[")[0].strip()
        team2 = team2.split("[")[0].strip()
        score1 = score1.split("(")[0].strip()
        score2 = score2.split("(")[0].strip()
        cleaned = (team1, int(score1), team2, int(score2))
        games_final.append(cleaned)
    except Exception as error:
        print(error)
        breakpoint()

# Create graph
dg = nx.MultiDiGraph()
for game in games_final:
    try:
        team1 = game[0]
        team1_score = int(game[1])
        team2 = game[2]
        team2_score = int(game[3])

        if team1_score > team2_score:
            winner = team1
            loser = team2
            margin = team1_score - team2_score
        else:
            winner = team2
            loser = team1
            margin = team2_score - team1_score

        dg.add_edge(
            loser,
            winner,
            weight=margin,
            winner_score=max(team1_score, team2_score),
            loser_score=min(team1_score, team2_score),
        )
    except Exception as error:
        print(error)
        breakpoint()

# print the graph
pos = nx.spring_layout(dg, seed=42, k=1.2)

plt.figure(figsize=(16, 12))

nx.draw_networkx_edges(dg, pos, arrows=True, alpha=0.35, width=0.8)

nx.draw_networkx_nodes(
    dg, pos,
    nodelist=[n for n in dg.nodes if n in sec_teams],
    node_size=1400
)

nx.draw_networkx_nodes(
    dg, pos,
    nodelist=[n for n in dg.nodes if n not in sec_teams],
    node_size=450,
    alpha=0.45
)

nx.draw_networkx_labels(
    dg, pos,
    labels={n: n for n in sec_teams if n in dg.nodes},
    font_size=10
)

plt.axis("off")
plt.savefig("sec_graph.png", bbox_inches="tight", dpi=200)
plt.close()

pr = nx.pagerank(dg, weight='weight')

for team, score in sorted(pr.items(), key=lambda x: x[1], reverse=True):
    print(f"{score:.4f} {team}")

print()
in_degree = dg.in_degree(weight="weight")
out_degree = dg.out_degree(weight="weight")
#print(f"In degree: {in_degree}")
#print(f"Out degree: {out_degree}")

in_sorted = sorted(
    dg.in_degree(weight="weight"),
    key=lambda x: x[1],
    reverse=True
)[:10]

pr_sorted = sorted(
    nx.pagerank(dg, weight="weight").items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print(in_sorted)
print()
print(pr_sorted)
