import os
import networkx as nx
import json
from pathlib import Path

sec_teams = {
    "Alabama", "Arkansas", "Auburn", "Florida", "Georgia", "Kentucky",
    "LSU", "Mississippi St.", "Missouri", "Ole Miss", "Oklahoma",
    "South Carolina", "Tennessee", "Texas", "Texas A&M", "Vanderbilt"
}

try:
    dg = nx.read_graphml("sec_2025.graphml", force_multigraph=True)
except Exception as error:
    print(error)
    breakpoint()

# Verify load
#print(type(dg))
#print(dg.number_of_nodes())
#print(dg.number_of_edges())
#for u, v, data in dg.edges(data=True):
#    print(u, "->", v, data)
#    break

# Analysis
print("Pagerank scores:")
pr = nx.pagerank(dg, weight='weight')
for team, score in sorted(pr.items(), key=lambda x: x[1], reverse=True):
    print(f"{score:.4f} {team}")

print()
in_degree = dg.in_degree(weight="weight")
out_degree = dg.out_degree(weight="weight")

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

print()
print("In degree sorted")
for game in in_sorted:
    print(game)

print()
print("Pagerank sorted")
for game in pr_sorted:
    print(game)





