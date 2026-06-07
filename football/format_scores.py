import os
import binascii
from pathlib import Path

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

games_final = []
for game in games:
    try:
        team1 = game[0][0].split("(")[0].strip()
        team2 = game[0][2].split("(")[0].strip()
        scores = game[0][1].split("-")[0].strip()
        cleaned = [(team1, score[0], team2, score[1])]
        games_final.append(cleaned)
    except:
        breakpoint()

breakpoint()
