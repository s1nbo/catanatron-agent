"""
Test for the ELO system using random, alphabeta, and PPO bots.
Uses the pre-trained PPO agents in start/.
"""
import os
import json
from catanatron import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from league import League

# --- Config ---
TEST_LEAGUE_FILE = "league_test.json"
NUM_GAMES = 10
COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

PPO_AGENTS = {
    "ppo_catanatron_01": "start/ppo_catanatron_01.zip",
    "ppo_catanatron_02": "start/ppo_catanatron_02.zip",
    "ppo_catanatron_03": "start/ppo_catanatron_03.zip",
}

# Clean up any previous test file
if os.path.exists(TEST_LEAGUE_FILE):
    os.remove(TEST_LEAGUE_FILE)
if os.path.exists(TEST_LEAGUE_FILE + ".lock"):
    os.remove(TEST_LEAGUE_FILE + ".lock")

# Create a fresh league backed by a test file
league = League(league_file=TEST_LEAGUE_FILE)

# Add a few named random bots
for name in ["random_blue", "random_orange", "random_white"]:
    league.add_player(name, "random")

# Add alphabeta bots with different depths
league.players["alphabeta_d1"] = {"type": "alphabeta", "depth": 1, "path": None, "elo": 1000, "games": 0}

# Add pre-trained PPO agents
for name, path in PPO_AGENTS.items():
    league.players[name] = {"type": "ppo", "path": path, "elo": 1000, "games": 0}

league.save_league()

print("Initial league state:")
print(json.dumps(league.players, indent=2))
print()

# Play games and update ELO
for game_num in range(1, NUM_GAMES + 1):
    # Sample 4 opponents (with replacement if needed)
    opponents = league.sample_opponents(n=4)
    # Pad/repeat if fewer than 4 players in league
    while len(opponents) < 4:
        opponents.append(opponents[0])
    opponents = opponents[:4]

    players = []
    for i, (name, data) in enumerate(opponents):
        color = COLORS[i]
        players.append((name, league.get_player_instance(name, color, data), color))

    game = Game([p for _, p, _ in players])
    game.play()

    # Determine winner by VP
    winner_color = game.winning_color()
    winner_name = next(
        (name for name, _, color in players if color == winner_color),
        None
    )
    loser_names = [name for name, _, color in players if color != winner_color]

    if winner_name:
        league.update_elo(winner_name, loser_names)
        print(f"Game {game_num:02d}: winner={winner_name}, losers={loser_names}")
    else:
        print(f"Game {game_num:02d}: no winner (draw/timeout)")

print()
print("Final league ELOs:")
# Reload to show saved state
with open(TEST_LEAGUE_FILE) as f:
    final = json.load(f)
for name, data in sorted(final.items(), key=lambda x: -x[1]["elo"]):
    print(f"  {name:20s}  ELO={data['elo']:.1f}  games={data['games']}")
