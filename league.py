import json
import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from filelock import FileLock
from catanatron import Player, Game
from catanatron.models.player import Color, RandomPlayer
# from catanatron.players.minimax import AlphaBetaPlayer # Temporarily disabled
from sb3_contrib import MaskablePPO
from ppo_player import PPOPlayer

LEAGUE_FILE = "league.json"
LOCK_FILE = "league.json.lock"

class League:
    def __init__(self, league_file=LEAGUE_FILE):
        self.league_file = league_file
        self.lock = FileLock(LOCK_FILE)
        self.players = self._load_league()

    def _load_league(self) -> Dict:
        with self.lock:
            if os.path.exists(self.league_file):
                with open(self.league_file, "r") as f:
                    return json.load(f)
            else:
                # Initialize with baseline players
                initial_players = {
                    "random_red": {"type": "random", "path": None, "elo": 1000, "games": 0},
                    # "alphabeta_deep": {"type": "alphabeta", "path": None, "elo": 1200, "games": 0},
                }
                with open(self.league_file, "w") as f:
                    json.dump(initial_players, f, indent=4)
                return initial_players

    def save_league(self):
        with self.lock:
            with open(self.league_file, "w") as f:
                json.dump(self.players, f, indent=4)

    def add_player(self, name: str, player_type: str, path: Optional[str] = None):
        with self.lock:
            # Reload to get latest state before writing
            if os.path.exists(self.league_file):
                with open(self.league_file, "r") as f:
                     self.players = json.load(f)

            if name not in self.players:
                # Start with average ELO of current league or 1000
                elos = [p["elo"] for p in self.players.values()]
                elo = np.mean(elos) if elos else 1000
                self.players[name] = {
                    "type": player_type,
                    "path": path,
                    "elo": elo,
                    "games": 0
                }
                # Write back
                with open(self.league_file, "w") as f:
                    json.dump(self.players, f, indent=4)

    def update_elo(self, winner: str, losers: List[str]):
        with self.lock:
            # Reload league to get fresh ELOs
            if os.path.exists(self.league_file):
                with open(self.league_file, "r") as f:
                     self.players = json.load(f)

            K = 32
            winner_data = self.players.get(winner)
            # If winner is not in league (e.g. current training agent), we skip updating its ELO 
            # but still update losers IF they are in league.
            
            # Actually, usually we track the training agent as a temporary entry or just don't update it yet.
            # But the losers must be updated.
            
            for loser in losers:
                loser_data = self.players.get(loser)
                if not loser_data: continue

                # If winner is unknown (e.g. current agent), treat as average ELO or fixed?
                # Better: assume winner has ELO equal to average of losers + 100?
                # Or just fetch if available.
                
                winner_elo = winner_data["elo"] if winner_data else 1200 # Default if training agent unknown
                loser_elo = loser_data["elo"]

                # Expected win prob for winner
                ea = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
                eb = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
                
                # Winner wins (score = 1), Loser loses (score = 0)
                if winner_data:
                    winner_data["elo"] += K * (1 - ea)
                    winner_data["games"] += 1
                
                loser_data["elo"] += K * (0 - eb)
                loser_data["games"] += 1

            with open(self.league_file, "w") as f:
                json.dump(self.players, f, indent=4)

    def sample_opponents(self, n=3) -> List[Tuple[str, Dict]]:
        # Reload league to get latest players
        with self.lock:
             if os.path.exists(self.league_file):
                with open(self.league_file, "r") as f:
                     self.players = json.load(f)

        names = list(self.players.keys())
        if not names:
             return []
             
        if len(names) < n:
            selected_names = [random.choice(names) for _ in range(n)]
        else:
            selected_names = random.sample(names, n)
            
        return [(name, self.players[name]) for name in selected_names]

    def get_player_instance(self, name, color, data=None):
        if not data:
            data = self.players.get(name)
        
        if not data:
            return RandomPlayer(color)
        
        if data["type"] == "random":
            return RandomPlayer(color)
        #elif data["type"] == "alphabeta":
        #    return AlphaBetaPlayer(color)
        elif data["type"] == "ppo":
            return PPOPlayer(color, model_path=data["path"])
        else:
            return RandomPlayer(color)

