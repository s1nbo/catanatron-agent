import json
import math
import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from filelock import FileLock
from catanatron import Player, Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
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
                    "alphabeta_d1": {"type": "alphabeta", "path": None, "depth": 1, "elo": 3000, "games": 0},
                }
                with open(self.league_file, "w") as f:
                    json.dump(initial_players, f, indent=4)
                return initial_players

    def save_league(self):
        with self.lock:
            with open(self.league_file, "w") as f:
                json.dump(self.players, f, indent=4)

    def add_player(self, name: str, player_type: str, path: Optional[str] = None, initial_elo: Optional[float] = None):
        with self.lock:
            # Reload to get latest state before writing
            if os.path.exists(self.league_file):
                with open(self.league_file, "r") as f:
                     self.players = json.load(f)

            if name not in self.players:
                if initial_elo is not None:
                    elo = initial_elo
                else:
                    # Fall back to league mean or 1000
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
    
    def prune_league(self, max_size: int = 32):
        """
        Removes oldest PPO bots from the league to keep size manageable.
        Always preserves non-PPO bots (like random, alphabeta).
        Deletes the model files from disk.
        """
        with self.lock:
            # Reload
            if os.path.exists(self.league_file):
                with open(self.league_file, "r") as f:
                    self.players = json.load(f)
            
            # Separate permanent agents from prune-able ones
            ppo_agents = []
            others = {}
            
            for name, data in self.players.items():
                if data.get("type") == "ppo":
                    ppo_agents.append((name, data))
                else:
                    others[name] = data

            # If we don't need to prune, exit
            if len(ppo_agents) <= max_size:
                return

            # Sort ppo agents by ELO ascending so we remove the weakest ones
            def sort_key(item):
                _, data = item
                return data.get("elo", 1000)

            # Sort ascending: lowest ELO first
            ppo_agents.sort(key=sort_key)

            # Determine how many to remove
            num_to_remove = len(ppo_agents) - max_size
            to_remove = ppo_agents[:num_to_remove]
            to_keep = ppo_agents[num_to_remove:]
            
            print(f"[League] Pruning {len(to_remove)} weakest agents...")

            # Reconstruct the players dict
            self.players = others
            for name, data in to_keep:
                self.players[name] = data
            
            # Delete files and clean up
            for name, data in to_remove:
                path = data.get("path")
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"Deleted properties for {name} (ELO {data.get('elo'):.1f}) and file: {path}")
                    except OSError as e:
                        print(f"Error deleting {path}: {e}")
            
            # Save
            with open(self.league_file, "w") as f:
                json.dump(self.players, f, indent=4)


    def update_elo(self, winner: str, losers: List[str]):
        with self.lock:
            # Reload league to get fresh ELOs
            if os.path.exists(self.league_file):
                with open(self.league_file, "r") as f:
                     self.players = json.load(f)

            K = 1

            # Ensure current_training_agent exists in the league so its ELO is tracked
            for participant in [winner] + losers:
                if participant == "current_training_agent" and participant not in self.players:
                    elos = [p["elo"] for p in self.players.values()]
                    self.players[participant] = {
                        "type": "training",
                        "path": None,
                        "elo": float(np.mean(elos)) if elos else 1000.0,
                        "games": 0,
                    }

            winner_data = self.players.get(winner)
            
            for loser in losers:
                loser_data = self.players.get(loser)
                if not loser_data: continue

                # If winner is unknown (e.g. current agent), treat as average ELO or fixed?
                # Better: assume winner has ELO equal to average of losers + 100?
                # Or just fetch if available.
                
                winner_elo = winner_data["elo"] if winner_data else 1000
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

    def sample_opponents(self, n=3, hero_elo: Optional[float] = None) -> List[Tuple[str, Dict]]:
        """
        Sample n opponents using ELO-weighted probabilities when hero_elo is provided.
        Opponents closer in ELO to the hero are sampled more often, which prevents
        a much-stronger opponent (e.g. AlphaBeta) from dominating the pool before
        the agent is ready to learn from it.
        """
        # Reload league to get latest players
        with self.lock:
            if os.path.exists(self.league_file):
                with open(self.league_file, "r") as f:
                    self.players = json.load(f)

        # Exclude the training agent itself from the opponent pool
        names = [k for k in self.players.keys() if k != "current_training_agent"]
        if not names:
            return []

        if hero_elo is not None:
            # Weight by proximity in ELO: exp(-|delta_elo| / 400)
            # This gives ~37% relative weight at 400 pts difference, ~14% at 800 pts.
            weights = [
                math.exp(-abs(self.players[name]["elo"] - hero_elo) / 400)
                for name in names
            ]
            total = sum(weights)
            probs = [w / total for w in weights]
        else:
            probs = None

        replace = len(names) < n
        selected_names = list(
            np.random.choice(names, size=n, replace=replace, p=probs)
        )

        return [(name, self.players[name]) for name in selected_names]

    def get_player_instance(self, name, color, data=None):
        if not data:
            data = self.players.get(name)
        
        if not data:
            return RandomPlayer(color)
        
        if data["type"] == "random":
            return RandomPlayer(color)
        elif data["type"] == "alphabeta":
            depth = data.get("depth", 2) # default to 2 if not specified
            return AlphaBetaPlayer(color, depth=depth)
        elif data["type"] == "ppo":
            return PPOPlayer(color, model_path=data["path"])
        else:
            return RandomPlayer(color)

