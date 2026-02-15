from catanatron import Player, Game, Color
from catanatron.cli import register_cli_player
import numpy as np
import gymnasium
import catanatron.gym
from sb3_contrib import MaskablePPO

# Import random bot 
from agent import MyRandomBot
from catanatron.cli.cli_players import AlphaBetaPlayer

# Register a stronger AlphaBeta bot
# depth=3 is significantly harder than depth=2. prunning=True makes it faster.
register_cli_player("STRONG_AB", lambda color: AlphaBetaPlayer(color, depth=3, prunning=True))

# Import helpers from our local copy to map actions
# This file (env_copy.py) must exist in the directory
import env_copy

class MyGymBot(Player):
    def __init__(self, color):
        super().__init__(color)
        print(f"Loading model for {color}...")
        try:
            self.model = MaskablePPO.load("ppo_catanatron")
        except Exception as e:
            print(f"FAILED TO LOAD MODEL: {e}")
            raise e
            
        # Helper env helper to create observations
        self.env = gymnasium.make("catanatron/Catanatron-v0")
        self.unwrapped = self.env.unwrapped

    def decide(self, game, playable_actions):
        # 1. Inject current game state
        self.unwrapped.game = game
        # Find the player object corresponding to self.color
        # game.state.players operates by color index usually or dictionary?
        # catanatron color is enum.
        # Let's inspect how to get player object safely.
        # game.state.players is a list ordered by turn order usually?
        # Safer lookup matching color
        player_obj = next(p for p in game.state.players if p.color == self.color)
        # If players is a dict or list?
        # In catanatron source, players is usually list of Player objects.
        # And players are indexed by color value (0,1,2,3).
        self.unwrapped.p0 = player_obj
        
        # 2. Get observation
        obs = self.unwrapped._get_observation()
        
        # 3. Construct action mask
        valid_indices = self.unwrapped.get_valid_actions()
        action_mask = np.zeros(self.unwrapped.action_space.n, dtype=bool)
        action_mask[valid_indices] = True
        
        # 4. Predict action
        action_idx, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        
        # 5. Convert to Action
        try:
            return env_copy.from_action_space(int(action_idx), playable_actions)
        except Exception as e:
            print(f"Error mapping action {action_idx}: {e}")
            import random
            return random.choice(playable_actions)

register_cli_player("MYBOT", MyGymBot)
