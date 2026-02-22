import gymnasium as gym
import numpy as np
import copy
from catanatron.gym.envs.catanatron_env import CatanatronEnv
from catanatron.models.player import Color
from league import League

class SelfPlayEnv(CatanatronEnv):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = {}
        
        self.league = League()
        self.hero_name = "current_training_agent" 
        
        # The reset() method will sample actual opponents from the league and update this config before each episode starts.
        config["enemies"] = [
            self.league.get_player_instance("random_red", Color.RED, {"type": "random"}),
            self.league.get_player_instance("random_red", Color.ORANGE, {"type": "random"}),
            self.league.get_player_instance("random_red", Color.WHITE, {"type": "random"}),
        ]
        
        super().__init__(config=config, **kwargs)

    def reset(self, seed=None, options=None):
        # Sample new enemies
        enemy_data = self.league.sample_opponents(3)
        self.current_enemy_names = [name for name, _ in enemy_data]
        
        # Assign colors to enemies
        # Assumes Hero is BLUE. We assign other colors to enemies.
        colors = [Color.RED, Color.ORANGE, Color.WHITE]
        enemies = []
        for i, (name, data) in enumerate(enemy_data):
            player = self.league.get_player_instance(name, colors[i], data)
            enemies.append(player)
            
        # Update internal game configuration so super().reset() uses new enemies
        self.config["enemies"] = enemies
        
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if terminated or truncated:
            winning_color = info.get("winning_color") 
            if winning_color is None and hasattr(self, "game"):
                winning_color = getattr(self.game, "winning_color", None)

            if winning_color:
                # Determine winner name
                winner_name = None
                loser_names = []
                
                # A We are BLUE
                if winning_color == Color.BLUE:
                    winner_name = self.hero_name
                    loser_names = self.current_enemy_names
                else:
                    enemy_colors = [Color.RED, Color.ORANGE, Color.WHITE]
                    if winning_color in enemy_colors:
                        idx = enemy_colors.index(winning_color)
                        winner_name = self.current_enemy_names[idx]
                        loser_names = [n for j, n in enumerate(self.current_enemy_names) if j != idx] + [self.hero_name]

                if winner_name:
                    self.league.update_elo(winner_name, loser_names)
                
        return obs, reward, terminated, truncated, info



