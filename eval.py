import numpy as np
from typing import List, Tuple
from sb3_contrib import MaskablePPO
from catanatron import Player, Game, Action
from catanatron.cli import register_cli_player
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.catanatron_env import to_action_space, ACTION_SPACE_SIZE

class MyBot(Player):
    _models = {}
    _features_ordering = None

    def __init__(self, color, model_path="ppo_catanatron_02.zip"):
        super().__init__(color)
        self.model_path = model_path
        self._load_resources()

    def _load_resources(self):
        if self.model_path not in MyBot._models:
            MyBot._models[self.model_path] = MaskablePPO.load(self.model_path)
        
        if MyBot._features_ordering is None:
            MyBot._features_ordering = get_feature_ordering(4)

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        observation = self._get_observation(game)
        action_mask, action_mapping = self._get_action_mask(playable_actions)
        
        try:
            predicted_idx = self._predict(observation, action_mask)
            mapped_action = action_mapping.get(predicted_idx)
            if mapped_action is not None:
                return mapped_action
        except Exception as e:
            print(f"Error during prediction: {e}. Falling back to random choice.")
        
        return np.random.choice(playable_actions)

    def _get_observation(self, game: Game) -> np.ndarray:
        sample = create_sample(game, self.color)
        return np.array([float(sample[f]) for f in MyBot._features_ordering])

    def _get_action_mask(self, playable_actions: List[Action]) -> Tuple[np.ndarray, dict]:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        action_mapping = {}
        
        for action in playable_actions:
            try:
                idx = to_action_space(action)
                mask[idx] = True
                action_mapping[idx] = action
            except ValueError:
                continue
                
        return mask, action_mapping

    def _predict(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        model = MyBot._models[self.model_path]
        action_idx, _ = model.predict(
            observation, 
            action_masks=action_mask, 
            deterministic=True
        )
        return int(action_idx)

# Create factory functions/classes for registration
def create_bot_class(model_path, name):
    class Bot(MyBot):
        def __init__(self, color):
            super().__init__(color, model_path=model_path)
    Bot.__name__ = name
    return Bot

Bot02 = create_bot_class("ppo_catanatron_02.zip", "Bot02")
Bot01 = create_bot_class("ppo_catanatron_01.zip", "Bot01")
Bot00 = create_bot_class("ppo_catanatron_00.zip", "Bot00")

register_cli_player("0", Bot02)
register_cli_player("1", Bot01)
register_cli_player("2", Bot00)
register_cli_player("3", Bot00)