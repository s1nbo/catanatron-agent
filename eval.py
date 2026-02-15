import numpy as np
from typing import List, Tuple
from sb3_contrib import MaskablePPO
from catanatron import Player, Game, Action
from catanatron.cli import register_cli_player
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.catanatron_env import to_action_space, ACTION_SPACE_SIZE

class MyBot(Player):
    MODEL_PATH = "ppo_catanatron_01.zip"
    _model = None
    _features_ordering = None

    def __init__(self, color):
        super().__init__(color)
        self._load_resources()

    @classmethod
    def _load_resources(cls):
        if cls._model is None:
            cls._model = MaskablePPO.load(cls.MODEL_PATH)
        if cls._features_ordering is None:
            cls._features_ordering = get_feature_ordering(4)

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        observation = self._get_observation(game)
        action_mask, action_mapping = self._get_action_mask(playable_actions)
        
        try:
            predicted_idx = self._predict(observation, action_mask)
            mapped_action = action_mapping.get(predicted_idx)
            if mapped_action is not None:
                return mapped_action
            print(f"Warning: Predicted action {predicted_idx} is not valid. Falling back to random choice.")
        except Exception as e:
            print(f"Error during prediction: {e}. Falling back to random choice.")
        
        return np.random.choice(playable_actions)

    def _get_observation(self, game: Game) -> np.ndarray:
        sample = create_sample(game, self.color)
        return np.array([float(sample[f]) for f in self._features_ordering])

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
        action_idx, _ = self._model.predict(
            observation, 
            action_masks=action_mask, 
            deterministic=True
        )
        return int(action_idx)


register_cli_player("ME", MyBot)