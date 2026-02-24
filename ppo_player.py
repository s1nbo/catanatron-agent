import numpy as np
import os
from typing import Tuple
from catanatron import Player, Game, Action
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.catanatron_env import to_action_space, ACTION_SPACE_SIZE
from sb3_contrib import MaskablePPO

class PPOPlayer(Player):
    _models = {}
    _features_ordering = None

    def __init__(self, color, model_path=None):
        super().__init__(color)
        self.model_path = model_path
        self._load_resources()

    def _load_resources(self):
        if self.model_path and os.path.exists(self.model_path):
            if self.model_path not in PPOPlayer._models:
                PPOPlayer._models[self.model_path] = MaskablePPO.load(self.model_path)
        
        if PPOPlayer._features_ordering is None:
            PPOPlayer._features_ordering = get_feature_ordering(4)

    def decide(self, game: Game, playable_actions: np.ndarray) -> Action:
        observation = self._get_observation(game)
        action_mask, action_mapping = self._get_action_mask(playable_actions)
        
        try:
            if self.model_path in PPOPlayer._models:
                model = PPOPlayer._models[self.model_path]
                action_idx, _ = model.predict(
                    observation, 
                    action_masks=action_mask, 
                    deterministic=True
                )
                mapped_action = action_mapping.get(int(action_idx))
                if mapped_action is not None:
                    return mapped_action
        except Exception as e:
            # Fallback to random if model fails or path invalid
            pass
        
        return np.random.choice(playable_actions)

    def _get_observation(self, game: Game) -> np.ndarray:
        sample = create_sample(game, self.color)
        return np.array([float(sample[f]) for f in PPOPlayer._features_ordering])

    def _get_action_mask(self, playable_actions: np.ndarray) -> Tuple[np.ndarray, dict]:
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
