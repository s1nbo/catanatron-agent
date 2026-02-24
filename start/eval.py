import os
import numpy as np
from typing import List, Tuple
from sb3_contrib import MaskablePPO
from catanatron import Player, Game, Action
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.cli import register_cli_player
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.catanatron_env import to_action_space, ACTION_SPACE_SIZE

# League models directory (one level up from start/)
LEAGUE_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "league_models")

def league_model(name: str) -> str:
    """Return absolute path for a model in league_models/."""
    filename = name if name.endswith(".zip") else name + ".zip"
    return os.path.join(LEAGUE_MODELS_DIR, filename)

class MyBot(Player):
    _models = {}
    _features_ordering = None

    def __init__(self, color, model_path=None):
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

# --- Configure which league models to use here ---
Bot01 = create_bot_class(league_model("v2_gen_9"),         "Bot01_v2_gen_9")
Bot02 = create_bot_class(league_model("v1_gen_25"),        "Bot02_v1_gen_26")
Bot03 = create_bot_class(league_model("real_run_gen_14"),  "Bot03_real_run_gen_14")
Bot04 = create_bot_class(league_model("v1_gen_21"),        "Bot04_v1_gen_21")

class D1(AlphaBetaPlayer):
    def __init__(self, color):
        super().__init__(color, depth=1, prunning=True)

register_cli_player("1", Bot01)
register_cli_player("2", Bot02)
register_cli_player("3", Bot03)
register_cli_player("4", Bot04)
register_cli_player("d1", D1)