The catanatron gymnasium environment is defined locally in env_copy.py. Here are the details for the actions, observations, and rewards.

1. Action Space
The environment uses a Discrete action space (size ~290), representing all possible moves in Catan.

0: ROLL
Move Robber: Moves robber to a specific tile (19 actions).
Discard: Discard half of resources (1 action, simplified to random).
Build Road: Build a road on a specific edge (72 actions).
Build Settlement: Build on a specific node (54 actions).
Build City: Upgrade a settlement to a city (54 actions).
Buy Development Card: (1 action).
Play Development Card:
PLAY_KNIGHT_CARD
PLAY_YEAR_OF_PLENTY (various resource combinations)
PLAY_ROAD_BUILDING
PLAY_MONOPOLY (one for each resource type)
Maritime Trade: Trade resources 4:1, 3:1, or 2:1 with ports (various combinations).
End Turn: Pass turn to the next player.
2. Observation Space
The observation depends on the configuration (vector or mixed). The default is Vector (flat array).

Vector Representation (Box):
A flat array of numerical features representing the game state from the current player's perspective (P0).

Bank State: Resources and Dev cards available in the bank.
Board State:
Ownership of valid Roads (Edges), Settlements (Nodes), and Cities (Nodes).
Port locations and types.
Robber position and tile resource types/probabilities.
Turn State: Flags for IS_DISCARDING, IS_MOVING_ROBBER, HAS_ROLLED, PLAYED_DEV_CARD.
Player State (P0 - You):
Actual Victory Points (including hidden ones).
Exact resource and development card counts in hand.
Pieces left to build (Roads, Settlements, Cities).
Opponent State (P1...Pn):
Public information only (e.g., total card counts, public VPs, longest road length).
Exact resources/dev cards are hidden (represented as total counts).
Mixed Representation (Dict):

"board": A 3D tensor (channels x 21 x 11) encoding the spatial board state.
"numeric": A 1D array containing the non-spatial features (cards in hand, bank state, etc.).
3. Reward Function
The default reward function is sparse and zero-sum:

+1: You win the game.
-1: You lose the game (opponent wins).
0: The game is ongoing.
Note: The environment also calculates an invalid action reward (default -1) if the agent attempts an illegal move, though widely used wrappers like ActionMasker typically prevent this.