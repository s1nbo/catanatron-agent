import random
from catanatron import Game, Color, Player, RandomPlayer

# TODO Change to PPO agent once trained, for now just a random bot to test the environment and game loop.
class MyRandomBot(Player):
    def decide(self, game, playable_actions):
        return random.choice(playable_actions)

if __name__ == "__main__":
    print("Starting game with MyRandomBot against 3 RandomPlayers...")
    
    # Setup players: MyRandomBot vs 3 RandomPlayers
    players = [
        MyRandomBot(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        RandomPlayer(Color.WHITE),
    ]

    game = Game(players)
    winning_color = game.play()
    
    print(f"Game finished. Winner: {winning_color}")
