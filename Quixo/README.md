# ExtendedGame

### Overview

The ExtendedGame class extends the functionality of a basic game represented by the Game class. It introduces additional functions that do not change the logic of the game but are useful to implement a player such as MinMaxPlayer.

### Class Overview

ExtendedGame
Methods:

- **possible_moves(self, playerId: int) -> tuple[tuple[int, int], Move]:** Returns a tuple of possible moves for a given player in the current state of the game
- **create_new_state(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> "ExtendedGame":** Creates a new game state performing a move

- **_switch_player()** switch the current player after a move

# Players

### Overview

- RandomPlayer: A player that makes random moves.
- HumanPlayer: A player that allows a human to interactively make moves.
- MinMaxPlayer: An AI player using the Minimax algorithm with Alpha-Beta Pruning to make strategic moves.

### Player Classes

- RandomPlayer
  This class represents a player that randomly selects moves on the game board. It is implemented with the make_move method, where it generates random positions and a random move direction.

- HumanPlayer
  This class represents a player that allows a human to interactively make moves. The make_move method prompts the user to input the position to move from and the direction to move.

- MinMaxPlayer
  This class represents an AI player using the Minimax algorithm with Alpha-Beta Pruning to make optimal moves. The make_move method implements the Minimax algorithm to evaluate possible moves and choose the best one. The evaluate method assigns scores to different game states, and the minmax method recursively explores possible moves while considering alpha-beta pruning for optimization.

### MinMaxPlayer Configuration

The MinMaxPlayer class takes three parameters during instantiation:

- game: The game object (an instance of ExtendedGame) on which the player will make moves.
- max_depth: The maximum depth to explore in the Minimax algorithm.

The agent is proposed with a __max_depth = 10__ even if tests show that the agent as good results also with other lower even numbers.

# Useful functions and Testing

In order to test the performance of the game some useful functions are provided.

- __test_agent(num_games)__ to simply test the Minmax player against a RandomPlayer 100 times

- __test_agent_depths(num_games, max_depths)__ to test the MinMaxPlayer performance against a RandomPlayer, playing 100 times as first player (simbol 0) and 100 times as second player (simbol 1)

- __plot_results(results, filename, title)__ to print some histograms about results obtained with the previous function.

- __play_against_ai()__ to play a real time game against the MinMaxPlayer.

# Sources

- The structure for the __make_move(self, game)__ in MinMaxPlayer was taken and modified from https://www.geeksforgeeks.org/finding-optimal-move-in-tic-tac-toe-using-minimax-algorithm-in-game-theory/ 

- The structure for the alpha-beta pruning inside __minmax(self,
        game,
        depth,
        alpha, 
        beta, 
        maximizingPlayer)__ was taken from https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/ and https://github.com/Berkays/Quixo/blob/master/ai.py

