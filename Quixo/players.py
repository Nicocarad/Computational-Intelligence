import random
import numpy as np
from Game.game import Move, Player
from Game.ExtendedGame import ExtendedGame


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.name = "RandomPlayer"

    def make_move(self, game: "ExtendedGame") -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.name = "HumanPlayer"

    def make_move(self, game: "ExtendedGame") -> tuple[tuple[int, int], Move]:
        # Get the current player
        player = game.get_current_player()

        # Get the list of possible moves
        possible_moves = game.possible_moves(player)

        print("BOARD:")
        game.print()

        # Print the list of possible moves
        print("Possible moves:")
        for move in possible_moves:
            print(f"From position {move[0]} move {move[1]}")

        # Ask the user for their move
        from_pos = tuple(
            map(int, input("Enter the position to move from (row, col): ").split(","))
        )
        move = Move[
            input("Enter the direction to move (TOP, BOTTOM, LEFT, RIGHT): ").upper()
        ]
        return from_pos, move


class MinMaxPlayer(Player):
    def __init__(self, game: "ExtendedGame", max_depth=10) -> None:
        super().__init__()
        self.name = "MinMaxPlayer"
        self.game = game
        self.max_depth = max_depth
        self.infinity = float("inf")

    def evaluate(self, game: "ExtendedGame") -> int:
        player = (
            1 - game.get_current_player()
        )  # restore the player of the prevoius state since create_new_state() switch it.
        score = 0
        board = game.get_board()

        # Check rows
        for row in board:
            score += self.evaluate_line(row, player)

        # Check columns
        for col in board.T:
            score += self.evaluate_line(col, player)

        # Check main diagonal
        main_diag = np.diagonal(board)
        score += self.evaluate_line(main_diag, player)

        # Check secondary diagonal
        secondary_diag = np.diagonal(np.fliplr(board))
        score += self.evaluate_line(secondary_diag, player)

        return score

    @staticmethod
    def evaluate_line(line: list[int], player_id: int) -> int:
        line_score = 0

        # Count occurrences of player's symbol and opponent's symbol
        player_count = np.sum(line == player_id)
        opponent_count = np.sum(line == 1 - player_id)

        # Assign scores based on counts
        if player_count > 0:
            line_score += 10**player_count
        if opponent_count > 0:
            line_score -= 10**opponent_count

        return line_score

    def minmax(
        self,
        game: "ExtendedGame",
        depth: int,
        alpha: float,
        beta: float,
        isMaximizingPlayer: bool,
    ) -> tuple[int, float, float]:
        # Base case: if we have reached the maximum depth or the game is over,
        # return the evaluation of the game state

        if depth == 0 or game.check_winner() != -1:
            return self.evaluate(game), alpha, beta

        # Decrease the depth
        depth -= 1

        player = game.get_current_player()

        # If we are the maximizing player
        if isMaximizingPlayer:
            # Initialize the maximum evaluation to negative infinity
            bestVal = -self.infinity
            # Iterate over all possible moves
            for move in game.possible_moves(player):
                # Create a new game state by making the move
                new_state = game.create_new_state(move[0], move[1], player)
                # Call minmax recursively on the new state
                value, alpha, beta = self.minmax(new_state, depth, alpha, beta, False)
                # Update the maximum evaluation
                bestVal = max(bestVal, value)
                # Update alpha
                alpha = max(alpha, value)
                # If beta is less than or equal to alpha, break the loop (alpha-beta pruning)
                if alpha >= beta:
                    break
            # The result is the maximum evaluation, alpha and beta
            result = bestVal, alpha, beta

        # If we are the minimizing player
        else:
            # Initialize the minimum evaluation to positive infinity
            bestVal = self.infinity
            # Iterate over all possible moves
            for move in game.possible_moves(player):
                # Create a new game state by making the move
                new_state = game.create_new_state(move[0], move[1], player)
                # Call minmax recursively on the new state
                value, alpha, beta = self.minmax(new_state, depth, alpha, beta, True)
                # Update the minimum evaluation
                bestVal = min(bestVal, value)
                # Update beta
                beta = min(beta, value)
                # If beta is less than or equal to alpha, break the loop (alpha-beta pruning)
                if alpha >= beta:
                    break
            # The result is the minimum evaluation, alpha and beta
            result = bestVal, alpha, beta

        return result

    def make_move(self, game: "ExtendedGame") -> tuple[tuple[int, int], Move]:
        # Initialize the best move to None and the best evaluation to negative infinity
        bestMove = None
        bestVal = -self.infinity

        # Get the current player, it will be the index of the MinmaxPlayer
        player = self.game.get_current_player()

        # Get the list of possible moves for MinmaxPlayer
        possible_moves = list(game.possible_moves(player))

        # Iterate over all possible moves
        for move in possible_moves:
            # Create a new game state by making the move
            new_state = self.game.create_new_state(move[0], move[1], False)

            # Early return if a move is a winning move for MinmaxPlayer
            if new_state.check_winner() == player:
                bestMove = move
                break

            # Call the Minimax function on the new state to get the evaluation of the state
            value = self.minmax(
                new_state, self.max_depth, -self.infinity, self.infinity, False
            )[0]

            # If the evaluation of the state is greater than the best evaluation, update the best evaluation and the best move
            if value > bestVal:
                bestVal = value
                bestMove = move

        return bestMove
