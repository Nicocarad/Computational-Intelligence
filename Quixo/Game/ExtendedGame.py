from copy import deepcopy
from Game.game import Game, Move


class ExtendedGame(Game):
    def __init__(self):
        super().__init__()

    def possible_moves(self, playerId: int) -> tuple[tuple[int, int], Move]:
        """Return a tuple of possible moves for a given player in a given state of the game"""

        # Define the edges of the game grid
        perimeter = [0, 4]
        # Initialize an empty list to store possible moves
        possible_moves = []
        # Get the current game board
        board = self.get_board()

        # Iterate over the edges of the game grid
        for index in perimeter:
            # Iterate over the columns of the game grid
            for col in range(5):
                # If the current cell belongs to the current player or is empty
                if board[col][index] in {playerId, -1}:
                    # If we are not on the first column, we can move up
                    if col != 0:
                        possible_moves.append(((index, col), Move.TOP))
                    # If we are not on the last column, we can move down
                    if col != 4:
                        possible_moves.append(((index, col), Move.BOTTOM))
                    # If we are not on the first row, we can move left
                    if index != 0:
                        possible_moves.append(((index, col), Move.LEFT))
                    # If we are not on the last row, we can move right
                    if index != 4:
                        possible_moves.append(((index, col), Move.RIGHT))

            # Iterate over the rows of the game grid
            for row in range(5):
                # If the current cell belongs to the current player or is empty
                if board[index][row] in {playerId, -1}:
                    # If we are not on the first column, we can move up
                    if index != 0:
                        possible_moves.append(((row, index), Move.TOP))
                    # If we are not on the last column, we can move down
                    if index != 4:
                        possible_moves.append(((row, index), Move.BOTTOM))
                    # If we are not on the first row, we can move left
                    if row != 0:
                        possible_moves.append(((row, index), Move.LEFT))
                    # If we are not on the last row, we can move right
                    if row != 4:
                        possible_moves.append(((row, index), Move.RIGHT))

        # Return the possible moves as a tuple
        return tuple(possible_moves)

    def create_new_state(
        self, from_pos: tuple[int, int], slide: Move, player_id: int
    ) -> "ExtendedGame":
        """Return a new game state after applying a move"""

        # Swap the position coordinates
        from_pos = (from_pos[1], from_pos[0])
        # Create a new instance of the ExtendedGame
        new_game = ExtendedGame()
        new_game.current_player_idx = player_id
        # Copy the current game board to the new game
        new_game._board = deepcopy(self._board)
        new_game._take(from_pos, player_id)
        new_game._slide(from_pos, slide)
        new_game._switch_player()

        # Return the new game state
        return new_game

    def _switch_player(self):
        self.current_player_idx = 1 - self.current_player_idx
