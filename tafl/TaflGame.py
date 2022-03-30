import copy
import random
from enum import IntEnum

import numpy as np

from Game import Game
from tafl.TaflBoard import Outcome, Player, TaflBoard, TileState


class MovementType(IntEnum):
    horizontal = 0
    vertical = 1

class TaflGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self, size, prune):
        if size != 11 and size != 9 and size != 7:
            raise ValueError
        self.size = size
        self.prune = prune
        self.prune_prob = 0.1

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return TaflBoard(self.size)

    def getBoardSize(self):
        """
            Returns:
                (x,y): a tuple of board dimensions
        """
        return self.size, self.size

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        # size*size to select the piece to move
        # size for horizontal movement, size for vertical movement, so size*2 to select the action to take
        return self.size*self.size*self.size*2+1

    def getNextState(self, board, player, action, copy_board=False):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """

        if action == self.getActionSize() - 1:
            board.outcome = Outcome.black if player == Player.white else Player.black
            # assert board.outcome != Outcome.ongoing, str(player) + " selected 'no action', but had still moves left\n" \
            #                                          + str(board) + "\n" + str(list(board.get_valid_actions(player)))
        else:
            explicit = action_conversion__index_to_explicit(action, self.size)
            assert action_conversion__explicit_to_index(explicit, self.size) == action
            if copy_board:
                board = copy.deepcopy(board)
            board.do_action(explicit, player)
        next_player = -1 if player == 1 else 1
        return board, next_player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        array = np.zeros(self.getActionSize())  # see comment in getActionSize()
                                            # use 0 for horizontal movement indexing, 1 for vertical movement indexing

        if True:
            non_losing_moves = []
            # white:
            # preferences:
            #   1. king to corner
            #   2. king next to corner or king to empty side
            #   3. go to (2,2) or symmetrical equivalents when king can't be captured and there is no piece on
            #       (2,1) or (1,2)
            #   4. force same board state for the third time
            # 	5. prevent king capture
            if player == Player.white:
                # 1., 2. and 3.
                winning_move = get_king_escape_move(board)
                if winning_move is None:
                    move_set = board.get_valid_actions(player)
                    # 4.
                    for action in move_set:
                        if would_next_board_be_third(board, action):
                            winning_move = action
                            break
                    # 5.
                    if winning_move is None:
                        for action in move_set:
                            if not would_next_board_lead_to_opponent_winning(board, action, Player.white):
                                non_losing_moves.append(action)
            # black:
            # preferences:
            #   1. capture king
            #   2. force same board state for the third time
            # 	3. prevent king to corner
            # 	4. prevent king next to corner and prevent king to empty side
            #   5. prevent king going to (2,2) or symmetrical equivalents when king can't be captured and there is no
            #       piece on  (2,1) or (1,2)
            else:
                move_set = board.get_valid_actions(player)
                # 1.
                winning_move = get_king_capture_move(board, move_set)
                if winning_move is None:
                    # 2.
                    for action in move_set:
                        if would_next_board_be_third(board, action):
                            winning_move = action
                            break
                    # 3., 4. and 5.
                    if winning_move is None:
                        for action in move_set:
                            if not would_next_board_lead_to_opponent_winning(board, action, Player.black):
                                non_losing_moves.append(action)

            # set winning move if it exists
            if winning_move is not None:
                index = action_conversion__explicit_to_index(winning_move, self.size)
                array[index] = 1
            else:
                # set non losing moves if they exist, but not a winning move
                for explicit in non_losing_moves:
                    index = action_conversion__explicit_to_index(explicit, self.size)
                    array[index] = 1
            # set any move if both don't exist
            if winning_move is None and non_losing_moves == []:
                if len(move_set) == 0:
                    index = self.getActionSize()-1
                else:
                    index = action_conversion__explicit_to_index(random.choice(move_set), self.size)
                array[index] = 1

        else:
            no_immediate_loss_possible = False
            move_set = board.get_valid_actions(player)
            for action in move_set:
                if board.would_next_board_be_third(action):
                    array = np.zeros(self.getActionSize())
                    index = action_conversion__explicit_to_index(action, self.size)
                    assert action_conversion__index_to_explicit(index, self.size) == action
                    array[index] = 1
                    return array
                elif not board.would_next_board_lead_to_third(action, player):
                    index = action_conversion__explicit_to_index(action, self.size)
                    assert action_conversion__index_to_explicit(index, self.size) == action
                    array[index] = 1
                    no_immediate_loss_possible = True
            # if all possible moves lead to a loss...
            if not no_immediate_loss_possible:
                # ... check if there are actually moves that can be made. If not, just choose an impossible move and the
                # board class will report the correct outcome. This is necessary because it is only in this method here
                # that it is checked whether there are actually moves left. Meaning that the Outcome is set in the board
                # class already, but the network is still asked to select a move. Therefore we need to give the program
                # at least one move to choose from.
                if len(move_set) == 0:
                    index = self.getActionSize()-1
                else:
                    index = action_conversion__explicit_to_index(random.choice(move_set), self.size)
                array[index] = 1
        return array

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        if board.outcome == Outcome.ongoing:
            return 0
        elif board.outcome == Outcome.draw:
            return 0.000001
        elif board.outcome == Outcome.black:
            return 1 if player == Player.black else -1
        else:
            return -1 if player == Player.black else 1

    def getCanonicalForm(self, board, player, copy_board=False):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        # canonical form isn't really possible because of the asymmetric nature of tafl
        if copy_board:
            board = copy.deepcopy(board)
        return board

    def getSymmetries(self, board, pi, king_position):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        actions_and_probs = [(index, prob)for index, prob in enumerate(pi) if prob != 0]
        king_x, king_y = king_position

        symmetries = []
        # occurrences = np.zeros(self.size*self.size*self.size*2)
        # for index, prob in actions_and_probs:
        #    ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
        #     explicit = (self.size + 1 - x_from, y_from), (self.size + 1 - x_to, y_to)
        #     occurrences[index] = 1 if board.would_next_board_be_second_third(2, explicit) else 0

        #original
        temp_board = np.copy(board.board[1:self.size + 1, 1:self.size + 1])
        symmetries.append((temp_board, pi, (king_x, king_y)))

        # horizontal flip
        temp_board = np.flip(temp_board, 0)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (self.size + 1 - x_from, y_from), (self.size + 1 - x_to, y_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (self.size + 1 - king_x, king_y)))

        # horizontal and vertical flip
        temp_board = np.flip(temp_board, 1)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (self.size + 1 - x_from, self.size + 1 - y_from), (self.size + 1 - x_to, self.size + 1 - y_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (self.size + 1 - king_x, self.size + 1 - king_y)))

        # vertical flip
        temp_board = np.flip(temp_board, 0)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (x_from, self.size + 1 - y_from), (x_to, self.size + 1 - y_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (king_x, self.size + 1 - king_y)))

        # rotation
        temp_board = np.flip(temp_board, 1)
        temp_board = np.rot90(temp_board)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (self.size + 1 - y_from, x_from), (self.size + 1 - y_to, x_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (self.size + 1 - king_y, king_x)))

        # rotation and horizontal flip
        temp_board = np.flip(temp_board, 0)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (y_from, x_from), (y_to, x_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (king_y, king_x)))

        # rotation and horizontal and vertical flip
        temp_board = np.flip(temp_board, 1)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (y_from, self.size + 1 - x_from), (y_to, self.size + 1 - x_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (king_y, self.size + 1 - king_x)))

        # rotation and vertical flip
        temp_board = np.flip(temp_board, 0)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (self.size + 1 - y_from, self.size + 1 - x_from), (self.size + 1 - y_to, self.size + 1 - x_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (self.size + 1 - king_y, king_x)))

        return symmetries

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.bytes()
        # return str(board)


def action_conversion__explicit_to_index(explicit, size):
    (x_from, y_from), (x_to, y_to) = explicit
    movement_type = MovementType.horizontal if y_from == y_to else MovementType.vertical
    to = x_to if movement_type == MovementType.horizontal else y_to
    result = (((x_from - 1) * size + y_from - 1) * size + to - 1) * 2 + movement_type   # all coordinates -1 because of the border
    assert 0 <= result < size*size*size * 2
    return result


def action_conversion__index_to_explicit(action, size):
    from_x, from_y, to, movement_type = np.unravel_index(action, (size, size, size, 2))
    if movement_type == MovementType.horizontal:
        action = ((from_x + 1, from_y + 1), (to + 1, from_y + 1))  # all coordinates + 1 because of the border
    else:
        action = ((from_x + 1, from_y + 1), (from_x + 1, to + 1))  # all coordinates + 1 because of the border
    return action


# checks whether the next move would lead to a board state where the opponent of the turn player could make a move
# such that the resulting board state will be seen for the third time. This would lead to a win for the opponent
def would_next_board_lead_to_third(board, move, turn_player):
    move_from, move_to = move
    x_to, y_to = move_to
    previous_from = board.board[move_from]
    previous_to = board.board[x_to, y_to]

    # check captures
    own_tile_state = TileState.black if previous_from & TileState.black != 0 else TileState.white | TileState.king
    # TileState.white for Player.black, TileState.black for Player.white
    # this way is necessary because capturing the king works differently and is done further below
    opponent_pawn_tile_state = own_tile_state ^ (TileState.black | TileState.white)

    throne_check = TileState.empty if board.board[board.king_position] & TileState.throne != 0 else TileState.throne
    # check capture right
    if board.board[x_to + 1, y_to] & opponent_pawn_tile_state != 0 \
            and board.board[x_to + 2, y_to] & (own_tile_state | TileState.corner | throne_check) != 0:
        return False
    # check capture left
    if board.board[x_to - 1, y_to] & opponent_pawn_tile_state != 0 \
            and board.board[x_to - 2, y_to] & (own_tile_state | TileState.corner | throne_check) != 0:
        return False
    # check capture bottom
    if board.board[x_to, y_to + 1] & opponent_pawn_tile_state != 0 \
            and board.board[x_to, y_to + 2] & (own_tile_state | TileState.corner | throne_check) != 0:
        return False
    # check capture top
    if board.board[x_to, y_to - 1] & opponent_pawn_tile_state != 0 \
            and board.board[x_to, y_to - 2] & (own_tile_state | TileState.corner | throne_check) != 0:
        return False

    # if nothing is captured and the current board state has been seen two times already,
    # then the next player can just revert the currently checked move and win
    board_bytes = board.board.tobytes()
    if board_bytes in board.board_states_dict and board.board_states_dict[board_bytes] == 2:
        return True

    # see regular move method for a short explanation
    board.board[move_to] = (board.board[move_to] & TileState.throne) | \
                          (board.board[move_from] & (TileState.white | TileState.black | TileState.king))
    board.board[move_from] = board.board[move_from] & \
                            ~(TileState.white | TileState.black | TileState.king)  # remove piece from tile
    result = False
    for action in board.get_valid_actions(-1 * turn_player):
        result = result or would_next_board_be_third(board, action)
    board.board[move_from] = previous_from
    board.board[move_to] = previous_to
    return result


# checks whether the next move would lead to a board state that has occurred two times already
def would_next_board_be_third(board, action):
    move_from, move_to = action
    x_to, y_to = move_to
    previous_from = board.board[move_from]
    previous_to = board.board[x_to, y_to]

    # check captures
    own_tile_state = TileState.black if previous_from & TileState.black != 0 else TileState.white | TileState.king
    # TileState.white for Player.black, TileState.black for Player.white
    # this way is necessary because capturing the king works differently and is done further below
    opponent_pawn_tile_state = own_tile_state ^ (TileState.black | TileState.white)

    throne_check = TileState.empty if board.board[board.king_position] & TileState.throne != 0 else TileState.throne
    # check capture right
    if board.board[x_to + 1, y_to] & opponent_pawn_tile_state != 0 \
            and board.board[x_to + 2, y_to] & (own_tile_state | TileState.corner | throne_check) != 0:
        return False
    # check capture left
    if board.board[x_to - 1, y_to] & opponent_pawn_tile_state != 0 \
            and board.board[x_to - 2, y_to] & (own_tile_state | TileState.corner | throne_check) != 0:
        return False
    # check capture bottom
    if board.board[x_to, y_to + 1] & opponent_pawn_tile_state != 0 \
            and board.board[x_to, y_to + 2] & (own_tile_state | TileState.corner | throne_check) != 0:
        return False
    # check capture top
    if board.board[x_to, y_to - 1] & opponent_pawn_tile_state != 0 \
            and board.board[x_to, y_to - 2] & (own_tile_state | TileState.corner | throne_check) != 0:
        return False

    # see regular move method for a short explanation
    board.board[move_to] = (board.board[move_to] & TileState.throne) | \
                             (board.board[move_from] & (TileState.white | TileState.black | TileState.king))
    board.board[move_from] = board.board[move_from] & \
                                 ~(TileState.white | TileState.black | TileState.king)  # remove piece from tile

    board_bytes = board.board.tobytes()
    result = board_bytes in board.board_states_dict and board.board_states_dict[board_bytes] == 2
    board.board[move_from] = previous_from
    board.board[move_to] = previous_to
    return result


# returns a winning move for white in the sense that either the move itself or the next move wins the game for white
# checks made in this function:
#   1. king to corner
#   2. king next to corner
#   3. king to empty edge
#   4. king to (2,2) or symmetrical equivalent when there is no piece to capture the king and no piece on (2,1) and
#       (1,2)
def get_king_escape_move(board):
    king_moves = board.get_valid_actions_for_piece(board.king_position)
    king_x, king_y = board.king_position

    # check if king is on an edge
    if king_x == 1 or king_x == board.size or king_y == 1 or king_y == board.size:
        # moves to corner
        if (board.king_position, (1, 1)) in king_moves:
            return board.king_position, (1, 1)
        if (board.king_position, (1, board.size)) in king_moves:
            return board.king_position, (1, board.size)
        if (board.king_position, (board.size, 1)) in king_moves:
            return board.king_position, (board.size, 1)
        if (board.king_position, (board.size, board.size)) in king_moves:
            return board.king_position, (board.size, board.size)

    # moves next to corner
    if king_x in [1, 2, board.size - 1, board.size] or king_y in [1, 2, board.size - 1, board.size]:
        if (board.king_position, (1, king_y)) in king_moves:
            return board.king_position, (1, king_y)
        if (board.king_position, (king_x, 1)) in king_moves:
            return board.king_position, (king_x, 1)
        if (board.king_position, (board.size, king_y)) in king_moves:
            return board.king_position, (board.size, king_y)
        if (board.king_position, (king_x, board.size)) in king_moves:
            return board.king_position, (king_x, board.size)

    # moves to empty edge
    # top
    if (board.king_position, (1, king_y)) in king_moves \
            and sum(board.board[1, 3:board.size - 1]) == TileState.empty:
        return board.king_position, (1, king_y)
    # bottom
    if (board.king_position, (board.size, king_y)) in king_moves \
            and sum(board.board[board.size, 3:board.size - 1]) == TileState.empty:
        return board.king_position, (board.size, king_y)
    # left
    if (board.king_position, (king_x, 1)) in king_moves \
            and sum(board.board[3:board.size - 1, 1]) == TileState.empty:
        return board.king_position, (king_x, 1)
    # right
    if (board.king_position, (king_x, board.size)) in king_moves \
            and sum(board.board[3:board.size - 1, board.size]) == TileState.empty:
        return board.king_position, (king_x, board.size)

    # all the interesting moves are not dependent on the move the king makes
    black_move_end_points = [move_to for move_from, move_to in board.get_valid_actions(Player.black)]

    # moves to an edge where the king can escape during the next turn and no black piece can block it despite the
    # edge not being empty
    # top -> right
    if (board.king_position, (1, king_y)) in king_moves \
            and sum(board.board[1, king_y + 1:board.size]) == TileState.empty \
            and [to_position for to_position in black_move_end_points
                 if to_position in [(1, y) for y in range(king_y + 1, board.size)]] == []:
        return board.king_position, (1, king_y)
    # top -> left
    if (board.king_position, (1, king_y)) in king_moves \
            and sum(board.board[1, king_y - 1:1]) == TileState.empty \
            and [to_position for to_position in black_move_end_points
                 if to_position in [(1, y) for y in range(king_y - 1, 1)]] == []:
        return board.king_position, (1, king_y)
    # bottom -> right
    if (board.king_position, (board.size, king_y)) in king_moves \
            and sum(board.board[board.size, king_y + 1:board.size]) == TileState.empty \
            and [to_position for to_position in black_move_end_points
                 if to_position in [(board.size, y) for y in range(king_y + 1, board.size)]] == []:
        return board.king_position, (board.size, king_y)
    # bottom -> left
    if (board.king_position, (board.size, king_y)) in king_moves \
            and sum(board.board[board.size, king_y - 1:1]) == TileState.empty \
            and [to_position for to_position in black_move_end_points
                 if to_position in [(board.size, y) for y in range(king_y - 1, 1)]] == []:
        return board.king_position, (board.size, king_y)
    # left -> top
    if (board.king_position, (king_x, 1)) in king_moves \
            and sum(board.board[king_x - 1:1, 1]) == TileState.empty \
            and [to_position for to_position in black_move_end_points
                 if to_position in [(x, 1) for x in range(king_x - 1, 1)]] == []:
        return board.king_position, (king_x, 1)
    # left -> bottom
    if (board.king_position, (king_x, 1)) in king_moves \
            and sum(board.board[king_x + 1:board.size, 1]) == TileState.empty \
            and [to_position for to_position in black_move_end_points
                 if to_position in [(x, 1) for x in range(king_x + 1, board.size)]] == []:
        return board.king_position, (king_x, 1)
    # right -> top
    if (board.king_position, (king_x, board.size)) in king_moves \
            and sum(board.board[king_x - 1:1, board.size]) == TileState.empty \
            and [to_position for to_position in black_move_end_points
                 if to_position in [(x, board.size) for x in range(king_x - 1, 1)]] == []:
        return board.king_position, (king_x, board.size)
    # right -> bottom
    if (board.king_position, (king_x, board.size)) in king_moves \
            and sum(board.board[king_x + 1:board.size, board.size]) == TileState.empty \
            and [to_position for to_position in black_move_end_points
                 if to_position in [(x, board.size) for x in range(king_x + 1, board.size)]] == []:
        return board.king_position, (king_x, board.size)

    # moves to (2,2) and their symmetries, if there is no piece on (2,1) and (1,2) and no black piece that can
    # capture the king
    no_black_top = np.bitwise_or.reduce(board.board[1, 3: board.size - 1]) & TileState.black == TileState.empty
    no_black_bottom = np.bitwise_or.reduce(board.board[board.size, 3: board.size - 1]) & TileState.black == TileState.empty
    no_black_left = np.bitwise_or.reduce(board.board[3: board.size - 1, 1]) & TileState.black == TileState.empty
    no_black_right = np.bitwise_or.reduce(board.board[3: board.size - 1, board.size]) & TileState.black == TileState.empty
    if (board.king_position, (2, 2)) in king_moves \
            and board.board[(1, 2)] | board.board[(2, 1)] == TileState.empty \
            and (no_black_top or board.board[(3, 2)] != TileState.black) \
            and (no_black_left or board.board[(2, 3)] != TileState.black):
        return board.king_position, (2, 2)
    if (board.king_position, (board.size - 1, 2)) in king_moves \
            and board.board[(board.size, 2)] | board.board[(board.size - 1, 1)] == TileState.empty \
            and (no_black_bottom or board.board[(board.size - 2, 2)] != TileState.black) \
            and (no_black_left or board.board[(board.size - 1, 3)] != TileState.black):
        return board.king_position, (board.size - 1, 2)
    if (board.king_position, (2, board.size - 1)) in king_moves \
            and board.board[(1, board.size - 1)] | board.board[(2, board.size)] == TileState.empty \
            and (no_black_top or board.board[(2, board.size - 2)] != TileState.black) \
            and (no_black_right or board.board[(3, board.size - 1)] != TileState.black):
        return board.king_position, (2, board.size - 1)
    if (board.king_position, (board.size - 1, board.size - 1)) in king_moves \
            and board.board[(board.size, board.size - 1)] | board.board[(board.size - 1, board.size)] == TileState.empty \
            and (no_black_bottom or board.board[(board.size - 2, board.size - 1)] != TileState.black) \
            and (no_black_right or board.board[(board.size - 1, board.size - 2)] != TileState.black):
        return board.king_position, (board.size - 1, board.size - 1)

    return None


# returns a move that captures the king if possible, else returns None
# valid_actions is passed as an arguments so that it doesn't need to be calculated again
def get_king_capture_move(board, valid_actions):
    king_x, king_y = board.king_position
    king_capture_positions = []

    # check if king is on or next to throne
    if (board.board[king_x, king_y] | board.board[king_x + 1, king_y] | board.board[king_x - 1, king_y] |
            board.board[king_x, king_y + 1] | board.board[king_x, king_y - 1]) & TileState.throne != 0:
        for around_king_position in [(king_x + 1, king_y), (king_x - 1, king_y), (king_x, king_y + 1), (king_x, king_y - 1)]:
            if board.board[around_king_position] & (TileState.black | TileState.throne) == 0:
                if not king_capture_positions:
                    king_capture_positions = [around_king_position]
                else:
                    # at least two spots are empty, so no capture possible
                    return None
    else:
        if board.board[king_x + 1, king_y] == TileState.black and board.board[king_x - 1, king_y] == TileState.empty:
            king_capture_positions.append((king_x - 1, king_y))
        elif board.board[king_x - 1, king_y] == TileState.black and board.board[king_x + 1, king_y] == TileState.empty:
            king_capture_positions.append((king_x + 1, king_y))
        if board.board[king_x, king_y + 1] == TileState.black and board.board[king_x, king_y - 1] == TileState.empty:
            king_capture_positions.append((king_x, king_y - 1))
        elif board.board[king_x, king_y - 1] == TileState.black and board.board[king_x, king_y + 1] == TileState.empty:
            king_capture_positions.append((king_x, king_y + 1))

    if not king_capture_positions:
        for action in valid_actions:
            if action[1] in king_capture_positions:
                return action
    return None


# checks whether the next board after the given move would lead to a board where the opponent of the turn player
# could win the game
def would_next_board_lead_to_opponent_winning(board, move, turn_player):
    move_from, move_to = move
    x, y = move_to
    previous_from = board.board[move_from]
    previous_to = board.board[x, y]

    # see regular move method for a short explanation
    board.board[move_to] = (board.board[move_to] & TileState.throne) | \
                           (board.board[move_from] & (TileState.white | TileState.black | TileState.king))
    board.board[move_from] = board.board[move_from] & \
                             ~(TileState.white | TileState.black | TileState.king)  # remove piece from tile

    captured_pieces = []
    own_tile_state = TileState.white | TileState.king if turn_player == Player.white else TileState.black
    # TileState.white for Player.black, TileState.black for Player.white
    # this way is necessary because capturing the king works differently and is done further below
    opponent_pawn_tile_state = TileState.black if turn_player == Player.black else TileState.white

    throne_check = TileState.empty if board.board[board.king_position] & TileState.throne != 0 else TileState.throne
    # check capture bottom
    if board.board[x + 1, y] & opponent_pawn_tile_state != 0 \
            and board.board[x + 2, y] & (own_tile_state | TileState.corner | throne_check) != 0:
        board.board[x + 1, y] = TileState.empty
        captured_pieces.append((x + 1, y))

    # check capture top
    if board.board[x - 1, y] & opponent_pawn_tile_state != 0 \
            and board.board[x - 2, y] & (own_tile_state | TileState.corner | throne_check) != 0:
        board.board[x - 1, y] = TileState.empty
        captured_pieces.append((x - 1, y))

    # check capture right
    if board.board[x, y + 1] & opponent_pawn_tile_state != 0 \
            and board.board[x, y + 2] & (own_tile_state | TileState.corner | throne_check) != 0:
        board.board[x, y + 1] = TileState.empty
        captured_pieces.append((x, y + 1))

    # check capture left
    if board.board[x, y - 1] & opponent_pawn_tile_state != 0 \
            and board.board[x, y - 2] & (own_tile_state | TileState.corner | throne_check) != 0:
        board.board[x, y - 1] = TileState.empty
        captured_pieces.append((x, y - 1))

    if turn_player == Player.black:
        result = get_king_capture_move(board, board.get_valid_actions(Player.black)) is None
    else:
        result = get_king_escape_move(board) is None
    for position in captured_pieces:
        board.board[position] = opponent_pawn_tile_state
    board.board[move_from] = previous_from
    board.board[move_to] = previous_to
    return result
