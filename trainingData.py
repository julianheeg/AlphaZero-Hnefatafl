import numpy as np
import pickle
from collections import deque
from tafl.TaflBoard import TaflBoard, Player, Outcome, TileState
from tafl.TaflGame import TaflGame, MovementType, action_conversion__explicit_to_index


# reads the data and removes all the games which do not clearly show a winner or are inconsistent with our rules
def generate_training_example(game, board, action, turn_player):
    king_position = board.king_position
    pi = np.zeros(7*7*7*2+1)

    (x_from, y_from), (x_to, y_to) = action
    movement_type = MovementType.horizontal if y_from == y_to else MovementType.vertical
    to = x_to if movement_type == MovementType.horizontal else y_to
    index = (((x_from - 1) * 7 + y_from - 1) * 7 + to - 1) * 2 + movement_type  # all coordinates -1 because of the border
    assert 0 <= index < 7 * 7 * 7 * 2

    pi[index] = 1
    return game.getSymmetries(board, pi, king_position)


def read_data(args):
    def king_capture_check(board):
        # check capture king
        first = False
        second = False
        king_x, king_y = board.king_position
        # check: (king is on or next to throne and surrounded on all for sides)
        # or (between to black pieces in vertical direction)
        # or (between to black pieces in horizontal direction)
        if (board.board[king_x, king_y] | board.board[king_x + 1, king_y] | board.board[king_x - 1, king_y] |
            board.board[king_x, king_y + 1] | board.board[king_x, king_y - 1]) & TileState.throne != 0:
            if board.board[king_x + 1, king_y] & (TileState.black | TileState.throne) != 0 \
                    and board.board[king_x - 1, king_y] & (TileState.black | TileState.throne) != 0 \
                    and board.board[king_x, king_y + 1] & (TileState.black | TileState.throne) != 0 \
                    and board.board[king_x, king_y - 1] & (TileState.black | TileState.throne) != 0:
                first = True
        elif board.board[king_x + 1, king_y] & TileState.black != 0 \
                and board.board[king_x - 1, king_y] & TileState.black != 0 \
                or board.board[king_x, king_y + 1] & TileState.black != 0 \
                and board.board[king_x, king_y - 1] & TileState.black != 0:
            second = True
        return "throne check: %s, other check: %s"%(first, second)

    training_data = pickle.load(open("full_game_stats.p", "rb"))
    outcomes = training_data['outcome']
    games = training_data['games']

    training_data_white = deque([], maxlen=args.maxlenOfQueue)
    training_data_black = deque([], maxlen=args.maxlenOfQueue)
    training_data_white_list = []
    training_data_black_list = []
    trainExamples_white = []
    trainExamples_black = []

    game = TaflGame(7, args.prune)

    assert len(outcomes) == len(games)

    move_conversion_table = {
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
    }

    outcome_conversion_table = {
        'black won': Outcome.black,
        'white won': Outcome.white,
    }

    usable_games = 0

    for i in range(len(games)):
        # filter out ongoing and resigned games
        if outcomes[i] == 'ongoing' or 'resigned' in games[i] or 'timeout' in games[i]:
            continue

        # print("---------------------------------------------" + str(i))
        # if i in np.array([15, 19, 99, 366, 551, 557, 593, 690, 832, 873, 960, 1034, 1039, 1041]):  # no capture against throne
        #    continue

        if i in np.array([136, 143, 327, 387, 484, 571, 1089]):  # wrong format
            continue

        if i in np.array([14, 16, 98, 103, 132, 218, 307, 431, 432, 433, 449, 473, 500, 514, 516, 525, 536, 550, 569, 595, 621,
                 623, 679, 684, 711, 728, 736, 763, 815, 825, 839, 849, 872, 878, 896, 904, 919, 942, 987, 995, 1020,
                 1046, 1058, 1067, 1097, 1099, 1110, 1113, 1120, 1121, 1124, 1125]):   # king is captured against the corner
            continue

        if i in np.array([247, 305, 333, 428, 458, 486]):  # game goes on although same board state has occurred 3 times
            continue
        usable_games += 1

        board = TaflBoard(7)
        board.print_game_over_reason = False
        turn_player = Player.black
        for string in games[i]:
            try:
                action = ((move_conversion_table[string[0]], move_conversion_table[string[1]]),
                          (move_conversion_table[string[3]], move_conversion_table[string[4]]))
            except:
                print(games[i])
                print(i)
                raise Exception
            # print(str(action) + "  " + string)
            assert board.outcome == Outcome.ongoing, str(i)

            symmetries = generate_training_example(game, board, action, turn_player)
            player_train_examples = trainExamples_white if turn_player == Player.white else trainExamples_black
            for b, p, scalar_values in symmetries:
                player_train_examples.append([b, p, scalar_values])

            board.do_action(action, turn_player)
            # print(board)
            turn_player *= -1

        assert outcome_conversion_table[outcomes[i]] == board.outcome, "\n" + str(board) + "\nexpected: " \
                                   + str(board.outcome) + ", actual: " + str(outcome_conversion_table[outcomes[i]]) \
                                   + "\n" + king_capture_check(board) + "\n example number:" + str(i)

        training_data_white += [(x[0], x[1], board.outcome, x[2]) for x in trainExamples_white]
        training_data_black += [(x[0], x[1], board.outcome, x[2]) for x in trainExamples_black]

        # split up into list format every "numEps" games
        if args.split_player_examples_into_episodes and usable_games % args.numEps == 0:
            training_data_white_list.append(training_data_white)
            training_data_black_list.append(training_data_black)
            training_data_white = deque([], maxlen=args.maxlenOfQueue)
            training_data_black = deque([], maxlen=args.maxlenOfQueue)

    training_data_white_list.append(training_data_white)
    training_data_black_list.append(training_data_black)
    return training_data_white_list, training_data_black_list
