import cProfile
import os
from pickle import Pickler

from pytorch_classification.utils import Bar, AverageMeter
import time


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None, replay=False):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.replay = replay
        self.game_id = 0

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        actions = []
        while self.game.getGameEnded(board, curPlayer)==0:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer), curPlayer)

            # valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),curPlayer)

            # don't know why it works when we just comment this out...
            # if valids[action] == 0:
            #    print("\nArena bug occured in turn " + str(it) + ":\naction: "
            #          + str(self.game.action_conversion__index_to_explicit(action))
            #          + ", turn player: " + str(Player(curPlayer)))
            #    print("board:\n" + str(board))
            #    assert 1. in self.game.getValidMoves(board, curPlayer)
            #    assert valids[action] > 0
            #    return None
            board.print_game_over_reason = False
            if self.replay:
                board.print_game_over_reason = True
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            board.print_game_over_reason = False
            actions.append(action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        if not self.replay:
            if not os.path.exists('./arena_replays/'):
                os.makedirs('./arena_replays/')
            with open("./arena_replays/arena_replay_" + str(self.game_id).zfill(3) + ".taflreplay", "wb+") as f:
                Pickler(f).dump(actions)
            self.game_id += 1
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, profile, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        if self.replay:
            self.playGame()
            return None

        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        oneWhiteWon = 0     # number of times the first player won as white
        oneBlackWon = 0     # number of times the first player won as black
        twoWhiteWon = 0     # number of times the second player won as white
        twoBlackWon = 0     # number of times the second player won as black
        if profile:
            prof = cProfile.Profile()
            prof.enable()
        for _ in range(num):
            gameResult = None
            while gameResult is None:
                gameResult = self.playGame(verbose=verbose)
            if gameResult==1:
                oneWon+=1
                oneBlackWon+=1
            elif gameResult==-1:
                twoWon+=1
                twoWhiteWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1
        
        for _ in range(num):
            gameResult = None
            while gameResult is None:
                gameResult = self.playGame(verbose=verbose)
            if gameResult==-1:
                oneWon+=1
                oneWhiteWon+=1
            elif gameResult==1:
                twoWon+=1
                twoBlackWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
            
        bar.finish()
        if profile:
            prof.disable()
            prof.print_stats(sort=2)

        return oneWon, twoWon, draws, oneWhiteWon, oneBlackWon, twoWhiteWon, twoBlackWon
