import cProfile
import copy
import math
from collections import deque
from multiprocessing.pool import Pool

from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle

from tafl.TaflBoard import Player
from trainingData import read_data


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, white_nnet, black_nnet, args):
        self.game = game
        self.white_nnet = white_nnet
        self.black_nnet = black_nnet
        self.white_pnet = self.white_nnet.__class__(self.game)  # the competitor network
        self.black_pnet = self.black_nnet.__class__(self.game)
        self.args = args
        self.mcts = MCTS(self.game, self.white_nnet, self.black_nnet, self.args)
        # self.trainExamplesHistory = []  ###########
        self.trainExamplesHistory_white = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.trainExamplesHistory_black = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples_white = []
        trainExamples_black = []
        # trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            # print("turn " + str(episodeStep))
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            try:
                pi = self.mcts.getActionProb(canonicalBoard, self.curPlayer, temp=temp)
            except ZeroDivisionError:
                print("ZeroDivisionError while building training example. continue with next iteration")
                return [], []
            sym = self.game.getSymmetries(canonicalBoard, pi, canonicalBoard.king_position)

            player_train_examples = trainExamples_white if self.curPlayer == Player.white else trainExamples_black
            for b,p, scalar_values in sym:
                player_train_examples.append([b, self.curPlayer, p, scalar_values])

            action = np.random.choice(len(pi), p=pi)
            if action == 0:
                print(pi)

            board.print_game_over_reason = False
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            board.print_game_over_reason = False

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                # if board.outcome == Outcome.black:
                #     print(" black wins")
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer)), x[3]) for x in trainExamples_white], \
                       [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer)), x[3]) for x in trainExamples_black]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        self.game.prune_prob = self.args.prune_starting_prob
        train_black = self.args.train_black_first

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.args.skip_first_self_play or i>1:
                iterationTrainExamples_white = deque([], maxlen=self.args.maxlenOfQueue)
                iterationTrainExamples_black = deque([], maxlen=self.args.maxlenOfQueue)
    
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                if self.args.profile_coach:
                    prof = cProfile.Profile()
                    prof.enable()

                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.white_nnet, self.black_nnet, self.args)   # reset search tree

                    white_examples, black_examples = self.executeEpisode()

                    iterationTrainExamples_white += white_examples
                    iterationTrainExamples_black += black_examples

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()
                if self.args.profile_coach:
                    prof.disable()
                    prof.print_stats(sort=2)

                # save the iteration examples to the history 
                self.trainExamplesHistory_white.append(iterationTrainExamples_white)
                self.trainExamplesHistory_black.append(iterationTrainExamples_black)
                
            while len(self.trainExamplesHistory_white) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory_white), " => remove the oldest trainExamples")
                self.trainExamplesHistory_white.pop(0)
                self.trainExamplesHistory_black.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)

            # training new network, keeping a copy of the old one
            self.white_nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp_white.pth.tar')
            self.black_nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp_black.pth.tar')
            self.white_pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp_white.pth.tar')
            self.black_pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp_black.pth.tar')

            pmcts = MCTS(self.game, self.white_pnet, self.black_pnet, self.args)

            if not self.args.train_both:
                if train_black:
                    # shuffle examples before training
                    trainExamples = []
                    for e in self.trainExamplesHistory_black:
                        trainExamples.extend(e)
                    shuffle(trainExamples)
                    self.black_nnet.train(trainExamples)
                else:
                    # shuffle examples before training
                    trainExamples = []
                    for e in self.trainExamplesHistory_white:
                        trainExamples.extend(e)
                    shuffle(trainExamples)
                    self.white_nnet.train(trainExamples)
            else:
                # shuffle examples before training
                trainExamples = []
                for e in self.trainExamplesHistory_black:
                    trainExamples.extend(e)
                shuffle(trainExamples)
                self.black_nnet.train(trainExamples)

                # shuffle examples before training
                trainExamples = []
                for e in self.trainExamplesHistory_white:
                    trainExamples.extend(e)
                shuffle(trainExamples)
                self.white_nnet.train(trainExamples)

            nmcts = MCTS(self.game, self.white_nnet, self.black_nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda board, turn_player: np.argmax(pmcts.getActionProb(board, turn_player, temp=0)),
                          lambda board, turn_player: np.argmax(nmcts.getActionProb(board, turn_player, temp=0)),
                          self.game)
            pwins, nwins, draws, pwins_white, pwins_black, nwins_white, nwins_black \
                = arena.playGames(self.args.arenaCompare, self.args.profile_arena)

            print('NEW/PREV WINS (white, black) : (%d,%d) / (%d,%d) ; DRAWS : %d' % (nwins_white, nwins_black, pwins_white, pwins_black, draws))

            if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold \
                    or nwins_black < pwins_black or nwins_white < pwins_white:
                print('REJECTING NEW MODEL')
                if not self.args.train_both:
                    if train_black:
                        self.black_nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp_black.pth.tar')
                    else:
                        self.white_nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp_white.pth.tar')
                else:
                    self.black_nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp_black.pth.tar')
                    self.white_nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp_white.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                if not self.args.train_both:
                    if train_black:
                        # self.black_nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i, Player.black))
                        self.black_nnet.save_checkpoint(folder=self.args.checkpoint, filename='best_black.pth.tar')
                        # if nwins_white == 0 or nwins_black / nwins_white >= self.args.train_other_network_threshold:
                        #     train_black = False
                        print("training white neural net next")
                        train_black = False
                    else:
                        # self.white_nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i, Player.white))
                        self.white_nnet.save_checkpoint(folder=self.args.checkpoint, filename='best_white.pth.tar')
                        # if nwins_black == 0 or nwins_white / nwins_black > self.args.train_other_network_threshold:
                        #     train_black = True
                        print("training black neural net next")
                        train_black = True
                else:
                    self.black_nnet.save_checkpoint(folder=self.args.checkpoint, filename='best_black.pth.tar')
                    self.white_nnet.save_checkpoint(folder=self.args.checkpoint, filename='best_white.pth.tar')
                self.game.prune_prob += self.args.prune_prob_gain_per_iteration
                self.args.arenaCompare = math.floor(self.args.arenaCompare * 1.05)
            # self.args.numEps = math.floor(self.args.numEps * 1.1)
            self.args.numMCTSSims = math.floor(self.args.numMCTSSims * 1.1)
            print("prune probability: " + str(self.game.prune_prob) + ", episodes: " + str(self.args.numEps) +
                  ", sims: " + str(self.args.numMCTSSims) + ", arena compare: " + str(self.args.arenaCompare))

    def getCheckpointFile(self, iteration, player=None):
        return 'checkpoint_' + ('white_' if player == Player.white else 'black_' if player == Player.black else '') + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_white = os.path.join(folder, "training_white.examples")
        filename_black = os.path.join(folder, "training_black.examples")
        with open(filename_white, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory_white)
        with open(filename_black, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory_black)

    def loadTrainExamples(self):
        folder = self.args.checkpoint
        filename_white = os.path.join(folder, "training_white.examples")
        filename_black = os.path.join(folder, "training_black.examples")
        if not os.path.isfile(filename_white) or not os.path.isfile(filename_black):
            print(filename_white)
            print(filename_black)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(filename_white, "rb") as f:
                self.trainExamplesHistory_white = Unpickler(f).load()
            with open(filename_black, "rb") as f:
                self.trainExamplesHistory_black = Unpickler(f).load()
            # examples based on the model were already collected (loaded)

    def load_expert_examples(self):
        white, black = read_data(self.args)
        self.trainExamplesHistory_white.extend(white)
        self.trainExamplesHistory_black.extend(black)
