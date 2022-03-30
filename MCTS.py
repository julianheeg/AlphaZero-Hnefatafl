import copy
import math
import random
import time

import numpy as np

from tafl.TaflBoard import Player
from tafl.TaflGame import MovementType, action_conversion__index_to_explicit

EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, white_nnet, black_nnet, args):
        self.game = game
        self.size = game.getBoardSize()[0]
        self.white_nnet = white_nnet
        self.black_nnet = black_nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, this_player, temp=1, time=None):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        if time is None:
            for i in range(self.args.numMCTSSims):
                # print("    search number " + str(i))
                self.search(copy.deepcopy(canonicalBoard), this_player)
        else:
            timeout = time.time() + time
            while time.time() < timeout:
                self.search(copy.deepcopy(canonicalBoard), this_player)

        # bytes are much faster
        s = self.game.stringRepresentation(canonicalBoard) + this_player.to_bytes(1, byteorder='big', signed=True)
        # s = self.game.stringRepresentation(canonicalBoard) + str(this_player)   # this addition is needed so that
        # the search algorithm doesn't get confused when the same board state as before is reached, but it's the
        # other player's turn
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            maximum = max(counts)
            argmaxs = [(index, count) for index, count in enumerate(counts) if count == maximum]
            bestA, count = random.choice(argmaxs)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, canonicalBoard, this_player):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        def player_net(player):
            return self.white_nnet if player == Player.white else self.black_nnet

        value_stack = []
        next_player = this_player
        iteration = 0

        # build stack
        while True:
            # workaround because in some cases this function doesn't stop (bug might exist in the original version also)
            # print("             iteration " + str(iteration))
            iteration += 1
            if iteration > 1000:
                print("more MCTS search iterations than the maximum, breaking out of possibly infinite loop!")
                return

            # workaround end

            # bytes are much faster
            s = self.game.stringRepresentation(canonicalBoard) + next_player.to_bytes(1, byteorder='big', signed=True)
            # s = self.game.stringRepresentation(canonicalBoard) + str(next_player)   # this addition is needed so that
            # the search algorithm doesn't get confused when the same board state as before is reached, but it's the
            # other player's turn

            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(canonicalBoard, next_player)
            if self.Es[s] != 0:
                # terminal node
                last_iteration_v = -self.Es[s]
                break

            if s not in self.Ps:
                # leaf node
                valids = self.game.getValidMoves(canonicalBoard, next_player)

                # occurrences = np.zeros(self.size * self.size * self.size * 2)
                # for index, action in enumerate(valids):
                #     if action == 1 and index != self.size * self.size * self.size * 2:
                #         explicit = action_conversion__index_to_explicit(index, self.size)
                #        occurrences[index] = 1 if canonicalBoard.would_next_board_be_second_third(2, explicit) else 0

                self.Ps[s], v = player_net(next_player).predict(canonicalBoard, np.array([canonicalBoard.king_position[0], canonicalBoard.king_position[1]]))
                # valids = self.game.getValidMoves(canonicalBoard, next_player)
                self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[s])
                if sum_Ps_s > 0:
                    self.Ps[s] /= sum_Ps_s  # renormalize
                else:
                    # if all valid moves were masked make all valid moves equally probable

                    # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                    # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                    print("All valid moves were masked, do workaround.")
                    print(valids)
                    self.Ps[s] = self.Ps[s] + valids
                    self.Ps[s] /= np.sum(self.Ps[s])

                self.Vs[s] = valids
                self.Ns[s] = 0
                last_iteration_v = -v
                break

            valids = self.Vs[s]
            cur_best = -float('inf')
            best_act = -1

            # pick the action with the highest upper confidence bound
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    if (s, a) in self.Qsa:
                        u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                                    1 + self.Nsa[(s, a)])
                    else:
                        u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act

            next_s, next_player = self.game.getNextState(canonicalBoard, next_player, a)
            canonicalBoard = self.game.getCanonicalForm(next_s, next_player)

            value_stack.append((s, a))

        # take from stack
        while len(value_stack) > 0:
            s, a = value_stack.pop()

            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + last_iteration_v) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1

            else:
                self.Qsa[(s, a)] = last_iteration_v
                self.Nsa[(s, a)] = 1

            self.Ns[s] += 1
            last_iteration_v = -last_iteration_v

        return -last_iteration_v
