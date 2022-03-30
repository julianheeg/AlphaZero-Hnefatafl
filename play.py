from MCTS import MCTS
from tafl.TaflGame import TaflGame
from tafl.pytorch.NNet import NNetWrapper as nn
from utils import dotdict
import numpy as np


def get_player(time):
    g = TaflGame(7, True)
    white_nnet = nn(g)
    black_nnet = nn(g)
    white_nnet.load_checkpoint('./tafl_model_1/', 'white.pth.tar')
    black_nnet.load_checkpoint('./tafl_model_1/', 'white.pth.tar')
    args = dotdict({'numMCTSSims': 10000, 'cpuct': 1.1})
    mcts = MCTS(g, white_nnet, black_nnet, args)
    return lambda board, turn_player: np.argmax(mcts.getActionProb(board, turn_player, temp=0, time=time))
