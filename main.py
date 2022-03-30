from tafl.pytorch.NNet import NNetWrapper as nn
from Coach import Coach
from tafl.TaflGame import TaflGame
from utils import dotdict

args = dotdict({
    'numIters': 1000,
    'numEps': 100,           # 200
    'tempThreshold': 15,    # 700
    'updateThreshold': 0.57,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 800,      # 900
    'arenaCompare': 50,     # 100
    'cpuct': 1,
    'prune': True,
    'prune_starting_prob': 0.75,
    'prune_prob_gain_per_iteration': 0.05,

    'checkpoint': './temp/',
    'load_model': True,
    'split_player_examples_into_episodes': False,

    'load_folder_file_white': ('./temp/', 'best_white.pth.tar'),
    'load_folder_file_black': ('./temp/', 'best_black.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'train_both': True,
    'train_black_first': False,
    'skip_first_self_play': False,
    'train_other_network_threshold': 1,    # compared with (network that is currently trained wins)/(other network wins)
                                           # toggles the network being trained when threshold is reached
    'profile_coach': False,
    'profile_arena': False,
})

if __name__=="__main__":
    #  g = OthelloGame(6)
    g = TaflGame(7, args.prune)
    white_nnet = nn(g)
    black_nnet = nn(g)

    if args.load_model:
        white_nnet.load_checkpoint(args.load_folder_file_white[0], args.load_folder_file_white[1])
        black_nnet.load_checkpoint(args.load_folder_file_black[0], args.load_folder_file_black[1])
    else:
        white_nnet.save_checkpoint(folder=args.checkpoint, filename='temp_white.pth.tar')
        black_nnet.save_checkpoint(folder=args.checkpoint, filename='temp_black.pth.tar')

    c = Coach(g, white_nnet, black_nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    else:
        c.load_expert_examples()
    c.learn()
