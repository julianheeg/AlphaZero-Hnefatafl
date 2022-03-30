import argparse
import os
import shutil
import time
import numpy as np
import sys

from NeuralNet import NeuralNet
from pytorch_classification.utils import AverageMeter
from pytorch_classification.utils.progress.progress.bar import Bar
from utils import dotdict

sys.path.append('../../')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from .TaflNNet import TaflNNet as tnnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'num_scalar_values': 2,  # ## bei Änderung der Anzahl der eingegebenen skalaren Werte:
    #                               1. Hier die richtige Anzahl eintragen
    #                               2. Bei TaflGame.getSymmetries(...) Methode die zusätzlichen Werte ins letzte Tupel eintragen
    #                                       (dort wo aktuell "(x[1],)" steht)
    #                               3. Bei MCTS.search(...) in der Zeile, in der
    #                                     "self.Ps[s], v = self.nnet.predict(canonicalBoard, np.array([next_player]))"
    #                                  steht, in die Liste im Array-Constructor die zuseätzlichen Werte eintragen
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = tnnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs, scalar_values = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                scalar_values = torch.FloatTensor(np.array(scalar_values).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs, scalar_values \
                        = boards.contiguous().cuda(), target_pis.contiguous().cuda(),target_vs.contiguous().cuda(),\
                          scalar_values.contiguous().cuda()
                boards, target_pis, target_vs, scalar_values \
                    = Variable(boards), Variable(target_pis), Variable(target_vs), Variable(scalar_values)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet(boards, scalar_values)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.data.item(), boards.size(0))
                v_losses.update(l_v.data.item(), boards.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()


    def predict(self, board, scalar_values):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.board[1: self.board_x + 1, 1: self.board_y + 1].astype(np.float64))
        scalar_values = torch.FloatTensor(scalar_values)
        if args.cuda:
            board = board.contiguous().cuda()
            scalar_values = scalar_values.contiguous().cuda()
        with torch.no_grad():
            board = Variable(board)
            board = board.view(1, self.board_x, self.board_y)
            scalar_values = Variable(scalar_values)
            scalar_values = scalar_values.view(1, -1)
            # print(board)
            # print(player)
            self.nnet.eval()
            pi, v = self.nnet(board, scalar_values)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
