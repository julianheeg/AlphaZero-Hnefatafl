import time
from pickle import Unpickler

import Arena
from tafl.TaflGame import TaflGame
from tafl.rendering.render_utils import room_to_rgb

filename = "../arena_replays/arena_replay_001.taflreplay"
# filename = "../demo_replays/demo_replay_01.taflreplay"
# filename = "../demo_replays/demo_replay_02.taflreplay"
wait_time = 1.5  # seconds between moves

class ReplayAgent:
    def __init__(self, filename, wait_time):
        with open(filename, "rb") as f:
            self.moves = Unpickler(f).load()
        self.turn_counter = -1
        self.wait_time = wait_time
        self.viewer = None

    def play(self, board, player):
        img = room_to_rgb(board, board.size)
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        time.sleep(self.wait_time)
        # input()
        self.turn_counter += 1
        return self.moves[self.turn_counter]


# this code is run
g = TaflGame(7, False)
replay = ReplayAgent(filename, wait_time).play
arena = Arena.Arena(replay, replay, g, replay=True)
arena.playGames(1, profile=False)

