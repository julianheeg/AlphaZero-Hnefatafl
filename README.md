# AlphaZero-Hnefatafl
A machine learning project to train an agent to play Hnefatafl, a historic Scandinavian board game

# Description

This project was made as part of a machine learning seminar at University during the winter semester 2018/2019. The task was to create an agent playing Hnefatafl using Google's AlphaZero algorithm.

The game is played by two players on a 7x7 chess-like board. Initially, white has 5 pieces arranged in the middle of the board, and black has 8 pieces arranged at the sides. The white piece in the center of the board is the king, sitting upon the throne.
White's goal is to move the king to one of the four corners. Black's goal is to capture the king by surrounding him on all four sides. 
 
The players alternate making moves. Each piece can move like the rook in chess, however, the corners as well as the throne cannot be occupied by a piece other than the king. Black can capture white pieces by moving a black piece next to a white piece while there is another black piece, a corner, throne, or edge of the board on the opposite side of the white piece, and vice versa, essentially "trapping" the opponent's piece between two own pieces or an own piece and a "wall". When the same board state occurs for the third time during a game, the opponent also automatically wins.

Unfortunately, the project turned out rather unsuccesful. The reason is that  black's victory condition needs much more specific movements to achieve than white's. This is a problem since the neural network is initialized with random weights, leading to games that are won basically always by white. Hence, it is difficult to train the neural net because for the early stages of training, almost all actions lead to the same outcome, and are thus considered almost equally good.
