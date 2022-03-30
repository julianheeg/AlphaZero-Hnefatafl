import numpy as np
import pkg_resources
from scipy import misc
from PIL import Image

# your own image operations


def room_to_rgb(board, board_size):
    """
    Creates an RGB image of the room.
    :param room:
    :param room_structure:
    :return:
    """
    resource_package = __name__
    piece_unicode = {1: "♟", 2: "♟", 3: "♛"}
    # Load images, representing the corresponding situation
    white_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'whiteplayer1.png')))
    white = Image.open(white_filename).convert('RGB')
    #white =piece_unicode(1);
    black_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'blackplayer.png')))
    black = Image.open(black_filename).convert('RGB')

    king_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'king1.png')))
    king= Image.open(king_filename).convert('RGB')

    corner_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'corner.png')))
    corner = Image.open(corner_filename).convert('RGB')

    border_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'border.png')))
    border = Image.open(border_filename).convert('RGB')

    empty_filename = pkg_resources.resource_filename(resource_package,'/'.join(('surface', 'emptyfield1.png')))
    empty = Image.open(empty_filename).convert('RGB')

    throne_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'throne.png')))
    throne = misc.imread(throne_filename)

    # surfaces = [empty, white, black, king, throne, corner, border]
    surfaces = {0: empty, 1: white, 2: black, 4: king, 8: throne, 12: king, 16: corner, 32: border}
    # Assemble the new rgb_room, with all loaded images
    room_rgb = np.zeros(shape=((board_size + 2) * 32, (board_size + 2) * 32, 3), dtype=np.uint8)
    for i in range(board_size + 2):
        x_i = i * 32
        for j in range(board_size + 2):
            y_j = j * 32
            surfaces_id = int(board.board[i, j])
            room_rgb[x_i:(x_i + 32), y_j:(y_j + 32), :] = surfaces[surfaces_id]

    return room_rgb
