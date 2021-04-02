import chess
import chess.engine

# engine = chess.engine.SimpleEngine.popen_uci("C:/Users/b-the\Desktop\python\Project-KapiteinKlauw\data\stockfish\Stockfish-sf_12")
engine = chess.engine.SimpleEngine.popen_uci("C:/Users/b-the/Desktop/python/Project-KapiteinKlauw/data/stockfish/a.exe")
board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')


def get_best_move():
    result = engine.play(board, chess.engine.Limit(time=0.5))
    board.push(result.move)
    return result.move

def is_move_legal(move_made):
    move = chess.Move.from_uci(move_made)
    return move in board.legal_moves

def update_board(move_made):
    move = chess.Move.from_uci(move_made)
    board.push(move)

def print_board():
    print(board)

def get_move_direction(move):
    print(move)
    turn = board.turn

    tile1 = move[0:2]
    tile2 = move[2:4]
    ptile1 = chess.parse_square(tile1)
    ptile2 = chess.parse_square(tile2)

    # print(f'color {board.color_at(ptile1)}')
    # print(f'color {board.color_at(ptile2)}')
    print(f'turn {turn}, tile1 {tile1}, tile2 {tile2}')

    if turn:
    #     White's turn, white possible moves: W -> B or W -> None
        zerotile = (tile1 if board.color_at(ptile1) else tile2)
        onetile = (tile2 if board.color_at(ptile1) else tile1)
    else:
        zerotile = (tile1 if not board.color_at(ptile1) else tile2)
        onetile = (tile2 if not board.color_at(ptile1) else tile1)

    move = f'{zerotile}{onetile}'
    print(f'Move direction: {move}')
    return move