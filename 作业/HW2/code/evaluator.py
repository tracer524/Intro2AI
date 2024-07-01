from submission import *

if __name__ == "__main__":
    chess_board = Gobang(board_size=4, bound=4)
    chess_board.load("Q.pkl")
    chess_board.evaluate_agent_performance(episodes=10000)
