import sys
import numpy as np
import tkinter as tk
import copy
import random
from typing import *
import math
from tqdm import tqdm
import pickle


class UtilGobang:
    def __init__(self, board_size, bound):
        self.board_size, self.bound = board_size, bound
        self.board = np.zeros((board_size, board_size))
        self.window, self.canvas, self.cell_size = None, None, None
        self.action_space = [(i, j) for i in range(board_size) for j in range(board_size)]
        self.gamma = 0.5
        self.Q = {}
        self.s_a_visited = {}
        self.kernel = np.array([[1, 1],
                                [1, 1]])

    def restart(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.action_space = [(i, j) for i in range(self.board_size) for j in range(self.board_size)]

    def draw_board(self):
        self.window = tk.Tk()
        self.window.title("Gobang Board")
        self.canvas = tk.Canvas(self.window, width=400, height=400)
        self.canvas.pack()
        self.cell_size = 400 // self.board_size
        self.visualize_board()
        self.window.mainloop()

    def save(self, q_name):
        with open(q_name, 'wb') as f:
            pickle.dump(self.Q, f)

    def load(self, q_name):
        with open(q_name, 'rb') as f:
            self.Q = pickle.load(f)

    def learning(self, episodes=1000):
        for _ in tqdm(range(episodes)):
            self.restart()
            while True:
                color, end_up_gaming = self.update_board(episode_num=_)
                if end_up_gaming is True:
                    break
        self.restart()
        self.save(f"Q.pkl")
        print("learning ended.")

    def visualize_board(self):
        self.canvas.delete("all")
        color, end_up_gaming = self.update_board(learning=False)
        text = "Black wins." if color == 1 else "White wins." if color == 2 else "Tie." if color == 0 else None
        if text is not None:
            message = tk.Message(self.window, text=text, width=100)
            message.pack()
        for i in range(self.board_size):
            for j in range(self.board_size):
                x1 = i * self.cell_size
                y1 = j * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                if self.board[i][j] == 1:
                    self.canvas.create_oval(x1, y1, x2, y2, fill="black")
                elif self.board[i][j] == 2:
                    self.canvas.create_oval(x1, y1, x2, y2, fill="white")
        if end_up_gaming is True:
            print("Game ended.")
        else:
            self.window.after(1000, self.visualize_board)

    def judge_legal_position(self, x, y) -> bool:
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def count_max_connections_for_single_color(self, state, color) -> int:
        directions = [(1, 1), (1, 0), (0, 1), (1, -1)]
        max_connections = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                for direction_x, direction_y in directions:
                    current_pos_x, current_pos_y = i, j
                    current_connections = 0
                    while self.judge_legal_position(current_pos_x, current_pos_y):
                        if state[current_pos_x][current_pos_y] == color:
                            current_connections += 1
                        else:
                            break
                        current_pos_x += direction_x
                        current_pos_y += direction_y
                    max_connections = max(current_connections, max_connections)
        return max_connections

    def count_max_connections(self, state) -> Tuple[int, int]:
        return (self.count_max_connections_for_single_color(state, 1),
                self.count_max_connections_for_single_color(state, 2))

    @staticmethod
    def array_to_hashable(array):
        return tuple([tuple(r) for r in array])

    @staticmethod
    def hashable_to_array(hash_key):
        return np.array([list(r) for r in hash_key])

    """
    During learning, double sampling techniques were utilized to decrease the error of Q(s, a). Specifically, 
    we initially sample a state s, based on which we collect multiple actions repeatedly to ensure better explorations.
    """

    def update_board(self, episode_num: Optional[int] = sys.maxsize,
                     learning: bool = True, attempt: int = 8) -> Tuple[int, bool]:
        action_space = copy.deepcopy(self.action_space)
        (next_state_free_of_noise, next_state,
         current_black_connection, current_white_connection,
         next_black_connection, next_white_connection, reward) = [None, None, None, None, None, None, None]
        for _ in range(attempt if learning else 1):
            self.action_space = copy.deepcopy(action_space)
            action, noise = self.sample_action_and_noise(eps=1.0 / math.log(math.sqrt(episode_num + 1) + 1))
            (current_black_connection, current_white_connection,
             next_black_connection, next_white_connection, reward) = self.get_connection_and_reward(action, noise)
            next_state = self.get_next_state(action, noise)
            next_state_free_of_noise = self.get_next_state(action, None)
            if learning:
                self.q_learning_update(self.board, action, next_state, reward)
        self.board = next_state_free_of_noise if next_black_connection >= self.bound else next_state
        return ((1, True) if next_black_connection >= self.bound else
                (2, True) if next_white_connection >= self.bound else
                (0, True) if len(self.action_space) == 0 else
                (-1, False))

    def evaluate_agent_performance(self, episodes=1000):
        black_wins, white_wins, ties = 0, 0, 0
        for _ in tqdm(range(episodes)):
            self.restart()
            while True:
                color, end_up_gaming = self.update_board(learning=False)
                black_wins, white_wins, ties = ((black_wins, white_wins, ties) if end_up_gaming is False else
                                                (black_wins, white_wins, ties + 1) if color == 0 else
                                                (black_wins + 1, white_wins, ties) if color == 1 else
                                                (black_wins, white_wins + 1, ties))
                if end_up_gaming:
                    print(f"Black wins: {black_wins}, white wins: {white_wins}, and ties: {ties}.")
                    print(
                        f"The evaluated winning probability for the black pieces is "
                        f"{black_wins / (black_wins + white_wins + ties)}."
                    )
                    break
        self.restart()
        print(f"Evaluation finished. Black wins: {black_wins}, white wins: {white_wins}, and ties: {ties}.")
        print(
            f"The evaluated winning probability for the black pieces is "
            f"{black_wins / (black_wins + white_wins + ties)}."
        )
