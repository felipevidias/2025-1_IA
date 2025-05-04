import tkinter as tk
from tkinter import messagebox
import random
import heapq
from collections import deque
import time

# --------- Classe Lógica ---------
class EightPuzzle:
    def __init__(self):
        self.goal = list(range(1, 9)) + [0]
        self.state = self.shuffle()

    def shuffle(self):
        state = self.goal[:]
        while True:
            random.shuffle(state)
            if self.is_solvable(state) and state != self.goal:
                return state

    def is_solvable(self, state):
        inv_count = 0
        for i in range(8):
            for j in range(i + 1, 9):
                if state[i] and state[j] and state[i] > state[j]:
                    inv_count += 1
        return inv_count % 2 == 0

    def move(self, direction):
        index = self.state.index(0)
        row, col = divmod(index, 3)
        if direction == "Up" and row > 0:
            self.swap(index, index - 3)
        elif direction == "Down" and row < 2:
            self.swap(index, index + 3)
        elif direction == "Left" and col > 0:
            self.swap(index, index - 1)
        elif direction == "Right" and col < 2:
            self.swap(index, index + 1)

    def swap(self, i, j):
        self.state[i], self.state[j] = self.state[j], self.state[i]

    def is_solved(self):
        return self.state == self.goal

# --------- Algoritmos de Busca ---------

def get_neighbors(state):
    neighbors = []
    index = state.index(0)
    row, col = divmod(index, 3)
    directions = [("Up", -3), ("Down", 3), ("Left", -1), ("Right", 1)]

    for direction, delta in directions:
        new_index = index + delta
        if direction == "Up" and row == 0: continue
        if direction == "Down" and row == 2: continue
        if direction == "Left" and col == 0: continue
        if direction == "Right" and col == 2: continue
        new_state = list(state)
        new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
        neighbors.append((new_state, direction))
    return neighbors

def manhattan(state):
    dist = 0
    for i, val in enumerate(state):
        if val == 0: continue
        goal_row, goal_col = divmod(val - 1, 3)
        row, col = divmod(i, 3)
        dist += abs(goal_row - row) + abs(goal_col - col)
    return dist

def astar(start):
    goal = list(range(1, 9)) + [0]
    queue = [(manhattan(start), 0, start, [])]
    visited = set()
    while queue:
        _, cost, state, path = heapq.heappop(queue)
        if tuple(state) in visited:
            continue
        visited.add(tuple(state))
        if state == goal:
            return path
        for neighbor, direction in get_neighbors(state):
            if tuple(neighbor) not in visited:
                heapq.heappush(queue, (cost + 1 + manhattan(neighbor), cost + 1, neighbor, path + [direction]))
    return []

def bfs(start):
    goal = list(range(1, 9)) + [0]
    queue = deque([(start, [])])
    visited = set()
    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path
        visited.add(tuple(state))
        for neighbor, direction in get_neighbors(state):
            if tuple(neighbor) not in visited:
                queue.append((neighbor, path + [direction]))
    return []

def dfs(start):
    goal = list(range(1, 9)) + [0]
    stack = [(start, [])]
    visited = set()
    visited.add(tuple(start)) 

    while stack:
        state, path = stack.pop()
        if state == goal:
            return path
        for neighbor, direction in get_neighbors(state):
            t_neighbor = tuple(neighbor)
            if t_neighbor not in visited:
                visited.add(t_neighbor)  
                stack.append((neighbor, path + [direction]))
    return []


# --------- Interface ---------
class PuzzleGUI:
    def __init__(self, root):
        self.game = EightPuzzle()
        self.root = root
        self.root.title("🧩 8-Puzzle com Algoritmos")
        self.root.configure(bg="#f0f4f8")

        self.board_frame = tk.Frame(root, bg="#f0f4f8", padx=20, pady=20)
        self.board_frame.grid(row=0, column=0, columnspan=3)

        self.buttons = []
        for i in range(9):
            btn = tk.Button(
                self.board_frame,
                text="",
                font=("Helvetica", 24, "bold"),
                width=4,
                height=2,
                relief="flat",
                bg="#ffffff",
                fg="#333",
                activebackground="#e0e0e0",
                command=lambda i=i: self.handle_click(i)
            )
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.buttons.append(btn)

        self.control_frame = tk.Frame(root, bg="#f0f4f8", pady=10)
        self.control_frame.grid(row=1, column=0, columnspan=3)

        btn_style = {"font": ("Helvetica", 12), "bg": "#1976d2", "fg": "white",
                     "activebackground": "#115293", "relief": "raised", "width": 10}

        self.up_btn = tk.Button(self.control_frame, text="▲", command=lambda: self.move("Up"), **btn_style)
        self.down_btn = tk.Button(self.control_frame, text="▼", command=lambda: self.move("Down"), **btn_style)
        self.left_btn = tk.Button(self.control_frame, text="◀", command=lambda: self.move("Left"), **btn_style)
        self.right_btn = tk.Button(self.control_frame, text="▶", command=lambda: self.move("Right"), **btn_style)

        self.up_btn.grid(row=0, column=1, padx=10, pady=5)
        self.left_btn.grid(row=1, column=0, padx=10, pady=5)
        self.down_btn.grid(row=1, column=1, padx=10, pady=5)
        self.right_btn.grid(row=1, column=2, padx=10, pady=5)

        self.shuffle_btn = tk.Button(root, text="🔀 Embaralhar", command=self.shuffle, **btn_style)
        self.shuffle_btn.grid(row=2, column=0, pady=10)

        self.solve_a_star_btn = tk.Button(root, text="⭐ A*", command=self.solve_astar, **btn_style)
        self.solve_a_star_btn.grid(row=2, column=1, pady=10)

        self.solve_bfs_btn = tk.Button(root, text="🌐 BFS", command=self.solve_bfs, **btn_style)
        self.solve_bfs_btn.grid(row=2, column=2, pady=10)

        self.solve_dfs_btn = tk.Button(root, text="🌲 DFS", command=self.solve_dfs, **btn_style)
        self.solve_dfs_btn.grid(row=3, column=1, pady=10)

        self.update_ui()

    def update_ui(self):
        for i in range(9):
            num = self.game.state[i]
            btn = self.buttons[i]
            if num == 0:
                btn.config(text="", bg="#e0e0e0")
            else:
                btn.config(text=str(num), bg="#ffffff")

    def move(self, direction):
        self.game.move(direction)
        self.update_ui()

    def handle_click(self, index):
        blank = self.game.state.index(0)
        row1, col1 = divmod(blank, 3)
        row2, col2 = divmod(index, 3)
        if abs(row1 - row2) + abs(col1 - col2) == 1:
            self.game.swap(blank, index)
            self.update_ui()

    def shuffle(self):
        self.game.state = self.game.shuffle()
        self.update_ui()

    def animate_solution(self, solution, tempo_exec):
        if not solution:
            messagebox.showinfo("Falhou", "Nenhuma solução encontrada.")
            return

        total_moves = len(solution)

        def step():
            if solution:
                self.game.move(solution.pop(0))
                self.update_ui()
                self.root.after(300, step)
            else:
                messagebox.showinfo("Resolvido!",
                    f"Movimentos: {total_moves}\nTempo: {tempo_exec:.4f} segundos")

        step()

    def solve_astar(self):
        inicio = time.time()
        path = astar(self.game.state[:])
        fim = time.time()
        self.animate_solution(path, fim - inicio)

    def solve_bfs(self):
        inicio = time.time()
        path = bfs(self.game.state[:])
        fim = time.time()
        self.animate_solution(path, fim - inicio)

    def solve_dfs(self):
        inicio = time.time()
        path = dfs(self.game.state[:])
        fim = time.time()
        self.animate_solution(path, fim - inicio)

# --------- Iniciar Aplicativo ---------
if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleGUI(root)
    root.mainloop()
