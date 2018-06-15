# Backtrack-Sudoku-Solver
Simple Python Sudoku solver written i Python.

This is not a serious sudoku solver!
I have not read a single sentence about how real sudoku solvers work.
Instead, I have simply read about backtracking algorithms and implemented their core principle to to the problem of Sudoku puzzles.
While the algorithm easily solves a normal 9*9 sudoku in a few dozen ms it is completely unfeasible for a 16*16.

The goal of this project has been to invent and implement an algorithm from scratch, and to write a nice, clean, class based Python implementation.

One class handles the representation and manipulation of Sudoku puzzles while a second class forms the solver. A function can be used to read puzzles from .txt files.

When run as a scipt a filepath to such a txt file must be provided.
