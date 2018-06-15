'''Naive sodoku solver by Martin based on backtracking algorithm.
I have NOT at all studied how real solvers are constructed'''

import numpy as np
import collections
import copy


class Board():
    '''class for keeping track of sodoku puzzle, inserting numbers and checking validity of numbers'''

    def __init__(self, data, dimension):
        # initialise board based np matrix
        self.data = data
        self.dimension = dimension
        self.nRows = dimension**2
        self.valList = list(range(1, self.nRows+1))

        # generate list of lists (represented as matrix) telling if the i'th row, col or square contains number n
        # we add to the second dimension so we can index via the number on the board (1 indexing)
        self.rowContains = np.zeros((self.nRows, self.nRows+1), dtype=bool)
        self.colContains = np.zeros((self.nRows, self.nRows+1), dtype=bool)
        self.sqrContains = np.zeros((self.nRows, self.nRows+1), dtype=bool)

        # populate lists by looping over board
        for iCol in range(self.nRows):
            for iRow in range(self.nRows):
                val = self.data[iRow, iCol]
                if val>0:
                    self.rowContains[iRow, val] = True
                    self.colContains[iCol, val] = True
                    self.sqrContains[(iRow//self.dimension)*self.dimension + (iCol//self.dimension), val] = True


    def printBoard(self):
        '''print board in console'''
        print(self.data)


    def addNumber(self, cord, val):
        '''add the value val at the coordinate cord. Cord is a tupple'''
        self.data[cord] = val
        self.rowContains[cord[0], val] = True
        self.colContains[cord[1], val] = True
        self.sqrContains[(cord[0] // self.dimension) * self.dimension + (cord[1] // self.dimension), val] = True


    def popNumber(self, cord):
        '''empty the square of the board specified by cord'''
        curVal = self.data[cord]
        self.data[cord] = 0

        # adjust the occupation lists based on the deletion
        self.rowContains[cord[0], curVal] = False
        self.colContains[cord[1], curVal] = False
        self.sqrContains[(cord[0] // self.dimension) * self.dimension + (cord[1] // self.dimension), curVal] = False


    def isAllowed(self, cord, val):
        '''returns true if it is allowed to insert val in cord in the board'''
        if self.data[cord] > 0: # this check is superficial for certain solvers
            print('square taken')
            return False
        return (not self.rowContains[cord[0], val]) and (not self.colContains[cord[1], val]) and (not self.sqrContains[(cord[0]//self.dimension)*self.dimension + (cord[1]//self.dimension), val])


    def getAllowed(self, cord):
        '''return a list of the allowed values at the gives square'''
        allowedFilter = np.logical_not(self.rowContains[cord[0], :] + self.colContains[cord[1], :] + self.sqrContains[(cord[0]//self.dimension)*self.dimension + (cord[1]//self.dimension), :])
        return [i for i in range(1,10) if allowedFilter[i]]  # ok, this is probably very unpythonic. I could just return a np array instead of a python list



class BacktrackSolver():
    '''backtracking algorithm for solving sodoku'''

    def __init__(self, board):
        self.board = board # takes a Board object
        self.inputBoard = copy.deepcopy(board) # save a copy of the input
        self.dimension = self.board.dimension
        self.nRows = self.board.nRows
        self.nQueries = 0


    def reset(self):
        self.board = copy.deepcopy(self.inputBoard)

    def getVacantList(self, priority = True):
        '''return list of coordinates for empty squares'''
        vacList = []

        # method 1, return list of vacant squares in order of transversing one row at a time from top to bottom
        if not priority:
            for iRow in range(self.nRows):
                for iCol in range(self.nRows):
                    if self.board.data[iRow, iCol] == 0:
                        vacList.append((iRow, iCol))

        # method 2, return a list of vacant squares sorted so the squares with most (!) possible options appear first
        else:
            nAllowed = []
            for iRow in range(self.nRows):
                for iCol in range(self.nRows):
                    
                    cord = (iRow, iCol)
                    if self.board.data[cord] == 0:
                        vacList.append(cord)
                        nAllowed.append(len(self.board.getAllowed(cord)))

            # do the sorting
            print(sorted(zip(nAllowed, vacList)))
            vacList = [x for _,x in sorted(zip(nAllowed, vacList), reverse=True)]
            print(vacList)

        return vacList


    def solve(self, usePriority = True):

        # generate list of vacant Squares
        vacList = self.getVacantList(priority = usePriority)
        nVacs = len(vacList)  # count the number of vacant squares

        # initiate a stack for the history of added numbers
        histStack = collections.deque(maxlen=nVacs)

        # do some statistics
        nBacktracks = 0
        nQueries = 0
        nSolvedMax = 0

        ######################################
        # Main loop: Loop over vacant squares#
        ######################################

        iVac = 0  # index of current vacant square being examined
        curCord = vacList[iVac]  # coordinate of current vacant square
        n = 1  # value being considered

        while iVac < nVacs:

            foundAllowed = False # indicates if we have found a value allowed in the square being considered

            while n < self.nRows + 1:  # run from starting value up to n rows
                #print('Testing %d in (%d,%d)' % (n, curCord[0], curCord[1]))
                nQueries += 1

                if self.board.isAllowed(curCord, n):
                    self.board.addNumber(curCord, n)
                    #print('Inserting %d in (%d,%d)' % (n, curCord[0], curCord[1]))
                    histStack.append(n)
                    foundAllowed = True
                    break
                n += 1

            if foundAllowed:
                
                iVac += 1 # move to next square
                
                if iVac>nSolvedMax:
                    nSolvedMax += 1
                    print(iVac)
                if iVac < nVacs: # check there are still vacant squares to fill. Without this check the very final itteration gives an index out of bounds error
                    curCord = vacList[iVac]
                    n = 1 # reset value guess

                #self.board.printBoard()

            if not foundAllowed:
                # now we need to back track
                #print('Backtracking!')
                self.board.popNumber(vacList[iVac-1]) # remove the last value added
                iVac -= 1
                curCord = vacList[iVac]
                n = histStack.pop() + 1 # begin n guess one above the previously inserted value. Key part of backtracking algorithm
                nBacktracks += 1

                #self.board.printBoard()

        self.board.printBoard()
        print('\nBoard solved with %d queries and %d backtracks'%(nQueries, nBacktracks))



def mapSquares(s):
    if s == '*':
        return 0
    try:
        return int(s)
    except ValueError:
        return ord(s) - 65 + 10


def loadSodoku(path):
    '''read a textfile containing a sodoku board and return a sodoku board object'''
    with open(path) as f:
        data = []

        for line in f.readlines():
            data.append([mapSquares(s) for s in line.strip().split(' ')])

    npdata = np.array(data, dtype = int)
    dimension = int(np.sqrt(npdata.shape[0]))

    return Board(npdata, dimension)


if __name__ == '__main__':

    testBoard = loadSodoku('Sodoku_3x3_easy.txt')

    s = BacktrackSolver(testBoard)
    s.solve(usePriority=False)