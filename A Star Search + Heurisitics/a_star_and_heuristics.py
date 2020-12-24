'''
Homework 1
Machine Reasoning 
Fall 2020
Dr. R

This program uses the A* search algorithm to solve
the 8-puzzle. Three heuristic funtions are available 
to be used in the A* search algorithm.

@authors: Brad Shook
          Wilbert Garcia
'''


from random import shuffle, randint
from queue import PriorityQueue
from sys import getsizeof
from collections import deque
import csv
import time
'''
Puzzle class creates a sliding puzzle to be solved.
'''
class Puzzle:

    # Initialize a Puzzle object with the heuristic to be used.
    def __init__(self, heuristic):
        # initialize the goal state
        self.goalState = [1,2,3,
                          4,5,6,
                          7,8,0]

        # the heuristic function to be used in A*
        self.heuristic = heuristic
        
        # initialize the initial state with a random solvable board
        self.initialState = self.generateSolvableBoard()

    '''
    setInitialState() sets the initial state of the board 
                      to a given board
    
    @params - self
              board: an array representing the sliding puzzle
    
    @return - None
    '''
    def setInitialState(self, board):
        self.initialState = board

                
    '''
    generateSolvableBoard() creates a sliding puzzle which
                            is solvable.
    
    @params - self
    
    @return - board: an array representing the sliding puzzle
    '''
    def generateSolvableBoard(self):
        solvableBool = False
        # generate random boards until a solvable one is made
        while solvableBool != True:
            # generate random board
            board = self.generateRandomBoard()

            # check if board is solvable
            solvableBool = self.isSolvable(board)
        
        return board
    
    '''
    generateRandomBoard() creates a random sliding puzzle
    
    @params - self
    
    @return - board: an array representing the sliding puzzle
    '''
    def generateRandomBoard(self):
        board = [i for i in self.goalState]
        
        # shuffle values
        shuffle(board)
        
        return board

    '''
    isSolvable() checks if a given slidng puzzle
                 is solvable. 
                      
    @params - self
              board: an array representing the sliding puzzle
    
    @return - a boolean for if the inversion count is even
    '''
    def isSolvable(self, board): 
        
        inversion_count = 0
        # iterate through each index on the board except the last one
        for i in range(len(board) - 1):
            for j in range(i + 1, len(board)):
                # count pairs(i, j) such that i appears 
                # before j, but i > j.  [4, 3, 2, 1] has 6 inversions. The 4 has 3 inv. The 3 has two. The 2 has 1.
                if (board[j] and board[i] and board[i] > board[j]):
                    inversion_count += 1

        # return true if inv count is even, false if odd
        return (inversion_count % 2 == 0)
    
    '''
    printBoard() prints a properly formatted version of a
                 given sliding puzzle.
    
    @params - self
              board: an array representing the sliding puzzle
    
    @return - None
    '''
    def printBoard(self, board):
        print(board[0:3])
        print(board[3:6])
        print(board[6:9])

    '''
    goalTest() checks if a given state is 
               the goal state.
                   
    @params - self
              state: an array representing the sliding puzzle
    
    @return - boolean for if the given state is the goal state
    '''
    def goalTest(self, state):
        
        for i in range(len(state)):
            if state[i] != self.goalState[i]:
                return False
        
        return True

'''
Node class that contains functions and fields relevant
to individual nodes in a search algorithm.
'''
class Node:
    '''
    __init__() initializes a node object.
               
    @params - self
              heuristic: the heuristic function to be used
              currentState: the current state of the sliding puzzle
              pathCost: the total path cost to the current node
              f: the f cost of the current node
              path: the solution path to the current node
    
    @return - None
    '''
    def __init__(self, heuristic, currentState, pathCost, f=0, path=[]):
        self.currentState = currentState
        self.vacantSqIndex = self.currentState.index(0)
        self.heuristicFunction = heuristic
        self.children = []
        self.goalState = [1,2,3,
                          4,5,6,
                          7,8,0]
        
        # array of nodes leading to current node
        self.path = path

        # edge cost to get from start node to current node
        self.pathCost = pathCost
        self.heuristicCost = 0
        self.f = f

        # update f based on heuristic given
        if self.heuristicFunction == "h1":
            self.countMisplacedTiles(self.currentState)
            self.incrementPathCost()
            self.calculateF()
        elif self.heuristicFunction == "h2":
            self.calculateManhattanDistanceSum(self.currentState)
            self.incrementPathCost()
            self.calculateF()
        elif self.heuristicFunction == "h3":
            self.findLinearConflict(self.currentState)
            self.incrementPathCost()
            self.calculateF()
        elif self.heuristicFunction == "none":
            self.incrementPathCost()
    '''
    __lt__() defines how the less than operator works.
             It had to be redefined to fix the error 
             of a node having two children with the same
             f cost.
    '''
    def __lt__(self, other):
        return self.f < other.f

    '''
    calculateF() calculates the f cost of the node.
                   
    @params - self
    
    @return - None
    '''
    def calculateF(self):
        self.f = self.pathCost + self.heuristicCost
    
    '''
    incrementPathCost() adds 1 to the path cost of the node.
                        Used to keep track of the path cost
                        of each node.
                   
    @params - self
    
    @return - None
    '''
    def incrementPathCost(self):
        self.pathCost += 1
    
    '''
    countMisplacedTiles() counts how many tiles are misplaced 
                          in relation to the goal state and adds the count
                          to the node's heuristicCost field. This is
                          the first heuristic function, 'h1'.
                   
    @params - self
              board: an array representing the current state of the puzzle
    
    @return - None
    '''
    def countMisplacedTiles(self, board):
        count = 0
        # iterate through each index on the board
        for i in range(len(self.goalState)):
            # check if the current tile matches its position in the goal state
            if board[i] != self.goalState[i]:
                count += 1
        
        self.heuristicCost += count
    '''
    calculateManhattanDistanceSum() calculates the manhattan
                          distance from each tile to that tile's
                          goal position and sums these distances.
                          The sum is added to the node's heuristic
                          cost field. This is the second heuristic
                          function, 'h2'.
                   
    @params - self
              board: an array representing the current state of the puzzle.
    
    @return - None
    '''
    def calculateManhattanDistanceSum(self, board):
        total = 0

        for i in range(len(board)):
            
            # using 5 as test case
            value = board[i] # 0
            startIndex = i # 
            finalIndex = board[i] - 1 # 
            if finalIndex == -1:
                finalIndex = 8
            
            startIndexRowCol = self.findRowAndCol(startIndex)
            finalIndexRowCol = self.findRowAndCol(finalIndex)

            # rowIndex - rowIndex
            rowDiff = abs(startIndexRowCol[0] - finalIndexRowCol[0])
            colDiff = abs(startIndexRowCol[1] - finalIndexRowCol[1])
        
            total += rowDiff + colDiff

        self.heuristicCost += total
        return total
    '''
    findRowAndCol() finds what row and column a given
                    index belongs to in the puzzle.
                   
    @params - self
              index: an integer representing the position of a tile
    
    @return - array with the row and column the index belongs to
    '''
    def findRowAndCol(self, index):
        row = 0
        col = 0
        if index <= 2:
            row = 1
        elif index <= 5:
            row = 2
        else:
            row = 3
        
        if index in [0, 3, 6]:
            col = 1
        elif index in [1, 4, 7]:
            col = 2
        else:
            col = 3

        return [row, col]

    def findLinearConflict(self, board):
        lc = 0
        c = 0
        
        rowConflictCountDict, rowConflictTileDict = self.findConflictsInRows(board)
        colConflictCountDict, colConflictTileDict = self.findConflictsInCols(board)

        maxConflictInRows = max(rowConflictCountDict.values())
        while maxConflictInRows > 0:
            
            # get tile with most conflicts
            maxRowTile = max(rowConflictCountDict, key=rowConflictCountDict.get)
            
            # set max tile number of conflicts to 0
            rowConflictCountDict[maxRowTile] = 0
            # get the tiles that are conflicting with the max tile
            tileKConflictingTiles = rowConflictTileDict[maxRowTile]
            
            # iterate through the conflicting tiles, subtracting 1 from
            # their conflict counts
            for tile in tileKConflictingTiles: 
                rowConflictCountDict[tile] -= 1
                rowConflictTileDict[tile].remove(maxRowTile)
                rowConflictTileDict[maxRowTile] = []
            
            # add 1 to the total linear conflict
            lc += 1
            # find the new max conflict
            maxConflictInRows = max(rowConflictCountDict.values())

        maxConflictInCols = max(colConflictCountDict.values())
        while maxConflictInCols > 0:
            # get tile with most conflicts
            maxColTile = max(colConflictCountDict, key=colConflictCountDict.get)
            
            # set max tile number of conflicts to 0
            colConflictCountDict[maxColTile] = 0
            # get the tiles that are conflicting with the max tile
            tileKConflictingTiles = colConflictTileDict[maxColTile]
            
            for tile in tileKConflictingTiles: 
                colConflictCountDict[tile] -= 1
                colConflictTileDict[tile].remove(maxColTile)
                colConflictTileDict[maxColTile] = []
            
            # add 1 to the total linear conflict
            lc += 1
            # find the new max conflict
            maxConflictInCols = max(colConflictCountDict.values())
        
        final_lc = 2 * lc

        self.heuristicCost += final_lc + self.calculateManhattanDistanceSum(board)

    def findConflictsInRows(self, board):

        boardWithRows = []
        boardWithRows.append(board[0:3])
        boardWithRows.append(board[3:6])
        boardWithRows.append(board[6:9])
        offset = 0
        
        rowConflictCountDict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 0: 0}
        rowConflictTileDict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 0: []}
        # iterate through each row on board
        for rowI in boardWithRows:
            for tileJ in range(len(rowI)):
                for tileK in range(tileJ + 1, len(rowI)):
                    tileJValue = rowI[tileJ]
                    tileKValue = rowI[tileK]
                    goalConflict = False
                    
                    tileJCorrectIndex = self.findCorrectTileIndex(tileJValue)
                    tileKCorrectIndex = self.findCorrectTileIndex(tileKValue)

                    currentRowIndexes = [0 + offset, 1 + offset, 2 + offset]

                    if tileJCorrectIndex in currentRowIndexes and tileKCorrectIndex in currentRowIndexes:
                        # tile j and tile k aren't in correct spots
                        if (((tileJCorrectIndex > board.index(tileJValue)) and (tileKCorrectIndex < board.index(tileKValue))) 
                            # tile J is in correct spot but tile K needs to move to the left of tile J
                            or ((tileJCorrectIndex == board.index(tileJValue)) and (tileKCorrectIndex < tileJCorrectIndex)) 
                                # tile K is in correct spot but tile J needs to move to the right of tile K 
                                or ((tileKCorrectIndex == board.index(tileKValue)) and (tileJCorrectIndex > tileKCorrectIndex))):
                            
                            goalConflict = True
                        
                        if goalConflict:
                            rowConflictCountDict[tileJValue] += 1
                            rowConflictCountDict[tileKValue] += 1
                            rowConflictTileDict[tileJValue].append(tileKValue)
                            rowConflictTileDict[tileKValue].append(tileJValue)
            
            offset += 3
        
        return (rowConflictCountDict, rowConflictTileDict)

    def findConflictsInCols(self, board):
        
        boardWithCols = []
        firstCol = []
        secondCol = []
        thirdCol = []
        offset = 0
        
        colConflictCountDict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 0: 0}
        colConflictTileDict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 0: []}

        for i in range(len(board)):
            if i in [0,3,6]:
                firstCol.append(board[i])
            elif i in [1,4,7]:
                secondCol.append(board[i])
            else:
                thirdCol.append(board[i])
        
        boardWithCols = [firstCol, secondCol, thirdCol]

        for colI in boardWithCols:
            for tileJ in range(len(colI)):
                for tileK in range(tileJ + 1, len(colI)):

                    tileJValue = colI[tileJ]
                    tileKValue = colI[tileK]
                    goalConflict = False

                    tileJCorrectIndex = self.findCorrectTileIndex(tileJValue)
                    tileKCorrectIndex = self.findCorrectTileIndex(tileKValue)

                    currentColIndexes = [0 + offset, 3 + offset, 6 + offset]
                    
                    if tileJCorrectIndex in currentColIndexes and tileKCorrectIndex in currentColIndexes:
                        
                        # tile j and tile k aren't in correct spots
                        if (((tileJCorrectIndex > board.index(tileJValue)) and (tileKCorrectIndex < board.index(tileKValue))) 
                            # tile J is in correct spot but tile K needs to move to the left of tile J
                            or ((tileJCorrectIndex == board.index(tileJValue)) and (tileKCorrectIndex < tileJCorrectIndex)) 
                                # tile K is in correct spot but tile J needs to move to the right of tile K 
                                or ((tileKCorrectIndex == board.index(tileKValue)) and (tileJCorrectIndex > tileKCorrectIndex))):

                            goalConflict = True
                        
                        if goalConflict:
                            colConflictCountDict[tileJValue] += 1
                            colConflictCountDict[tileKValue] += 1
                            colConflictTileDict[tileJValue].append(tileKValue)
                            colConflictTileDict[tileKValue].append(tileJValue)

            offset += 1

        return (colConflictCountDict, colConflictTileDict)
        
                    
    def findCorrectTileIndex(self, tileValue):
        return self.goalState.index(tileValue)

    '''
    findNextStates() finds the possible successors of the
                     the current state and appends them
                     to the children field array.
                   
    @params - self
    
    @return - None
    '''
    def findNextStates(self):
        # move left
        if self.vacantSqIndex not in [0, 3, 6]:
            
            self.children.append(Node(self.heuristicFunction, self.moveVacantLeft(), 
                                        self.pathCost, self.f, self.path + [self.currentState]))

        # move right
        if self.vacantSqIndex not in [2, 5, 8]:
            
            self.children.append(Node(self.heuristicFunction, self.moveVacantRight(), 
                                        self.pathCost, self.f, self.path + [self.currentState]))
        
        # move down
        if self.vacantSqIndex < 6:
            
            self.children.append(Node(self.heuristicFunction, self.moveVacantDown(), 
                                        self.pathCost, self.f, self.path + [self.currentState]))
        
        # move up
        if self.vacantSqIndex > 2:
            
            self.children.append(Node(self.heuristicFunction, self.moveVacantUp(), 
                                        self.pathCost, self.f, self.path + [self.currentState]))

    '''
    moveVacantLeft() creates a new state array with the vacant
                     tile shifted left.   
     
    @params - self
    
    @return - state: an array representing the newly created state of the board.
    '''
    def moveVacantLeft(self):
        
        state = [i for i in self.currentState]
        leftSqVal = state[self.vacantSqIndex - 1]
        state[self.vacantSqIndex] = leftSqVal
        state[self.vacantSqIndex - 1] = 0
    
        return state
    '''
    moveVacantRight() creates a new state array with the vacant
                      tile shifted right.   
     
    @params - self

    @return - state: an array representing the newly created state of the board.
    '''
    def moveVacantRight(self):
        state = [i for i in self.currentState]
        rightSqVal = state[self.vacantSqIndex + 1]
        state[self.vacantSqIndex] = rightSqVal
        state[self.vacantSqIndex + 1] = 0

        return state
    
    '''
    moveVacantDown() creates a new state array with the vacant
                     tile shifted down.   
     
    @params - self

    @return - state: an array representing the newly created state of the board.
    '''
    def moveVacantDown(self):
        state = [i for i in self.currentState]
        belowSqVal = state[self.vacantSqIndex + 3]
        state[self.vacantSqIndex] = belowSqVal
        state[self.vacantSqIndex + 3] = 0
        
        return state

    '''
    moveVacantUp() creates a new state array with the vacant
                   tile shifted up.   
     
    @params - self

    @return - state: an array representing the newly created state of the board.
    '''
    def moveVacantUp(self):
        state = [i for i in self.currentState]
        aboveSqVal = state[self.vacantSqIndex - 3]
        state[self.vacantSqIndex] = aboveSqVal
        state[self.vacantSqIndex - 3] = 0
        
        return state

    def setBoardWithNumOfMoves(self, numMoves):
        movesMade = 0
        lastMove = 0
        while movesMade < numMoves:

            move = randint(1,4)
            # move left
            if self.vacantSqIndex not in [0, 3, 6] and move == 1 and lastMove != 2:
                
                self.currentState = self.moveVacantLeft()
                self.vacantSqIndex = self.currentState.index(0)
            
                lastMove = 1
                movesMade += 1
                
                
            # move right
            if self.vacantSqIndex not in [2, 5, 8] and move == 2 and lastMove != 1:
                
                self.currentState = self.moveVacantRight()
                self.vacantSqIndex = self.currentState.index(0)
                
                lastMove = 2
                movesMade += 1
            
            # move down
            if self.vacantSqIndex < 6 and move == 3 and lastMove != 4:
                
                self.currentState = self.moveVacantDown()
                self.vacantSqIndex = self.currentState.index(0)
                
                lastMove = 3
                movesMade += 1
                
            # move up
            if self.vacantSqIndex > 2 and move == 4 and lastMove != 3:
                
                self.currentState = self.moveVacantUp()
                self.vacantSqIndex = self.currentState.index(0)
                
                lastMove = 4
                movesMade += 1

def aSearch(puzzle, depthNum, iteration):
        # create initial node
        startTime = time.time()
        node = Node(puzzle.heuristic, puzzle.initialState, 0)
        
        # frontier is priority queue of nodes with f values as priority
        frontier = PriorityQueue()

        # put initial node in frontier
        frontier.put((0, node))
        
        explored = set()
        counter = 0
        bfSum = 0
        nodeSize = getsizeof(node.currentState)
        memSize = 0

        while 1:
            # print("\nNodes Generated: ")
            # print(len(explored))

            # if len(explored) > 100000:
            #     print("Memory Error: Too many nodes have been generated.")
            #     exit()

            # print("\nExplored Size: ")
            # print(memSize)
            if frontier == []:
                return "Failure: Solution not found."
            
            node = frontier.get()
            # print(puzzle.initialState)
            # print("\nPath Cost/Depth: ")
            # print(node[1].pathCost)
            # puzzle.printBoard(node[1].state)
            memSize += nodeSize
            if puzzle.goalTest(node[1].currentState) == True:
                # write to csv - start state, path length, EBF, nodes generated
                # print("Initial State: ")
                # puzzle.printBoard(puzzle.initialState)
                # print()
                # print("Path Length: ")
                # print(len(node[1].path))
                # print("\nSolution Path: ")
                # for i in node[1].path:
                #     puzzle.printBoard(i)
                #     print()
                # puzzle.printBoard(puzzle.goalState)
                # print("\nPath Length: ")
                # print(len(node[1].path))
                # ebf = bfSum / len(explored)
                
                # print("\nEBF: " + str(ebf))

                endTime = time.time()

                totalTime = endTime - startTime

                with open("CorrectEdgeDepth" + str(depthNum) + "Analysis.csv", 'a+', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([iteration, node[1].heuristicFunction, len(node[1].path), len(explored), totalTime])

                return node[1].path
            
            # add the hashed value for the state
            explored.add(hash(tuple(node[1].currentState)))
            
            # find children of current node
            node[1].findNextStates()
            
            # iterate through current node's children
            for child in node[1].children:

                if child not in frontier.queue and hash(tuple(child.currentState)) not in explored:
                    frontier.put((child.f, child))
                    bfSum += 1

def IDS(puzzle):
    max_depth = 4
    for i in range(max_depth):
        DLS(puzzle, i)

def DLS(puzzle, limit):
        node = Node(puzzle.heuristic, puzzle.initialState, 0)
        depth = 0

        print("\nInitial Board:")
        puzzle.printBoard(node.currentState)
        
        frontier = []
        frontier.append(node)
        
        explored = set()
        counter = 0
        
        while 1:

            print(len(explored))
            
            if frontier == []:
                return False
            
            node = frontier.pop()

            print("\nPath Cost: ")
            print(node.pathCost)
            print("\nNode state " + str(counter))
            puzzle.printBoard(node.currentState)
            
            if puzzle.goalTest(node.currentState) == True:
                for i in node.path:
                    puzzle.printBoard(i)
                    print()
                print("\nPath Length: ")
                print(len(node.path))
                return node.path
            
            explored.add(hash(tuple(node.currentState)))
            
            node.findNextStates()

            for child in node.children:
                print("\nChild State")
                puzzle.printBoard(child.currentState)
                if child not in frontier and hash(tuple(child.currentState)) not in explored and node.pathCost <= limit:
                    frontier.append(child)
            
            print("NEXT")


                
                


h1Puzzle = Puzzle("h1")
h2Puzzle = Puzzle("h2")
h3Puzzle = Puzzle("h3")


depthNum = 10

# h1Puzzle.setInitialState([1,2,3,4,5,6,7,8,0])
# node = Node(h1Puzzle.heuristic, h1Puzzle.initialState, 0)
# node.setBoardWithNumOfMoves(depthNum)
# h2Puzzle.initialState = node.currentState
# aSearch(h2Puzzle, 4, 4)



for i in range(800):

    h1Puzzle.setInitialState([1,2,3,4,5,6,7,8,0])
    node = Node(h1Puzzle.heuristic, h1Puzzle.initialState, 0)
    node.setBoardWithNumOfMoves(depthNum)

    h1Puzzle.initialState = node.currentState
    h2Puzzle.initialState = node.currentState
    h3Puzzle.initialState = node.currentState

    aSearch(h1Puzzle, depthNum, i)
    aSearch(h2Puzzle, depthNum, i)
    aSearch(h3Puzzle, depthNum, i)