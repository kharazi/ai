# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def learnHowToSearch(problem):
    from game import Directions
    import random

    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    state = problem.getStartState() #to get initial state

    print 'First element of state tuple is the current position of Pacman:'
    print state[0]
    print 'Second element of state tuple is current state of Foods:'
    print state[1]
    print 'You can have position of walls:'
    print problem.walls

    path = [] #list used to return the desired actions

    while problem.isGoalState(state) == False: #repeat until Pacman finds all the foods
    	successors = problem.getSuccessors(state) #get successors of current state
    	count = len(successors) #count of successors
    	index = random.randint(0, count-1) #find next successor index randomly
    	nextSuc = successors[index]
    	state = nextSuc[0] #first element of successor is state of that successor

    	if nextSuc[1] == 'South': #second element of successor is a string specify the direction to that successor
    		path.append(s)
    	elif nextSuc[1] == 'North':
    		path.append(n)
    	elif nextSuc[1] == 'West':
    		path.append(w)
    	else:
    		path.append(e)

    return path


def depthFirstSearch(problem):
    stack = util.Stack()
    state = problem.getStartState()
    stack.push((state, list(), list()))
    # print state
    # for i in problem.getSuccessors(state):
    #     print i
    while not stack.isEmpty():
        current_state, actions, visited = stack.pop()
        for pos, dir, step in problem.getSuccessors(current_state):
            if not pos in visited:
                if problem.isGoalState(pos):
                    return actions + [dir]
                stack.push((pos, actions + [dir], visited + [current_state] ))


def breadthFirstSearch(problem):

    queue = util.Queue()
    state = problem.getStartState()
    queue.push((state, list()))

    visited = list()
    while not queue.isEmpty():
        node, actions = queue.pop()

        for pos, dir, steps in problem.getSuccessors(node):
            if not pos in visited:
                visited.append(pos)
                if problem.isGoalState(pos):
                    return actions + [dir]
                queue.push((pos, actions + [dir]))


# def DLS(node, goal, depth):
#     if depth == 0 and node == goal:

# def iterativeDeepeningSearch(problem):
#     util.raiseNotDefined()

def uniformCostSearch(problem):

    pqueue = util.PriorityQueue()
    state = problem.getStartState()
    pqueue.push((state, list()), 0)

    visited = []

    while not pqueue.isEmpty():
        current_state, actions = pqueue.pop()
        if problem.isGoalState(current_state):
            return actions

        visited.append(current_state)

        for pos, dir, steps in problem.getSuccessors(current_state):
            if not pos in visited:
                new_actions = actions + [dir]
                pqueue.push((pos, new_actions), problem.getCostOfActions(new_actions))


def manhattanHeuristicFunction(state, problem=None):
    xy1 = state[0]
    for i in range(state[1].height):
        for j in range(state[1].width):
            if state[1].data[j][i]:
                xy1 = (j,i)
                break
    xy2 = state[0]
    return  abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


import math
import sys

def heuristicFunction(state, problem=0):
    food_count = 0
    xy1 = state[0]
    result = sys.maxint
    for i in range(state[1].height):
        for j in range(state[1].width):
            if state[1].data[j][i]:
                xy2 = (j,i)
                food_count += 1
    xy2 = state[0]
    result = abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
    if food_count <= 1:
        return 0
    return result + food_count


def aStarSearch(problem, heuristic=manhattanHeuristicFunction): # or heuristic=heuristicFunction

    closedset = []
    fringe = util.PriorityQueue()
    start = problem.getStartState()
    fringe.push((start, []), heuristic(start, problem))

    while not fringe.isEmpty():
        node, actions = fringe.pop()

        if problem.isGoalState(node):
            return actions

        closedset.append(node)

        for coord, direction, cost in problem.getSuccessors(node):
            if not coord in closedset:
                new_actions = actions + [direction]
                score = problem.getCostOfActions(new_actions) + heuristic(coord, problem)
                fringe.push( (coord, new_actions), score)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
