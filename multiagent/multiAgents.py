# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import sys
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # FINAL FUNCIO
        # return childGameState.getScore()
        total_score = 0
        curr_food = currentGameState.getFood().asList()
        # food considerations
        for food_pos in curr_food:
            d = manhattanDistance(food_pos, newPos)
            total_score += 2000 if d == 0 else 1.0 / (d ** 2)

        # ghost considerations
        for ghost in newGhostStates:
            d = manhattanDistance(ghost.getPosition(), newPos)
            if d > 1:
                continue
            total_score += 3000 if ghost.scaredTimer != 0 else -3000

        return total_score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        print(f'Depth{depth}'.center(50, '='))

    def terminal_test(self, state, depth):
        return depth == 0 or state.isWin() or state.isLose()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        v = float("-inf")
        actions = []
        for a in gameState.getLegalActions(agentIndex=0):
            succ = gameState.getNextState(agentIndex=0, action=a)
            u = self.min_value(
                succ, agent=1, depth=self.depth
            )
            if u == v:
                actions.append(a)
            elif u > v:
                v = u
                actions = [a]
        
        # return random.choice(actions)
        return actions[0]

    def min_value(self, gameState, agent, depth):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("inf")
        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=a)
            if agent == gameState.getNumAgents() - 1:
                v = min(
                    v, self.max_value(succ, agent=0, depth=depth - 1)
                )

            else:
                v = min(
                    v, self.min_value(succ, agent=agent + 1, depth=depth)
                )
        return v

    def max_value(self, gameState, agent, depth):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("-inf")
        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=a)
            v = max(
                v, self.min_value(succ, agent=1, depth=depth)
            )
        return v



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    best_action = None

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float("-inf")
        beta = float("inf")
        self.max_value(gameState, agent=0, alpha=alpha, beta=beta, depth=self.depth)
        return self.best_action

    def min_value(self, gameState, agent, alpha, beta, depth):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("inf")
        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=a)
            if agent == gameState.getNumAgents() - 1:
                v = min(
                    v, self.max_value(succ, agent=0, alpha=alpha, beta=beta, depth=depth - 1)
                )

            else:
                v = min(
                    v, self.min_value(succ, agent=agent + 1, alpha=alpha, beta=beta, depth=depth)
                )

            # check if prune
            if v < alpha:
                return v

            # update beta
            beta = min(beta, v)
        return v

    def max_value(self, gameState, agent, alpha, beta, depth):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("-inf")
        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=a)
            v = max(
                v, self.min_value(succ, agent=1, alpha=alpha, beta=beta, depth=depth)
            )

            # check if can prune
            if v > beta:
                return v

            # update alpha
            if v > alpha:
                alpha = v
                # update best action
                self.best_action = a  # current level best action
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        v = float("-inf")
        actions = []
        for a in gameState.getLegalActions(agentIndex=0):
            succ = gameState.getNextState(agentIndex=0, action=a)
            u = self.chance_value(
                succ, agent=1, depth=self.depth
            )
            if u == v:
                actions.append(a)
            elif u > v:
                v = u
                actions = [a]

        return actions[0]

    def max_value(self, gameState, agent, depth):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)

        v = float("-inf")
        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=a)
            v = max(
                v, self.chance_value(succ, agent=1, depth=depth)
            )
        return v

    def chance_value(self, gameState, agent, depth):
        if self.terminal_test(gameState, depth):
            return self.evaluationFunction(gameState)
        v = 0
        n_actions = len(gameState.getLegalActions(agent))
        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=a)
            if agent == gameState.getNumAgents() - 1:
                v += 1/n_actions * self.max_value(succ, 0, depth-1)
            else:
                v += 1/n_actions * self.chance_value(succ, agent+1, depth)
        return v


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    foodList = currentGameState.getFood().asList()
    pacman = currentGameState.getPacmanPosition()
    # Amount of food
        # The less amount of food left, the better
    food_amount_score = len(foodList)
    # Pacman distance to food
    food_distance_score = 0
    min_food = float("inf")
    for food in foodList:
        d = (manhattanDistance(pacman, food) ** 2)
        if d < min_food:
            min_food = d
        food_distance_score += d

    ghosts = 0
    for ghost_position in currentGameState.getGhostPositions():
        ghost_distance = manhattanDistance(pacman, ghost_position)

        if ghost_distance < 2:
            ghosts = 1

    # Weighted average
    features = [
        food_amount_score,
        food_distance_score,
        1.0 / min_food,
        currentGameState.getScore(),
        ghosts,
    ]

    weights = [
        -100,
        -1,
        200,
        200,
        10,
    ]

    score = sum([(feat * weight) for feat, weight in zip(features, weights)])
    print([(feat * weight) for feat, weight in zip(features, weights)])
    print("SCORE: ", score)
    return score


# Abbreviation
better = betterEvaluationFunction


class IterativeMinMax(MultiAgentSearchAgent):
    """
      Mini Max interative version
    """

    class Node:
        depth = None
        action = None
        actor = None
        value = None
        state = None
        best_action=None
        expanded=False
        agent = None
        parent = None

        def __init__(self, state, agent, action, actor, depth, value, parent):
            self.state = state
            self.action = action,
            self.actor = actor
            self.depth = depth
            self.value = value
            self.agent = agent
            self.parent = parent

    def getAction(self, gameState):
        """
        Returns the minmax action using self.depth and self.evaluationFunction
        """
        start_node = self.Node(
            state=gameState,
            agent=0,
            action=None,
            actor="max",
            depth=self.depth,
            value=float("-inf"),
            parent=None
        )
        # apply min max algorithm
        self.stack = [start_node]
        while self.stack:
            current_node = self.stack[-1]
            # check if the current state it is a terminal state
            if self.terminal_test(current_node.state, current_node.depth):
                self.pop_node(current_node, is_terminal=True)
            elif not current_node.expanded:
                self.expand_node(current_node)
            else:
               self.pop_node(current_node)

        return self.best_action

    def expand_node(self, node):
        # check how expand the node
        if node.actor == "max":
           self.max_player_expand(node)
        else:
            self.min_player_expand(node)
        # mark the node as expanded
        node.expanded = True

    def max_player_expand(self, node):
        gameState = node.state
        agent = node.agent
        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=a)
            next_node = self.Node(
                state=succ, 
                agent=1,
                action=a,
                actor="min",
                depth=node.depth,
                value=float("inf"),
                parent=node
            )
            self.stack.append(next_node)

    def min_player_expand(self, node):
        gameState = node.state
        agent = node.agent
        for a in gameState.getLegalActions(agent):
            succ = gameState.getNextState(agent, action=a)
            if agent == gameState.getNumAgents() - 1:
                next_node = self.Node(
                    state=succ, 
                    agent=0,
                    action=a,
                    actor="max",
                    depth=node.depth - 1,
                    value=float("-inf"),
                    parent=node
                )
            else:
                next_node = self.Node(
                    state=succ, 
                    agent=node.agent + 1,
                    action=a,
                    actor="min",
                    depth=node.depth,
                    value=float("inf"),
                    parent=node
                )
            self.stack.append(next_node)

    def pop_node(self, node, is_terminal=False):
        # pop node
        self.stack.pop()

        # if the node it is the first update the best_action
        if node.parent is None:
            self.best_action = node.best_action[0]
            return

        # update parent best action
        value = self.evaluationFunction(node.state) if is_terminal else node.value
        parent_node = node.parent
        if parent_node.actor == "max" and value >= parent_node.value:
            parent_node.value = value
            parent_node.best_action = node.action
        elif parent_node.actor == "min" and value <= parent_node.value:
            parent_node.value = value
            parent_node.best_action = node.action


