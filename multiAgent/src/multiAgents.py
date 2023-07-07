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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # We use nearest food as an positive factor
        nearest_food_distance = float(min([manhattanDistance(food, newPos) for food in newFood.asList()])) if len(newFood.asList()) != 0 else 10000
        # We use Distance of ghosts as an negative factor
        # Also, If a ghost is going to be so so close next step, it's very bad!
        ghosts = successorGameState.getGhostPositions()
        total_ghost_distance = 0.1
        near_ghosts_score = 0.0
        for ghost in ghosts:
            ghost_dist = manhattanDistance(ghost, newPos)
            total_ghost_distance += ghost_dist
            if ghost_dist <= 1:
                near_ghosts_score += -1
        # We consider scared time avg as an positive factor too!
        scared_time_score = sum(newScaredTimes) / len(newScaredTimes)
        total_score = successorGameState.getScore()
        total_score += (1 / nearest_food_distance)
        total_score += -3 * (1 / total_ghost_distance)
        total_score += 5 * near_ghosts_score
        total_score += scared_time_score
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]

    def maxValue(self, gameState, agentIndex, depth):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            actions.append((self.minimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0], action))   
        return max(actions)

        
    
    def minValue(self, gameState, agentIndex, depth):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            actions.append((self.minimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0], action))    
        return min(actions)
    
    def minimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return ( self.evaluationFunction(gameState), "Stop")
        
        agentsNum = gameState.getNumAgents()
        agentIndex %=  agentsNum
        if agentIndex == agentsNum - 1:
            depth -= 1

        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)
    


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]


    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            v = self.minimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta)[0]
            actions.append((v, action))
            if v > beta:
                return (v, action)
            alpha = max(alpha, v)
        return max(actions)

        
    
    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            v = self.minimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta)[0]
            actions.append((v, action))
            if v < alpha:
                return (v, action)
            beta = min(beta, v)
        return min(actions)



    def minimax(self, gameState, agentIndex, depth, alpha = -999999, beta = 999999):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return ( self.evaluationFunction(gameState), "Stop")
        
        agentsNum = gameState.getNumAgents()
        agentIndex %=  agentsNum
        if agentIndex == agentsNum - 1:
            depth -= 1

        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

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
        "*** YOUR CODE HERE ***"
        return self.Expectimax(gameState, 0, self.depth)[1]

    def maxValue(self, gameState, agentIndex, depth):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            actions.append((self.Expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0], action))   
        return max(actions)


    def minValue(self, gameState, agentIndex, depth):
        actions = []
        total = 0
        for action in gameState.getLegalActions(agentIndex):
            v = self.Expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0]
            total += v
            actions.append((v, action))
        
        return (total / len(actions), )
    
    def Expectimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return ( self.evaluationFunction(gameState), "Stop")
        
        agentsNum = gameState.getNumAgents()
        agentIndex %=  agentsNum
        if agentIndex == agentsNum - 1:
            depth -= 1

        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    
    "*** YOUR CODE HERE ***"
    food_score = 10000
    for food in foods.asList():
        distance = manhattanDistance(pacmanPosition, food)
        if distance < food_score:
            food_score = float(distance)

    total_ghost_distance = 0.1
    near_ghost_score = 0.0
    for ghost in ghostPositions:
        distance = util.manhattanDistance(pacmanPosition, ghost)
        total_ghost_distance += distance
        if distance <= 1:
            near_ghost_score += -1

    scared_time_score = sum(scaredTimers) / len(scaredTimers)

    score = currentGameState.getScore()
    score += 1 / food_score
    score += -3 * (1 / total_ghost_distance)
    score += 5 * near_ghost_score
    score += scared_time_score

    return score

# Abbreviation
better = betterEvaluationFunction
