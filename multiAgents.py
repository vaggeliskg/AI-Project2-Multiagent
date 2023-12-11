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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        if currentGameState.isWin():
            return float('inf')
    
        if currentGameState.isLose():
            return float('-inf')
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        food_list = newFood.asList()
        food_dist = []
        ghost_dist = []
        score = 0                                               # initialize score with 0
        ghost_pos = successorGameState.getGhostPositions()      # get ghost position

        for food in food_list:
            food_dist.append(manhattanDistance(newPos,food))    # calculate manhattan distances to foods

        if len(food_dist):                                      # if food_dist is not empty
            closest_food = min(food_dist)                       # take the min food 
            score += 1/closest_food                             # and do +1/closest_food so that it goes to the closest food first

        for ghost in ghost_pos:
            ghost_dist.append(manhattanDistance(newPos,ghost))
            
        closest_ghost =  min(ghost)                             # if a ghost is very close to you then subtract 
        if closest_ghost <= 1:                                  # a sufficient amount from the score to avoid going there 
            score -= 100/closest_ghost 
        
        "*** YOUR CODE HERE***"
        return score + successorGameState.getScore()            # finally return the score and + the successorGameState.getScore() to improte results 

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        # when its pacmans turn
        def max_val(gameState,depth):
            if depth == self.depth or gameState.isLose() or gameState.isWin():  # check terminal conditions
                return (self.evaluationFunction(gameState),None)                # it needs to be a tuple just because we use result[0] later 
            m = -float('inf')                                                   # which will produce a typerror for float accessing if i dont always return tuples
            optimal_action = None

            for action in gameState.getLegalActions(0):                         # check all actions
                successor = gameState.generateSuccessor(0, action)              # get the successor
                result = min_val(successor,1,depth)                             # and call min (ghost's turn)
                first_element = result[0]
                if first_element > m:                                           # update the neccesary value
                    m =  first_element
                    optimal_val, optimal_action = m,action
            return (optimal_val,optimal_action)
        
        def min_val(gameState,agentIndex,depth):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)            
            m = float('inf')
            optimal_action = None
            
            if(agentIndex == gameState.getNumAgents() - 1):                    # check if the next agent will be pacman
                next_agent = 0
            else:
                next_agent = agentIndex + 1                                    # else add + 1 to the index for the next ghost
            
            for action in gameState.getLegalActions(agentIndex):
                    if(next_agent == 0):
                        successor = gameState.generateSuccessor(agentIndex,action)  
                        result = max_val(successor,depth + 1)                  # if pacman is next increase depth and call max
                        first_element = result[0]
                    else:
                        successor = gameState.generateSuccessor(agentIndex,action)
                        result = min_val(successor,agentIndex + 1 ,depth)     # if a ghost is next just recursively call min
                        first_element = result[0]
                    if first_element < m:
                        m = first_element 
                        optimal_val, optimal_action = m,action
            return (optimal_val,optimal_action)
        
        first_act = max_val(gameState,0)[1]                                   # start at root by calling max_val
        return first_act                                                      # return actions

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # same algorithm used in minimax but no we need a,b to decide when to prune
        a = -float('inf')                       
        b =  float('inf')

        def max_val(gameState,depth,a,b):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)             
            m = -float('inf')
            optimal_action = None

            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                result = min_val(successor,1,depth,a,b)
                first_element = result[0]
                if first_element > m:
                    m =  first_element
                    optimal_val, optimal_action = m,action

                if m > b:                                      # pruning using the pseudocode given
                    return (m,None)

                a = max(a,m)
            return (optimal_val,optimal_action)
        
        def min_val(gameState,agentIndex,depth,a,b):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)            
            m = float('inf')
            optimal_action = None
            
            if(agentIndex == gameState.getNumAgents() - 1):
                next_agent = 0
            else:
                next_agent = agentIndex + 1
            
            for action in gameState.getLegalActions(agentIndex):
                    if(next_agent == 0):
                        successor = gameState.generateSuccessor(agentIndex,action)
                        result = max_val(successor,depth + 1,a,b)
                        first_element = result[0]
                    else:
                        successor = gameState.generateSuccessor(agentIndex,action)
                        result = min_val(successor,agentIndex + 1 ,depth,a,b)
                        first_element = result[0]

                    if first_element < m:                                 
                        m = first_element 
                        optimal_val, optimal_action = m,action
                    
                    if m < a:                                        # pruning using the pseudocode given
                        return (m,None)
                    
                    b = min(b,m)

            return (optimal_val,optimal_action)
        
        first_act = max_val(gameState,0,a,b)[1]
        return first_act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # same algorithm but now min is using expectiminimax values(probabilities)
        def max_val(gameState,depth):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)               
            m = float('-inf')
            optimal_action = None
            

            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                result = min_val(successor,1,depth)
                first_element = result[0]
                if first_element > m:
                    m =  first_element
                    optimal_action = action
            return (m,optimal_action)
        
        def min_val(gameState,agentIndex,depth):
            if depth == self.depth or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState),None)            
            value = 0                                                               # initialize the value
            random_action = None
            
            if(agentIndex == gameState.getNumAgents() - 1):
                next_agent = 0
            else:
                next_agent = agentIndex + 1
                
            probability = 1 / len(gameState.getLegalActions(agentIndex))        # calculate action probability
            for action in gameState.getLegalActions(agentIndex):
                if(next_agent == 0):
                    successor = gameState.generateSuccessor(agentIndex,action)
                    result = max_val(successor,depth + 1)
                    first_element = result[0]
                    value += first_element * probability                       # update value using probability and the value returned
                    random_action = action                                     # update actions
                
                else:
                    successor = gameState.generateSuccessor(agentIndex,action)
                    result = min_val(successor,agentIndex + 1 ,depth)
                    first_element = result[0]
                    value += first_element * probability
                    random_action = action
            
            return (value,random_action)
        
        first_act = max_val(gameState,0)[1]
        return first_act
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    # check terminal conditions

    if currentGameState.isWin():
        return float('inf')
    
    if currentGameState.isLose():
       return float('-inf')
    
    pos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    food_list = newFood.asList()
    food_dist = []
    ghost_dist = []
    score = 0                                               # initialize score to make decisions later              
    ghost_pos = currentGameState.getGhostPositions()      

    # calculate distance to closest food
    for food in food_list:
        food_dist.append(manhattanDistance(pos,food))                             
    closest_food = min(food_dist)                                        

    # calculate distance to closest ghost
    for ghost in ghost_pos:
        ghost_dist.append(manhattanDistance(pos,ghost))
    closest_ghost =  min(ghost)                                                              
    
    # finally update the score and add to currentGameState.getScore() like q13
    score += 1/closest_food - (1/100 *closest_ghost)
    return score + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
