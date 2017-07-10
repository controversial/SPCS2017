from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent

"""
General information
    Pac-Man is always agent 0, and the agents move in order of increasing agent
    index. Use self.index in your minimax implementation, but only Pac-Man will
    actually be running your MinimaxAgent.

    Functions are provided to get legal moves for Pac-Man or the ghosts and to
    execute a move by any agent. See GameState in pacman.py for details.

    All states in minimax should be GameStates, either passed in to getAction
    or generated via GameState.generateSuccessor. In this project, you will not
    be abstracting to simplified states.

    Use self.evaluationFunction in your definition of Vmax,min wherever you
    used Eval(s) in your on paper description

    The minimax values of the initial state in the minimaxClassic layout are 9,
    8, 7, -492 for depths 1, 2, 3 and 4 respectively. You can use these numbers
    to verify if your implementation is correct. Note that your minimax agent
    will often win (just under 50% of the time for us--be sure to test on a
    large number of games using the -n and -q flags) despite the dire
    prediction of depth 4 minimax.

    Here is an example call to run the program:
            python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
    Note that this is designed for python 2.7 based on print statements so you
    may need to replace "python" with "python2.7"

    You can assume that you will always have at least one action from which to
    choose in getAction. You are welcome to, but not required to, remove
    Directions.STOP as a valid action (this is true for any part of the
    assignment).

    If there is a tie between multiple actions for the best move, you may break
    the tie however you see fit.

    THIS IS THE ONLY FILE YOU NEED TO EDIT.
    pacman.py runs the pacman game and describes the GameState type
    game.py contains logic about the world - agent, direction, grid.
    util.py: data structures for implementing search algirthms
    graphicsDisplay.py does what the title says
    graphicsUtils.py is just more support for graphics
    textDisplay is just for ASCII displays
    ghostAgents are the agents that control the ghosts
    keyboardAgents is what allows you to control pacman
    layout.py reads files and stores their contents.
"""


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.    You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation
        function.

        getAction takes a GameState and returns some Directions.X for some X in
        the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food,
        capsules, agent configurations and score changes. In this function, the
        |gameState| argument is an object of GameState class. Following are a
        few of the helper methods that you can use to query a GameState object
        to gather information about the present state of Pac-Man, the ghosts
        and the maze.

        gameState.getLegalActions():
            Returns the legal actions for the agent specified. Returns
            Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the
            action. Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position state.direction
            gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look
        into that for other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [
            self.evaluationFunction(gameState, action) for action in legalMoves
        ]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores))
            if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are
        better.

        The code below extracts some useful information from the state, like
        the remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return successorGameState.getScore()


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
    This class provides some common elements to all of your multi-agent
    searchers. Any methods defined here will be available to the
    MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to add
    functionality to all your adversarial search agents. Please do not remove
    anything, however.

    Note: this is an abstract class: one that should not be instantiated. It's
    only partially specified, and designed to be extended. Agent (game.py) is
    another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

###############################################################################
# Problem 1: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. Terminal states can be found by one of the
        following: pacman won, pacman lost or there are no legal moves.

        Here are some method calls that might be useful when implementing
        minimax.

        gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

        Directions.STOP:
            The stop direction, which is always legal

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game

        gameState.isWin():
            Returns True if it's a winning state

        gameState.isLose():
            Returns True if it's a losing state

        self.depth:
            The depth to which search should continue


        This function should use Vmax,min to determine the best action for
        Pac-Man. Consider defining another function within this function that
        you just call to get your result. Hint: That function should be
        recursive, and the initial call should include self.depth as a
        parameter.

        One thing to consider is when the "depth" should decrement by one. Why
        are you decrementing? If you scroll up in init you can see that the
        default is "2" which means that you should go to depths 0, 1, and 2.
        It's easiest to do so by starting at depth 2, then going to depth 1,
        then depth 0, and on depth 0 doing "something special" (think about
        what is reasonable). Another thing to consider is when you should
        "stop."
        """

        def max_agent(state, depth):
            """Calculate the best move that pacman could make."""
            next_actions = state.getLegalActions(0)
            # Stopping is always a choice even when the program says it's not
            if len(next_actions) == 0: next_actions.insert(0, Directions.STOP)
            # Call min agent if the game is running and we're not at depth
            if depth <= self.depth and not (state.isWin() or state.isLose()):
                # Make a game state for each of the actions we could take
                next_states = [state.generateSuccessor(0, action) for action in next_actions]
                next_scores = [min_agent(state, depth + 1)[0] for state in next_states]
                m = max(next_scores)
                return m, next_scores.index(m)
            # Otherwise use score
            else:
                return self.evaluationFunction(state), 0


        def min_agent(state, depth):
            """Calculate the worst move that each of the ghosts could make."""
            # Record the possible next steps for each agent in a list of tuples
            next_actions = []
            for ghostNumber in range(1, state.getNumAgents()):
                for legalAction in state.getLegalActions(ghostNumber):
                    next_actions.append((ghostNumber, legalAction))
            if depth <= self.depth and not (state.isWin() or state.isLose()):
                # Make a game state for each of the actions we could take
                next_states = [state.generateSuccessor(actor, action) for actor, action in next_actions]
                next_scores = [max_agent(state, depth + 1)[0] for state in next_states]
                m = min(next_scores)
                return m, next_scores.index(m)
            else:
                return self.evaluationFunction(state), 0

        index_of_best_move = max_agent(gameState, 0)[1]

        next_actions = gameState.getLegalActions(0)
        if len(next_actions) == 0: next_actions.insert(0, Directions.STOP)

        return next_actions[index_of_best_move]


class GraphNode:
    def __init__(self, name='Node'):
        self.graph = None  # parent graph
        self.neighbors = {}  # adjacent nodes
        self.distance = float('inf')  # distance from node is infinity
        self.prev_node = None
        self.id = name

    def __float__(self):
        return float(self.distance)

    def __str__(self):
        return str(self.id)


class Graph:
    def __init__(self, nodes):
        self.neighbors = dict(zip(nodes, [{} for _ in nodes]))

    def __iter__(self):
        return iter(self.neighbors.keys())

    def add_connection(self, a, b, cost=1):
        self.neighbors[a][b] = cost
        self.neighbors[b][a] = cost
        a.neighbors = self.neighbors[a]
        b.neighbors = self.neighbors[b]


def run_dijkstra(graph, start, end):
    if start not in graph or end not in graph:
        raise ValueError(
            'Both the start and nodes must be in the provided graph'
        )
    visited = set()
    unvisited = set(graph)

    start.distance = 0
    while end.distance == float('inf'):
        current = min(unvisited, key=lambda x: x.distance)
        unvisited.remove(current)
        visited.add(current)
        adjs = current.neighbors.keys()
        for a in adjs:
            tentativeDistance = current.distance+current.neighbors[a]
            if tentativeDistance < a.distance:
                a.distance = tentativeDistance
                a.prev_node = current
    path = [end]
    # Trace path back to beginning
    while 1:
        current = path[-1]
        if current.prev_node:
            path.append(current.prev_node)
        elif current == start:
            break
    path.reverse()
    return path
