from util import manhattanDistance
from game import Directions
import random
import util
import time

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


###############################################################################


###############################################################################


###############################################################################


# ################### FINAL PROJECT PACMAN AGENT FUNCTION ################### #


###############################################################################


###############################################################################


###############################################################################


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

    def add_connection(self, a, b, cost=1):
        self.neighbors[a][b] = cost
        self.neighbors[b][a] = cost
        a.neighbors = self.neighbors[a]
        b.neighbors = self.neighbors[b]

    def __iter__(self):
        return iter(self.neighbors.keys())

    def __str__(self):
        out = ""
        for n in self.neighbors:
            out += str(n)
            out += ": "
            out += str([str(l) for l in self.neighbors[n]])
            out += "\n"
        return out

    def dijkstra(self, start, end):
        if start not in self or end not in self:
            raise ValueError(
                "Both the start and end nodes must be in the provided graph"
            )
        visited = set()
        unvisited = set(self)

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
            else:
                return []
        path.reverse()
        return path


class Test7PacmanAgent(MultiAgentSearchAgent):
    def buildGraph(self, gameState):
        """Rebuild the graph of path squares to pathfind through"""
        w = gameState.data.layout.width
        h = gameState.data.layout.height
        self.layoutCoords = [(x, y) for x in range(w) for y in range(h)]
        # Add to a list all the coordinates of the path tiles.
        self.pathCoords = []
        for x, y in self.layoutCoords:
            if not gameState.getWalls()[x][y]:
                self.pathCoords.append((x, y))
        # Make a GraphNode for each path coordinate
        self.pathNodes = [GraphNode(coord) for coord in self.pathCoords]
        # Build a graph from the list of path tiles
        self.pathGraph = Graph(self.pathNodes)
        # Connect adjacent path tiles in the graph
        for x, y in self.pathCoords:
            neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            for neighbor in neighbors:
                if neighbor in self.pathCoords:
                    this = self.getPathNode(x, y)
                    other = self.getPathNode(*neighbor)
                    self.pathGraph.add_connection(this, other)

    def getPathNode(self, x, y):
        """Get a Node object in self.pathGraph for an (x, y) point"""
        if (x, y) in self.pathCoords:
            return self.pathNodes[self.pathCoords.index((x, y))]
        else:
            raise IndexError("{0} is not a path node".format((x, y)))

    def pathfind(self, a, b):
        """Find the path between two points on the grid"""
        self.buildGraph(self.gameState)
        # When ghosts are scared, they can appear halfway between grid points
        # because they move slower. Flooring is a simple solution to this
        # problem.
        a = (int(a[0]), int(a[1]))
        b = (int(b[0]), int(b[1]))
        return self.pathGraph.dijkstra(
            self.getPathNode(*a),
            self.getPathNode(*b)
        )

    def pathfindFromPacman(self, x, y):
        """Find the path from pacman to any path point"""
        return self.pathfind(
            self.gameState.getPacmanPosition(),
            (x, y)
        )

    def manhattan(self, a, b):
        """Find the manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def manhattanFromPacman(self, x, y):
        return self.manhattan(
            self.gameState.getPacmanPosition(),
            (x, y)
        )

    def getClosestFoodToPacman(self):
        """Return the (x, y) coordinate of the geographically closest food on
        the board."""
        return min(
            self.gameState.getFood().asList(),
            key=lambda loc: self.manhattanFromPacman(*loc)
        )

    def getClosestGhostToPacman(self):
        """Return the (x, y) coordinate of the closest ghost on the board."""
        return min(
            self.gameState.getGhostPositions(),
            key=lambda loc: len(self.pathfindFromPacman(*loc))
        )

    def getScaredGhosts(self):
        return [
            self.gameState.getGhostPosition(i)
            for i in range(1, self.gameState.getNumAgents())
            if self.gameState.getGhostState(i).scaredTimer > 0
        ]

    def getClosestScaredGhostToPacman(self):
        """Return the (x, y) coordinate of the closest scared ghost on the
        board. Errors if there aren't any, so check getScaredGhosts first."""
        # List of positions of all ghosts
        return min(
            self.getScaredGhosts(),
            key=lambda loc: len(self.pathfindFromPacman(*loc))
        )

    def getClosestCapsuleToPacman(self):
        """Return the (x, y) coordinate of the geographically closest ghost on
        the board."""
        return min(
            self.gameState.getCapsules(),
            key=lambda loc: len(self.pathfindFromPacman(*loc))
        )

    def getOptimalCapsulePosition(self):
        """If both pacman and a ghost are near a capsule, return the position of
        that capsule. Otherwise, return None."""
        capsulePositions = self.gameState.getCapsules()
        ghostPositions = self.gameState.getGhostPositions()

        # Build a list of places where pacman, a ghost, and a capsule are all
        # within 5 spaces of eachother. Includes the ghost and capsule
        # locations.
        workingCombos = []
        for gp in ghostPositions:
            for cp in capsulePositions:
                # Is the ghost within 10 spaces of the capsule
                ghostCapsulePass = len(self.pathfind(gp, cp)) < 10
                # Is pacman within 10 spaces of the capsule
                pacmanCapsulePass = len(self.pathfindFromPacman(*cp)) < 10
                # Is pacman within 10 spaces of the ghost
                # pacmanGhostPass = len(self.pathfindFromPacman(*gp)) < 10
                # If all 3 are within 5 of eachother
                if all([ghostCapsulePass, pacmanCapsulePass]):
                    workingCombos.append((gp, cp))
        if len(workingCombos) is 0:
            return None
        else:
            return workingCombos[0][1]

    def getActionToCoords(self, coords):
        """Get the action that moves pacman to the provided coordinates.
        Coordinates should be valid path coordinates that are directly
        adjacent to pacman."""
        goalX, goalY = coords
        pacX, pacY = self.gameState.getPacmanPosition()

        if (pacX - 1 == goalX and pacY == goalY):
            return Directions.WEST
        if (pacY - 1 == goalY and pacX == goalX):
            return Directions.SOUTH
        if (pacX + 1 == goalX and pacY == goalY):
            return Directions.EAST
        if (pacY + 1 == goalY and pacX == goalX):
            return Directions.NORTH
        else:
            raise ValueError(
                "No action leads pacman directly to {0}".format(coords)
            )

    def getActionTowards(self, coords):
        """Get the action that takes pacman towards the provided coordinates"""
        pathToFood = [x.id for x in self.pathfindFromPacman(*coords)]
        firstStep = pathToFood[1]  # index 0 is pacman
        return self.getActionToCoords(firstStep)

    def getAction(self, gameState):
        """Decide which action pacman should take"""
        self.gameState = gameState

        data = gameState.data
        layout = data.layout

        # NOTE: COORDS ARE FROM BOTTOM LEFT
        # NOTE: state.getGhostPositions() -> list of tuples
        # NOTE: state.getPacmanPosition() -> tuple
        # NOTE: state.getCapsules() -> list of tuples
        # NOTE: state.getFood().asList() -> list of tuples
        # NOTE: state.getWalls().asList() -> list of tuples
        # NOTE: if a ghost is scared, state.getGhostState(i).scaredTimer > 0

        t1 = time.time()

        # Decisions!
        optcap = self.getOptimalCapsulePosition()
        if len(self.getScaredGhosts()) > 0:
            print("[info] Pathfinding to closest scared ghost")
            closestScared = self.getClosestScaredGhostToPacman()
            answer = self.getActionTowards(closestScared)
        elif optcap:
            print('[info] Pathfinding towards pellet to eat proximate ghost')
            answer = self.getActionTowards(optcap)
        else:
            print("[info] Pathfinding towards closest food")
            answer = self.getActionTowards(self.getClosestFoodToPacman())

        print("[info] Chose move in %.4f seconds" % (time.time() - t1))
        return answer
