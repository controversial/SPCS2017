from __future__ import print_function
import time
from game import Directions, Agent


oppositeActions = {
    Directions.NORTH: Directions.SOUTH,
    Directions.EAST: Directions.WEST,
    Directions.SOUTH: Directions.NORTH,
    Directions.WEST: Directions.EAST
}

counterclockwiseActions = {
    Directions.NORTH: Directions.WEST,
    Directions.WEST: Directions.SOUTH,
    Directions.SOUTH: Directions.EAST,
    Directions.EAST: Directions.NORTH
}


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


###############################################################################


###############################################################################


###############################################################################


# ######################## FINAL PROJECT PACMAN AGENT ####################### #


###############################################################################


###############################################################################


###############################################################################


class Test7PacmanAgent(Agent):
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

    def pathfind(self, a, b, state=None):
        """Find the path between two points on the grid"""
        if state is None:
            state = self.gameState
        self.buildGraph(state)
        # When ghosts are scared, they can appear halfway between grid points
        # because they move slower. Flooring is a simple solution to this
        # problem.
        a = (int(a[0]), int(a[1]))
        b = (int(b[0]), int(b[1]))
        return self.pathGraph.dijkstra(
            self.getPathNode(*a),
            self.getPathNode(*b)
        )

    def pathfindFromPacman(self, x, y, state=None):
        """Find the path from pacman to any path point"""
        if state is None:
            state = self.gameState
        return self.pathfind(
            state.getPacmanPosition(),
            (x, y),
            state
        )

    def manhattan(self, a, b):
        """Find the manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def manhattanFromPacman(self, x, y, state=None):
        if state is None:
            state = self.gameState
        return self.manhattan(
            state.getPacmanPosition(),
            (x, y)
        )

    def getClosestFoodToPacman(self, state=None):
        """Return the (x, y) coordinate of the geographically closest food on
        the board."""
        if state is None:
            state = self.gameState
        return min(
            state.getFood().asList(),
            key=lambda loc: self.manhattanFromPacman(*loc, state=state)
        )

    def getClosestNonScaredGhostToPacman(self, state=None):
        """Return the (x, y) coordinate of the closest ghost on the board that
        isn't scared."""
        if state is None:
            state = self.gameState
        nonScaredGhosts = set(state.getGhostPositions()) - set(self.getScaredGhosts(state))
        if len(nonScaredGhosts) == 0:
            return None
        else:
            return min(
                nonScaredGhosts,
                key=lambda loc: len(self.pathfindFromPacman(*loc, state=state))
            )

    def getScaredGhosts(self, state=None):
        if state is None:
            state = self.gameState
        return [
            state.getGhostPosition(i)
            for i in range(1, state.getNumAgents())
            if state.getGhostState(i).scaredTimer > 0
        ]

    def getClosestScaredGhostToPacman(self, state=None):
        """Return the (x, y) coordinate of the closest scared ghost on the
        board. Errors if there aren't any, so check getScaredGhosts first."""
        # List of positions of all ghosts
        if state is None:
            state = self.gameState
        return min(
            self.getScaredGhosts(state),
            key=lambda loc: len(self.pathfindFromPacman(*loc, state=state))
        )

    def getClosestCapsuleToPacman(self, state=None):
        """Return the (x, y) coordinate of the geographically closest ghost on
        the board."""
        if state is None:
            state = self.gameState
        return min(
            state.getCapsules(),
            key=lambda loc: len(self.pathfindFromPacman(*loc, state=state))
        )

    def getOptimalCapsulePosition(self, state=None):
        """If both pacman and a ghost are near a capsule, return the position of
        that capsule. Otherwise, return None."""
        if state is None:
            state = self.gameState
        capsulePositions = state.getCapsules()
        ghostPositions = state.getGhostPositions()

        # Build a list of places where pacman, a ghost, and a capsule are all
        # within 5 spaces of eachother. Includes the ghost and capsule
        # locations.
        workingCombos = []
        for gp in ghostPositions:
            for cp in capsulePositions:
                # Is the ghost within 10 spaces of the capsule
                ghostCapsulePass = len(self.pathfind(gp, cp, state)) < 10
                # Is pacman within 10 spaces of the capsule
                pacmanCapsulePass = len(
                    self.pathfindFromPacman(*cp, state=state)
                ) < 10
                # Is pacman within 10 spaces of the ghost
                # pacmanGhostPass = len(self.pathfindFromPacman(*gp)) < 10
                # If all 3 are within 5 of eachother
                if all([ghostCapsulePass, pacmanCapsulePass]):
                    workingCombos.append((gp, cp))
        if len(workingCombos) is 0:
            return None
        else:
            return workingCombos[0][1]

    def getActionToCoords(self, coords, state=None):
        """Get the action that moves pacman to the provided coordinates.
        Coordinates should be valid path coordinates that are directly
        adjacent to pacman."""
        if state is None:
            state = self.gameState
        goalX, goalY = coords
        pacX, pacY = state.getPacmanPosition()

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

    def getActionTowards(self, coords, state=None):
        """Get the action that takes pacman towards the provided coordinates"""
        if state is None:
            state = self.gameState
        pathTo = [x.id for x in self.pathfindFromPacman(*coords, state=state)]
        firstStep = pathTo[1]  # index 0 is pacman's position
        return self.getActionToCoords(firstStep, state)

    def getActionAwayFrom(self, coords, state=None):
        """Get the action that takes pacman away from the provided
        coordinates"""
        if state is None:
            state = self.gameState
        possibleMoves = state.getLegalActions(0)
        pacX, pacY = state.getPacmanPosition()
        actionTowardsGhost = self.getActionTowards(coords)
        idealAction = oppositeActions[actionTowardsGhost]
        action = idealAction
        tries = 0
        while (action not in possibleMoves) or (action == actionTowardsGhost):
            action = counterclockwiseActions[action]
            tries += 1
            if tries > 4:
                print("... SUICIDE")
                return actionTowardsGhost  # If trapped, don't waste time
        return action

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

        # There's a nearby ghost - run
        nonScaredGhost = self.getClosestNonScaredGhostToPacman()
        if nonScaredGhost and len(self.pathfindFromPacman(*nonScaredGhost)) < 5:
            print("Avoiding ghost", end="")
            answer = self.getActionAwayFrom(nonScaredGhost)
        # There's a scared ghost on the board - chase
        elif len(self.getScaredGhosts()) > 0:
            print("Pathfinding to closest scared ghost", end="")
            closestScared = self.getClosestScaredGhostToPacman()
            answer = self.getActionTowards(closestScared)
        # Pacman and a ghost are both close to a capsule: target capsule
        elif optcap:
            print("Pathfinding towards pellet to eat proximate ghost", end="")
            answer = self.getActionTowards(optcap)
        # Otherwise idly aim for any food on the board
        else:
            print("Pathfinding towards closest food", end="")
            answer = self.getActionTowards(self.getClosestFoodToPacman())

        print("... %.4f seconds" % (time.time() - t1))
        return answer
