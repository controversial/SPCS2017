from game import Agent
import random


class Team7GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, gameState):
        return random.choice(gameState.getLegalActions(self.index))
