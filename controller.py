from pacman import Pacman
from constants import *


DIRECTIONS = [LEFT, UP, RIGHT, DOWN]

class Controller(Pacman):
    def __init__(self, node):
        Pacman.__init__(self, node)
    
    def update(self, dt):
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt
        if self.overshotTarget():
            self.node = self.target
        for d in DIRECTIONS:
            if self.validDirection(d):
                self.direction = d
                self.target = self.getNewTarget(d)
                break
        