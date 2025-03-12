from ghosts import GhostGroup
from nodes import Node
from pacman import Pacman
from constants import *
from pellets import PelletGroup, Pellet
from math import inf

DIRECTIONS = [LEFT, UP, RIGHT, DOWN]

class Controller(Pacman):
    def __init__(self, node: Node) -> None:
        Pacman.__init__(self, node)
        self.ghosts: GhostGroup = None
        self.pellets: PelletGroup = None
        self.pellet_map: list[list[bool]] = [[False] * NROWS] * NCOLS
    
    def update(self, dt) -> None:
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt
        if self.overshotTarget():
            print("ping")
            if self.target.neighbors[PORTAL] is not None:
                self.node = self.target.neighbors[PORTAL]
                self.target = self.node.neighbors[self.direction]
            else:
                self.node = self.target
                self.next_move()
            self.setPosition()
    
    def set_ghosts(self, ghosts: GhostGroup) -> None:
        self.ghosts = ghosts
        
    def set_pellets(self, pellets: PelletGroup) -> None:
        self.pellets = pellets
        for p in self.pellets.pelletList:
            self.pellet_map[p.position.x//TILEWIDTH][p.position.y//TILEHEIGHT] = True
    
    def next_move(self) -> None:
        best = (inf, STOP) # The best move, minimum weight
        for d, n in self.node.neighbors.items():
            if n is None or d == PORTAL:
                continue
            weight = 0
            self.target: Node = n
            self.direction = d
            
            best = min(best, (weight, d))
            
        self.direction = best[1]
        self.target = self.getNewTarget(self.direction)

    def eatPellets(self, pellet_list: list[Pellet]) -> None | Pellet:
        for pellet in pellet_list:
            if self.collideCheck(pellet):
                self.remove_pellet(pellet)
                return pellet
        return None

    def remove_pellet(self, pellet: Pellet) -> None:
        self.pellet_map[pellet.position.x//TILEWIDTH][pellet.position.y//TILEHEIGHT] = False
    
    def pellets_between(self, n1: Node, n2: Node) -> int:
        c = 0
        minx = min(n1.position.x, n2.position.x) // TILEWIDTH
        maxx = max(n1.position.x, n2.position.x) // TILEWIDTH
        miny = min(n1.position.y, n2.position.y) // TILEHEIGHT
        maxy = max(n1.position.y, n2.position.y) // TILEHEIGHT
        
        if miny == maxy:
            for x in range(minx, maxx+1):
                if self.pellet_map[x][miny]:
                    c += 1
        
        elif minx == maxx:
            for y in range(miny, maxy+1):
                if self.pellet_map[minx][y]:
                    c += 1
        
        return c
