from ghosts import GhostGroup
from nodes import Node
from pacman import Pacman
from constants import *
from pellets import PelletGroup, Pellet
from math import inf

DIRECTIONS = [LEFT, UP, RIGHT, DOWN]
MAX_DEPTH = 10
GREED_FACTOR = 1.01 
MAX_FEAR = 5 # Don't take ghost further than this many nodes into account
FEAR_FACTOR = .5

class Controller(Pacman):
    def __init__(self, node: Node) -> None:
        Pacman.__init__(self, node)
        self.ghosts: GhostGroup = None
        self.pellets: PelletGroup = None
        self.pellet_map: list[list[bool]] = [[False for _ in range(NCOLS)] for _ in range(NROWS)]
    
    def update(self, dt) -> None:
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt
        if self.overshotTarget():
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
            self.pellet_map[p.position.y//TILEHEIGHT][p.position.x//TILEWIDTH] = True
    
    def next_move(self) -> None:
        best = (inf, STOP) # The best move, minimum weight
        for d, n in self.node.neighbors.items():
            if n is None or d == PORTAL:
                continue
            def get_weight(cur: Node, prev: Node, depth: int = 0, seen: set[Node] | None = None) -> float:
                if seen is None:
                    seen = set()
                seen = seen.copy()
                seen.add(prev)
                if depth == MAX_DEPTH:
                    return 0
                if self.has_ghost(prev, cur):
                    if depth >= MAX_FEAR:
                        return 0
                    else:
                        return (MAX_FEAR - depth)**2 * FEAR_FACTOR
                s = -self.pellets_between(prev, cur) / ((depth + 1)**2)
                for next in cur.neighbors.values():
                    if next is None or next in seen:
                        continue
                    s += get_weight(next, cur, depth + 1, seen)
                return s * GREED_FACTOR
            
            weight = get_weight(n, self.node)
            
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
        self.pellet_map[pellet.position.y//TILEHEIGHT][pellet.position.x//TILEWIDTH] = False
    
    # This may double count pellets on nodes. I don't care enough to account for it.
    def pellets_between(self, n1: Node, n2: Node) -> int:
        c = 0
        minx = int(min(n1.position.x, n2.position.x) // TILEWIDTH)
        maxx = int(max(n1.position.x, n2.position.x) // TILEWIDTH)
        miny = int(min(n1.position.y, n2.position.y) // TILEHEIGHT)
        maxy = int(max(n1.position.y, n2.position.y) // TILEHEIGHT)
        
        if miny == maxy:
            for x in range(minx, maxx+1):
                if self.pellet_map[miny][x]:
                    c += 1
        
        elif minx == maxx:
            for y in range(miny, maxy+1):
                if self.pellet_map[y][minx]:
                    c += 1
        
        return c
    
    def has_ghost(self, n1: Node, n2: Node) -> bool:
        if n1.position.x == n2.position.x:
            for g in self.ghosts:
                if g.mode.current == FREIGHT:
                    continue
                miny = int(min(n1.position.y, n2.position.y))
                maxy = int(max(n1.position.y, n2.position.y))
                if g.position.x == n1.position.x and miny <= g.position.y <= maxy:
                    return True
        else:
            for g in self.ghosts:
                if g.mode.current == FREIGHT:
                    continue
                minx = int(min(n1.position.x, n2.position.x))
                maxx = int(max(n1.position.x, n2.position.x))
                if g.position.y == n1.position.y and minx <= g.position.x <= maxx:
                    return True
        return False
