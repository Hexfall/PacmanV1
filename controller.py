from ghosts import GhostGroup, Ghost
from nodes import Node
from pacman import Pacman
from constants import *
from pellets import PelletGroup, Pellet
from math import inf
import numpy as np

MAX_DEPTH = 12
GREED_FACTOR = 4.5
MAX_FEAR = 7 # Don't take ghost further than this many nodes into account
FEAR_FACTOR = 7.5

class Controller(Pacman):
    def __init__(self, node: Node) -> None:
        Pacman.__init__(self, node)
        self.ghosts: GhostGroup = None
        self.pellets: PelletGroup = None
        self.pellet_map: list[list[bool]] = [[False for _ in range(NCOLS)] for _ in range(NROWS)]
        self.power_pellet_map: list[list[bool]] = [[False for _ in range(NCOLS)] for _ in range(NROWS)]
    
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
        for p in self.pellets.powerpellets:
            self.power_pellet_map[p.position.y//TILEHEIGHT][p.position.x//TILEWIDTH] = True
    
    def next_move(self) -> None:
        best = (inf, STOP) # The best move, minimum weight
        for d, n in self.node.neighbors.items():
            if n is None or d == PORTAL:
                continue
            if not PACMAN in self.node.access[d]:
                continue
            def get_weight(cur: Node, prev: Node, depth: int = 0, seen: set[Node] | None = None, power: bool = False) -> float:
                if seen is None:
                    seen = set()
                seen = seen.copy()
                seen.add(prev)
                if depth == MAX_DEPTH:
                    return 0
                ghost = self.has_ghost(prev, cur)
                if not power and ghost is not None:
                    ghost_dir = self.directions[ghost.direction]
                    ghost_dir = np.array([ghost_dir.x, ghost_dir.y])
                    to_pacman = self.position - ghost.position
                    to_pacman = np.array([to_pacman.x, to_pacman.y])
                    to_pacman = to_pacman / np.linalg.norm(to_pacman)
                    dir_dot = np.dot(to_pacman, ghost_dir)
                    #if dir_dot > 0:
                    #    return 0
                    if depth == 0 and dir_dot < .8:
                        return inf
                    if depth >= MAX_FEAR:
                        return 0
                    else:
                        return ((MAX_FEAR - depth)/3)**5 * FEAR_FACTOR * max((dir_dot + 1)/2, 0)
                s = -self.pellets_between(prev, cur) / ((depth + 1)**3) * GREED_FACTOR
                p = False
                if self.power_pellet_between(prev, cur) and depth < 4:
                    p = True
                for next in cur.neighbors.values():
                    if next is None or next in seen:
                        continue
                    s += get_weight(next, cur, depth + 1, seen, power or p)
                return s 
            
            weight = get_weight(n, self.node)
            
            best = min(best, (weight, d))
            
        print(f"Weight: {best[0]}")
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
        self.power_pellet_map[pellet.position.y//TILEHEIGHT][pellet.position.x//TILEWIDTH] = False
    
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

    def power_pellet_between(self, n1: Node, n2: Node) -> bool:
        minx = int(min(n1.position.x, n2.position.x) // TILEWIDTH)
        maxx = int(max(n1.position.x, n2.position.x) // TILEWIDTH)
        miny = int(min(n1.position.y, n2.position.y) // TILEHEIGHT)
        maxy = int(max(n1.position.y, n2.position.y) // TILEHEIGHT)

        if miny == maxy:
            for x in range(minx, maxx+1):
                if self.pellet_map[miny][x]:
                    return True

        elif minx == maxx:
            for y in range(miny, maxy+1):
                if self.pellet_map[y][minx]:
                    return True

        return False
    
    def has_ghost(self, n1: Node, n2: Node) -> Ghost | None:
        if n1.position.x == n2.position.x:
            for g in self.ghosts:
                if g.mode.current in [FREIGHT, SPAWN]:
                    continue
                miny = int(min(n1.position.y, n2.position.y))
                maxy = int(max(n1.position.y, n2.position.y))
                if g.position.x == n1.position.x and miny <= g.position.y <= maxy:
                    return g
        else:
            for g in self.ghosts:
                if g.mode.current in [FREIGHT, SPAWN]:
                    continue
                minx = int(min(n1.position.x, n2.position.x))
                maxx = int(max(n1.position.x, n2.position.x))
                if g.position.y == n1.position.y and minx <= g.position.x <= maxx:
                    return g
        return None
