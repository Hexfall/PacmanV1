from ghosts import GhostGroup, Ghost
from nodes import Node
from pacman import Pacman
from constants import *
from pellets import PelletGroup, Pellet
from math import inf
import numpy as np

from vector import Vector2

MAX_DEPTH = 14
GREED_FACTOR = 25
GHOST_VALUE = -20
MAX_FEAR = 8 # Don't take ghost further than this many nodes into account
FEAR_FACTOR = 1.5
BOARD_RADIUS = Vector2(NROWS * TILEHEIGHT, NCOLS * TILEWIDTH).magnitude()

class Controller(Pacman):
    def __init__(self, node: Node) -> None:
        Pacman.__init__(self, node)
        self.ghosts: GhostGroup = None
        self.pellets: PelletGroup = None
        self.pellet_map: list[list[bool]] = [[False for _ in range(NCOLS)] for _ in range(NROWS)]
        self.power_pellet_map: list[list[bool]] = [[False for _ in range(NCOLS)] for _ in range(NROWS)]
        self.col: int = node.position.x // TILEWIDTH
        self.row: int = node.position.y // TILEHEIGHT
        self.flipped = False
        self.ghost_multiplier: int = 1
        self.ghosts_killed: set[Ghost] = set()
    
    # Updated col and row values, returns true if values changed, false otherwise.
    def update_coords(self) -> bool:
        c = self.position.x // TILEWIDTH
        r = self.position.y // TILEHEIGHT
        if c != self.col or r != self.row:
            self.col = c
            self.row = r
            return True
        return False
    
    def update(self, dt) -> None:
        self.sprites.update(dt)
        self.position += self.directions[self.direction]*self.speed*dt
        
        # Update Ghost multiplier
        for g in self.ghosts:
            if g in self.ghosts_killed:
                continue
            if g.mode.current == SPAWN:
                self.ghosts_killed.add(g)
                self.ghost_multiplier *= 4
        
        if self.overshotTarget():
            self.flipped = False
            if self.target.neighbors[PORTAL] is not None:
                self.node = self.target.neighbors[PORTAL]
                self.target = self.node.neighbors[self.direction]
            else:
                self.node = self.target
                self.next_move()
            self.setPosition()
        elif self.update_coords():
            self.eval_flip()
    
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
            
            weight = self.get_weight(n, self.node)
            
            best = min(best, (weight, d))
            
        print(f"Weight: {best[0]}")
        self.direction = best[1]
        self.target = self.getNewTarget(self.direction)

    def get_weight(self, cur: Node, prev: Node, depth: int = 0, seen: set[Node] | None = None, power: bool = False, between: bool = False) -> float:
        s = 0
        if seen is None:
            seen = set()
        seen = seen.copy()
        seen.add(prev)
        if depth == MAX_DEPTH:
            return 0
        ghosts = self.has_ghost(prev, cur)
        if not power and len(ghosts) != 0:
            dir_dot = -1
            dangerous = False
            danger = 0
            for ghost in ghosts:
                if ghost.mode.current == SPAWN:
                    continue
                if ghost.mode.current == FREIGHT:
                    if ghost.mode.timer / ghost.mode.time < .85:
                        s += GHOST_VALUE * GREED_FACTOR * self.ghost_multiplier
                        continue
                dangerous = True
                ghost_dir = self.directions[ghost.direction]
                ghost_dir = np.array([ghost_dir.x, ghost_dir.y])
                to_node = prev.position - ghost.position
                to_node = np.array([to_node.x, to_node.y])
                to_node = to_node / np.linalg.norm(to_node)
                dir_dot = max(np.dot(to_node, ghost_dir), dir_dot)
                if depth == 0 and dir_dot > .8:
                    return inf
                danger = max(
                    danger,
                    ((MAX_FEAR - depth+4)/3)**5 * FEAR_FACTOR * max((dir_dot + 1)/2, 0) / (1 + (self.position - ghost.position).magnitude() / BOARD_RADIUS)
                )
            #if dir_dot > 0:
            #    return 0
            if dangerous:
                if depth >= MAX_FEAR:
                    return 0
                else:
                    return danger
        if between:
            pellets = 0
        else:
            pellets = -self.pellets_between(prev, cur)
        s += pellets * GREED_FACTOR
        p = False
        if self.power_pellet_between(prev, cur) and depth < 4:
            p = True
        for next in cur.neighbors.values():
            if next is None or next in seen:
                continue
            s += self.get_weight(next, cur, depth + 1, seen, power or p)
        return s / (1 + cur.position.magnitude() / BOARD_RADIUS) / (depth + 1)

    def eatPellets(self, pellet_list: list[Pellet]) -> None | Pellet:
        for pellet in pellet_list:
            if self.collideCheck(pellet):
                self.remove_pellet(pellet)
                return pellet
        return None

    def remove_pellet(self, pellet: Pellet) -> None:
        self.pellet_map[pellet.position.y//TILEHEIGHT][pellet.position.x//TILEWIDTH] = False
        if self.power_pellet_map[pellet.position.y//TILEHEIGHT][pellet.position.x//TILEWIDTH]:
            self.ghost_multiplier = 1
            self.ghosts_killed = set()
            for g in self.ghosts:
                if g.mode.current == SPAWN:
                    self.ghosts_killed.add(g)
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
                if self.power_pellet_map[miny][x]:
                    return True

        elif minx == maxx:
            for y in range(miny, maxy+1):
                if self.power_pellet_map[y][minx]:
                    return True

        return False
    
    def has_ghost(self, n1: Node, n2: Node) -> list[Ghost]:
        l = []
        if n1.position.x == n2.position.x:
            for g in self.ghosts:
                miny = int(min(n1.position.y, n2.position.y))
                maxy = int(max(n1.position.y, n2.position.y))
                if g.position.x == n1.position.x and miny <= g.position.y <= maxy:
                    l.append(g)
        else:
            for g in self.ghosts:
                minx = int(min(n1.position.x, n2.position.x))
                maxx = int(max(n1.position.x, n2.position.x))
                if g.position.y == n1.position.y and minx <= g.position.x <= maxx:
                    l.append(g)
        return l

    def eval_flip(self):
        if self.get_weight(self.target, self.node, between=self.flipped) > self.get_weight(self.node, self.target, between=not self.flipped):
            self.direction = -self.direction
            self.target, self.node = self.node, self.target
            self.flipped = not self.flipped
