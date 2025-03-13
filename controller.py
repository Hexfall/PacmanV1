from ghosts import GhostGroup, Ghost
from nodes import Node
from pacman import Pacman
from constants import *
from pellets import PelletGroup, Pellet
from math import inf
import numpy as np

from vector import Vector2

MAX_DEPTH = 14
GREED_FACTOR = 80
GHOST_VALUE = -20
MAX_FEAR = 8 # Don't take ghost further than this many nodes into account
FEAR_FACTOR = 1.5
BOARD_RADIUS = Vector2(NROWS * TILEHEIGHT, NCOLS * TILEWIDTH).magnitude()

class Controller(Pacman):
    def __init__(self, node: Node) -> None:
        Pacman.__init__(self, node)
        self.ghosts: GhostGroup = None   # Ghost collection
        self.pellets: PelletGroup = None # Pellet collection
        self.pellet_map: list[list[bool]] = [[False for _ in range(NCOLS)] for _ in range(NROWS)]       # List of pellet locations
        self.power_pellet_map: list[list[bool]] = [[False for _ in range(NCOLS)] for _ in range(NROWS)] # List of power pellet locations
        self.col: int = node.position.x // TILEWIDTH  # Pacman's column
        self.row: int = node.position.y // TILEHEIGHT # Pacman's row
        self.flipped: bool = False             # For keeping track of whether the pellets between nodes can be collected or not
        self.ghost_multiplier: int = 1         # Score multiplier for killing ghosts
        self.ghosts_killed: set[Ghost] = set() # Ghosts that have already been killed, for keeping track of multiplier
    
    # Updates col and row values, returns true if values changed, false otherwise
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
            # Reached intersection, choose new best direction
            self.flipped = False
            if self.target.neighbors[PORTAL] is not None:
                # If crossing portal, update location.
                self.node = self.target.neighbors[PORTAL]
                self.target = self.node.neighbors[self.direction]
            else:
                # Decide on new move
                self.node = self.target
                self.next_move()
            # Fix offset (and change position to match portal)
            self.setPosition()
        elif self.update_coords():
            # Evaluate whether we should flip our direction. Only happens upon coord update so it's not called too often
            self.eval_flip()
    
    # Set ghost list
    def set_ghosts(self, ghosts: GhostGroup) -> None:
        self.ghosts = ghosts
    
    # Set pellet list and update pellet maps
    def set_pellets(self, pellets: PelletGroup) -> None:
        self.pellets = pellets
        for p in self.pellets.pelletList:
            self.pellet_map[p.position.y//TILEHEIGHT][p.position.x//TILEWIDTH] = True
        for p in self.pellets.powerpellets:
            self.power_pellet_map[p.position.y//TILEHEIGHT][p.position.x//TILEWIDTH] = True
    
    # Get the next best move at an intersection
    def next_move(self) -> None:
        best = (inf, STOP) # The best move, minimum weight
        
        for d, n in self.node.neighbors.items():
            # Check each potential neighbor
            if n is None or d == PORTAL:
                continue
            if not PACMAN in self.node.access[d]:
                continue
            
            weight = self.get_weight(n, self.node)
            best = min(best, (weight, d))
            
        self.direction = best[1]
        self.target = self.getNewTarget(self.direction)

    # Where the magic happens. Determine the desirability of going towards the cur node. Recursively checks branches in DFS. Deeper branches count for less
    def get_weight(self, cur: Node, prev: Node, depth: int = 0, seen: set[Node] | None = None, power: bool = False, between: bool = False) -> float:
        if depth == MAX_DEPTH:
            # We've dug too deep and greedily
            return 0
        
        s = 0 # Initial value of move
        # Variable for keeping track of where we've been, so we don't backtrack
        if seen is None:
            seen = set()
        seen = seen.copy()
        seen.add(prev)
        
        # Check if there are ghosts in the path
        ghosts = self.has_ghost(prev, cur)
        # If there are no ghosts, or we expect to have a power up, it's fine
        if not power and len(ghosts) != 0:
            dir_dot = -1
            dangerous = False
            danger = 0
            for ghost in ghosts:
                # Spawning and frightened ghosts are harmless (mostly)
                if ghost.mode.current == SPAWN:
                    if (ghost.position - ghost.goal).magnitudeSquared() > 9:
                        continue
                if ghost.mode.current == FREIGHT:
                    if ghost.mode.timer / ghost.mode.time < .85:
                        # Frigtened ghosts are delicious, actually
                        s += GHOST_VALUE * GREED_FACTOR * self.ghost_multiplier
                        continue
                # Reached if there are any dangerous ghosts on the path 
                dangerous = True
                ghost_dir = self.directions[ghost.direction]
                ghost_dir = np.array([ghost_dir.x, ghost_dir.y])
                to_node = prev.position - ghost.position
                to_node = np.array([to_node.x, to_node.y])
                to_node = to_node / np.linalg.norm(to_node)
                dir_dot = max(np.dot(to_node, ghost_dir), dir_dot) # Angle between ghost's direction and direction of path. Ghost is harmless if it's ahead of us and moving away.
                if depth == 0 and dir_dot > .8:
                    # Right next to us and moving towards us. Death is imminent this way
                    return inf
                danger = max(
                    danger,
                    # Complicated and reached through trial and error. Essentially, ghost is closer = ghost is more bad
                    ((MAX_FEAR - depth+4)/3)**5 * FEAR_FACTOR * max((dir_dot + 1)/2, 0) / (1 + (self.position - ghost.position).magnitude() / BOARD_RADIUS)
                )
                
            if dangerous:
                if depth >= MAX_FEAR:
                    # Ghost is far enough away to be harmless, but we shouldn't consider pellets in this direction as potential rewards, so return to cut pellet check short
                    return 0
                else:
                    return danger
                
        if between:
            # For turning around. There can be no pellets along the path you just came
            pellets = 0
        else:
            # Amount of pellets along path
            pellets = -self.pellets_between(prev, cur)
        s += pellets * GREED_FACTOR
        p = False # Power pellet along path?
        if self.power_pellet_between(prev, cur) and depth < 4:
            p = True
        
        # Check neighbor node branches 
        for next in cur.neighbors.values():
            if next is None or next in seen:
                continue
            s += self.get_weight(next, cur, depth + 1, seen, power or p)
        # Return value for full branch, decrease by distance
        return s / (1 + cur.position.magnitude() / BOARD_RADIUS) / (depth + 1)

    # Same as in Pacman, expect updates local pellet maps
    def eatPellets(self, pellet_list: list[Pellet]) -> None | Pellet:
        for pellet in pellet_list:
            if self.collideCheck(pellet):
                self.remove_pellet(pellet)
                return pellet
        return None

    # Update pellet maps
    def remove_pellet(self, pellet: Pellet) -> None:
        self.pellet_map[pellet.position.y//TILEHEIGHT][pellet.position.x//TILEWIDTH] = False
        if self.power_pellet_map[pellet.position.y//TILEHEIGHT][pellet.position.x//TILEWIDTH]:
            # Ate power pellet. Reset multiplier and killed ghosts
            self.ghost_multiplier = 1
            self.ghosts_killed = set()
            for g in self.ghosts:
                # Add any ghosts already dead to killed list. This avoids multiplier going higher than it should.
                if g.mode.current == SPAWN:
                    self.ghosts_killed.add(g)
            self.power_pellet_map[pellet.position.y//TILEHEIGHT][pellet.position.x//TILEWIDTH] = False
    
    # This may double count pellets on nodes. I don't care enough to account for it. Get number of pellets along path, inclusive ends.
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

    # Check whether there is a power pellet along path
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
    
    # Finds all ghosts along path and returns them in a list
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

    # Evaluate whether we should flip direction on account of new information
    def eval_flip(self):
        # Simple check on whether flipping is better or not, based on weight of current node and target
        if self.get_weight(self.target, self.node, between=self.flipped) > self.get_weight(self.node, self.target, between=not self.flipped):
            self.direction = -self.direction
            self.target, self.node = self.node, self.target
            self.flipped = not self.flipped
