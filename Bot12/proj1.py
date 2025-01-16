import os
import itertools as it
import numpy as np
from collections import namedtuple
from collections import deque
from collections import defaultdict
import copy
import random
from time import sleep
from termcolor import colored
from matplotlib import pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import time

# Total number of Bot Types
NUM_BOTS=4
# Compute Limit
COMPUTE_LIMIT=10000

class GridAttrib:
    __slots__ = ('open', 'bot_occupied', 'traversed', 'alien_id',
                 'captain_slot', 'crew_belief', 'alien_belief')

    def __init__(self):
        self.open = False
        self.bot_occupied = False
        self.traversed = False
        self.alien_id = -1
        self.captain_slot = False
        self.crew_belief = 1.0 # Start out with a uniform belief of 1.0
        self.alien_belief = 1.0 # Start out with a uniform belief of 1.0

class Grid:
    def __init__(self, D=30, debug=True):
        self.D = D
        self.grid = []
        self.debug = debug
        self.gen_grid()

    def valid_index(self, ind):
        return not (ind[0] >= self.D or ind[0] < 0 or ind[1] >= self.D or ind[1] < 0)

    def get_neighbors(self, ind):
        neighbors = []
        left = (ind[0] - 1, ind[1])
        right = (ind[0] + 1, ind[1])
        up = (ind[0], ind[1] + 1)
        down = (ind[0], ind[1] - 1)
        indices = [left, right, up, down]
        for index in indices:
            if self.valid_index(index):
                neighbors.append(index)
        return neighbors
    def get_open_neighbors(self, ind):
        neighbors = []
        left = (ind[0] - 1, ind[1])
        right = (ind[0] + 1, ind[1])
        up = (ind[0], ind[1] + 1)
        down = (ind[0], ind[1] - 1)
        indices = [left, right, up, down]
        return [index for index in indices if self.valid_index(index) and self.grid[index[1]][index[0]].open]

    # Gets only the unvisited open neighbors. Used mainly for path planning.
    def get_untraversed_open_neighbors(self, ind):
        neighbors = []
        left = (ind[0] - 1, ind[1])
        right = (ind[0] + 1, ind[1])
        up = (ind[0], ind[1] + 1)
        down = (ind[0], ind[1] - 1)
        indices = [left, right, up, down]
        return [index for index in indices if self.valid_index(index) and self.grid[index[1]][index[0]].open and not self.grid[index[1]][index[0]].traversed]
    
    # The steps to be iterated over and over till they cannot be done are implemented here
    def gen_grid_iterate(self):
        cells_to_open = []
        # Get all blocked cells with one open neighbors
        for j in range(self.D):
            for i in range(self.D):
                if self.grid[j][i].open == True:
                    continue
                neighbors_ind = self.get_neighbors((i, j))
                open_neighbors = []
                for neighbor_ind in neighbors_ind:
                    if self.grid[neighbor_ind[1]][neighbor_ind[0]].open is True:
                        open_neighbors.append(neighbor_ind)
                if len(open_neighbors) == 1:
                    cells_to_open.append((i, j))
        # Randomly open one of those cells
        if len(cells_to_open) > 0:
            index = random.choice(cells_to_open)
            self.grid[index[1]][index[0]].open = True
        if self.debug:
            print("After one iteration")
            print(self)
            print(f"Cells to open: {len(cells_to_open)}")
        return len(cells_to_open) != 0

    # Grid generation happens here
    def gen_grid(self):
        for j in range(self.D):
            row = []
            for i in range(self.D):
                row.append(GridAttrib())
            self.grid.append(row)

        # Open Random Cell
        rand_ind = np.random.randint(0, self.D, 2)
        self.grid[rand_ind[1]][rand_ind[0]].open = True
        if self.debug:
            print(self)
        # Iterate on the grid
        while self.gen_grid_iterate():
            pass
        # Randomly open half the dead ends
        cells_to_open = []
        for j in range(self.D):
            for i in range(self.D):
                    all_neighbors = self.get_neighbors((i,j))
                    open_neighbors = [ind for ind in all_neighbors if self.grid[ind[1]][ind[0]].open]
                    closed_neighbors = [ind for ind in all_neighbors if not self.grid[ind[1]][ind[0]].open]
                    # randint is used here to maintain a ~50% chance of any dead end opening
                    if self.grid[j][i].open and random.randint(0, 1) == 1 and len(open_neighbors) == 1:
                        cells_to_open.append(random.choice(closed_neighbors))
        for ind in cells_to_open:
            self.grid[ind[1]][ind[0]].open = True
        if self.debug:
            print("After dead end opening")
            print(self)

    # A bunch of simple helper functions

    def place_alien(self, ind, alien_id):
        self.grid[ind[1]][ind[0]].alien_id = alien_id
    def remove_alien(self, ind):
        self.grid[ind[1]][ind[0]].alien_id = -1
    # k tells us how deep to look from the index
    def has_alien(self, ind, k=1):
        if k == 1:
            return self.grid[ind[1]][ind[0]].alien_id != -1
        elif k==2:
            depth1 = self.grid[ind[1]][ind[0]].alien_id != -1
            if depth1:
                return True
            neighbors = self.get_open_neighbors(ind)
            alien_exists = any([self.has_alien(n) for n in neighbors])
            del neighbors
            return alien_exists

        else:
            # This was implemented in case k >= 3 was needed.
            print("SHOULD NOT HAPPEN")
            traversed = {}
            children = deque([])
            current = deque([ind])
            while k >= 1:
                for ind in current:
                    traversed[ind] = 1
                    if self.grid[ind[1]][ind[0]].alien_id != -1:
                        return True
                    neighbors = self.get_open_neighbors(ind)
                    neighbors = [neighbor for neighbor in neighbors if neighbor not in traversed]
                    children.extend(neighbors)
                current = children
                children = deque([])
                k -= 1
            return False
        
    def place_bot(self, ind):
        self.grid[ind[1]][ind[0]].bot_occupied = True
    def remove_bot(self, ind):
        self.grid[ind[1]][ind[0]].bot_occupied = False
    def set_traversed(self, ind):
        self.grid[ind[1]][ind[0]].traversed = True
    def remove_all_traversal(self):
        for j in range(self.D):
            for i in range(self.D):
                self.grid[j][i].traversed = False

    def get_open_indices(self):
        return [(i, j) for i in range(self.D) for j in range(self.D) if self.grid[j][i].open == True]

    def get_unoccupied_open_indices(self):
        return [(i, j) for i in range(self.D) for j in range(self.D) if self.grid[j][i].open == True and self.grid[j][i].alien_id == -1
                and self.grid[j][i].bot_occupied == False]
    # End of all helper functions

    # A function to reset grid
    # Used to speed up data gathering using the same grid for all bots
    def reset_grid(self):
        for j in range(self.D):
            for i in range(self.D):
                self.grid[j][i].alien_id = -1
                self.grid[j][i].bot_occupied = False
                self.grid[j][i].captain_slot = False
                self.grid[j][i].traversed = False
                self.grid[j][i].crew_belief = 1.0
                self.grid[j][i].alien_belief = 1.0

    def __str__(self):
        s = ""
        for j in range(self.D):
            for i in range(self.D):
                if self.grid[j][i].open == True:
                    if self.grid[j][i].captain_slot:
                        s += colored('C', 'magenta')
                    elif self.grid[j][i].alien_id != -1:
                        s += colored('A', 'red')
                    elif self.grid[j][i].bot_occupied:
                        s += colored('B', 'yellow')
                    elif self.grid[j][i].traversed:
                        s += colored('P', 'blue')
                    else:
                        s += colored('O', 'green')
                else:
                    s += 'X'
            s += "\n"
        return s

class Alien:
    # This alien_id is used to keep track of every alien
    alien_id = 0
    def __init__(self, grid):
        self.grid = grid
        indices = self.grid.get_unoccupied_open_indices()
        ind = random.choice(indices)
        self.ind = ind
        self.alien_id = Alien.alien_id
        self.grid.place_alien(ind, Alien.alien_id)
        Alien.alien_id += 1

    def move(self):
        # Get all possible locations for the alien
        neighbors = self.grid.get_open_neighbors(self.ind)
        # Filter out the ones that are occupied by other aliens
        neighbors_without_aliens = [neighbor for neighbor in neighbors if self.grid.grid[neighbor[1]][neighbor[0]].alien_id == -1]
        # Randomly choose any of the locations
        if len(neighbors_without_aliens) > 0:
            rand_ind = np.random.randint(0, len( neighbors_without_aliens ))
            self.grid.remove_alien(self.ind)
            self.ind = neighbors_without_aliens[rand_ind]
            self.grid.place_alien(self.ind, self.alien_id)

# Used for parent tracking with BFS
class PathTreeNode:
    def __init__(self):
        self.children = []
        self.parent = None
        self.data = None

class Bot1:
    def __init__(self, grid, captain_ind, debug=True):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = None
        self.debug = debug

    def plan_path(self):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.ind
        path_deque = deque([path_tree])
        destination = None
        visited = set()
        compute_counter = 0
        while not captain_found:
            if len(path_deque) == 0 or compute_counter >= COMPUTE_LIMIT:
                self.grid.remove_all_traversal()
                return
            compute_counter += 1
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid.set_traversed(ind)
            if ind == self.captain_ind:
                destination = node
                break
            neighbors_ind = self.grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that do not hit an alien
                if not self.grid.has_alien(neighbor_ind):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
            print(self.grid)

    def move(self):
        if self.path is None:
            self.plan_path()
        if len(self.path) == 0:
            if self.debug:
                print("No path found!")
            return

        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)
            
class Bot2:
    def __init__(self, grid, captain_ind, debug = True):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = deque([])
        self.debug = debug

    def plan_path(self):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.ind
        path_deque = deque([path_tree])
        destination = None
        compute_counter = 0
        visited = set()
        while not captain_found:
            if len(path_deque) == 0 or compute_counter > COMPUTE_LIMIT:
                self.grid.remove_all_traversal()
                return
            compute_counter += 1
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid.set_traversed(ind)
            if ind == self.captain_ind:
                destination = node
                break
            neighbors_ind = self.grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that do not hit an alien
                if not self.grid.has_alien(neighbor_ind):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
            print(self.grid)

    def move(self):
        self.plan_path()
        if len(self.path) == 0:
            if self.debug:
                print("No path found!")
            return
        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)

class Bot3:
    def __init__(self, grid, captain_ind, debug = True):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = deque([])
        self.debug = debug

    def plan_path(self, k=2):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.ind
        path_deque = deque([path_tree])
        destination = None
        compute_counter = 0
        visited = set()
        while not captain_found:
            if len(path_deque) == 0 or compute_counter > COMPUTE_LIMIT//2:
                self.grid.remove_all_traversal()
                return
            compute_counter += 1
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid.set_traversed(ind)
            if ind == self.captain_ind:
                destination = node
                break
            neighbors_ind = self.grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that do not hit an alien
                if not self.grid.has_alien(neighbor_ind, k = k):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
            print(self.grid)

    def move(self):
        self.plan_path(2)
        if len(self.path) == 0:
            if self.debug:
                print("Reverting...")
            self.plan_path(1)
            if len(self.path) == 0:
                if self.debug:
                    print("No path found")
                return
        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)

class Bot4:
    def __init__(self, grid, captain_ind, debug = True):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = deque([])
        self.debug = debug
        self.risk_limit = 0.0
        self.K = 0
        for j in range(self.grid.D):
            for i in range(self.grid.D):
                if self.grid.grid[j][i].alien_id != -1:
                    self.K += 1

    def plan_path(self, k=2):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.ind
        path_deque = deque([path_tree])
        destination = None
        visited = set()
        while not captain_found:
            if len(path_deque) == 0:
                self.grid.remove_all_traversal()
                return
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid.set_traversed(ind)
            if ind == self.captain_ind:
                destination = node
                break
            neighbors_ind = self.grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that do not hit an alien
                if not self.grid.has_alien(neighbor_ind, k = k):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
            print(self.grid)
    def distance(self, ind):
        return (ind[1] - self.ind[1])**2 + (ind[0] - self.ind[0])**2
    def evade(self, offset = 2):
        # From the current position
        # Search +/- offset for aliens
        # Evade in a way that escapes the closest alien
        x_offsets = [i for i in range(-offset, offset + 1)]
        y_offsets = [i for i in range(-offset, offset + 1)]
        aliens = []
        for j in y_offsets:
            for i in x_offsets:
                if i == 0 and j == 0:
                    continue
                ind = (self.ind[0] + i, self.ind[1] + j)
                if self.grid.valid_index(ind) and self.grid.has_alien(ind):
                    aliens.append((self.ind[0] + i, self.ind[1] + j))
        if not aliens:
            # If there are no aliens nearby just wait
            return
        aliens.sort(key=self.distance)
        closest_alien = aliens[0]
        ideal_move_dir = (self.ind[0] - closest_alien[0], self.ind[1] - closest_alien[1])
        def condition_dir(d):
            if d == 0:
                return 0
            elif d > 0:
                return 1
            elif d < 0:
                return -1
        if ideal_move_dir[0] == ideal_move_dir[1] or abs(ideal_move_dir[0]) > abs(ideal_move_dir[1]):
            new_ind = (self.ind[0] + condition_dir(ideal_move_dir[0]), self.ind[1])
            if self.grid.valid_index(new_ind) and self.grid.grid[new_ind[1]][new_ind[0]].open:
                self.grid.remove_bot(self.ind)
                self.ind = new_ind
                self.grid.place_bot(self.ind)
            else:
                return -1
        else:
            new_ind = (self.ind[0], self.ind[1] + condition_dir(ideal_move_dir[1]))
            if self.grid.valid_index(new_ind) and self.grid.grid[new_ind[1]][new_ind[0]].open:
                self.grid.remove_bot(self.ind)
                self.ind = new_ind
                self.grid.place_bot(self.ind)
            else:
                return -1

        return 0

    # Calculates a danger level so that we can
    # evade in the direction that lets us be in
    # the safest possible position
    def calculate_danger(self, ind, offset):
        danger = 0
        for j in range(-offset, offset):
            for i in range(-offset, offset):
                x = ind[0] + i
                y = ind[1] + j
                if self.grid.valid_index((x, y)) and self.grid.has_alien((x, y)):
                    # The closer the more dangerous. The 0.001 at the end is to avoid div by 0
                    danger += 1/(abs(i) + abs(j) + 0.001)
        return danger

    def evade2(self, offset=2):
        possible_positions = self.grid.get_open_neighbors(self.ind)
        # The current position might be the safest.
        # We may not want to move in that case so we
        # add it to the possible positions
        possible_positions.append(self.ind)
        dangers = [self.calculate_danger(p, offset) for p in possible_positions]
        min_position_i = min(enumerate(dangers), key=lambda x: x[1])[0]
        next_position = possible_positions[min_position_i]
        self.grid.remove_bot(self.ind)
        self.ind = next_position
        self.grid.place_bot(self.ind)

    def move(self):
        self.plan_path(2)
        if len(self.path) == 0:
            # The closer the captain is the higher the risk
            # Scaled such that if the captain is under 3 units, (rf = 0.8)
            # the bot should take high risk
            # If far away(> 10), it should focus more on evading (rf = 0.2)
            # We use a linear model
            d = self.distance(self.captain_ind)
            alpha = random.uniform(0.01, 1.0)
            m = -0.086
            c = 1.06
            self.risk_limit = m * d + c
            if self.risk_limit > 1:
                self.risk_limit = 1
            elif self.risk_limit < 0:
                self.risk_limit = 0
            if(alpha > self.risk_limit):
                self.evade2()
                return
            else:
                print("REVERT")
                if self.debug:
                    print("Reverting...")
                self.plan_path(1)
                if len(self.path) == 0:
                    self.evade2()
                    if self.debug:
                        print("No path found")
                    return

        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)

class Bot5:
    def __init__(self, grid, captain_ind, debug = True):
        self.grid = grid
        self.captain_ind = captain_ind
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.path = deque([])
        self.debug = debug
        self.risk_limit = 0.0
        self.computation_limit = 5000
        self.K = 0
        for j in range(self.grid.D):
            for i in range(self.grid.D):
                if self.grid.grid[j][i].alien_id != -1:
                    self.K += 1

    def plan_path(self, k=2):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.ind
        path_deque = deque([path_tree])
        destination = None
        visited=set()
        compute_counter = 0
        iters = 0
        while not captain_found:
            # Divide by 2 because this can be run twice
            if len(path_deque) == 0 or compute_counter > COMPUTE_LIMIT//2:
                self.grid.remove_all_traversal()
                return
            compute_counter += 1
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid.set_traversed(ind)
            if ind == self.captain_ind:
                destination = node
                break
            neighbors_ind = self.grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that do not hit an alien
                if not self.grid.has_alien(neighbor_ind, k = k):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
            print(self.grid)
    def distance(self, ind):
        return (ind[1] - self.ind[1])**2 + (ind[0] - self.ind[0])**2
    # Calculates a danger level so that we can
    # evade in the direction that lets us be in
    # the safest possible position
    def calculate_danger(self, ind, offset):
        danger = 0
        for j in range(-offset, offset):
            for i in range(-offset, offset):
                x = ind[0] + i
                y = ind[1] + j
                if self.grid.valid_index((x, y)) and self.grid.has_alien((x, y)):
                    # The closer the more dangerous. The 0.001 at the end is to avoid div by 0
                    danger += 1/(abs(i) + abs(j) + 0.001)
        return danger

    def evade2(self, offset=2):
        possible_positions = self.grid.get_open_neighbors(self.ind)
        # The current position might be the safest.
        # We may not want to move in that case so we
        # add it to the possible positions
        possible_positions.append(self.ind)
        dangers = [self.calculate_danger(p, offset) for p in possible_positions]
        min_position_i = min(enumerate(dangers), key=lambda x: x[1])[0]
        next_position = possible_positions[min_position_i]
        self.grid.remove_bot(self.ind)
        self.ind = next_position
        self.grid.place_bot(self.ind)

    def move(self):
        self.plan_path(2)
        if len(self.path) == 0:
            # The closer the captain is the higher the risk
            # Scaled such that if the captain is under 3 units, (rf = 0.8)
            # the bot should take high risk
            # If far away(> 10), it should focus more on evading (rf = 0.2)
            # We use a linear model
            d = self.distance(self.captain_ind)
            alpha = random.uniform(0.01, 1.0)
            m = -0.086
            c = 1.06
            self.risk_limit = m * d + c
            if self.risk_limit > 1:
                self.risk_limit = 1
            elif self.risk_limit < 0:
                self.risk_limit = 0
            if(alpha > self.risk_limit):
                self.evade2()
                return
            else:
                print("REVERT")
                if self.debug:
                    print("Reverting...")
                self.plan_path(1)
                if len(self.path) == 0:
                    self.evade2()
                    if self.debug:
                        print("No path found")
                    return

        next_dest = self.path.popleft()
        self.grid.remove_bot(self.ind)
        self.ind = next_dest
        self.grid.place_bot(self.ind)

class WorldState:
    def __init__(self, debug=True):
        self.debug = debug
    def gen_grid(self):
        self.grid = Grid(debug=self.debug)
    def set_iters(self, iters):
        self.iters = iters
    def set_K_range(self, K_start, K_end, K_skip):
        self.K_start = K_start
        self.K_end = K_end
        self.K_skip = K_skip
    def gen_world(self, K):
        self.captain_ind = random.choice(self.grid.get_open_indices())
        self.aliens = [Alien(self.grid) for _ in range(K)]
        self.captain_found = False
        self.bot_caught = False
    def simulate_world(self, bot):
        for _ in range(1000):
            if self.bot_caught:
                break
            bot.move()
            if bot.ind == self.captain_ind:
                self.captain_found = True
                break
            for alien in self.aliens:
                if bot.ind == alien.ind:
                    self.bot_caught = True
                    break
                alien.move()
                if bot.ind == alien.ind:
                    self.bot_caught = True
                    break
            if self.debug:
                print("Next Iteration")
                print(self.grid)
                sleep(0.016)
        if self.captain_found:
            return 0
            if self.debug:
                print("Success")
        elif self.bot_caught:
            return -2
        else:
            return -1
            if self.debug:
                print("Failure")

def proc_fun(ws):
    temp_dict = {}
    data_dict = {}
    for K in range(ws.K_start, ws.K_end, ws.K_skip):
        for b in range(NUM_BOTS):
            temp_dict[(b, K)] = [0, 0]
    for _ in range(ws.iters):
        for K in range(ws.K_start, ws.K_end, ws.K_skip):
            ws.gen_grid()
            for b in range(NUM_BOTS):
                print(f"Process({os.getpid()}) Working on iter: {_}, K: {K}, Bot: {b + 1}...")
                ws.grid.reset_grid()
                ws.gen_world(K)
                bot = None
                if b == 0:
                    bot = Bot1(ws.grid, ws.captain_ind, debug=ws.debug)
                elif b == 1:
                    bot = Bot2(ws.grid, ws.captain_ind, debug=ws.debug)
                elif b == 2:
                    bot = Bot3(ws.grid, ws.captain_ind, debug=ws.debug)
                else:
                    bot = Bot5(ws.grid, ws.captain_ind, debug=ws.debug)
                ret = ws.simulate_world(bot)
                if ret == 0:
                    temp_dict[(b, K)][0] += 1
                elif ret == -1:
                    temp_dict[(b, K)][1] += 1
                elif ret == -2:
                    pass
                else:
                    print("Shouldn't be here")
    for b in range(NUM_BOTS):
        data_dict[b] = [[], []]
        for K in range(ws.K_start, ws.K_end, ws.K_skip):
            data_dict[b][0].append(temp_dict[(b, K)][0])
            data_dict[b][1].append(temp_dict[(b, K)][1])
    return data_dict

class World:
    def __init__(self, debug=True, track_time = False, jobs=1):
        self.debug = debug
        self.jobs = jobs
        if jobs > 1:
            self.states = [WorldState(debug=debug) for _ in range(jobs)]

    def gen_grid(self):
        self.grid = Grid(debug=self.debug)

    def gen_world(self, K):
        self.captain_ind = random.choice(self.grid.get_open_indices())
        self.aliens = [Alien(self.grid) for _ in range(K)]
        self.captain_found = False
        self.bot_caught = False

    def gather_data(self, iters=20, K_range=(0, 20, 1), batch=0):
        K_start = K_range[0]
        K_end = K_range[1]
        K_skip = K_range[2]
        self.K_start = K_range[0]
        self.K_end = K_range[1]
        self.K_skip = K_range[2]
        self.K_skip = K_skip
        self.data_dict = {}
        self.batch = 0
        for b in range(NUM_BOTS):
            for K in range(K_start, K_end, K_skip):
                self.data_dict[b] = [[], []]
        if self.jobs > 1:
            iters_per_job = iters//self.jobs
            for ws in self.states:
                ws.set_iters(iters_per_job)
                ws.set_K_range(K_start, K_end, K_skip)
            p = Pool(self.jobs)
            data = p.map(proc_fun, self.states)
            for b in range(NUM_BOTS):
                for i, K in enumerate(range(K_start, K_end, K_skip)):
                    successes = 0
                    survivals = 0
                    for job_data in data:
                        successes += job_data[b][0][i]
                        survivals += job_data[b][1][i]
                    self.data_dict[b][0].append(successes/(self.jobs * iters_per_job))
                    self.data_dict[b][1].append(survivals/(self.jobs * iters_per_job))
            for b in range(NUM_BOTS):
                self.data_dict[b][0] = np.array(self.data_dict[b][0])
                self.data_dict[b][1] = np.array(self.data_dict[b][1])
            print(self.data_dict)
        else:
            temp_dict = {}
            for K in range(K_start, K_end, K_skip):
                for b in range(NUM_BOTS):
                    temp_dict[(b, K)] = [0, 0]
            for i in range(iters):
                for K in range(K_start, K_end, K_skip):
                    self.gen_grid()
                    for b in range(NUM_BOTS):
                        self.grid.reset_grid()
                        self.gen_world(K)
                        bot = None
                        if b == 0:
                            bot = Bot1(self.grid, self.captain_ind, debug=self.debug)
                        elif b == 1:
                            bot = Bot2(self.grid, self.captain_ind, debug=self.debug)
                        elif b == 2:
                            bot = Bot3(self.grid, self.captain_ind, debug=self.debug)
                        else:
                            bot = Bot5(self.grid, self.captain_ind, debug=self.debug)
                        ret = self.simulate_world(bot)
                        print(f"K={K},Bot {b + 1}, iter={i}, res={ret}")
                        if ret == 0:
                            temp_dict[(b, K)][0] += 1
                        elif ret == -1:
                            temp_dict[(b, K)][1] += 1
                        elif ret == -2:
                            pass
                        else:
                            print("Shouldn't happen")
            for b in range(NUM_BOTS):
                for K in range(K_start, K_end, K_skip):
                    self.data_dict[b][0].append(temp_dict[(b, K)][0]/iters)
                    self.data_dict[b][1].append(temp_dict[(b, K)][1]/iters)
            for b in range(NUM_BOTS):
                for i in range(2):
                    self.data_dict[b][1] = np.array(self.data_dict[b][1])
                    self.data_dict[b][0] = np.array(self.data_dict[b][0])
            print(temp_dict)
            print(self.data_dict)
    def plot_data(self):
        x = np.arange(self.K_start, self.K_end, self.K_skip)
        plt.subplot(211)
        for b in range(NUM_BOTS):
            plt.plot(x, self.data_dict[b][0], label=f"Bot {5 if b == 3 else b + 1} Success")
        plt.xlabel("No. of Aliens")
        plt.ylabel("Success Rate")
        plt.legend()
        plt.ylim(0.001, 1.2)
        plt.subplot(212)
        for b in range(NUM_BOTS):
            plt.plot(x, self.data_dict[b][1], label=f"Bot {5 if b==3 else b + 1} Survival")
        plt.xlabel("No. of Aliens")
        plt.ylabel("Survival Rate")
        plt.legend()
        plt.ylim(0.001, 1.2)
        plt.show()

    def simulate_world(self, bot):
        for _ in range(1000):
            if self.bot_caught:
                break
            bot.move()
            if bot.ind == self.captain_ind:
                self.captain_found = True
                break
            for alien in self.aliens:
                if bot.ind == alien.ind:
                    self.bot_caught = True
                    if self.debug:
                        print("Failure")
                    break
                alien.move()
                if bot.ind == alien.ind:
                    self.bot_caught = True
                    if self.debug:
                        print("Failure")
                    break
            if self.debug:
                print("Next Iteration")
                print(self.grid)
                sleep(0.016)
        if self.captain_found:
            return 0
            if self.debug:
                print("Success")
        elif self.bot_caught:
            return -2
        else:
            return -1
            if self.debug:
                print("Failure")
def sim_worst_case_bfs(const_func = lambda x : sleep(0.0005)):
    print("Doing a worse case BFS")
    grid = Grid(debug=False)
    for j in range(grid.D):
        for i in range(grid.D):
            grid.grid[j][i].open
    ind = (0, 0)
    for _ in range(1000):
        print(f"Iter: {_}")
        visited = set()
        parent_map = {ind: None}
        fringe = deque([ind])
        while True:
            if len(fringe) == 0:
                break
            ind = fringe.popleft()
            if ind in visited:
                print("ERROR")
            print(ind)
            visited.add(ind)
            neighbors = grid.get_open_neighbors(ind)
            neighbors = [n for n in neighbors if n not in visited]
            for n in neighbors:
                if n in visited:
                    print("ERROR")
                parent_map[n] = ind
            const_func(ind)
            fringe.extend(neighbors)
    #print(f"Overall Time Taken for 1000 worst case BFSes: {end_time - start_time}")


#plt.style.use('ggplot')
#w = World(debug=False, jobs=1)
#w.gather_data(iters=100, K_range=(0, 100, 10), batch=20)
#w.plot_data()
