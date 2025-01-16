import numpy as np
import matplotlib.pyplot as plt
from proj1 import GridAttrib, Grid, PathTreeNode
from numpy import random as nprd
import random
import random as rd
from PIL import Image
from collections import deque
import os
from math import exp
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue

plt.style.use('ggplot')

D=35
COMPUTE_LIMIT = 5000
BB_SIZE=40
BB_SIZE_4D = 50
ALPHA=0.1

def grid_sum(D, num_g):
    s = 0
    for j in range(D):
        for i in range(D):
            s += num_g[j][i]
    return s

def zeros(x, y):
    ret = []
    for j in range(y):
        ret.append([])
        for i in range(x):
            ret[-1].append(0)
    return ret

class Alien:
    # This alien_id is used to keep track of every alien
    alien_id = 0
    def __init__(self, grid, bot, p=None):
        self.grid = grid
        indices = self.grid.get_unoccupied_open_indices()
        ind = random.choice(indices)
        while bot.within_alien_sensor(ind):
            ind = random.choice(indices)
        self.ind = ind if p is None else p
        self.alien_id = Alien.alien_id
        self.grid.place_alien(self.ind, Alien.alien_id)
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

class Grid2:
    def __init__(self, D=35, debug=1, o_crew_pos=None, o_crew_pos2=None, alpha=ALPHA):
        self._grid = Grid(D, debug=debug - 1>0)
        self.D = D
        self.grid = self._grid.grid

        self.crew_pos = rd.choice(self._grid.get_open_indices()) if o_crew_pos is None else o_crew_pos
        self.crew_pos2 = None if o_crew_pos2 is None else o_crew_pos2
        while self.crew_pos2 is None or self.crew_pos2 == self.crew_pos:
            self.crew_pos2 = rd.choice(self._grid.get_open_indices())

        self.beliefs = {}
        open_cells = self._grid.get_unoccupied_open_indices()

        s = len(open_cells) * (len(open_cells) - 1) / 2
        for one_cell in open_cells:
            for two_cell in open_cells:
                if (two_cell, one_cell) in self.beliefs or one_cell == two_cell:
                    continue
                self.beliefs[(one_cell, two_cell)] = 1 / s

        # got from bot4.py grid2 init function
        X = range(self.D)
        Y = range(self.D)
        prob_grid = []
        for j in range(self.D):
            prob_grid.append([])
            for i in range(self.D):
                p1 = exp(-alpha * self.distance(self.crew_pos, (i, j)))
                p2 = exp(-alpha * self.distance(self.crew_pos2, (i, j)))
                ptotal = p1 + p2 - p1*p2
                prob_grid[-1].append(ptotal)
        #prob_sum = np.sum(prob_grid)
        prob_sum = grid_sum(self.D, prob_grid)
        for j in Y:
            for i in X:
                prob_grid[j][i] /= prob_sum
        plt.imshow(prob_grid)
        plt.title("Probability of getting beeps")


    def distance(self, pos1, pos2):
        d = abs(pos1[1] - pos2[1])
        d += abs(pos1[0] - pos2[0])
        return d
    
    def distance_to_crew(self, pos):
        d1, d2 = None, None
        # print(f"pos: {pos}, crew1: {self.crew_pos}, crew2: {self.crew_pos2}")
        if self.crew_pos is not None:
            d1 = self.distance(pos, self.crew_pos)
        if self.crew_pos2 is not None:
            d2 = self.distance(pos, self.crew_pos2)
        
        return d1, d2
    
    def reset_grid(self):
        self._grid.reset_grid()
        open_cells = self._grid.get_open_indices()
        s = len(open_cells) * (len(open_cells) - 1) / 2
        for one_cell in open_cells:
            for two_cell in open_cells:
                if (two_cell, one_cell) in self.beliefs or one_cell == two_cell:
                    continue
                self.beliefs[(one_cell, two_cell)] = 1 / s

class bot3:
    def __init__(self, grid, alpha = ALPHA, k=5, debug=1, p=None):
        self.grid = grid
        self.pos = p
        while self.pos in [self.grid.crew_pos, self.grid.crew_pos2] or self.pos is None:
            self.pos = rd.choice(self.grid._grid.get_open_indices())
        self.alpha = alpha
        self.debug=debug
        self.tick=0
        self.k=k

    def within_alien_sensor(self, pos):
        return abs(pos[0] - self.pos[0]) <= self.k and abs(pos[1] - self.pos[1]) <= self.k

    def alien_sensor_edge(self, pos, offset):
        return ( abs(pos[0] - self.pos[0]) == self.k + offset and abs(pos[1] - self.pos[1]) <= self.k ) or (abs(pos[0] - self.pos[0]) <= self.k and abs(pos[1] - self.pos[1]) == self.k + offset)

    def in_danger(self, offset=1):
        for i in range(-offset, offset):
            for j in range(-offset, offset):
                # Skip the current bot location
                if i == 0 and j == 0:
                    continue
                if self.grid.grid[j][i].open and self.grid.grid[j][i].alien_belief > 0.1/self.grid.D:                    
                    return True
        return False


    def crew_sensor(self):
        c1 = rd.random()
        c2 = rd.random()

        d1, d2 = self.grid.distance_to_crew(self.pos)
        a, b = False, False

        if d1 is not None:
            a = c1 <= exp(-self.alpha * (d1 - 1))

        if d2 is not None:
            b = c2 <= exp(-self.alpha * (d2 - 1))

        return a or b


    def alien_sensor(self):
        found_alien = 0
        for j in range(-self.k, self.k + 1):
            if found_alien == 1:
                break
            for i in range(-self.k, self.k + 1):
                pos = [ self.pos[0] + i, self.pos[1] + j ]
                if pos[0] > self.grid.D - 1:
                    pos[0] = self.grid.D - 1
                elif pos[0] < 0:
                    pos[0] = 0
                if pos[1] > self.grid.D - 1:
                    pos[1] = self.grid.D - 1
                elif pos[1] < 0:
                    pos[1] = 0
                if self.grid.grid[pos[1]][pos[0]].alien_id != -1:
                    found_alien = 1
                    break
        return found_alien == 1

    def diffuse_alien_prob(self, alien_found):
        choose_fun = None
        if alien_found:
            choose_fun = lambda x: self.within_alien_sensor(x)
        else:
            choose_fun = lambda x: not self.within_alien_sensor(x)
        open_cells = self.grid._grid.get_open_indices()
        # Cells inside the alien sensor and just outside
        # The probability will diffuse among these
        filtered_open_cells = [oc for oc in open_cells if ( choose_fun(oc) or self.alien_sensor_edge(oc, 1 if alien_found else 0) )]
        alien_belief = np.zeros((self.grid.D, self.grid.D))

        # Diffuse through the edge cells
        for ci in filtered_open_cells:
            neighbors = self.grid._grid.get_neighbors(ci)
            neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open and choose_fun(n) ]
            # Diffuse the probability at the current square into the
            # neighbors that the alien can move to
            for n in neighbors:
                alien_belief[n[1]][n[0]] += self.grid.grid[ci[1]][ci[0]].alien_belief/len(neighbors)
        # Normalizs
        total_belief = np.sum(alien_belief)
        for ci in open_cells:
            alien_belief[ci[1]][ci[0]] /= total_belief
        # Update the original probabilities
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = alien_belief[ci[1]][ci[0]]

    def restrict_alien_prob(self, alien_found):
        choose_fun = None
        if alien_found:
            choose_fun = lambda x: self.within_alien_sensor(x)
        else:
            choose_fun = lambda x: not self.within_alien_sensor(x)

        open_cells = self.grid._grid.get_open_indices()
        filtered_open_cells = [oc for oc in open_cells if not choose_fun(oc)]
        #print(f"Cells to set to 0: {len(filtered_open_cells)}")
        for ci in filtered_open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = 0.0
        # Normalize
        total_belief = 0
        for ci in open_cells:
            total_belief += self.grid.grid[ci[1]][ci[0]].alien_belief
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief /= total_belief



    def update_belief(self, beep, alien_found):
        # Crew Belief
        generative_fn = lambda x: np.exp(-self.alpha*(x - 1)) if beep else (1 - np.exp(-self.alpha*(x-1)))
        open_cells = self.grid._grid.get_open_indices()
        for ci in open_cells:
            if ci == self.pos:
                continue
            gen_res = generative_fn(self.grid.distance(ci, self.pos))
            self.grid.grid[ci[1]][ci[0]].crew_belief *= gen_res
        # Normalize
        flat_beliefs = [self.grid.grid[ci[1]][ci[0]].crew_belief for ci in open_cells]
        belief_sum = sum(flat_beliefs)
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].crew_belief /= belief_sum

        # Alien Belief
        # The alien belief consists of two steps-
        # If there is no detection, we diffuse everything outside of the detection square
        # If there is a detection, we set everything outside the square to 0 and leave
        # everything inside the square as is
        alien_belief = np.zeros(( self.grid.D, self.grid.D ))
        self.diffuse_alien_prob(alien_found)
        self.restrict_alien_prob(alien_found)
        #print("Alien detected" if alien_found else "Alien Not Detected")

    def plan_path(self, dest):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid._grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.pos
        path_deque = deque([path_tree])
        destination = None
        visited = set()
        compute_counter = 0
        while not captain_found:
            if len(path_deque) == 0 or compute_counter >= COMPUTE_LIMIT:
                self.grid._grid.remove_all_traversal()
                return
            compute_counter += 1
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid._grid.set_traversed(ind)
            if ind == dest:
                destination = node
                break
            neighbors_ind = self.grid._grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that start with no aliens nearby
                if (self.grid.grid[neighbor_ind[1]][neighbor_ind[0]].alien_belief == 0) or (compute_counter > 2):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid._grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid._grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")

    def move(self):
        self.update_belief(self.crew_sensor(), self.alien_sensor())

        neighbors = self.grid._grid.get_open_neighbors(self.pos)
        neighbors = [n for n in neighbors if not self.grid.crew_pos == n]
        neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        open_cells = self.grid._grid.get_unoccupied_open_indices()

        self.grid._grid.remove_bot(self.pos)
        dest_cell = max(open_cells, key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        self.plan_path(dest_cell)
        if len(self.path) != 0:
            self.pos = self.path[0]
        # If no path is found, we automatically shift to evasion, and the evasion strategy is basic
        # Go to the cell with the lowest alien probability
        else:
            if self.debug:
                print("Evasion!!")
            neighbors = self.grid._grid.get_neighbors(self.pos)
            open_neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open]
            open_neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].alien_belief)
            self.pos = open_neighbors[0]
        #elif self.grid.grid[neighbors[0][1]][neighbors[0][0]].crew_belief == self.grid.grid[neighbors[-1][1]][neighbors[-1][0]].crew_belief:
        #    self.pos = rd.choice(neighbors)
        #else:
        #    self.pos = neighbors[-1]
        
        self.grid._grid.place_bot(self.pos)
        
        if self.pos != self.grid.crew_pos and self.pos != self.grid.crew_pos2:
            self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0
        elif self.pos == self.grid.crew_pos:
            self.grid.crew_pos = None
        elif self.pos == self.grid.crew_pos2:
            self.grid.crew_pos2 = None

        if not self.grid._grid.has_alien(self.pos):
            self.grid.grid[self.pos[1]][self.pos[0]].alien_belief = 0.0
        self.tick += 1

# Bot 2
# The improvements will be mainly in decision making
# We want to explore as much of the the grid as possible before making any changes
# As soon as we sense that the beliefs are concentrating into one spot
# we stop exploring and switch to shortest path planning
class bot4:
    def __init__(self, grid, alpha = 0.15, k=5, debug=1, bot_pos=None):
        self.grid = grid
        self.pos = None if bot_pos is None else bot_pos
        while self.pos in [self.grid.crew_pos, self.grid.crew_pos2] or self.pos is None:
            self.pos = rd.choice(self.grid._grid.get_open_indices())
        self.debug=debug
        if self.debug:
            print(self.pos)
        self.alpha = alpha
        
        self.tick=0
        self.k=k
        self.found_crew = None
        self.found1 = False
        self.found2 = False
        self.found_all_crew = False
        self.switch_to_single = False

    def crew_sensor(self):
        c1 = rd.random()
        c2 = rd.random()
        d1, d2 = self.grid.distance_to_crew(self.pos)
        a, b = False, False

        if d1 is not None:
            a = c1 <= exp(-self.alpha* (d1 - 1))
        if d2 is not None:
            b = c2 <= exp(-self.alpha* (d2 - 1))

        return a or b
    
    def within_alien_sensor(self, pos):
        return abs(pos[0] - self.pos[0]) <= self.k and abs(pos[1] - self.pos[1]) <= self.k
    
    def alien_sensor_edge(self, pos, offset):
        return ( abs(pos[0] - self.pos[0]) == self.k + offset and abs(pos[1] - self.pos[1]) <= self.k ) or (abs(pos[0] - self.pos[0]) <= self.k and abs(pos[1] - self.pos[1]) == self.k + offset)

    def in_danger(self, offset=1):
        for i in range(-offset, offset):
            for j in range(-offset, offset):
                # Skip the current bot location
                if i == 0 and j == 0:
                    continue
                if self.grid.grid[j][i].open and self.grid.grid[j][i].alien_belief > 0.1/self.grid.D:
                    return True
        return False
    
    def alien_sensor(self):
        found_alien = 0
        for j in range(-self.k, self.k + 1):
            if found_alien == 1:
                break
            for i in range(-self.k, self.k + 1):
                pos = [ self.pos[0] + i, self.pos[1] + j ]
                if pos[0] > self.grid.D - 1:
                    pos[0] = self.grid.D - 1
                elif pos[0] < 0:
                    pos[0] = 0
                if pos[1] > self.grid.D - 1:
                    pos[1] = self.grid.D - 1
                elif pos[1] < 0:
                    pos[1] = 0
                if self.grid.grid[pos[1]][pos[0]].alien_id != -1:
                    found_alien = 1
                    break
        return found_alien == 1
    

    def diffuse_alien_prob(self, alien_found):
        choose_fun = None
        if alien_found:
            choose_fun = lambda x: self.within_alien_sensor(x)
        else:
            choose_fun = lambda x: not self.within_alien_sensor(x)
        open_cells = self.grid._grid.get_open_indices()
        # Cells inside the alien sensor and just outside
        # The probability will diffuse among these
        filtered_open_cells = [oc for oc in open_cells if ( choose_fun(oc) or self.alien_sensor_edge(oc, 1 if alien_found else 0) )]
        alien_belief = zeros(self.grid.D, self.grid.D)

        # Diffuse through the edge cells
        for ci in filtered_open_cells:
            neighbors = self.grid._grid.get_neighbors(ci)
            neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open and choose_fun(n) ]
            # Diffuse the probability at the current square into the
            # neighbors that the alien can move to
            for n in neighbors:
                alien_belief[n[1]][n[0]] += self.grid.grid[ci[1]][ci[0]].alien_belief/len(neighbors)
        # Normalizs
        total_belief = grid_sum(self.grid.D, alien_belief)
        for ci in open_cells:
            alien_belief[ci[1]][ci[0]] /= total_belief
        # Update the original probabilities
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = alien_belief[ci[1]][ci[0]]

    def restrict_alien_prob(self, alien_found):
        choose_fun = None
        if alien_found:
            choose_fun = lambda x: self.within_alien_sensor(x)
        else:
            choose_fun = lambda x: not self.within_alien_sensor(x)

        open_cells = self.grid._grid.get_open_indices()
        filtered_open_cells = [oc for oc in open_cells if not choose_fun(oc)]
        #print(f"Cells to set to 0: {len(filtered_open_cells)}")
        for ci in filtered_open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = 0.0
        # Normalize
        total_belief = 0
        for ci in open_cells:
            total_belief += self.grid.grid[ci[1]][ci[0]].alien_belief
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief /= total_belief

    def update_helper(self, crew_member):
        '''
            this resets the probability after one of the crew members has been found
        '''
        # Use this switch to revert back to one crew behavior.
        # Could be done more elegantly with a list, lack of time forced this hacky solution
        self.switch_to_single = True
        crew = None
        if crew_member == 1:
            self.found_crew = self.grid.crew_pos
        elif crew_member == 2:
            self.found_crew = self.grid.crew_pos2
            
        # now we have to remove all the dict keys that don't have this crew coordinate
        new_dict = {}
        open_cells = self.grid._grid.get_open_indices()
        for oc in open_cells:
            max_belief = 0
            for k in self.grid.beliefs:
                if oc in k and self.grid.beliefs[k] > max_belief:
                    max_belief = self.grid.beliefs[k]
            new_dict[oc] = max_belief
            self.grid.grid[oc[1]][oc[0]].crew_belief = 1.0 #max_belief

        total_belief = sum([self.grid.grid[oc[1]][oc[0]].crew_belief for oc in open_cells])
        for oc in open_cells:
            self.grid.grid[oc[1]][oc[0]].crew_belief /= total_belief


    def update_belief(self, beep, alien_found):
        generative_fn = lambda x: exp(-self.alpha * (x - 1))# if beep else (1 - (np.exp(-self.alpha * (x - 1))))
        if self.switch_to_single:
            open_cells = self.grid._grid.get_open_indices()
            for ci in open_cells:
                if ci == self.pos:
                    continue
                dist = self.grid.distance(ci, self.pos)
                gen_res = generative_fn(dist)
                if not beep:
                    gen_res = 1.0 - gen_res
                if gen_res == 0:
                    pass
                    #print("DANGER!!!")
                    #print(f"Distance: {self.grid.distance(ci, self.pos)}, Beep: {beep}")
                self.grid.grid[ci[1]][ci[0]].crew_belief *= gen_res
            # Normalize
            flat_beliefs = [self.grid.grid[ci[1]][ci[0]].crew_belief for ci in open_cells]
            belief_sum = sum(flat_beliefs)
            for ci in open_cells:
                self.grid.grid[ci[1]][ci[0]].crew_belief /= belief_sum

        else:
            # Crew Belief
            for key, _ in self.grid.beliefs.items():
                one_cell, two_cell = key
                self.runs = [[], [], []]
                self.captures = [0, 0, 0]
                self.fails = [0, 0, 0]
                self.turns = [0, 0, 0]
                gen_crew_one, gen_crew_two = 0, 0

                gen_crew_one = generative_fn(self.grid.distance(one_cell, self.pos))
                gen_crew_two = generative_fn(self.grid.distance(two_cell, self.pos))
                if beep:
                    total_prob = gen_crew_one + gen_crew_two - gen_crew_one * gen_crew_two
                else:
                    total_prob = gen_crew_one + gen_crew_two - gen_crew_one * gen_crew_two
                    total_prob = 1 - total_prob
                self.grid.beliefs[(one_cell, two_cell)] *= total_prob


            # Normalize
            sum_beliefs = sum(self.grid.beliefs.values())
            for key, value in self.grid.beliefs.items():
                self.grid.beliefs[key] = value / sum_beliefs

        # Alien Belief

        alien_belief = zeros(self.grid.D, self.grid.D)
        self.diffuse_alien_prob(alien_found)
        self.restrict_alien_prob(alien_found)
        #print("Alien detected" if alien_found else "Alien Not Detected")

    def plan_path(self, dest):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid._grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.pos
        path_deque = deque([path_tree])
        destination = None
        visited = set()
        compute_counter = 0
        while not captain_found:
            if len(path_deque) == 0 or compute_counter >= COMPUTE_LIMIT:
                self.grid._grid.remove_all_traversal()
                return
            compute_counter += 1
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid._grid.set_traversed(ind)
            if ind == dest:
                destination = node
                break
            neighbors_ind = self.grid._grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that start with no aliens nearby and go through paths with a low alien probability
                if (self.grid.grid[neighbor_ind[1]][neighbor_ind[0]].alien_belief == 0 ) or (compute_counter > 1):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid._grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid._grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
    
    def test(self):
        consolidated_prob = {}
        for oc in self.grid._grid.get_open_indices():
            consolidated_prob[oc] = 0.0
        for k, v in self.grid.beliefs.items():
            consolidated_prob[k[0]] += v
            consolidated_prob[k[1]] += v
        overall_probs = list(consolidated_prob)
        overall_probs.sort(key=lambda x: consolidated_prob[x])
        print(f"Highest indices: {overall_probs[-1:-10:-1]}")

    def move(self):        
        beep = self.crew_sensor()
        self.update_belief(beep, self.alien_sensor())
        if self.debug:
            print("BEEP" if beep else "NO BEEP")

        neighbors = self.grid._grid.get_open_neighbors(self.pos)
        neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
        open_cells = self.grid._grid.get_unoccupied_open_indices()

        self.grid._grid.remove_bot(self.pos)

        if self.switch_to_single:
            dest_cell = max(open_cells, key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
            print(f"Dest Cell: {dest_cell}, Actual Crew: {self.grid.crew_pos}, Current Pos: {self.pos}, Distance to Crew: {self.grid.distance_to_crew(self.pos)}")
        else:
            max_belief = max(self.grid.beliefs.values())
            sorted_positions = [key for key in self.grid.beliefs.keys()]
            sorted_positions.sort(key=lambda x: self.grid.beliefs[x])
            position = sorted_positions[-1]
            dest_cell = min(position[0], position[1], 
                            key=lambda x: abs(x[0] - self.pos[0]) + abs(x[1] - self.pos[1])
                        ) if self.found_crew is None else (position[0] if self.found_crew == position[1] else position[1])
            if self.debug:
                print(f"No. of pairs: {len(self.grid.beliefs)}")
                print(f"Top 3 position pairs: {sorted_positions[-1]} : {self.grid.beliefs[sorted_positions[-1]]}, {sorted_positions[-2]} : {self.grid.beliefs[sorted_positions[-2]]}, {sorted_positions[-3]} : {self.grid.beliefs[sorted_positions[-3]]}")
                print(f"Dest Cell: {dest_cell}, actual crew: {self.grid.crew_pos}, {self.grid.crew_pos2}")
        self.plan_path(dest_cell)
        if len(self.path) != 0:
            self.pos = self.path[0]
        # elif self.grid.grid[neighbors[0][1]][neighbors[0][0]].crew_belief == self.grid.grid[neighbors[-1][1]][neighbors[-1][0]].crew_belief:
            # self.pos = rd.choice(neighbors)
        else:
            self.pos = neighbors[-1]
        self.grid._grid.place_bot(self.pos)

        # if self.pos != self.grid.crew_pos:
            # self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0

        if not self.grid._grid.has_alien(self.pos):
            self.grid.grid[self.pos[1]][self.pos[0]].alien_belief = 0.0

        if not self.switch_to_single:
            if (self.pos != self.found_crew) and \
                (self.pos != self.grid.crew_pos) and (self.pos != self.grid.crew_pos2):
                self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0
                keys_to_delete = []
                for key, _ in self.grid.beliefs.items():
                    if self.pos in key:
                        self.grid.beliefs[key] = 0
                        keys_to_delete.append(key)
                for k in keys_to_delete:
                    del self.grid.beliefs[k]
            
            if self.pos == self.grid.crew_pos:
                self.found1 = True
                self.update_helper(1)
                self.grid.crew_pos = self.grid.crew_pos2
                self.grid.crew_pos2 = None
            elif self.pos == self.grid.crew_pos2:
                self.found2 = True
                self.update_helper(2)
                self.grid.crew_pos2 = None
        else:
            if self.pos != self.grid.crew_pos:
                self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0
            else:
                self.found_all_crew = True

        self.tick += 1
        if self.grid.crew_pos == None and self.grid.crew_pos2 == None:
            print("Success!")
            pass

class bot5:
    def __init__(self, grid, alpha = ALPHA, k=5, debug=1, p=None):
        self.grid = grid
        self.pos = p
        while self.pos in [self.grid.crew_pos, self.grid.crew_pos2] or self.pos is None:
            self.pos = rd.choice(self.grid._grid.get_open_indices())
        self.alpha = alpha
        self.debug=debug
        self.tick=0
        self.k=k
        self.DECISION_EXPLORE = 0
        self.DECISION_CLOSE_IN = 1
        self.coarse_grid_size = 7
        self.coarse_grid = [[0 for _ in range(self.coarse_grid_size)] for __ in range(self.coarse_grid_size)]
        self.decision_state = None
        self.dest_cell = None
        self.visited_cg = set()
        self.found_crew = None
        self.found1 = False
        self.found2 = False
        self.found_all_crew = False
        self.switch_to_single = False


    def within_alien_sensor(self, pos):
        return abs(pos[0] - self.pos[0]) <= self.k and abs(pos[1] - self.pos[1]) <= self.k

    def alien_sensor_edge(self, pos, offset):
        return ( abs(pos[0] - self.pos[0]) == self.k + offset and abs(pos[1] - self.pos[1]) <= self.k ) or (abs(pos[0] - self.pos[0]) <= self.k and abs(pos[1] - self.pos[1]) == self.k + offset)

    def in_danger(self, offset=1):
        for i in range(-offset, offset):
            for j in range(-offset, offset):
                # Skip the current bot location
                if i == 0 and j == 0:
                    continue
                if self.grid.grid[j][i].open and self.grid.grid[j][i].alien_belief > 0.1/self.grid.D:                    
                    return True
        return False

    def distance_4d(self, p1, p2):
        p1_1, p1_2 = p1
        p2_1, p2_2 = p2
        d1 = self.grid.distance(p1_1, p2_1) + self.grid.distance(p1_2, p2_2)
        d2 = self.grid.distance(p1_1, p2_2) + self.grid.distance(p1_2, p2_1)
        return min(d1, d2)

    def crew_sensor(self):
        c1 = rd.random()
        c2 = rd.random()

        d1, d2 = self.grid.distance_to_crew(self.pos)
        a, b = False, False

        if d1 is not None:
            a = c1 <= exp(-self.alpha * (d1 - 1))

        if d2 is not None:
            b = c2 <= exp(-self.alpha * (d2 - 1))

        return a or b
    
    def alien_sensor(self):
        found_alien = 0
        for j in range(-self.k, self.k + 1):
            if found_alien == 1:
                break
            for i in range(-self.k, self.k + 1):
                pos = [ self.pos[0] + i, self.pos[1] + j ]
                if pos[0] > self.grid.D - 1:
                    pos[0] = self.grid.D - 1
                elif pos[0] < 0:
                    pos[0] = 0
                if pos[1] > self.grid.D - 1:
                    pos[1] = self.grid.D - 1
                elif pos[1] < 0:
                    pos[1] = 0
                if self.grid.grid[pos[1]][pos[0]].alien_id != -1:
                    found_alien = 1
                    break
        return found_alien == 1

    def diffuse_alien_prob(self, alien_found):
        choose_fun = None
        if alien_found:
            choose_fun = lambda x: self.within_alien_sensor(x)
        else:
            choose_fun = lambda x: not self.within_alien_sensor(x)
        open_cells = self.grid._grid.get_open_indices()
        # Cells inside the alien sensor and just outside
        # The probability will diffuse among these
        filtered_open_cells = [oc for oc in open_cells if ( choose_fun(oc) or self.alien_sensor_edge(oc, 1 if alien_found else 0) )]
        alien_belief = np.zeros((self.grid.D, self.grid.D))

        # Diffuse through the edge cells
        for ci in filtered_open_cells:
            neighbors = self.grid._grid.get_neighbors(ci)
            neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open and choose_fun(n) ]
            # Diffuse the probability at the current square into the
            # neighbors that the alien can move to
            for n in neighbors:
                alien_belief[n[1]][n[0]] += self.grid.grid[ci[1]][ci[0]].alien_belief/len(neighbors)
        # Normalizs
        total_belief = np.sum(alien_belief)
        for ci in open_cells:
            alien_belief[ci[1]][ci[0]] /= total_belief
        # Update the original probabilities
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = alien_belief[ci[1]][ci[0]]

    def restrict_alien_prob(self, alien_found):
        choose_fun = None
        if alien_found:
            choose_fun = lambda x: self.within_alien_sensor(x)
        else:
            choose_fun = lambda x: not self.within_alien_sensor(x)

        open_cells = self.grid._grid.get_open_indices()
        filtered_open_cells = [oc for oc in open_cells if not choose_fun(oc)]
        #print(f"Cells to set to 0: {len(filtered_open_cells)}")
        for ci in filtered_open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief = 0.0
        # Normalize
        total_belief = 0
        for ci in open_cells:
            total_belief += self.grid.grid[ci[1]][ci[0]].alien_belief
        for ci in open_cells:
            self.grid.grid[ci[1]][ci[0]].alien_belief /= total_belief



    def update_helper(self, crew_member):
        '''
            this resets the probability after one of the crew members has been found
        '''
        # Use this switch to revert back to one crew behavior.
        # Could be done more elegantly with a list, lack of time forced this hacky solution
        self.switch_to_single = True
        crew = None
        if crew_member == 1:
            self.found_crew = self.grid.crew_pos
        elif crew_member == 2:
            self.found_crew = self.grid.crew_pos2
            
        # Now we know one of the coordinates. We use that coordinate as one part of the pair
        # and find all the keys which include it. We then use the other coordinate and use
        # it to set the remaining crew member's belief
        new_dict = {}
        open_cells = self.grid._grid.get_open_indices()
        for oc in open_cells:
            self.grid.grid[oc[1]][oc[0]].crew_belief = 0.0 #max_belief
        for k, v in self.grid.beliefs.items():
            if self.found_crew in k:
                other_pos = None
                if k[0] == self.found_crew:
                    other_pos = k[1]
                elif k[1] == self.found_crew:
                    other_pos = k[0]
                else:
                    print("Shouldn't happen!")
                    exit(-1)
                self.grid.grid[other_pos[1]][other_pos[0]].crew_belief = v
        
        # Normalize
        total_belief = sum([self.grid.grid[oc[1]][oc[0]].crew_belief for oc in open_cells])
        for oc in open_cells:
            self.grid.grid[oc[1]][oc[0]].crew_belief /= total_belief


    def update_belief(self, beep, alien_found):
        generative_fn = lambda x: exp(-self.alpha * (x - 1))# if beep else (1 - (np.exp(-self.alpha * (x - 1))))
        if self.switch_to_single:
            open_cells = self.grid._grid.get_open_indices()
            for ci in open_cells:
                if ci == self.pos:
                    continue
                dist = self.grid.distance(ci, self.pos)
                gen_res = generative_fn(dist)
                if not beep:
                    gen_res = 1.0 - gen_res
                if gen_res == 0:
                    pass
                    #print("DANGER!!!")
                    #print(f"Distance: {self.grid.distance(ci, self.pos)}, Beep: {beep}")
                self.grid.grid[ci[1]][ci[0]].crew_belief *= gen_res
            # Normalize
            flat_beliefs = [self.grid.grid[ci[1]][ci[0]].crew_belief for ci in open_cells]
            belief_sum = sum(flat_beliefs)
            for ci in open_cells:
                self.grid.grid[ci[1]][ci[0]].crew_belief /= belief_sum

        else:
            # Crew Belief
            for key, _ in self.grid.beliefs.items():
                one_cell, two_cell = key
                gen_crew_one, gen_crew_two = 0, 0

                gen_crew_one = generative_fn(self.grid.distance(one_cell, self.pos))
                gen_crew_two = generative_fn(self.grid.distance(two_cell, self.pos))
                if beep:
                    total_prob = gen_crew_one + gen_crew_two - gen_crew_one * gen_crew_two
                else:
                    total_prob = gen_crew_one + gen_crew_two - gen_crew_one * gen_crew_two
                    total_prob = 1 - total_prob
                self.grid.beliefs[(one_cell, two_cell)] *= total_prob


            # Normalize
            sum_beliefs = sum(self.grid.beliefs.values())
            for key, value in self.grid.beliefs.items():
                self.grid.beliefs[key] = value / sum_beliefs

        # Alien Belief

        alien_belief = zeros(self.grid.D, self.grid.D)
        self.diffuse_alien_prob(alien_found)
        self.restrict_alien_prob(alien_found)
        #print("Alien detected" if alien_found else "Alien Not Detected")

    #def update_belief(self, beep, alien_found):
    #    # Crew Belief
    #    generative_fn = lambda x: np.exp(-self.alpha*(x - 1)) if beep else (1 - np.exp(-self.alpha*(x-1)))
    #    open_cells = self.grid._grid.get_open_indices()
    #    for ci in open_cells:
    #        if ci == self.pos:
    #            continue
    #        gen_res = generative_fn(self.grid.distance(ci, self.pos))
    #        self.grid.grid[ci[1]][ci[0]].crew_belief *= gen_res
    #    # Normalize
    #    flat_beliefs = [self.grid.grid[ci[1]][ci[0]].crew_belief for ci in open_cells]
    #    belief_sum = sum(flat_beliefs)
    #    for ci in open_cells:
    #        self.grid.grid[ci[1]][ci[0]].crew_belief /= belief_sum

    #    # Alien Belief
    #    # The alien belief consists of two steps-
    #    # If there is no detection, we diffuse everything outside of the detection square
    #    # If there is a detection, we set everything outside the square to 0 and leave
    #    # everything inside the square as is
    #    alien_belief = np.zeros(( self.grid.D, self.grid.D ))
    #    self.diffuse_alien_prob(alien_found)
    #    self.restrict_alien_prob(alien_found)
    #    #print("Alien detected" if alien_found else "Alien Not Detected")

    def plan_path(self, dest):
        if self.debug:
            print("Planning Path...")  # If path is empty we plan one
        self.path = deque([])
        self.grid._grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.pos
        path_deque = deque([path_tree])
        destination = None
        visited = set()
        compute_counter = 0
        while not captain_found:
            if len(path_deque) == 0 or compute_counter >= COMPUTE_LIMIT:
                self.grid._grid.remove_all_traversal()
                return
            compute_counter += 1
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid._grid.set_traversed(ind)
            if ind == dest:
                destination = node
                break
            neighbors_ind = self.grid._grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                # Add all possible paths that start with no aliens nearby
                if (self.grid.grid[neighbor_ind[1]][neighbor_ind[0]].alien_belief == 0) or (compute_counter > 2):
                    new_node = PathTreeNode()
                    new_node.data = neighbor_ind
                    new_node.parent = node
                    node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid._grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.path.extend(reversed(reverse_path))
        for ind in self.path:
            self.grid._grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")

    def measure_belief_bb_size(self):
        highest_prob_pair = max(self.grid.beliefs.keys(), key=lambda x: self.grid.beliefs[x])
        highest_prob = self.grid.beliefs[highest_prob_pair]
        thresh = 0.3
        binarized_pairs = [k for k in self.grid.beliefs.keys() if self.grid.beliefs[k] > highest_prob*thresh]
        max_dist = 0
        for pair in binarized_pairs:
            dist = self.distance_4d(highest_prob_pair, pair)
            if dist > max_dist:
                max_dist = dist
        return max_dist

    def make_decision(self):
        #print("New Decision!")
        # We make these decisions based on how concentrated the beliefs are
        # The easiest way to do this is to create a bounding box of the beliefs after updating it
        # If the manhattan distance between the two diagonal corners is small, we concentrate
        # in on that blob. Otherwise we focus on exploration

        # Let us look at the beliefs and see whether they are concentrated enough:
        if self.decision_state == self.DECISION_EXPLORE or self.decision_state is None:
            self.decision_state = self.DECISION_EXPLORE
            #print("Exploration")
            bb_size = self.measure_belief_bb_size()
            if bb_size <= BB_SIZE_4D:
                return self.DECISION_CLOSE_IN, None
            # We first subdivide the grid into a coarser grid and then we are going choose a random point in each of these
            # coarser cells. We travel there and keep on going further till the beliefs condense. For the time being we are setting
            # the coarse grid size to 5x5

            # USE THE BELOW FOR BOT 5---------------------------------
            #coarse_grid_size = 5
            #cells_per_coarse_cell = self.grid.D // coarse_grid_size
            #if self.grid.D % coarse_grid_size != 0:
            #    print("[ERROR] Invalid coarse grid size")
            #crew_prob_coarse = [[0.0 for _ in range(coarse_grid_size)] for __ in range(coarse_grid_size)]
            #for jc in range( coarse_grid_size ):
            #    for ic in range( coarse_grid_size ):
            #        coarse_cell_prob = 0.0
            #        for j_f in range(cells_per_coarse_cell):
            #            for i_f in (cells_per_coarse_cell):
            #                fine_cell = self.grid.grid[jc * cells_per_coarse_cell + j_f][ic * cells_per_coarse_cell + i_c]
            #                if fine_cell.open:
            #                    coarse_cell_prob += fine_cell.prob_crew
            #        crew_prob_coarse[jc][ic] = coarse_cell_prob
            # --------------------------------------------------------
            stride = self.grid.D / self.coarse_grid_size
            stride = int(stride)
            #print(f"Stride: {stride}")
            coarse_positions = [(i, j) for i in range(self.coarse_grid_size) for j in range(self.coarse_grid_size) if (i, j) not in self.visited_cg]
            if len(coarse_positions) == 0:
                return self.DECISION_CLOSE_IN, None
            coarse_pos = rd.choice(coarse_positions)
            self.visited_cg.add(coarse_pos)
            dest = rd.choice([(coarse_pos[0] + i, coarse_pos[1] + j) for i in range(stride) for j in range(stride) if self.grid.grid[coarse_pos[1] + j][coarse_pos[0] + i].open])
            return self.DECISION_EXPLORE, dest
        return self.DECISION_CLOSE_IN, None
        
    def move(self):
        self.update_belief(self.crew_sensor(), self.alien_sensor())
        bb_size = self.measure_belief_bb_size()
        print(f"Bounding Box size: {bb_size}")
        if bb_size <= BB_SIZE_4D:
            if self.decision_state == self.DECISION_EXPLORE:
                print(f"Turns till convergence: {self.tick}")
            self.decision_state = self.DECISION_CLOSE_IN
        if self.pos == self.dest_cell and self.decision_state == self.DECISION_EXPLORE:
            self.decision_state, self.dest_cell = self.make_decision()
        if self.decision_state == None:
            self.decision_state, self.dest_cell = self.make_decision()
        if self.decision_state == self.DECISION_CLOSE_IN:
            #print("Closing In!!")
            #neighbors = self.grid._grid.get_open_neighbors(self.pos)
            #neighbors = [n for n in neighbors if not self.grid.crew_pos == n]
            #neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
            #open_cells = self.grid._grid.get_unoccupied_open_indices()

            #self.grid._grid.remove_bot(self.pos)
            #dest_cell = max(open_cells, key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
            #self.plan_path(dest_cell)
            #if len(self.path) != 0:
            #    self.pos = self.path[0]
            ## If no path is found, we automatically shift to evasion, and the evasion strategy is basic
            ## Go to the cell with the lowest alien probability
            #else:
            #    if self.debug:
            #        print("Evasion!!")
            #    neighbors = self.grid._grid.get_neighbors(self.pos)
            #    open_neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open]
            #    open_neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].alien_belief)
            #    self.pos = open_neighbors[0]
            ##elif self.grid.grid[neighbors[0][1]][neighbors[0][0]].crew_belief == self.grid.grid[neighbors[-1][1]][neighbors[-1][0]].crew_belief:
            ##    self.pos = rd.choice(neighbors)
            ##else:
            ##    self.pos = neighbors[-1]
            #
            #self.grid._grid.place_bot(self.pos)

            neighbors = self.grid._grid.get_open_neighbors(self.pos)
            neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
            open_cells = self.grid._grid.get_unoccupied_open_indices()

            self.grid._grid.remove_bot(self.pos)

            if self.switch_to_single:
                dest_cell = max(open_cells, key=lambda x: self.grid.grid[x[1]][x[0]].crew_belief)
                print(f"Dest Cell: {dest_cell}, Actual Crew: {self.grid.crew_pos}, Current Pos: {self.pos}, Distance to Crew: {self.grid.distance_to_crew(self.pos)}")
            else:
                max_belief = max(self.grid.beliefs.values())
                sorted_positions = [key for key in self.grid.beliefs.keys()]
                sorted_positions.sort(key=lambda x: self.grid.beliefs[x])
                position = sorted_positions[-1]
                dest_cell = min(position[0], position[1], 
                                key=lambda x: abs(x[0] - self.pos[0]) + abs(x[1] - self.pos[1])
                            ) if self.found_crew is None else (position[0] if self.found_crew == position[1] else position[1])
                if self.debug:
                    print(f"No. of pairs: {len(self.grid.beliefs)}")
                    print(f"Top 3 position pairs: {sorted_positions[-1]} : {self.grid.beliefs[sorted_positions[-1]]}, {sorted_positions[-2]} : {self.grid.beliefs[sorted_positions[-2]]}, {sorted_positions[-3]} : {self.grid.beliefs[sorted_positions[-3]]}")
                    print(f"Dest Cell: {dest_cell}, actual crew: {self.grid.crew_pos}, {self.grid.crew_pos2}")
            self.plan_path(dest_cell)
            if len(self.path) != 0:
                self.pos = self.path[0]
            # elif self.grid.grid[neighbors[0][1]][neighbors[0][0]].crew_belief == self.grid.grid[neighbors[-1][1]][neighbors[-1][0]].crew_belief:
                # self.pos = rd.choice(neighbors)
            else:
                self.pos = neighbors[-1]
            self.grid._grid.place_bot(self.pos)

            # if self.pos != self.grid.crew_pos:
                # self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0

            if not self.grid._grid.has_alien(self.pos):
                self.grid.grid[self.pos[1]][self.pos[0]].alien_belief = 0.0

            if not self.switch_to_single:
                if (self.pos != self.found_crew) and \
                    (self.pos != self.grid.crew_pos) and (self.pos != self.grid.crew_pos2):
                    self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0
                    keys_to_delete = []
                    for key, _ in self.grid.beliefs.items():
                        if self.pos in key:
                            self.grid.beliefs[key] = 0
                            keys_to_delete.append(key)
                    for k in keys_to_delete:
                        del self.grid.beliefs[k]
                
                if self.pos == self.grid.crew_pos:
                    self.found1 = True
                    self.update_helper(1)
                    self.grid.crew_pos = self.grid.crew_pos2
                    self.grid.crew_pos2 = None
                elif self.pos == self.grid.crew_pos2:
                    self.found2 = True
                    self.update_helper(2)
                    self.grid.crew_pos2 = None
            else:
                if self.pos != self.grid.crew_pos:
                    self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0
                else:
                    self.found_all_crew = True
        else:
            self.grid._grid.remove_bot(self.pos)
            self.plan_path(self.dest_cell)
            if len(self.path) != 0:
                self.pos = self.path[0]
            # If no path is found, we automatically shift to evasion, and the evasion strategy is basic
            # Go to the cell with the lowest alien probability
            else:
                if self.debug:
                    print("Evasion!!")
                neighbors = self.grid._grid.get_neighbors(self.pos)
                open_neighbors = [n for n in neighbors if self.grid.grid[n[1]][n[0]].open]
                open_neighbors.sort(key=lambda x: self.grid.grid[x[1]][x[0]].alien_belief)
                self.pos = open_neighbors[0]
            self.grid._grid.place_bot(self.pos)
        
        #if self.pos != self.grid.crew_pos and self.pos != self.grid.crew_pos2:
        #    self.grid.grid[self.pos[1]][self.pos[0]].crew_belief = 0.0
        #elif self.pos == self.grid.crew_pos:
        #    self.grid.crew_pos = None
        #elif self.pos == self.grid.crew_pos2:
        #    self.grid.crew_pos2 = None

        #if not self.grid._grid.has_alien(self.pos):
        #    self.grid.grid[self.pos[1]][self.pos[0]].alien_belief = 0.0
        self.tick += 1

class WorldState:
    def __init__(self, max_runs=100, max_turns=400, alpha_list=[i/100 for i in range(1, 5)] + [i/10 for i in range(1, 5)]):
        self.MAX_RUNS = max_runs
        self.MAX_TURNS = max_turns
        self.runs = [[], [], []]
        self.captures = [0, 0, 0]
        self.fails = [0, 0, 0]
        self.turns = [0, 0, 0]
        self.data = [[], [], []]
        self.ret_turns = [[], [], []]
        self.ret_fails = [[],[], []]
        self.ret_captures = [[],[], []]
        self.alpha_list = alpha_list

    def simulate(self, q=None):
        for alpha in self.alpha_list:
            self.runs = [[], [], []]
            self.captures = [0, 0, 0]
            self.fails = [0, 0, 0]
            self.turns = [0, 0, 0]
            for __ in range(self.MAX_RUNS):
                self.g = Grid2(debug=False, alpha=alpha)
                b = bot3(self.g, alpha=alpha, debug=False)
                a = Alien(self.g._grid, b)
                alien_pos = a.ind
                bot_pos = b.pos
                crew_pos1 = self.g.crew_pos
                crew_pos2 = self.g.crew_pos2
                #succ, run = simulate(g, b, a)
                for _ in range(self.MAX_TURNS):
                    print(f"Alpha: {alpha}, Turn {_}")
                    b.move()
                    if self.g.grid[a.ind[1]][a.ind[0]].alien_belief == 0:
                        print("Alien belief 0 at alien position!!!!")
                    if a.ind == b.pos:
                        print("FAILURE: Alien Capture!")
                        self.captures[0] += 1
                        break
                    #plot_world_state(g, b)
                    #plt.show(
                    a.move()
                    if PLOT:
                        plot_world_state(self.g, b)
                        plt.savefig(f"tmp{_}.png", dpi=200)
                        plt.close()
                        #plt.show()
                        gif_coll.append(Image.open(f"tmp{_}.png"))
                    self.turns[0] += 1
                    if self.g.crew_pos == None and self.g.crew_pos2 == None:
                        print("SUCCES: Crew member reached!")
                        self.runs[0].append(_)
                        break
                    if a.ind == b.pos:
                        print("FAILURE: Alien Capture!")
                        self.captures[0] += 1
                        break
                    if _ == self.MAX_TURNS - 1:
                        self.fails[0] += 1
                        break

                # bot 4 runs now
                del b
                del a
                print(f"Alien Pos: {alien_pos}")
                print(f"Bot Pos: {bot_pos}")
                #del g
                self.g.reset_grid()
                self.g.crew_pos = crew_pos1
                self.g.crew_pos2 = crew_pos2
                #g = Grid2(debug=False)
                b = bot4(self.g, alpha=alpha, debug=False, bot_pos=bot_pos)
                a = Alien(self.g._grid, b, p=alien_pos)
                for _ in range(self.MAX_TURNS):
                    print(f"Turn {_}")
                    b.move()
                    if self.g.grid[a.ind[1]][a.ind[0]].alien_belief == 0:
                        print("Alien belief 0 at alien position!!!!")
                    if a.ind == b.pos:
                        print("FAILURE: Alien Capture!")
                        self.captures[1] += 1
                        break
                    #plot_world_state(g, b)
                    #plt.show()
                    a.move()
                    if PLOT:
                        plot_world_state(self.g, b)
                        plt.savefig(f"tmp{_}.png", dpi=200)
                        plt.close()
                        #plt.show()
                        gif_coll.append(Image.open(f"tmp{_}.png"))
                    self.turns[1] += 1
                    if b.found_all_crew:
                        print("SUCCES: Crew member reached!")
                        self.runs[1].append(_)
                        break
                    if a.ind == b.pos:
                        print("FAILURE: Alien Capture!")
                        self.captures[1] += 1
                        break
                    if _ == self.MAX_TURNS - 1:
                        self.fails[1] += 1
                        break

                # bot 5 runs now
                del b
                del a
                print(f"Alien Pos: {alien_pos}")
                print(f"Bot Pos: {bot_pos}")
                #del g
                self.g.reset_grid()
                self.g.crew_pos = crew_pos1
                self.g.crew_pos2 = crew_pos2
                #g = Grid2(debug=False)
                b = bot5(self.g, alpha=alpha, debug=False, p=bot_pos)
                a = Alien(self.g._grid, b, p=alien_pos)
                for _ in range(self.MAX_TURNS):
                    print(f"Turn {_}")
                    b.move()
                    if self.g.grid[a.ind[1]][a.ind[0]].alien_belief == 0:
                        print("Alien belief 0 at alien position!!!!")
                    if a.ind == b.pos:
                        print("FAILURE: Alien Capture!")
                        self.captures[2] += 1
                        break
                    #plot_world_state(g, b)
                    #plt.show()
                    a.move()
                    if PLOT:
                        plot_world_state(self.g, b)
                        plt.savefig(f"tmp{_}.png", dpi=200)
                        plt.close()
                        #plt.show()
                        gif_coll.append(Image.open(f"tmp{_}.png"))
                    self.turns[2] += 1
                    if b.found_all_crew:
                        print("SUCCES: Crew member reached!")
                        self.runs[2].append(_)
                        break
                    if a.ind == b.pos:
                        print("FAILURE: Alien Capture!")
                        self.captures[2] += 1
                        break
                    if _ == self.MAX_TURNS - 1:
                        self.fails[2] += 1
                        break
                
            # bot 3 data stuff
            print(f"Length Check: {len(self.runs[0])}")
            self.data[0].append(sum(self.runs[0])/len(self.runs[0]) if len(self.runs[0]) > 0 else float('nan'))
            self.ret_captures[0].append(self.captures[0])
            self.ret_fails[0].append(self.fails[0])
            self.ret_turns[0].append(self.turns[0])
            
            
            # bot 4 data stuff
            self.data[1].append(sum(self.runs[1])/len(self.runs[1]) if len(self.runs[1]) > 0 else float('nan'))
            self.ret_captures[1].append(self.captures[1])
            self.ret_fails[1].append(self.fails[1])
            self.ret_turns[1].append(self.turns[1])
            
            
            # bot 5 data stuff
            self.data[2].append(sum(self.runs[2])/len(self.runs[2]) if len(self.runs[2]) > 0 else float('nan'))
            self.ret_captures[2].append(self.captures[2])
            self.ret_fails[2].append(self.fails[2])
            self.ret_turns[2].append(self.turns[2])

            if q is not None:
                print("PUSHING")
                ret_dict = {}
                ret_dict["data"] = self.data
                ret_dict["captures"] = self.ret_captures
                ret_dict["fails"] = self.ret_fails
                ret_dict["turns"] = self.ret_turns
                q.put(ret_dict)
                print("DONE PUSHING")
            else:
                print("Something is wrong!")
                exit(-1)

        return (self.data, self.alpha_list)



def dispatch_jobs(jobs=6, alpha_list=[i/100 for i in range(1, 20, 2)]):
    if len(alpha_list) % jobs > 0:
        print("Not properly divisible!")
        exit(-1)
    queues = [Queue() for i in range(jobs)]
    rets = [0 for i in range(jobs)]
    alpha_lists = [alpha_list[i*(len(alpha_list)//jobs):(i + 1)*(len(alpha_list)//jobs)] for i in range(jobs)]
    states = [WorldState(alpha_list=alpha_lists[i]) for i in range(jobs)]
    print(alpha_lists[0])
    processes = [Process(target=states[i].simulate, args=(), kwargs={"q": queues[i]}) for i in range(jobs)]
    for p in processes:
        p.start()
    for i, q in enumerate(queues):
        print("Getting data")
        rets[i] = q.get()
        print("Got data")
    for p in processes:
        p.join()
    captures = [[], [], []]
    fails = [[], [], []]
    avg_turns = [[], [], []]
    for i, r in enumerate(rets):
        print(f"Return {i}: {r}")
        avg_turns[0].extend(r["data"][0])
        avg_turns[1].extend(r["data"][1])
        avg_turns[2].extend(r["data"][2])
        fails[0].extend(r["fails"][0])
        fails[1].extend(r["fails"][1])
        fails[2].extend(r["fails"][2])
        captures[0].extend(r["captures"][0])
        captures[1].extend(r["captures"][1])
        captures[2].extend(r["captures"][2])
    print(f"Bot 1 Avg Runs: {list(zip(avg_turns[0], alpha_list))}")
    print(f"Bot 1 Captures: {captures[0]}")
    print(f"Bot 1 Fails: {fails[0]}")
    print(f"Bot 2 Avg Runs: {list(zip(avg_turns[1], alpha_list))}")
    print(f"Bot 2 Captures: {captures[1]}")
    print(f"Bot 2 Fails: {fails[1]}")
    print(f"Bot 3 Avg Runs: {list(zip(avg_turns[2], alpha_list))}")
    print(f"Bot 3 Captures: {captures[2]}")
    print(f"Bot 3 Fails: {fails[2]}")
    plt.plot(alpha_list, avg_turns[0], label="Bot 1")
    plt.plot(alpha_list, avg_turns[1], label="Bot 2")
    plt.plot(alpha_list, avg_turns[2], label="Bot 3")
    plt.title("Average Turns Till Rescue")
    plt.xlabel("Alpha")
    plt.ylabel("Average Turns")
    plt.legend()
    plt.show()

gif_coll = []
def plot_world_state(grid, bot):
    red = [1., 0., 0.]
    orange = [1.0, 0.7, 0.0]
    purple = [0.7, 0.0, 1.0]
    blue = [0., 0., 1.]
    green = [0., 1., 0.]
    yellow = [1., 1., 0.]
    white = [1., 1., 1.]
    black = [0., 0., 0.]
    grid_img = []
    grid_img2 = []
    grid_img3 = []
    open_cells = grid._grid.get_open_indices()
    beliefs_flat = [grid.grid[oc[1]][oc[0]].crew_belief for oc in open_cells]
    alien_beliefs_flat = [grid.grid[oc[1]][oc[0]].alien_belief for oc in open_cells]
    max_belief = max(beliefs_flat)
    max_alien_belief = max(alien_beliefs_flat)
    print(f"Max Crew Belief: {max_belief}")
    print(f"Max Alien Belief: {max_alien_belief}")
    for j in range(grid.D):
        grid_img.append([])
        grid_img2.append([])
        grid_img3.append([])
        for i in range(grid.D):
            if grid.crew_pos == (i, j) or grid.crew_pos2 == (i, j):
                grid_img[-1].append(green)
            elif bot.pos == (i, j):
                grid_img[-1].append(yellow)
            elif grid._grid.has_alien((i,j)):
                grid_img[-1].append(red)
            elif grid.grid[j][i].traversed:
                grid_img[-1].append(purple)
            elif grid.grid[j][i].open:
                #grid_img[-1].append([c*grid.grid[j][i].crew_belief/max_belief for c in blue])
                #if grid.grid[j][i].crew_belief < 0:
                #    print("TOO LOW")
                grid_img[-1].append(black)
            else:
                grid_img[-1].append(white)

            if grid.grid[j][i].open:
                grid_img2[-1].append([c*grid.grid[j][i].crew_belief/max_belief for c in blue])
                if grid.grid[j][i].crew_belief < 0:
                    print("TOO LOW")
            else:
                grid_img2[-1].append(white)

            if grid.grid[j][i].open:
                grid_img3[-1].append([c*grid.grid[j][i].alien_belief/max_alien_belief for c in orange])
                if grid.grid[j][i].alien_belief < 0:
                    print("TOO LOW")
            else:
                grid_img3[-1].append(white)
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(grid_img)
    plt.subplot(132)
    plt.imshow(grid_img2)
    plt.subplot(133)
    plt.imshow(grid_img3)
    #plt.show()

MAX_TURNS = 600
MAX_RUNS = 10
PLOT = False


if __name__ == "__main__":
    dispatch_jobs(jobs=10)

#def simulate(gr, bo, al):
#    turns = 0
#    #runs = []
#    #captures = 0
#    #fails = 0
#    for _ in range(MAX_TURNS):
#        print(f"Turn {_}")
#        bo.move()
#        if al.ind == bo.pos:
#            print("FAILURE: Alien Capture!")
#            return False, -1
#            break
#        if gr.grid[al.ind[1]][al.ind[0]].alien_belief == 0:
#            print("Alien belief 0 at alien position!!!!")
#            abel = [[gr.grid[j][i].alien_belief for i in range(g.D)] for j in range(g.D)]
#            plt.imshow(abel)
#            plt.show()
#            exit(-1)
#        al.move()
#        if PLOT:
#            plot_world_state(gr, bo)
#            plt.savefig(f"tmp{_}.png", dpi=200)
#            plt.close()
#            #plt.show()
#            #gif_coll.append(Image.open(f"tmp{_}.png"))
#        turns += 1
#        if gr.crew_pos == bo.pos:
#            print("SUCCES: Crew member reached!")
#            return True, _
#            break
#        if al.ind == bo.pos:
#            print("FAILURE: Alien Capture!")
#            return False, -1
#            break
#        if _ == MAX_TURNS - 1:
#            return False, -2
#            break


#runs = [[], []]
#captures = [0, 0]
#fails = [0, 0]
#turns = [0, 0]
#for __ in range(MAX_RUNS):
#    g = Grid2(debug=False)
#    b = bot5(g, alpha=0.1, debug=True)
#    a = Alien(g._grid, b)
#    alien_pos = a.ind
#    bot_pos = b.pos
#    #succ, run = simulate(g, b, a)
#    for _ in range(MAX_TURNS):
#        print(f"Turn {_}")
#        b.move()
#        if g.grid[a.ind[1]][a.ind[0]].alien_belief == 0:
#            print("Alien belief 0 at alien position!!!!")
#        if a.ind == b.pos:
#            print("FAILURE: Alien Capture!")
#            captures[0] += 1
#            break
#        #plot_world_state(g, b)
#        #plt.show()
#        a.move()
#        turns[0] += 1
#        if b.found_all_crew:
#            print("Success!")
#            runs[0].append(_)
#            break
#        if a.ind == b.pos:
#            print("FAILURE: Alien Capture!")
#            captures[0] += 1
#            break
#        if _ == MAX_TURNS - 1:
#            fails[0] += 1
#            break
#print(f"Bot 5 Runs: {runs[0]}")
#print(f"Bot 5 Fails: {fails[0]}")
#print(f"Bot 5 Captures: {fails[1]}")
#gif_coll[0].save('animated.gif', save_all=True, append_images=gif_coll, duratin=len(gif_coll)*0.2, loop=0)
#if PLOT:
#    print("Saving gif...")
#    os.system("ffmpeg -r 10 -i tmp%01d.png -vcodec mpeg4 -y -vb 400M movie.mp4")
#    for _ in range(turns[0]):
#        os.remove(f"tmp{_}.png")
#plt.style.use("ggplot")
#w = WorldState()
#data, alpha_list = w.simulate()
#plt.xlabel("Alpha")
#plt.ylabel("Avg Turns to Capture")
#plt.plot(alpha_list, data[0], label="Bot3")
#plt.plot(alpha_list, data[1], label="Bot4")
#plt.plot(alpha_list, data[2], label="Bot5")
#
#print(f"data[0]: {data[0]}")
#print(f"data[1]: {data[1]}")
#print(f"data[2]: {data[2]}")
#
#plt.legend()
#plt.savefig(f"bot345_bb70.png", dpi=200)
#print("Bot 1 Output:")
#print(f"Run output: {w.runs[0]}\nAlienCaptures: {w.captures[0]}\nFails: {w.fails[0]}")
#print(f"Average steps: {sum(w.runs[0])/len(w.runs[0])}")
#print("Bot 2 Output:")
#print(f"Run output: {w.runs[1]}\nAlienCaptures: {w.captures[1]}\nFails: {w.fails[1]}")
#print(f"Average steps: {sum(w.runs[1])/len(w.runs[1])}")
