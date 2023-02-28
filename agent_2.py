import sys
import time
from typing import List

import numpy as np
# import matplotlib.pyplot as plt
import pygame
import random
import heapq as heap


class Node:

    def __init__(self, pos) -> None:
        self.pos = pos

        self.g = 0
        self.h = 0
        self.f = 0
        self.ghost_value = 0

    def __eq__(self, __o: object) -> bool:
        return self.pos == __o.pos

    def __hash__(self):
        return hash(str(self))


def check_min_path_grid(got_created_numpy_maze, got_stale_numpy_maze) -> bool:
    got_stale_numpy_maze = got_stale_numpy_maze

    # Could use dict here
    unchecked_nodes = []
    checked_nodes = []

    heap.heapify(checked_nodes)

    start_axis = (0, 0)
    goal_axis = (51, 51)

    start_node = Node(start_axis)
    end_node = Node(goal_axis)

    unchecked_nodes.append(start_node)

    movable_direc = {'see_left': (0, -1), 'see_right': (0, 1), 'see_up': (-1, 0), 'see_down': (1, 0)}

    num_of_iterations = 0
    max_iterations = (len(created_numpy_maze) // 2) ** 4

    while len(unchecked_nodes) != 0:

        num_of_iterations += 1

        cur_node_obj = unchecked_nodes[0]
        cur_index = 0

        for index, obj_item in enumerate(unchecked_nodes):
            if obj_item.f < cur_node_obj.f:
                cur_node_obj = obj_item
                cur_index = index

        if num_of_iterations > max_iterations:
            print("No solution for this maze. Repeat!")
            num_of_iterations = 0
            re_created_numpy_maze = maze_creation(got_stale_numpy_maze)
            check_min_path_grid(re_created_numpy_maze, got_stale_numpy_maze)

        if len(unchecked_nodes) == 0:
            print("No solution for this maze. Repeat!")
            num_of_iterations = 0
            re_created_numpy_maze = maze_creation(got_stale_numpy_maze)
            check_min_path_grid(re_created_numpy_maze, got_stale_numpy_maze)

        cur_node_neighbours = []

        unchecked_nodes.pop(cur_index)
        checked_nodes.append(cur_node_obj)

        if cur_node_obj == end_node:
            print("There is a path available here!")
            return True

        for single_step in movable_direc.values():
            cur_pos_neighbour = tuple(np.add(cur_node_obj.pos, single_step))
            if cur_pos_neighbour[0] < 0 or cur_pos_neighbour[1] < 0 or cur_pos_neighbour[0] > 51 or \
                    cur_pos_neighbour[1] > 51 or got_created_numpy_maze[cur_pos_neighbour] == 0:
                continue

            new_node = Node(cur_pos_neighbour)

            cur_node_neighbours.append(new_node)

        for neigh in cur_node_neighbours:

            if neigh in checked_nodes:
                continue

            neigh.g = cur_node_obj.g + 1
            # Omitted sqrt because it's costly and doesn't give great adv
            neigh.h = ((neigh.pos[0] - end_node.pos[0]) ** 2) + ((neigh.pos[1] - end_node.pos[1]) ** 2)
            neigh.f = neigh.g + neigh.h

            for sing_node_obj in unchecked_nodes:
                if neigh == sing_node_obj and neigh.g > sing_node_obj.g:
                    break

            else:
                unchecked_nodes.append(neigh)


'''Agent 2 has a 2x2 grid proximity around him. So, if there are any ghosts within his proximity he will try to avoid that ghost
by a choosing a neighbour node with the highest ghost euclidean distance effectively averting from choosing a neighbour position
that is closer to a ghost'''
'''Agent 2 if there is a ghost in his proximity will calculate f values of his neighbour node positions but will penalize
the neighbour position with very high f value (ghost value along with g,h,f values) if that position is near to a ghost based
on euclidean distance to that ghost'''


def agent_2(got_created_numpy_maze):
    unchecked_nodes = []
    checked_nodes = []

    heap.heapify(checked_nodes)

    start_axis = (0, 0)
    goal_axis = (51, 51)

    start_node = Node(start_axis)
    end_node = Node(goal_axis)

    unchecked_nodes.append(start_node)

    movable_direc = {'see_left': (0, -1), 'see_right': (0, 1), 'see_up': (-1, 0), 'see_down': (1, 0)}

    num_of_iterations = 0
    max_iterations = (len(created_numpy_maze) // 2) ** 4

    PROB_GHOST_GO = 0.50
    PROB_GHOST_STAY = 0.50

    agent_2_x_2_grid_search_pos = [(0, 1), (0, 2),
                                   (0, -1), (0, -2),
                                   (1, 0), (2, 0),
                                   (-1, 0), (-2, 0),
                                   (-1, -1), (-2, -2),
                                   (-1, 1), (-2, 2),
                                   (1, -1), (2, -2),
                                   (1, 1), (2, 2)]

    while len(unchecked_nodes) != 0:

        time.sleep(0.2)

        num_of_iterations += 1

        cur_node_obj = unchecked_nodes[0]
        cur_index = 0

        for index, obj_item in enumerate(unchecked_nodes):
            if obj_item.f < cur_node_obj.f:
                cur_node_obj = obj_item
                cur_index = index

        if num_of_iterations > max_iterations:
            print("Taking too long to find path. Agent 2 Failed!")
            return None

        cur_node_neighbours = []
        nearby_ghost_in_proximity_pos = 0

        unchecked_nodes.pop(cur_index)
        checked_nodes.append(cur_node_obj)
        # print("Cur Pos = ", cur_node_obj.pos)

        for g in ghosts_positions.copy().keys():
            pick_random_direc = random.choice(tuple(movable_direc.values()))
            move_random_direc = tuple(np.add(g, pick_random_direc))
            if move_random_direc[0] < 0 or move_random_direc[1] < 0 or move_random_direc[0] > 51 or \
                    move_random_direc[1] > 51:
                continue
            if got_created_numpy_maze[move_random_direc] == 9:
                # print('Hi neighbour friend')
                continue
            if got_created_numpy_maze[move_random_direc] == 0:
                # print('I cant move freely maybe')
                ghost_random_step = random.choices(('C', 'A'), [PROB_GHOST_GO, PROB_GHOST_STAY], k=1)
                if ghost_random_step == ['C']:
                    # print('I dont wanna move')
                    continue
                elif ghost_random_step == ['A']:
                    # print('I want to move no matter what')
                    actual_about_to_move_ghost_pos_value = got_created_numpy_maze[move_random_direc]
                    got_created_numpy_maze[move_random_direc] = 9
                    if ghosts_positions[g] == 0:
                        got_created_numpy_maze[g] = 0
                        ghosts_positions.pop(g)
                        ghosts_positions[tuple(move_random_direc)] = actual_about_to_move_ghost_pos_value
                        pygame.draw.rect(screen, DIM_GREY, [(MARGIN + WIDTH) * g[1] +
                                                            MARGIN, (MARGIN + HEIGHT) * g[0] + MARGIN, WIDTH, HEIGHT])

                    elif ghosts_positions[g] == 1:
                        got_created_numpy_maze[g] = 1
                        ghosts_positions.pop(g)
                        ghosts_positions[tuple(move_random_direc)] = actual_about_to_move_ghost_pos_value
                        pygame.draw.rect(screen, WHITE, [(MARGIN + WIDTH) * g[1] +
                                                         MARGIN, (MARGIN + HEIGHT) * g[0] + MARGIN, WIDTH, HEIGHT])

                    # ghosts_positions.append(move_random_direc)

                    pygame.draw.rect(screen, RED, [(MARGIN + WIDTH) * move_random_direc[1] +
                                                   MARGIN, (MARGIN + HEIGHT) * move_random_direc[0] + MARGIN, WIDTH,
                                                   HEIGHT])

                    pygame.display.update()
                    # break
            elif got_created_numpy_maze[move_random_direc] == 1:
                # print('lol I can move freely')
                actual_about_to_move_ghost_pos_value = got_created_numpy_maze[move_random_direc]
                got_created_numpy_maze[move_random_direc] = 9
                if ghosts_positions[g] == 0:
                    got_created_numpy_maze[g] = 0
                    ghosts_positions.pop(g)
                    ghosts_positions[tuple(move_random_direc)] = actual_about_to_move_ghost_pos_value
                    pygame.draw.rect(screen, DIM_GREY, [(MARGIN + WIDTH) * g[1] +
                                                        MARGIN, (MARGIN + HEIGHT) * g[0] + MARGIN, WIDTH, HEIGHT])
                elif ghosts_positions[g] == 1:
                    got_created_numpy_maze[g] = 1
                    ghosts_positions.pop(g)
                    ghosts_positions[tuple(move_random_direc)] = actual_about_to_move_ghost_pos_value
                    pygame.draw.rect(screen, WHITE, [(MARGIN + WIDTH) * g[1] +
                                                     MARGIN, (MARGIN + HEIGHT) * g[0] + MARGIN, WIDTH, HEIGHT])

                # ghosts_positions.append(move_random_direc)
                pygame.draw.rect(screen, RED, [(MARGIN + WIDTH) * move_random_direc[1] +
                                               MARGIN, (MARGIN + HEIGHT) * move_random_direc[0] + MARGIN, WIDTH,
                                               HEIGHT])

                pygame.display.update()

        for s in agent_2_x_2_grid_search_pos:
            my_search_for_ghosts_grid = tuple(np.add(cur_node_obj.pos, s))
            if my_search_for_ghosts_grid[0] < 0 or my_search_for_ghosts_grid[1] < 0 or my_search_for_ghosts_grid[
                0] > 51 or \
                    my_search_for_ghosts_grid[1] > 51:
                continue
            if got_created_numpy_maze[my_search_for_ghosts_grid] == 9:
                nearby_ghost_in_proximity_pos = my_search_for_ghosts_grid
                print('There is a ghost in my proximity = ', my_search_for_ghosts_grid)
                # NEED TO POP GHOSTS AS WELL

        for d in checked_nodes:
            pygame.draw.rect(screen, TEAL,
                             [(MARGIN + WIDTH) * d.pos[1] + MARGIN, (MARGIN + HEIGHT) * d.pos[0] + MARGIN, WIDTH,
                              HEIGHT])

        pygame.display.update()

        if cur_node_obj.pos in ghosts_positions:
            print("YOU ARE DEAD!")
            pygame.draw.rect(screen, BLACK,
                             [(MARGIN + WIDTH) * cur_node_obj.pos[1] + MARGIN, (MARGIN + HEIGHT) * cur_node_obj.pos[0] +
                              MARGIN, WIDTH, HEIGHT])

            pygame.display.update()
            return checked_nodes

        if cur_node_obj == end_node:
            print("DONEEEE!!")
            pygame.display.set_caption("AGENT WON!")
            return checked_nodes

        for single_step in movable_direc.values():
            cur_pos_neighbour = tuple(np.add(cur_node_obj.pos, single_step))

            if cur_pos_neighbour[0] < 0 or cur_pos_neighbour[1] < 0 or cur_pos_neighbour[0] > 51 or \
                    cur_pos_neighbour[1] > 51 or got_created_numpy_maze[cur_pos_neighbour] == 0 or \
                    got_created_numpy_maze[cur_pos_neighbour] == 9:
                continue

            new_node = Node(cur_pos_neighbour)

            cur_node_neighbours.append(new_node)

        neigh_dict = {}
        for neigh in cur_node_neighbours:

            if neigh in checked_nodes:
                continue

            neigh.g = cur_node_obj.g + 1
            # Omitted sqrt because it's costly and doesn't give great adv
            neigh.h = ((neigh.pos[0] - end_node.pos[0]) ** 2) + ((neigh.pos[1] - end_node.pos[1]) ** 2)
            if nearby_ghost_in_proximity_pos != 0:
                ghost_dist = ((neigh.pos[0] - nearby_ghost_in_proximity_pos[0]) ** 2) + (
                            (neigh.pos[1] - nearby_ghost_in_proximity_pos[1]) ** 2)
                neigh.ghost_value = (neigh.ghost_value + ghost_dist)

            neigh.f = neigh.g + neigh.h

            for sing_node_obj in unchecked_nodes:
                if neigh == sing_node_obj and neigh.f > sing_node_obj.f:
                    break

            else:
                neigh_dict[neigh] = neigh.ghost_value

        xx = sorted(neigh_dict.items(), key=lambda item: item[1], reverse=True)
        # max_one = hu[0]
        # for index, kk in enumerate(hu):
        #     if index == 0:
        #         max_one = kk
        #     elif index == 1:
        #         if kk.ghost_value > max_one.ghost_value:
        #             max_one = kk
        #     elif index == 2:
        #         if kk.ghost_value > max_one.ghost_value:
        #             max_one = kk
        for index, i in enumerate(xx):
            if i[1] > 0:
                if index == 0:
                    i[0].f = (10 * 2)
                    # unchecked_nodes.append(i[0])
                if index == 1:
                    i[0].f = (10 * 3)
                    # unchecked_nodes.append(i[0])
                if index == 2:
                    i[0].f = (10 * 4)
                    # unchecked_nodes.append(i[0])
                if index == 3:
                    i[0].f = (10 * 5)
                    # unchecked_nodes.append(i[0])
                if index == 4:
                    i[0].f = (10 * 6)
                    # unchecked_nodes.append(i[0])
            else:
                unchecked_nodes.append(i[0])
        xx.clear()

        # if i[1] > 0:
        #     if index == 0:
        #         unchecked_nodes.append(i[0])
        #     if index == 1:
        #         unchecked_nodes.append(i[0])
        #
        # elif i[1] == 0:
        #     unchecked_nodes.append(i[0])
        # if index == 2:
        #     break
        # unchecked_nodes.append(i[0])


def maze_creation(numpy_maze) -> np.ndarray:
    for index_row, item_row in enumerate(numpy_maze):

        for index_col, item_col in enumerate(item_row):

            my_random = random.choices(PROB_LIST, [PROB_BlCK, PROB_UNBLCK], k=1)

            if 0 in my_random:
                if (index_row not in range(0, 3) or index_col not in range(0, 3)) and \
                        (index_row not in range(len(numpy_maze) - 3, len(numpy_maze)) or index_col not in range(
                            len(numpy_maze) - 3, len(numpy_maze))):
                    numpy_maze[index_row][index_col] = 0

            elif 1 in my_random:
                numpy_maze[index_row][index_col] = 1

    return numpy_maze


if __name__ == '__main__':

    PROB_BlCK = 0.28
    PROB_UNBLCK = 0.72

    PROB_LIST = [0, 1]

    BLACK = (0, 0, 0)
    # DARK_GREY = (169, 169, 169)
    DIM_GREY = (105, 105, 105)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 128, 0)
    BLUE = (0, 0, 255)
    TEAL = (0, 128, 128)

    WIDTH = 10
    HEIGHT = 10

    MARGIN = 2

    matrix = []

    for row in range(0, 52):
        matrix.append([])

        for col in range(0, 52):
            matrix[row].append(1)

    stale_numpy_maze = np.array(matrix)

    created_numpy_maze = maze_creation(stale_numpy_maze)

    # Ghost Creation
    ghost_random = random.randrange(10, 20)
    print(ghost_random)
    ghosts_positions = {}

    for iterate in range(ghost_random):
        x = random.randrange(4, 52)
        y = random.randrange(4, 52)

        if created_numpy_maze[x][y] != 9:
            actual_ghost_pos_value = created_numpy_maze[x][y]
            created_numpy_maze[x][y] = 9
            ghosts_positions[tuple((x, y))] = actual_ghost_pos_value
            # ghosts_positions.append(tuple((x, y)))
        else:
            print("same spot!")

    print(ghosts_positions)
    pygame.init()

    WINDOW_SIZE = [700, 700]
    screen = pygame.display.set_mode(WINDOW_SIZE)

    pygame.display.set_caption("Maze")

    screen.fill(BLACK)

    for row in range(52):
        for column in range(52):
            color = WHITE
            if created_numpy_maze[row][column] == 0:
                color = DIM_GREY

            if created_numpy_maze[row][column] == 9:
                color = RED

            pygame.draw.rect(screen, color,
                             [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT])

    pygame.display.update()

    there_is_path = check_min_path_grid(created_numpy_maze, stale_numpy_maze)

    if there_is_path:
        agent_2(created_numpy_maze)

    while there_is_path:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(type(pygame.event.get()))
                sys.exit()
            elif event.type == pygame.KEYUP:
                pygame.quit()
