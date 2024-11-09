import math
import numpy as np

class AStarPlanner:

    def __init__(self, occupancy_grid, resolution, rr, origin):
        """
        Initialize grid map for A* planning

        occupancy_grid: 2D numpy array of the occupancy grid (0: free, 1: obstacle)
        resolution: grid resolution [m]
        rr: robot radius [m]
        origin: list or tuple containing [x, y, theta] of the map origin
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = origin[0], origin[1]  # 맵의 origin 설정
        self.max_x = occupancy_grid.shape[1] * resolution + self.min_x
        self.max_y = occupancy_grid.shape[0] * resolution + self.min_y
        self.x_width = occupancy_grid.shape[1]
        self.y_width = occupancy_grid.shape[0]
        self.motion = self.get_motion_model()

        self.obstacle_map = occupancy_grid.astype(bool)
        # 로봇 반경을 고려하여 장애물 영역을 확장하려면 여기에서 추가 코드를 작성할 수 있습니다.
        # 현재는 occupancy_grid가 이미 로봇 크기를 고려한다고 가정함

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return f"{self.x},{self.y},{self.cost},{self.parent_index}"

    def planning(self, sx, sy, gx, gy):
        """
        A* path search

        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        # 시작점과 목표점이 유효한지 확인
        if not self.verify_node(start_node):
            print("Start node is invalid!")
            return None, None
        if not self.verify_node(goal_node):
            print("Goal node is invalid!")
            return None, None

        open_set, closed_set = dict(), dict()
        start_index = self.calc_grid_index(start_node)
        open_set[start_index] = start_node

        while True:
            if not open_set:
                print("Open set is empty..")
                return None, None

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o])
            )
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx[::-1], ry[::-1]

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return int(round((position - min_pos) / self.resolution))

    def calc_grid_index(self, node):
        return node.y * self.x_width + node.x

    def verify_node(self, node):
        if node.x < 0 or node.x >= self.x_width:
            return False
        if node.y < 0 or node.y >= self.y_width:
            return False

        if self.obstacle_map[node.y][node.x]:
            return False

        return True

    def verify_position(self, x, y):
        """
        주어진 위치가 맵 내에 있고 장애물이 아닌지 확인
        """
        grid_x = self.calc_xy_index(x, self.min_x)
        grid_y = self.calc_xy_index(y, self.min_y)
        node = self.Node(grid_x, grid_y, 0.0, -1)
        return self.verify_node(node)

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)]
        ]

        return motion
