import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from Robot import Robot

class Node:
    def __init__(self, coords, lmc=float('inf')):
        self.r = Robot()
        self.coords = np.array(self.r.compute_forward_kinematics(coords)[-1])
        self.lmc = lmc


    def __repr__(self):
        return f"Node(coords={self.coords}, lmc={self.lmc})"

class GraphPlanner:
    def __init__(self, init, goal):
        self.initial = Node(init, 0)
        self.goal = Node(goal)
        self.V = [self.initial]
        self.E = []
        self.X = {'free': [self.initial.coords, self.goal.coords], 'obstacles': []}
        self.relevant_nodes = [self.initial, self.goal]
        self.not_relevant_nodes = []

    def add_obs_node(self, node):
        r = Robot()
        temp = r.compute_forward_kinematics(node)[-1]
        #print(f"obs:temp is: {temp}")
        self.X['obstacles'].append(temp)

    def add_free_node(self, node):
        r = Robot()
        temp = r.compute_forward_kinematics(node)[-1]
        #print(f"free: temp is: {temp}")
        self.X['free'].append(temp)

    def modify_node_lmc(self, x_coords, new_lmc):
        for node in self.V:
            if x_coords == node.coords:
                node.lmc = new_lmc

    def add_new_conf(self, x_coords, lmc=float('inf')):
        node = Node(x_coords, lmc)
        self.V.append(node)

        # if check(node):
        #     self.add_free_node(node)
        # else:
        #     self.add_obs_node(node)

    def add_new_edge(self, node_from, node_to):
        self.E.append([node_from, node_to])

    def add_new_node(self, node, relevancy, lmc):
        node.lmc = lmc
        self.V.append(node)

        # if check(node):
        #     self.add_free_node(node)
        # else:
        #     self.add_obs_node(node)

        if relevancy:
            self.relevant_nodes.append(node)
        else:
            self.not_relevant_nodes.append(node)

    # def run(self, max_iter=1000):
    #     count = 0
    #     for iteration in range(max_iter):
    #         print(f"count is: {count}")
    #         count +=1
    #         xnew = Node([np.random.uniform(0, self.width), np.random.uniform(0, self.height)])
    #
    #         if iteration < 0.1 * max_iter:
    #             self.G[0].append(xnew)
    #             self.initialize_lmc()
    #             self.initial_nodes.append(xnew)
    #         else:
    #             relevant = self.IsRelevant(xnew)
    #             if relevant:
    #                 self.G[0].append(xnew)
    #                 self.initialize_lmc()
    #                 self.relevant_nodes.append(xnew)
    #             else:
    #                 self.not_relevant_nodes.append(xnew)
    #
    #     # Plotting at the end
    #     plt.figure()
    #     ax = plt.gca()
    #     ax.scatter(self.xinit.coords[0], self.xinit.coords[1], c='black')
    #     ax.scatter(self.goal_node.coords[0], self.goal_node.coords[1], c='purple')
    #
    #     # Plotting grid lines
    #     for i in range(self.num_divisions + 1):
    #         ax.axvline(i * self.cell_size_x, color='gray', linestyle='--', linewidth=0.5)
    #         ax.axhline(i * self.cell_size_y, color='gray', linestyle='--', linewidth=0.5)
    #
    #     width = 0.15
    #     height = 0.15
    #     for node in self.initial_nodes:
    #         rect = patches.Rectangle((node.coords[0] - width / 2, node.coords[1] - height / 2), width, height,
    #                                  linewidth=1,
    #                                  edgecolor='green', facecolor='green')
    #
    #         ax.add_patch(rect)
    #
    #     for node in self.relevant_nodes:
    #         rect = patches.Rectangle((node.coords[0] - width / 2, node.coords[1] - height / 2), width, height,
    #                                  linewidth=1,
    #                                  edgecolor='green', facecolor='green')
    #
    #         ax.add_patch(rect)
    #
    #     for node in self.not_relevant_nodes:
    #         rect = patches.Rectangle((node.coords[0] - width / 2, node.coords[1] - height / 2), width, height,
    #                                  linewidth=1,
    #                                  edgecolor='red', facecolor='green')
    #
    #         ax.add_patch(rect)
    #
    #     ax.legend()
    #
    #     ax.grid(True)
    #
    #     plt.show()

class RelevancyCheck:
    def __init__(self, initial_conf, goal_conf, max_iterations):
        self.G = GraphPlanner(initial_conf, goal_conf)
        self.max_iterations = max_iterations
        self.initial_nodes = []
        self.initialize_lmc()

    def KHo(self, x):
        return np.exp(-np.linalg.norm(x))

    def KHf(self, x):
        return np.exp(-np.linalg.norm(x))

    def calculate_hv(self, V, d):
        size_V = len(V)
        return (np.log(size_V) / size_V) ** (1 / d)

    def KHv(self, x, hv):
        Hv = np.diag([hv] * len(x))
        return np.exp(-np.linalg.norm(np.dot(Hv, x)))

    def NearNode(self, points, xnew, num_points):
        #print(f"near: points are: {points}")
        points_tuples = [tuple(point.coords) for point in points]
        #print(f"point tuples is: {points_tuples}")
        #print(f"xnew is: {xnew}")
        tree = KDTree(points_tuples)
        #print(f"Near: xnew is: {xnew}")
        _, idxs = tree.query(tuple(xnew.coords), k=min(num_points, len(points)))
        #print(f"Near: idxs is: {idxs}")
        return [points[idx] for idx in idxs]

    def NearCoords(self, points, xnew, num_points):
        #print(f"near: points are: {points}")
        points_tuples = [tuple(point) for point in points]
        #print(f"point tuples is: {points_tuples}")
        #print(f"xnew is: {xnew}")
        tree = KDTree(points_tuples)
        #print(f"Near: xnew is: {xnew}")
        _, idxs = tree.query(tuple(xnew.coords), k=min(num_points, len(points)))
        #print(f"Near: idxs is: {idxs}")
        return [points[idx] for idx in idxs]
    def g_hat(self, x):
        V, E = self.G.V, self.G.E
        g_hat_xnew = 0
        wtotal = 0
        Xnear = self.NearNode(V, x, len(V))
        #print(f"Xnear is:{Xnear}")
        # for i in Xnear:
        #     print(f"haha {i.coords}")
        hv = self.calculate_hv(V, len(x.coords))
        for x0 in Xnear:
            w = self.KHv(x.coords - x0.coords, hv)
            g_hat_xnew += self.lmc(x0) * w
            wtotal += w
        g_hat_xnew /= wtotal
        return g_hat_xnew

    def h(self, x):
        return np.linalg.norm(x.coords - self.G.goal.coords) / 10

    def Key(self, x):
        g_hat_x = self.g_hat(x)
        h_x = self.h(x)
        return g_hat_x + h_x, g_hat_x

    def lmc(self, x):
        V, E = self.G.V, self.G.E
        if np.array_equal(x.coords, self.G.initial.coords):
            return 0
        if x in V:
            return x.lmc
        else:
            return min(x_i.lmc + self.cost(x_i, x) for x_i in V)

    def cost(self, v1, v2):
        return np.linalg.norm(v1.coords - v2.coords)

    def initialize_lmc(self):
        V, E = self.G.V, self.G.E
        for v in V:
            if np.array_equal(v.coords, self.G.initial.coords):
                v.lmc = 0
        for v in V:
            if not np.array_equal(v.coords, self.G.initial.coords):
                v.lmc = min(x_i.lmc + self.cost(x_i, v) for x_i in V)

    def IsRelevant(self, new_node):
        #print("is relevant called")
        V, E = self.G.V, self.G.E
        Xobs, Xfree = self.G.X['obstacles'], self.G.X['free']

        Pp_obs = len(Xobs) / (len(Xobs) + len(Xfree))
        Pp_free = len(Xfree) / (len(Xobs) + len(Xfree))
        Sobs = self.NearCoords(Xobs, new_node, len(Xobs))
        Pc_obs = sum(self.KHo(new_node.coords - x) for x in Sobs) / len(Sobs)
        Sfree = self.NearCoords(Xfree, new_node, len(Xfree))
        Pc_free = sum(self.KHf(new_node.coords - x) for x in Sfree) / len(Sfree)
        qobs_xnew = Pc_obs * Pp_obs
        qfree_xnew = Pc_free * Pp_free
        if qfree_xnew >= qobs_xnew:
            key_xnew = self.Key(new_node)
            key_xgoal = self.Key(self.G.goal)
        # if self.cost(self.xinit, xnew) < 3:
        #     print(f"key new is: {key_xnew} and key goal is: {key_xgoal}")
            return key_xnew < key_xgoal
        return False

    def WrapperIsRelevant(self, xnew, iteration):
        new_node = Node(xnew)

        if iteration <= 0.1 * self.max_iterations:
           result = True
        else:
            result = self.IsRelevant(new_node)
        new_node = Node(xnew) #TODO: make is relevant return lmc and then delete this line
        self.G.add_new_node(new_node, result, self.lmc(new_node))
        # print(f"in iteration: {iteration}, graph is: ")
        # for i in self.G.V:
        #     print(i)
        if result is False and random.random() <= 0.1:
            return True
        else:
            return result




def main():
    # Initialize the start and goal configurations
    start_conf = [2, 2]
    goal_conf = [5, 5]
    # Create GraphPlanner and RelevancyCheck objects
    planner = GraphPlanner(start_conf, goal_conf)
    relevancy_check = RelevancyCheck(start_conf, goal_conf)

    # Set up the figure
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')

    # Add initial and goal nodes
    ax.scatter(planner.initial.coords[0], planner.initial.coords[1], c='black', label='Start')
    ax.scatter(planner.goal.coords[0], planner.goal.coords[1], c='purple', label='Goal')

    # Number of iterations for random sampling
    num_iterations = 100

    # Perform random sampling and relevancy checking
    for _ in range(num_iterations):
        xnew = [np.random.uniform(0, 10), np.random.uniform(0, 10)]
        if relevancy_check.WrapperIsRelevant(xnew, _):
            color = 'green'
        else:
            color = 'red'

        # Draw rectangle for each sampled configuration
        rect = patches.Rectangle((xnew[0] - 0.15 / 2, xnew[1] - 0.15 / 2), 0.15, 0.15,
                                 linewidth=1, edgecolor=color, facecolor=color, alpha=0.5)
        ax.add_patch(rect)

    # Add legend and show plot
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
