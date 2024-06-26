import numpy as np
from RRTTree import RRTTree
import time
from collections import deque

from AdaptiveSampler2D import AdaptiveSampler2D


class RRTMotionPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob, random=True):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

        # for extend mode E2
        self.eta = 0.5

        self.random_sampler = random
        self.count = 0
        if not random:
            # adaptive sampler
            self.sampler = AdaptiveSampler2D(self.planning_env.config_validity_checker, resolution=0.05)

    def plan(self):
        """
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        """
        start_time = time.time()

        # initialize an empty plan.
        plan = []

        self.tree.add_vertex(self.planning_env.start)
        while True:
            new_config = self.sample_legal()
            self.count += 1
            print(f'size: {self.count}')
            nearest_id, nearest_config = self.tree.get_nearest_config(new_config)
            extended_config, added_goal = self.extend(nearest_config, new_config)
            edge_cost = self.planning_env.robot.compute_distance(extended_config, nearest_config)

            if not self.planning_env.config_validity_checker(extended_config):
                continue

            if self.planning_env.edge_validity_checker(extended_config, nearest_config):
                new_i = self.tree.add_vertex(extended_config, (nearest_id, nearest_config))
                self.tree.add_edge(nearest_id, new_i, edge_cost)
                if added_goal:
                    plan = list(self.build_path(extended_config, new_i, nearest_config))
                    break

        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time() - start_time))

        return np.array(plan)

    def compute_cost(self, plan):
        """
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        """
        # TODO: Task 2.3
        if not plan:
            return float('inf')
        else:
            return self.tree.get_vertex_for_config(plan[-1]).cost

    def extend(self, near_config, rand_config):
        """
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        """
        rand_is_goal = (self.planning_env.goal == rand_config).all()

        if self.ext_mode == "E1":
            return rand_config, rand_is_goal
        else:
            # extend_length = self.planning_env.compute_distance(rand_config, near_config)
            vec = np.subtract(rand_config, near_config)
            vec_mag = np.linalg.norm(vec, 2)
            unit_vec = vec / vec_mag

            # Projection of the step size in the direction of the neighbor
            new_vec = self.eta * unit_vec
            new_config = near_config + new_vec  # New sample point

            if rand_is_goal:
                if vec_mag < self.eta:
                    return rand_config, True
            return new_config, False

    def sample_legal(self):
        p = np.random.uniform()
        config = None
        while True:
            if p < self.goal_prob:
                config = self.planning_env.goal
                return config
            else:
                if self.random_sampler:
                    config = np.random.uniform(low=-np.pi, high=np.pi, size=(2,))
                    if self.planning_env.config_validity_checker(config):
                        return config
                else:
                    # config = self.sampler.sample_point_and_update()
                    config = self.sampler.sample()

                    # print(f'size: {len(self.sampler.X_free) + len(self.sampler.X_obs)}')
                    if self.planning_env.config_validity_checker(np.array(config)):
                        return np.array(config)
            # if self.planning_env.config_validity_checker(config):
            #     return config

    def build_path(self, end_state, end_state_i, s):
        plan_deque = deque([end_state])
        child_i = end_state_i
        while self.tree.edges[child_i]:
            plan_deque.appendleft(s)
            child_i = self.tree.get_idx_for_config(s)
            s = self.tree.vertices[self.tree.edges[child_i]].config
        plan_deque.appendleft(s)
        return plan_deque
