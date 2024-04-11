import os
import time
from datetime import datetime
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches as pat
from matplotlib import collections as coll
from numpy.core.fromnumeric import size
from Robot import Robot
from shapely.geometry import Point, LineString, Polygon
import imageio
from AdaptiveSampler2D import AdaptiveSampler2D

class MapEnvironment(object):
    
    # options for origin: 'bottom_left', 'center'
    def __init__(self, json_file, origin='bottom_left'):

        # check if json file exists and load
        json_path = os.path.join(os.getcwd(), json_file)
        if not os.path.isfile(json_path):
            raise ValueError('Json file does not exist!')
        with open(json_path) as f:
            json_dict = json.load(f)

        # obtain boundary limit and start
        self.xlimit = [0, json_dict['WIDTH']]
        self.ylimit = [0, json_dict['HEIGHT']]
        self.start = np.array(json_dict['START'])
        self.load_obstacles(obstacles=json_dict['OBSTACLES'])


        self.goal = np.array(json_dict['GOAL'])

        # create a 2-DOF manipulator robot (2 links)
        self.robot = Robot()

        # origin of robot
        self.origin = [0, 0] # default is 'bottom_left'
        if origin == "center":
            self.origin = [self.xlimit[1] // 2, self.ylimit[1] // 2]

        # check that the start location is within limits and collision free
        if not self.config_validity_checker(config=self.start):
            raise ValueError('Start config must be within the map limits')

        # check that the goal location is within limits and collision free
        if not self.config_validity_checker(config=self.goal):
            raise ValueError('Goal config must be within the map limits')

        # if you want to - you can display starting map here
        self.visualize_map(config=self.start)

    def load_obstacles(self, obstacles):
        '''
        A function to load and verify scene obstacles.
        @param obstacles A list of lists of obstacles points.
        '''
        # iterate over all obstacles
        self.obstacles, self.obstacles_edges = [], []
        for obstacle in obstacles:
            non_applicable_vertices = [x[0] < self.xlimit[0] or x[0] > self.xlimit[1] or x[1] < self.ylimit[0] or x[1] > self.ylimit[1] for x in obstacle]
            if any(non_applicable_vertices):
                raise ValueError('An obstacle coincides with the maps boundaries!')
            
            # make sure that the obstacle is a closed form
            if obstacle[0] != obstacle[-1]:
                obstacle.append(obstacle[0])
                self.obstacles_edges.append([LineString([Point(x[0],x[1]),Point(y[0],y[1])]) for (x,y) in zip(obstacle[:-1], obstacle[1:])])
            self.obstacles.append(obstacle)

    def config_validity_checker(self, config):
        '''
        Verify that the config (given or stored) does not contain self collisions or links that are out of the world boundaries.
        Return false if the config is not applicable, and true otherwise.
        @param config The given configuration of the robot.
        '''
        # compute robot links positions
        robot_positions = self.robot.compute_forward_kinematics(given_config=config, origin=self.origin)

        # add pos of origin
        robot_positions = np.concatenate([np.array(self.origin).reshape((1,2)), robot_positions])

        # verify that the robot do not collide with itself
        if not self.robot.validate_robot(robot_positions=robot_positions):
            return False

        # verify that all robot joints (and links) are between world boundaries
        non_applicable_poses = [(x[0] < self.xlimit[0] or x[1] < self.ylimit[0] or x[0] > self.xlimit[1] or x[1] > self.ylimit[1]) for x in robot_positions]
        if any(non_applicable_poses):
            return False

        # verify that all robot links do not collide with obstacle edges
        # for each obstacle, check collision with each of the robot links
        robot_links = [LineString([Point(x[0],x[1]),Point(y[0],y[1])]) for x,y in zip(robot_positions.tolist()[:-1], robot_positions.tolist()[1:])]
        for obstacle_edges in self.obstacles_edges:
            for robot_link in robot_links:
                obstacle_collisions = [robot_link.crosses(x) for x in obstacle_edges]
                if any(obstacle_collisions):
                    return False

        return True

    def edge_validity_checker(self, config1, config2):
        '''
        A function to check if the edge between two configurations is free from collisions. The function will interpolate between the two states to verify
        that the links during motion do not collide with anything.
        @param config1 The source configuration of the robot.
        @param config2 The destination configuration of the robot.
        '''
        # interpolate between first config and second config to verify that there is no collision during the motion
        required_diff = 0.05
        interpolation_steps = int(np.linalg.norm(config2 - config1)//required_diff)
        if interpolation_steps > 0:
            interpolated_configs = np.linspace(start=config1, stop=config2, num=interpolation_steps)
            
            # compute robot links positions for interpolated configs
            configs_positions = np.apply_along_axis(self.robot.compute_forward_kinematics, 1, interpolated_configs, origin=self.origin)

            # compute edges between joints to verify that the motion between two configs does not collide with anything
            edges_between_positions = []
            for j in range(self.robot.dim):
                for i in range(interpolation_steps-1):
                    edges_between_positions.append(LineString([Point(configs_positions[i,j,0],configs_positions[i,j,1]),Point(configs_positions[i+1,j,0],configs_positions[i+1,j,1])]))

            # check collision for each edge between joints and each obstacle
            for edge_pos in edges_between_positions:
                for obstacle_edges in self.obstacles_edges:
                    obstacle_collisions = [edge_pos.crosses(x) for x in obstacle_edges]
                    if any(obstacle_collisions):
                        return False

            # add origin of robot
            duplicated_origin = np.tile(np.array(self.origin).reshape((1, 1, 2)), (len(configs_positions), 1, 1))
            configs_positions = np.concatenate([duplicated_origin, configs_positions], axis=1) 

            # verify that the robot do not collide with itself during motion
            for config_positions in configs_positions:
                if not self.robot.validate_robot(config_positions):
                    return False

            # verify that all robot joints (and links) are between world boundaries
            if len(np.where(configs_positions[:,:,0] < self.xlimit[0])[0]) > 0 or \
               len(np.where(configs_positions[:,:,1] < self.ylimit[0])[0]) > 0 or \
               len(np.where(configs_positions[:,:,0] > self.xlimit[1])[0]) > 0 or \
               len(np.where(configs_positions[:,:,1] > self.ylimit[1])[0]) > 0:
               return False

        return True

    def compute_angle_of_vector(self, vec):
        '''
        A utility function to compute the angle of the vector from the end-effector to a point.
        @param vec Vector from the end-effector to a point.
        '''
        vec = vec / np.linalg.norm(vec)
        if vec[1] > 0:
            return np.arccos(vec[0])
        else: # vec[1] <= 0
            return -np.arccos(vec[0])

    def check_if_angle_in_range(self, angle, ee_range):
        '''
        A utility function to check if an inspection point is inside the FOV of the end-effector.
        @param angle The angle beteen the point and the end-effector.
        @param ee_range The FOV of the end-effector.
        '''
        # ee range is in the expected order
        if abs((ee_range[1] - self.robot.ee_fov) - ee_range[0]) < 1e-5:
            if angle < ee_range.min() or angle > ee_range.max():
                return False
        # ee range reached the point in which pi becomes -pi
        else:
            if angle > ee_range.min() or angle < ee_range.max():
                return False

        return True

    def compute_union_of_points(self, points1, points2):
        '''
        Compute a union of two sets of inpection points.
        @param points1 list of inspected points.
        @param points2 list of inspected points.
        '''
        # TODO: Task 2.4
        if points1.size == 0: return points2
        if points2.size == 0: return points1
        return np.unique(np.vstack((points1, points2)), axis=0)
    

    # ------------------------#
    # Visualization Functions
    # ------------------------#
    def interpolate_plan(self, plan_configs):
        '''
        Interpolate plan of configurations - add steps between each to configs to make visualization smoother.
        @param plan_configs Sequence of configs defining the plan.
        '''
        required_diff = 0.05

        # interpolate configs list
        plan_configs_interpolated = []
        for i in range(len(plan_configs)-1):

            # number of steps to add from i to i+1
            interpolation_steps = int(np.linalg.norm(plan_configs[i+1] - plan_configs[i])//required_diff) + 1
            interpolated_configs = np.linspace(start=plan_configs[i], stop=plan_configs[i+1], endpoint=False, num=interpolation_steps)
            plan_configs_interpolated += list(interpolated_configs)

        # add goal vertex
        plan_configs_interpolated.append(plan_configs[-1])

        return plan_configs_interpolated

    def visualize_map(self, config, show_map=True):
        '''
        Visualize map with current config of robot and obstacles in the map.
        @param config The requested configuration of the robot.
        @param show_map If to show the map or not.
        '''
        # create empty background
        plt = self.create_map_visualization()

        # add obstacles
        plt = self.visualize_obstacles(plt=plt)

        # add start
        plt = self.visualize_point_location(plt=plt, config=self.start, color='y')

        # add goal
        plt = self.visualize_point_location(plt=plt, config=self.goal, color='g')

        # add robot
        plt = self.visualize_robot(plt=plt, config=config)

        # show map
        if show_map:
            #plt.show() # replace savefig with show if you want to display map actively
            plt.savefig('map.png')
            
        return plt

    def create_map_visualization(self):
        '''
        Prepare the plot of the scene for visualization.
        '''
        # create figure and add background
        plt.figure()
        back_img = np.ones((self.ylimit[1]+1, self.xlimit[1]+1, 3)) # white background
        plt.imshow(back_img, origin="lower", zorder=0)
        plt.xticks(np.arange(0, self.xlimit[1], self.xlimit[1]//10))
        plt.yticks(np.arange(0, self.ylimit[1], self.ylimit[1]//10))
        plt.grid(color='lightgrey', linestyle=':', linewidth=0.5)

        return plt

    def visualize_obstacles(self, plt):
        '''
        Draw the scene's obstacles on top of the given frame.
        @param plt Plot of a frame of the plan.
        '''
        # plot obstacles
        for obstacle in self.obstacles:
            obstacle_xs, obstacle_ys = zip(*obstacle)
            plt.fill(obstacle_xs, obstacle_ys, "r", zorder=5)

        return plt
    
    def visualize_point_location(self, plt, config, color):
        '''
        Draw a point of start/goal on top of the given frame.
        @param plt Plot of a frame of the plan.
        @param config The requested configuration of the point.
        @param color The requested color for the point.
        '''
        # compute point location in 2D
        point_loc = self.robot.compute_forward_kinematics(given_config=config, origin=self.origin)[-1]

        # draw the circle
        point_circ = plt.Circle(point_loc, radius=5, color=color, zorder=5)
        plt.gca().add_patch(point_circ)
    
        return plt

    def visualize_inspection_points(self, plt, inspected_points=None):
        '''
        Draw inspected and not inspected points on top of the plt.
        @param plt Plot of a frame of the plan.
        @param inspected_points list of inspected points.
        '''
        plt.scatter(self.inspection_points[:,0], self.inspection_points[:,1], color='lime', zorder=5, s=3)

        # if given inspected points
        if inspected_points is not None and len(inspected_points) > 0:
            plt.scatter(inspected_points[:,0], inspected_points[:,1], color='g', zorder=6, s=3)

        return plt

    def visualize_robot(self, plt, config):
        '''
        Draw the robot on top of the plt.
        @param plt Plot of a frame of the plan.
        @param config The requested configuration of the robot.
        '''
        # get robot joints and end-effector positions.
        robot_positions = self.robot.compute_forward_kinematics(given_config=config, origin=self.origin)

        # add init robot pos - origin
        robot_positions = np.concatenate([np.array(self.origin).reshape((1,2)), robot_positions])
        # print(robot_positions)

        # draw the robot
        plt.plot(robot_positions[:,0], robot_positions[:,1], 'coral', linewidth=3.0, zorder=10) # joints
        plt.scatter(robot_positions[:,0], robot_positions[:,1], zorder=15) # joints
        plt.scatter(robot_positions[-1:,0], robot_positions[-1:,1], color='cornflowerblue', zorder=15) # end-effector

        return plt

    def get_normalized_angle(self, angle):
        '''
        A utility function to get the normalized angle of the end-effector
        @param angle The angle of the robot's ee
        '''
        if angle > np.pi:
            return angle - 2*np.pi
        elif angle < -np.pi:
            return angle + 2*np.pi
        else:
            return angle

    def visualize_plan(self, plan):
        '''
        Visualize the final plan as a GIF and stores it.
        @param plan Sequence of configs defining the plan.
        '''
        # switch backend - possible bugfix if animation fails
        matplotlib.use('TkAgg')

        # interpolate plan and get inspected points
        plan = self.interpolate_plan(plan_configs=plan)

        # visualize each step of the given plan
        plan_images = []
        for i in range(len(plan)):

            # create background, obstacles, start
            plt = self.create_map_visualization()
            plt = self.visualize_obstacles(plt=plt)
            plt = self.visualize_point_location(plt=plt, config=self.start, color='r')

            # add goal 
            plt = self.visualize_point_location(plt=plt, config=self.goal, color='g')

            # add robot with current plan step
            plt = self.visualize_robot(plt=plt, config=plan[i])

            # convert plot to image
            canvas = plt.gca().figure.canvas
            canvas.draw()
            data = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(canvas.get_width_height()[::-1] + (3,))
            plan_images.append(data)
        
        # store gif
        plan_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        imageio.mimsave(f'plan_{plan_time}.gif', plan_images, 'GIF', duration=0.05)


    def draw_config_space(self, resolution=0.025):
        '''
        Calculate the configuration space for the robot and visualizes it
        (as a scatter plot... looks better than you might think...).
        @param resolution The resolution of the grid for each joint angle.
                          higher takes less time, but looks worse.
        '''
        angle_range = np.arange(0, 2 * np.pi, resolution)
        theta1_grid, theta2_grid = np.meshgrid(angle_range, angle_range)
        theta1_flat = theta1_grid.flatten()
        theta2_flat = theta2_grid.flatten()
        configurations = np.vstack((theta1_flat, theta2_flat)).T

        illegal_configurations = np.array([c for c in configurations if not self.config_validity_checker(c)])
        
        # plot the illegal configurations
        plt.figure()
        plt.scatter(illegal_configurations[:, 0], illegal_configurations[:, 1], c='r', s=1)
        plt.xlabel('first link angle (Radians)')
        plt.ylabel('second link angle (Radians)')
        plt.title('Configuration Space')
        plt.xlim(0, 2 * np.pi)
        plt.ylim(0, 2 * np.pi)
        # plt.grid(color='lightgrey', linestyle=':', linewidth=0.5)
        plt.show()

    def draw_sampled_config_space(self, iterations, resolution=0.025, uniform=False):
        """
        iterations - number of sampling iterations to draw
        resolution - config space drawing resolution (lower is better)
        """
        print("drawing the config space")
        angle_range = np.arange(0, 2 * np.pi, resolution)
        theta1_grid, theta2_grid = np.meshgrid(angle_range, angle_range)
        theta1_flat = theta1_grid.flatten()
        theta2_flat = theta2_grid.flatten()
        configurations = np.vstack((theta1_flat, theta2_flat)).T
        illegal_configurations = np.array([c for c in configurations if not self.config_validity_checker(c)])
        # plot the illegal configurations
        plt.figure()
        plt.scatter(illegal_configurations[:, 0], illegal_configurations[:, 1], c='r', s=1)
        plt.xlabel('first link angle (Radians)')
        plt.ylabel('second link angle (Radians)')
        plt.title('Configuration Space')
        plt.xlim(0, 2 * np.pi)
        plt.ylim(0, 2 * np.pi)

        print(f"running the sampler for {iterations} iterations")
        sampler = AdaptiveSampler2D(resolution=resolution, legal_config_func=self.config_validity_checker)
        X_obs, X_free = sampler.run(num_iterations = iterations, uniform=uniform)
        X_obs = np.array(list(X_obs))
        X_free = np.array(list(X_free))
        plt.scatter(X_obs[:, 0], X_obs[:, 1], c='k', s=1)  # Black for X_obs
        plt.scatter(X_free[:, 0], X_free[:, 1], c='g', s=1)  # Green for X_free

        plt.show()


