import itertools
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from shapely.geometry import Point, LineString

class Robot(object):
    
    def __init__(self):

        # define robot properties
        self.links = np.array([4,4])
        self.dim = len(self.links)
    

    def compute_distance(self, prev_config, next_config):
        '''
        Compute the euclidean distance betweeen two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        '''
        prev_config, next_config = np.array(prev_config), np.array(next_config)
        return np.linalg.norm(prev_config-next_config, ord=2)
    

    def compute_forward_kinematics(self, given_config, origin=[0, 0]):
        '''
        Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
        @param given_config Given configuration.
        '''
        x_init, y_init = origin[0], origin[1]
        theta = given_config[0]
        links_pos = np.zeros((self.dim, 2))
        links_pos[0] = [x_init + self.links[0] * np.cos(theta), y_init + self.links[0] * np.sin(theta)]

        for i in range(1, self.dim):
            theta = self.compute_link_angle(theta, given_config[i])
            links_pos[i] = [self.links[i] * np.cos(theta) + links_pos[i-1][0], self.links[i] * np.sin(theta) + links_pos[i-1][1]]

        return links_pos
        

    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        ee_angle = given_config[0]
        for i in range(1,len(given_config)):
            ee_angle = self.compute_link_angle(ee_angle, given_config[i])

        return ee_angle

    def compute_link_angle(self, link_angle, given_angle):
        '''
        Compute the 1D orientation of a link given the previous link and the current joint angle.
        @param link_angle previous link angle.
        @param given_angle Given joint angle.
        '''
        if link_angle + given_angle > np.pi:
            return link_angle + given_angle - 2*np.pi
        elif link_angle + given_angle < -np.pi:
            return link_angle + given_angle + 2*np.pi
        else:
            return link_angle + given_angle
        
    def validate_robot(self, robot_positions):
        '''
        Verify that the given set of links positions does not contain self collisions.
        @param robot_positions Given links positions.
        '''
        # TODO: Task 2.2
        links = []

        for i in range(len(robot_positions) - 1):
            next_link = LineString([robot_positions[i], robot_positions[i + 1]])

            for j, link in enumerate(links):

                if j < len(links) - 1 and next_link.intersects(link):
                    return False

                if next_link.overlaps(link):
                    return False

            if i == len(robot_positions) - 2 and next_link.equals(links[-1]):
                return False

            links.append(next_link)

        return True

        
    