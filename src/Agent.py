import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
from DataLoader import *
from CollisionsPerAgent import CollisionsPerAgent
import os

class Agent():
    def __init__(self, id, state, curr_time, capture_freq):
        self.id = id
        self.state = state
        self.time_step_history = [curr_time]
        self.collisions = 0
        self.start_state = state
        self.average_speed = 0.0
        self.average_velocity = 0.0
        self.average_acceleration = 0.0
        self.average_jerk = 0.0
        self.total_distance_travelled = 0
        self.total_displacement = 0.0
        self.total_time_moved = 0
        #must come from same config as in PercentageSlower metric
        self.congestion_threshold_speed = 0.5
        self.time_under_congestion = 0
        self.percent_under_congestion = 0.0
        self.closest_distance_to_other_agent = float('Inf')
        self.furthest_distance_to_other_agent = -1.0
        self.time_scaling_factor = capture_freq #how many seconds in a time_step
        self.is_stopped = False
        self.energy = 0.0
        self.motion_history = [state]
        self.stopped_trail_threshold = 2
        self.stopped_threshold = .01
        self.social_score_dict = {}


    def set_start_state(self, state, time_stamp):
        self.start_state = state
        self.motion_history.append(state)
        self.time_step_history.append(time_stamp)

    def get_state(self):
        return self.new_state

    #TODO: update so its not calculated every iteration
    def update_state(self, new_state, time_step):
        self.state = new_state
        self.time_step_history.append(time_step)
        self.motion_history.append(self.state)
        self.average_speed, self.average_velocity, self.average_acceleration, self.average_jerk = self.get_average_derivatives()
        self.update_time_under_congestion_and_energy()

    def update_time_under_congestion_and_energy(self):
        if len(self.time_step_history) >= 2:
            inst_speed = np.linalg.norm(np.asmatrix(self.motion_history[-1]) - np.asmatrix(self.motion_history[-2])) / \
                         (self.time_step_history[-1] - self.time_step_history[-2])
            if inst_speed < self.congestion_threshold_speed:
                self.time_under_congestion += self.time_scaling_factor
            self.energy += np.square(inst_speed)*self.time_scaling_factor
            total_time_travelled = self.time_step_history[-1] - self.time_step_history[0]
            self.percent_under_congestion = self.time_under_congestion/total_time_travelled

    def get_average_energy(self):
        if len(self.time_step_history) >= 2:
            total_time_travelled = self.time_step_history[-1] - self.time_step_history[0]
            return self.energy/total_time_travelled
        else:
            return 0.0

    def update_agent_proximity(self, dist):
        if dist.size == 1: #no other agents to compare to, no proximities to consider
            return
        dist = dist[dist >= 0.0]
        max = np.max(dist)
        min = np.min(dist)
        if min < self.closest_distance_to_other_agent and min >= 0.0:
            self.closest_distance_to_other_agent = np.min(dist)
        if max > self.furthest_distance_to_other_agent and max < float('Inf'):
            self.furthest_distance_to_other_agent = np.max(dist)

    def get_path_efficiency(self):
        #a return value of 0 indicates no distance travelled
        straight_path_dist = np.linalg.norm(np.asmatrix(self.motion_history[-1]) - np.asmatrix(self.motion_history[0]))
        if self.total_distance_travelled == 0:
            return 0.0
        return straight_path_dist / self.total_distance_travelled

    def get_closest_distance(self):
        return self.closest_distance_to_other_agent

    def get_furthest_distance(self):
        return self.furthest_distance_to_other_agent

    def is_robot_stopped(self):
        if(len(self.motion_history) < self.stopped_trail_threshold):
            return False
        else:
            recent_motion_diffs = np.mean(np.linalg.norm(np.diff(np.asmatrix(self.motion_history[-self.stopped_trail_threshold:]), axis=0), axis=1))
            return recent_motion_diffs < self.stopped_threshold

    def get_average_derivatives(self):
        #can only get average speed with 2 or more time values
        average_speed, average_velocity, average_acceleration, average_jerk = \
            self.average_speed, self.average_velocity, self.average_acceleration, self.average_jerk

        if not self.is_robot_stopped():
            if(len(self.time_step_history) >= 2):
                self.total_displacement = np.linalg.norm(np.asmatrix(self.motion_history[-1]) - np.asmatrix(self.motion_history[0]))
                total_position_diff = (np.diff(np.asmatrix(self.motion_history), axis=0))
                self.total_distance_travelled = np.linalg.norm(np.sum(abs(total_position_diff), axis=0))
                self.total_time_moved = self.time_step_history[-1] - self.time_step_history[0]
                if self.total_time_moved == 0:
                    print("Invalid time steps for agent %f", self.id)
                    average_speed = 0
                average_velocity = self.total_displacement / self.total_time_moved #norm is taken already
                average_speed = self.total_distance_travelled / self.total_time_moved
            if len(self.time_step_history) >= 3:
                total_speed_diff = (np.diff(total_position_diff, axis=0))
                total_speed_travelled = np.sum(total_speed_diff, axis=0)
                total_time_under_acceleration = self.time_step_history[-2] - self.time_step_history[0]
                average_acceleration = np.linalg.norm(total_speed_travelled / total_time_under_acceleration)
            if len(self.time_step_history) >= 4:
                total_acceleration_diff = (np.diff(total_speed_diff, axis=0))
                total_acceleration_diff = np.sum(total_acceleration_diff, axis=0)
                total_time_under_jerk = self.time_step_history[-3] - self.time_step_history[0]
                average_jerk = np.linalg.norm(total_acceleration_diff / total_time_under_jerk)
        return (average_speed, average_velocity, average_acceleration, average_jerk)

    def get_bearing_to_goal(self, curr_pos ):
        d_s = np.asmatrix(self.motion_history[-1]) - np.asmatrix(self.start_state)
        return np.arctan2(d_s[:,1], d_s[:,0])

    #unnecessary rotation between heading to goal and current heading along all timesteps
    def get_path_irregularity(self):
        if len(self.motion_history) >= 2:
            straight_path_vectors = np.repeat(np.asmatrix(self.motion_history[-1]), np.asmatrix(self.motion_history).shape[0] - 1, axis=0) - np.asmatrix(self.motion_history[:-1])
            straight_path_bearings = np.arctan2(straight_path_vectors[:,1], straight_path_vectors[:,0])
            sub_path_diffs = np.diff(np.asmatrix(self.motion_history), axis=0)
            sub_path_bearings = np.degrees( np.arctan2(sub_path_diffs[:,1], sub_path_diffs[:,0]) ) # changed to degrees
            sub_path_bearings[np.squeeze(np.asarray(~sub_path_diffs.any(axis=1)))] = straight_path_bearings[np.squeeze(np.asarray(~sub_path_diffs.any(axis=1)))]
            total_irregularity = np.sum(abs(sub_path_bearings - straight_path_bearings), axis=0)
            return total_irregularity.item()
        return 0.0

    def get_last_updated_time_step(self):
        return self.time_step_history[-1]

    def get_latest_straight_path_time(self, nominal_speed=-1):
        if nominal_speed != -1:
            speed = nominal_speed
        else:
            speed = self.average_speed
        if(speed == 0 and np.linalg.norm(self.total_displacement) == 0): #stationary agent
            return 0
        if len(self.time_step_history) >= 2:
            return float(np.linalg.norm(np.asmatrix(self.motion_history[-1]) - np.asmatrix(self.motion_history[0]))) / float(speed)
        else:
            return -1

    def get_latest_trajectory_time(self):
        if(np.linalg.norm(self.total_displacement) == 0):
            return 0
        if len(self.time_step_history) >= 2:
            return float(self.total_time_moved)
        else:
            return -1

    def get_total_distance_travelled(self):
        if len(self.time_step_history) >= 2:
            return float(self.total_distance_travelled)
        else:
            return -1


