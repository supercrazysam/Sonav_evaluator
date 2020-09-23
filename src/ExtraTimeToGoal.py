##############################################################
# ExtraTimeToGoal class
# Description:
#   This class is responsible for:
#       ExtraTimeToGoal - excess time taken to get to goal travelling at agent's average speed
#       PathEfficiency - ratio of straight line path to agent's path




import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import os

class ExtraTimeToGoal(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.nominal_speed = .5
        self.extra_time_to_goal_dict = dict()
        self.path_efficiency_dict = dict()

        #added time efficiency
        self.time_efficiency_dict = dict()
        
        self.time_step_done = False

    def evaluate(self, agents, curr_agents, time_stamp):
        return

    def set_extra_time_to_goal(self, agent_id):
        agent = self.agents[agent_id]
        agent_path_time = agent.get_latest_trajectory_time()         #actual time
        straight_path_time = agent.get_latest_straight_path_time()  #nominal time  #caculate (goal - start) / nominal speed
        extra_time = agent_path_time - straight_path_time
        self.extra_time_to_goal_dict[agent_id] = extra_time

        ideal_straight_path_time = agent.get_latest_straight_path_time() #use(1) nominal time if it follows 1 m/s speed setting (especially in population)

        self.time_efficiency_dict[agent_id] = ideal_straight_path_time /  (agent_path_time+0.0000001) #latest nominal time / latest actual time   

    def set_path_efficiency(self, agent_id):
        path_efficiency = self.agents[agent_id].get_path_efficiency()
        #don't want to consider case where agent doesn't move
        if path_efficiency > 0:
            self.path_efficiency_dict[agent_id] = path_efficiency

    def update_agents(self, agents):
        self.agents = agents

    #this evaluation metric has extra computation to perform at the end of the episode
    def write_out(self, agents):
        self.update_agents(agents)
        for agent_id in self.agents:
            self.set_extra_time_to_goal(agent_id)
            self.set_path_efficiency(agent_id)
        extra_time_to_goal_list = np.asarray(list(self.extra_time_to_goal_dict.values()))
        path_efficiency_list = np.asarray(list(self.path_efficiency_dict.values()))

        time_efficiency_list = np.asarray(list(self.time_efficiency_dict.values()))

        self.average_extra_time_to_goal = np.mean(extra_time_to_goal_list)
        self.std_dev_extra_time_to_goal = np.std(extra_time_to_goal_list)
        self.min_extra_time_to_goal = np.min(extra_time_to_goal_list)
        self.max_extra_time_to_goal = np.max(extra_time_to_goal_list)

        self.average_path_efficiency = np.mean(path_efficiency_list)
        self.std_dev_path_efficiency = np.std(path_efficiency_list)
        self.min_path_efficiency = np.min(path_efficiency_list)
        self.max_path_efficiency = np.max(path_efficiency_list)

        self.average_time_efficiency = np.mean(time_efficiency_list)
        self.std_dev_time_efficiency = np.std(time_efficiency_list)
        self.min_time_efficiency = np.min(time_efficiency_list)
        self.max_time_efficiency = np.max(time_efficiency_list)

        

        return {#'Mean Extra Time to Goal (s)': self.average_extra_time_to_goal,
                #'StdDev Extra Time to Goal (s)': self.std_dev_extra_time_to_goal,
                #'Max Extra Time to Goal (s)': self.max_extra_time_to_goal,
                #'Min Extra Time to Goal (s)': self.min_extra_time_to_goal,
                #'Mean Extra Time to Goal (s)_Scatter': self.extra_time_to_goal_dict,
                'Mean Path Efficiency': self.average_path_efficiency,
                'StdDev Path Efficiency': self.std_dev_path_efficiency,
                #'Max Path Efficiency': self.max_path_efficiency,
                #'Min Path Efficiency': self.min_path_efficiency,
                #'Mean Path Efficiency_Scatter': self.path_efficiency_dict,

                'Mean Time Efficiency': self.average_time_efficiency,
                'StdDev Time Efficiency': self.std_dev_time_efficiency}
                #'Max Time Efficiency': self.max_time_efficiency,
                #'Min Time Efficiency': self.min_time_efficiency, 
                #'Mean Time Efficiency_Scatter': self.time_efficiency_dict}#added

