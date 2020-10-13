import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import os


class PathIrregularity(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.collision_radius = .20
        self.collisions_per_agent = dict()
        self.total_collisions = 0
        self.agents = dict()
        self.time_step_done = False
        self.time_step = 0

    def evaluate(self, agents, curr_agents, time_step):
        self.update_agents(agents)

    def update_agents(self, agents):
        self.agents = agents

    def write_out(self, agents):
        irregularity_dict = dict()
        for id in self.agents:
            if self.agents[id].total_distance_travelled > 0:
                irregularity_dict[id] = \
                    self.agents[id].get_path_irregularity()/self.agents[id].total_distance_travelled
                if irregularity_dict[id] > 1000.0:
                    print("")
        irregularity_list = list(irregularity_dict.values())
        average_irregularity = np.mean(irregularity_list)
        std_dev_irregularity = np.std(irregularity_list)
        max_irregularity = np.max(irregularity_list)
        min_irregularity = np.min(irregularity_list)
        
        return {'Mean Path Irregularity (degrees/m)': average_irregularity,
                'StdDev Path Irregularity (degrees/m)': std_dev_irregularity,
                'Max Path Irregularity (degrees/m)': max_irregularity,
                'Min Path Irregularity (degrees/m)': min_irregularity  }
    
                #{'Mean Path Irregularity (radians/m)': average_irregularity },
                #'StdDev Path Irregularity (radians/m)': std_dev_irregularity,
                #'Max Path Irregularity (radians/m)': max_irregularity,
                #'Min Path Irregularity (radians/m)': min_irregularity,
                #'Mean Path Irregularity (radians/m)_Scatter': irregularity_dict}
