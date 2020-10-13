import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import xmltodict
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import os

class CongestionScore(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.agents = {}
        self.average_speeds_per_agent = {}
        self.average_speed_threshold = .5

    def update_agents(self, agents):
        self.agents = agents

    def evaluate(self, agents, curr_agents, time_stamp):
        self.update_agents(agents)

    def get_averaged_agent_speed(self):
        speeds = [self.agents[agent_id].average_speed for agent_id in self.agents]
        self.percentage_under_congestion = [self.agents[agent_id].percent_under_congestion for agent_id in self.agents]
        self.average_agent_speed = np.mean(speeds)

    #evaluation occurs at the end
    def write_out(self, agents):
        self.get_averaged_agent_speed()
        d = {id: self.agents[id].average_speed for id in self.agents}
        self.percentage_slower = np.count_nonzero((np.asarray(list(d.values())) < self.average_speed_threshold).astype(float)) / len(list(self.agents.keys()))
        mean_under_congestion = np.mean(self.percentage_under_congestion)
        std_dev_under_congestion = np.std(self.percentage_under_congestion)
        min_under_congestion = np.min(self.percentage_under_congestion)
        max_under_congestion = np.max(self.percentage_under_congestion)
        return {'Mean Congestion Score'   : self.percentage_slower ,
                'StdDev Congestion Score' : std_dev_under_congestion,
                'Min Congestion Score'    : min_under_congestion,
                'Max Congestion Score'    : max_under_congestion }#
    
        '''     {'Percentage of Agents Average Speeds Slower than %f m/s' % self.average_speed_threshold: self.percentage_slower * 100.0 },
                'Mean Percentage of Time Congested below %f m/s' % self.average_speed_threshold: mean_under_congestion * 100,
                'StdDev Percentage of Time Congested below %f m/s' % self.average_speed_threshold: std_dev_under_congestion * 100,
                'Min Percentage of Time Congested below %f m/s' % self.average_speed_threshold: min_under_congestion * 100,
                'Max Percentage of Time Congested below %f m/s' % self.average_speed_threshold: max_under_congestion * 100,
                'Mean Percentage of Time Congested below %f m/s_Scatter' % self.average_speed_threshold: \
                    {ii: self.percentage_under_congestion[ii] * 100.0 for ii in range(len(self.percentage_under_congestion))}}
        '''


