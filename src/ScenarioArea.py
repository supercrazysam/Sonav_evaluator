import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import os

#computes the closest proximity for the entire episode
class ScenarioArea(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.agents = dict()
        self.time_step_done = False
        self.min_x = float('Inf')
        self.max_x = -float('Inf')
        self.min_y = float('Inf')
        self.max_y = -float('Inf')

    def update_agents(self, agents):
        self.agents = agents

    def evaluate(self, agents, curr_agents, time_stamp):
        for agent_id in curr_agents:
            state = curr_agents[agent_id].state
            x = state[0]
            y = state[1]
            if x < self.min_x:
                self.min_x = x
            if x > self.max_x:
                self.max_x = x
            if y < self.min_y:
                self.min_y = y
            if y > self.max_y:
                self.max_y = y
        return

    def write_out(self, agents):
        return {#'Min X (m)': self.min_x,
                #'Max X (m)': self.max_x,
                #'Min Y (m)': self.min_y,
                #'Max Y (m)': self.max_y,
                'Mean X (m)': self.min_x, # will get averaged across all data files
                'Mean X (m)': self.max_x,
                'Mean Y (m)': self.min_y,
                'Mean Y (m)': self.max_y,
                'StdDev X (m)': self.min_x,
                'StdDev X (m)': self.max_x,
                'StdDev Y (m)': self.min_y,
                'StdDev Y (m)': self.max_y,}
