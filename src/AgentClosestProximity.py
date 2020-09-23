import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import os

#computes the closest proximity for the entire episode
class AgentClosestProximity(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.agents = dict()
        self.time_step_done = False
        self.closest_proximity_per_agent = dict()
        self.furthest_proximity_per_agent = dict()
        self.average_proximity = 0

    def update_agents(self, agents):
        self.agents = agents

    def evaluate(self, agents, curr_agents, time_stamp):
        #self.update_agents(agents)
        return

    def get_closest_proximity_to_agent(self):
        for agent_id in self.agents:
            if agent_id not in self.closest_proximity_per_agent:
                self.closest_proximity_per_agent[agent_id] = self.agents[agent_id].get_closest_distance()
                self.furthest_proximity_per_agent[agent_id] = self.agents[agent_id].get_furthest_distance()
                #no other agents were nearby for any timestep the agent was present
                if self.closest_proximity_per_agent[agent_id] == float('Inf'):
                    del self.closest_proximity_per_agent[agent_id]
        min_proximities = [self.closest_proximity_per_agent[agent_id] for agent_id in self.closest_proximity_per_agent]
        max_proximities = [self.furthest_proximity_per_agent[agent_id] for agent_id in self.furthest_proximity_per_agent]
        agents = list(self.closest_proximity_per_agent)

        self.average_min_proximity = np.mean(min_proximities)
        self.std_dev_min_proximity = np.std(min_proximities)
        closest_proximity_ind = np.argmin(min_proximities)
        self.closest_proximity = min_proximities[closest_proximity_ind]
        self.closest_proximity_ind = agents[closest_proximity_ind]

        self.average_max_proximity = np.mean(max_proximities)
        self.std_dev_max_proximity = np.std(max_proximities)
        furthest_proximity_ind = np.argmax(max_proximities)
        self.furthest_proximity = max_proximities[furthest_proximity_ind]
        self.furthest_proximity_ind = agents[furthest_proximity_ind]



    def write_out(self, agents):
        self.update_agents(agents)
        self.get_closest_proximity_to_agent()
        return {'Mean Closest Proximity (m)': self.average_min_proximity,
                'StdDev Closest Proximity (m)': self.std_dev_min_proximity} #

        ''',
                'StdDev Closest Proximity (m)': self.std_dev_min_proximity,
                'Closest Proximity (m)': self.closest_proximity,
                'Closest Proximity Agent_ID': self.closest_proximity_ind,
                'Mean Closest Proximity (m)_Scatter': self.closest_proximity_per_agent,
                'Mean Furthest Proximity (m)': self.average_max_proximity,
                'StdDev Furthest Proximity (m)': self.std_dev_max_proximity,
                'Furthest Proximity (m)': self.furthest_proximity,
                'Furthest Proximity Agent_ID': self.furthest_proximity_ind,
                'Mean Furthest Proximity (m)_Scatter': self.furthest_proximity_per_agent}
        '''
                
            # "[Average Closest Proximity]"  + "\n" + str(self.average_proximity) + " m " + "StdDev: " + str(self.std_dev_proximity) + " m " + "\n" + \
            #   "[Closest Proximity]" + "\n" + str(self.closest_proximity) + "\n" + "[Closest Proximity Agent_ID]" + "\n" + str(self.closet_proximity_ind) + "\n" +\
            #    "[Closest Proximity Per Agent]" + "\n" + str(self.closest_proximity_per_agent) +"\n"
