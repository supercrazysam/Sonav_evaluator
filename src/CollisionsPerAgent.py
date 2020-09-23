import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import os
import getpy as gp


class CollisionsPerAgent(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.collision_radius = 0.0 
        key_type = np.dtype('i8')
        value_type = np.dtype('i8')
        self.collisions_per_agent = gp.Dict(key_type, value_type, default_value=0)
        self.agent_radius = 0.2 #0.2 normal  0.1 for UNIV crowded
        self.total_collisions = 0
        self.agents = dict()
        self.num_agents_buffer = []
        self.time_step_done = False
        self.time_step = 0

    def evaluate(self, agents, curr_agents, time_step):
        #self.update_agents(agents)
        self.vectorized_agent_collisions(curr_agents)
        self.num_agents_buffer.append(len(curr_agents.keys()))
        if len(curr_agents.keys()) > 70:
            print("MAny agents!")
            print(curr_agents)
            print(time_step)
        # for agent in sorted(list(self.agents)):
        #         self.check_collisions(self.agents[agent], time_step)
        self.time_step = time_step

    def vectorized_agent_collisions(self, curr_agents_dict):
        curr_agents_locations = np.asmatrix([curr_agents_dict[agent].state for agent in curr_agents_dict])
        num_new_agents = curr_agents_locations.shape[0]
        focus_agent_tile = np.tile(curr_agents_locations, (num_new_agents, 1))
        other_agents_repeat = np.repeat(curr_agents_locations, num_new_agents, axis=0)
        distances = np.linalg.norm(focus_agent_tile - other_agents_repeat, axis=1).reshape(num_new_agents, num_new_agents)\
                    - 2*self.agent_radius
        np.fill_diagonal(distances, -1.0)
        for ii, curr_agent in enumerate(curr_agents_dict):
            curr_agents_dict[curr_agent].update_agent_proximity(distances[ii, :])
        collision_matrix = ((0.0 <= distances) & (distances < self.collision_radius)).astype(float)

        num_collisions = np.sum(collision_matrix, axis=1)
        agents = np.asarray(list(curr_agents_dict.keys()))
        self.collisions_per_agent[agents] += num_collisions.astype(int)

    def update_agents(self, agents):
        self.agents = agents

    def check_collision_time(self, agent_1, agent_2):
        return agent_1.get_last_updated_time_step() == agent_2.get_last_updated_time_step()

    def check_collision_radius(self, agent_1, agent_2):
        dist = np.linalg.norm(np.asmatrix(agent_1.state) - np.asmatrix(agent_2.state))
        return dist <= self.collision_radius, dist

    def write_out(self, agents):
        collision_list = np.asarray(list(self.collisions_per_agent.values()))
        total_agents = collision_list.size
        self.total_collisions = np.sum(collision_list) / 2
        return {'Total Collisions@%f m' % self.collision_radius: self.total_collisions,
                'Mean Collisions@%f m' % self.collision_radius: float(self.total_collisions) / total_agents,
  
                #'StdDev Collisions@%f m' % self.collision_radius: np.std(collision_list),
                #'Max Collisions@%f m' % self.collision_radius: np.max(collision_list),
                #'Min Collisions@%f m' % self.collision_radius: np.min(collision_list),
                #'Mean Collisions@%f m_Scatter' % self.collision_radius: self.collisions_per_agent,
                
                'Mean Num Agents per Timestep': np.mean(self.num_agents_buffer),
                'StdDev Num Agents per Timestep': np.std(self.num_agents_buffer) }
                #'Min Num Agents per Timestep': np.min(self.num_agents_buffer),
                #'Max Num Agents per Timestep': np.max(self.num_agents_buffer),
                #'Mean Num Agents Batch': total_agents,
                #'StdDev Num Agents Batch': total_agents,
                #'Min Num Agents Batch': total_agents,
                #'Max Num Agents Batch': total_agents}

