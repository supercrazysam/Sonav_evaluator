import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
from scipy.spatial import *
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import matplotlib.pyplot as plt
import os
import getpy as gp


class PersonalSpace(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.personal_space_dict = {}
        self.num_agents_to_sample = 10

    def evaluate(self, agents, curr_agents, time_step):
        self.time_step = time_step
        self.update_personal_space(curr_agents)

    #use voronoi cell to calculate personal space of max(self.num_agents_to_sample, len(curr_agents))
    #only want to consider convex hull personal space "bubbles", no edge voronoi regions
    def update_personal_space(self, curr_agents):
        if len(curr_agents.keys()) >= 3:
            all_agents_positions = [curr_agents[id].state for id in curr_agents]
            num_agents_to_sample = np.clip(self.num_agents_to_sample, 1, len(curr_agents.keys()))
            vor = Voronoi(all_agents_positions, qhull_options="QbB Qc Qz")
            choices = np.random.choice(len(vor.regions), size=(num_agents_to_sample)).astype(int)
            vertex_choices = np.asarray(vor.regions)[choices]
            filtered_choices = []
            for choice in range(len(vertex_choices)):
                #print(vertex_choices[choice])
                if np.all(np.asarray(vertex_choices[choice]) >= 0) and len(vertex_choices[choice]) > 0:
                    filtered_choices.append(np.array(vertex_choices[choice]))
            if len(filtered_choices) > 0:
                sampled_region_vertices = [(vor.vertices)[filtered_choices[ii]] for ii in range(len(filtered_choices))]
                areas = list(map(lambda x: ConvexHull(x).area, sampled_region_vertices))
                self.personal_space_dict[self.time_step] = np.mean(areas)

    def update_agents(self, agents):
        self.agents = agents

    def write_out(self, agents):
        area_list = np.asarray(list(self.personal_space_dict.values()))
        average_personal_space = -1
        std_dev_personal_space = -1
        min_personal_space = -1
        max_personal_space = -1
        if len(area_list) > 0:
            average_personal_space = np.mean(area_list)
            std_dev_personal_space = np.std(area_list)
            min_personal_space = np.min(area_list)
            max_personal_space = np.max(area_list)
        return {'Mean Personal Space (m^2)': average_personal_space,
                'StdDev Personal Space (m^2)': std_dev_personal_space,
                'Min Personal Space (m^2)': min_personal_space,
                'Max Personal Space (m^2)': max_personal_space,
                'Mean Personal Space (m^2)_Scatter': self.personal_space_dict}

