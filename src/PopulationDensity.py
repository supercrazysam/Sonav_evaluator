import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import xmltodict
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import os


class PopulationDensity(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.nominal_velocity = .5
        self.agents = dict()
        self.time_step_done = False
        self.average_density_per_timestep = []
        self.density_window = 2.0


    def evaluate(self, agents, curr_agents, time_stamp):
        if len(curr_agents.values()) == 0:
            self.average_density_per_timestep.append(0.0)
            return
        self.record_agent_density(curr_agents)

    def record_agent_density(self, curr_agents):
        all_states = np.asarray([curr_agents[agent].state for agent in curr_agents])
        x = all_states[:, 0].reshape(len(curr_agents.keys()))
        y = all_states[:, 1].reshape(len(curr_agents.keys()))
        mins = np.min(all_states, axis=0) - 1.5*self.density_window
        maxs = np.max(all_states, axis=0) + 1.5*self.density_window
        x_bins = np.arange(mins[0], maxs[0], self.density_window)
        y_bins = np.arange(mins[1], maxs[1], self.density_window)
        responses = np.asmatrix(plt.hist2d(list(x), list(y), [x_bins, y_bins])[0])
        responses[np.isnan(responses)] = 0
        self.average_density_per_timestep.append(np.mean(responses[responses > 0])/(self.density_window**2))

    #evaluation occurs at the end
    def write_out(self, agents):
        self.average_population_density = np.mean(self.average_density_per_timestep)
        self.std_dev_population_density = np.std(self.average_density_per_timestep)
        self.minimum_population_density = np.min(self.average_density_per_timestep)
        self.maximum_population_density = np.max(self.average_density_per_timestep)
        d = self.average_density_per_timestep
        return {'Mean Population Density (agents/m^2)': self.average_population_density,
                'StdDev Population Density (agents/m^2)': self.std_dev_population_density,
                'Min Population Density (agents/m^2)': self.minimum_population_density,
                'Max Population Density (agents/m^2)': self.maximum_population_density,
                'Mean Population Density (agents/m^2)_Scatter': \
                    dict(zip(range(len(self.average_density_per_timestep)), self.average_density_per_timestep))}


