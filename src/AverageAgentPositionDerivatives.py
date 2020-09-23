import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import xmltodict
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import os


class AverageAgentPositionDerivatives(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.nominal_velocity = .5
        self.agents = dict()
        self.time_step_done = False
        self.average_speeds_per_agent = dict()
        self.average_agent_speed = 0.0

    def update_agents(self, agents):
        self.agents = agents

    def evaluate(self, agents, curr_agents, time_stamp):
        self.update_agents(agents)

    def get_averaged_agent_speed(self):
        self.velocities = [self.agents[agent_id].average_velocity for agent_id in self.agents]
        self.accelerations = [self.agents[agent_id].average_acceleration for agent_id in self.agents]
        self.jerks = [self.agents[agent_id].average_jerk for agent_id in self.agents]
        self.energies = [self.agents[agent_id].get_average_energy() for agent_id in self.agents]

        self.average_agent_velocity = np.mean(self.velocities)
        self.std_dev_agent_velocity = np.std(self.velocities)
        self.minimum_agent_velocity = np.min(self.velocities)
        self.maximum_agent_velocity = np.max(self.velocities)

        self.average_agent_acceleration = np.mean(self.accelerations)
        self.std_dev_agent_acceleration = np.std(self.accelerations)
        self.minimum_agent_acceleration = np.min(self.accelerations)
        self.maximum_agent_acceleration = np.max(self.accelerations)

        self.average_agent_jerk = np.mean(self.jerks)
        self.std_dev_agent_jerk = np.std(self.jerks)
        self.minimum_agent_jerk = np.min(self.jerks)
        self.maximum_agent_jerk = np.max(self.jerks)

        self.average_agent_energy = np.mean(self.energies)
        self.std_dev_agent_energy = np.std(self.energies)
        self.minimum_agent_energy = np.min(self.energies)
        self.maximum_agent_energy = np.max(self.energies)

    #evaluation occurs at the end
    def write_out(self, agents):
        self.get_averaged_agent_speed()
        d1 = {ii: self.velocities[ii] for ii in range(len(self.velocities))}
        d2 = {ii: self.accelerations[ii] for ii in range(len(self.accelerations))}
        d3 = {ii: self.jerks[ii] for ii in range(len(self.jerks))}
        d4 = {ii: self.energies[ii] for ii in range(len(self.energies))}
        return {'Mean Agent velocity (m/s)': self.average_agent_velocity,
               'StdDev Agent velocity (m/s)': self.std_dev_agent_velocity,
               'Min Agent velocity (m/s)': self.minimum_agent_velocity,
                'Max Agent velocity (m/s)': self.maximum_agent_velocity,
                #'Mean Agent velocity (m/s)_Scatter': d1,

                'Mean Agent Accel (m/s^2)': self.average_agent_acceleration,
                'StdDev Agent Accel (m/s^2)': self.std_dev_agent_acceleration,
                'Min Agent Accel (m/s^2)': self.minimum_agent_acceleration,
                'Max Agent Accel (m/s^2)': self.maximum_agent_acceleration,
                #'Mean Agent Accel (m/s^2)_Scatter': d2,

                'Mean Agent Jerk (m/s^3)': self.average_agent_jerk,
                'StdDev Agent Jerk (m/s^3)': self.std_dev_agent_jerk,
                'Min Agent Jerk (m/s^3)': self.minimum_agent_jerk,
                'Max Agent Jerk (m/s^3)': self.maximum_agent_jerk,
                #,'Mean Agent Jerk (m/s^3)_Scatter': d3,

                'Mean Agent Energy (m^2/s^2)': self.average_agent_energy,
                'StdDev Agent Energy (m^2/s^2)': self.std_dev_agent_energy,
                'Min Agent Energy (m^2/s^2)': self.minimum_agent_energy,
                'Max Agent Energy (m^2/s^2)': self.maximum_agent_energy,
                #'Mean Agent Energy (m^2/s^2)_Scatter': d4
                }


