import pandas as pd
import numpy as np
import similaritymeasures
import xml.etree.ElementTree as ET
import xmltodict
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import os
import getpy as gp

scores_dict = {
    'Frechet Distance (m)': similaritymeasures.frechet_dist,
    'Dynamic Time Warping': similaritymeasures.dtw
}

class SimilarityScores(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.similarity_score_dict = dict()
        self.agents = dict()
        self.scores_to_measure = ['Frechet Distance (m)', 'Dynamic Time Warping']
        self.time_scaling_factor = .4


    def evaluate(self, agents, curr_agents, time_step):
        self.time_step = time_step

    def update_agents(self, agents):
        self.agents = agents

    def update_similarity_scores(self, agents):
        for score in self.scores_to_measure:
            self.similarity_score_dict[score] = {}
            for agent in agents:
                motion_history = agents[agent].motion_history
                traj = motion_history
                if np.linalg.norm(agents[agent].total_displacement) > 0.0:
                    #change .4 to framerate here
                    num_time_steps = np.ceil(np.linalg.norm(np.asmatrix(motion_history[0]) - np.asmatrix(motion_history[-1]))/agents[agent].average_speed/self.time_scaling_factor)
                    straight_line_traj = np.linspace(motion_history[0], motion_history[-1], int(num_time_steps))
                    if score == 'Dynamic Time Warping':
                        self.similarity_score_dict[score][agent], _ = scores_dict[score](traj, straight_line_traj)
                    else:
                        self.similarity_score_dict[score][agent] = scores_dict[score](traj, straight_line_traj)

    def write_out(self, agents):
        self.update_similarity_scores(agents)
        results_dict = {}
        for score in self.scores_to_measure:
            mean_scatter = self.similarity_score_dict[score]
            score_list = list(mean_scatter.values())
            mean_score = np.mean(score_list)
            std_dev_score = np.std(score_list)
            min_score = np.min(score_list)
            max_score = np.max(score_list)
            results_dict = {**results_dict, **{'Mean %s' % score: mean_score,
                                             'StdDev %s' % score: std_dev_score,
                                             'Min %s' % score: min_score,
                                             'Max %s' % score: max_score,
                                             'Mean %s_Scatter' % score: mean_scatter}}
        return results_dict

