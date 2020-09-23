import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
from EvaluationMetric import EvaluationMetric
from DataLoader import *
import os
import getpy as gp


# noinspection PyInterpreter
class SocialScore(EvaluationMetric):
    def __init__(self, data_format):
        super().__init__(data_format)
        self.nominal_speed = .5
        self.social_score_proximity_threshold_list = [0.2, 0.3, 0.5]
        key_type = np.dtype('i8')
        value_type = np.dtype('f8')
        self.social_score_dict = {}

        self.update_iteration_count_dict = {} 
        for thresh in self.social_score_proximity_threshold_list:
            self.social_score_dict[thresh] = gp.Dict(key_type, value_type, default_value=np.asarray(0.0).astype('f8'))
            
            self.update_iteration_count_dict[thresh] = gp.Dict(key_type, value_type, default_value=np.asarray(0.0).astype('f8'))
        self.time_step_done = False
        self.min_social_score_agent = 0
        self.min_social_score = [float('Inf') for _ in range(len(self.social_score_proximity_threshold_list))]
        self.social_score_collision_threshold = 0.0
        self.agent_radius = 0.2 #needs to be the same as in collision

        

    def set_social_scores(self, agent_id):
        agent = self.agents[agent_id]
        for ii, threshold in enumerate(self.social_score_proximity_threshold_list):
            if threshold not in agent.social_score_dict:
                agent.social_score_dict[threshold] = 0
            if agent.social_score_dict[threshold] < self.min_social_score[ii]:
                self.min_social_score[ii] = agent.social_score_dict[threshold]

    def update_agents(self, agents):
        self.agents = agents


    def evaluate(self, agents, curr_agents, time_step):
        self.vectorized_social_score_update(curr_agents)
        self.time_step = time_step

    def vectorized_social_score_update(self, curr_agents_dict):
##        for thresh in self.social_score_proximity_threshold_list:
##            self.update_iteration_count_dict[thresh][np.asarray(list(curr_agents_dict.keys())).astype('i8')] +=1
        
        for thresh in self.social_score_proximity_threshold_list:
            curr_agents_locations = np.asmatrix([curr_agents_dict[agent].state for agent in curr_agents_dict])
            num_new_agents = curr_agents_locations.shape[0]
            focus_agent_tile = np.tile(curr_agents_locations, (num_new_agents, 1))
            other_agents_repeat = np.repeat(curr_agents_locations, num_new_agents, axis=0)
            distances = np.linalg.norm(focus_agent_tile - other_agents_repeat, axis=1).reshape(num_new_agents, num_new_agents)\
                        - 2*self.agent_radius
            np.fill_diagonal(distances, -1.0)
            

            threshold_matrix = ((self.social_score_collision_threshold < distances) & (distances < thresh)).astype(float)
            
            threshold_encroachments = np.min(np.multiply(threshold_matrix, -thresh/2.0 + distances/2.0), axis=1) #bitmask

            for agent_index in range(len(threshold_encroachments)):
                #if social score is zero, dont count it during averaging, otherwise will affect everything.
                if threshold_encroachments[ agent_index ] != 0:
                    self.update_iteration_count_dict[thresh][ agent_index ] += 1
##                    print("1"*30)
##                    print(list(self.update_iteration_count_dict[thresh].values()))

                #if it is first time executing, there will be 0 in iteration count dict, force set them to 1, otherwise divide will cause error
                if self.update_iteration_count_dict[thresh][ agent_index ]  == 0:
                    self.update_iteration_count_dict[thresh][ agent_index ] = 1
##                    print("2"*30)
##                    print(list(self.update_iteration_count_dict[thresh].values()))

            total = (threshold_encroachments + self.social_score_dict[thresh][np.asarray(list(curr_agents_dict.keys())).astype('i8')] *  (self.update_iteration_count_dict[thresh][np.asarray(list(curr_agents_dict.keys())).astype('i8')] - 1 )   )
##            print("total")
##            print(total)

            #self.social_score_dict[thresh][np.asarray(list(curr_agents_dict.keys())).astype('i8')] = total  /  self.update_iteration_count_dict[thresh][np.asarray(list(curr_agents_dict.keys())).astype('i8')]

            for agent_index in range(len(total)):
                self.social_score_dict[thresh][agent_index] = total[agent_index]  /  self.update_iteration_count_dict[thresh][agent_index]

##            print("self.update_iteration_count_dict[thresh] ",thresh)
##            print(list(self.update_iteration_count_dict[thresh].values()))
##            
##            print("self.social_score_dict[thresh] ",thresh)
##            print(list(self.social_score_dict[thresh].values()))

            #print("self.update_iteration_count_dict[thresh][np.asarray(list(curr_agents_dict.keys())).astype('i8')]")
            #print(self.update_iteration_count_dict[thresh][np.asarray(list(curr_agents_dict.keys())).astype('i8')])

    #this evaluation metric has extra computation to perform at the end of the episode
    def write_out(self, agents):
        output_dict = {}
        for threshold in self.social_score_proximity_threshold_list:
            social_score_list = list(self.social_score_dict[threshold].values())
            self.average_social_score = np.mean(social_score_list)
            self.std_dev_social_score = np.std(social_score_list)
            worst_social_score_ind = np.argmin(social_score_list)
            self.min_social_score = social_score_list[worst_social_score_ind]
            self.max_social_score = np.max(social_score_list)
            output_dict = {**output_dict, **{'Mean Social Score@%s Threshold' % threshold : self.average_social_score,
                    'StdDev Social Score@%s Threshold' % threshold: self.std_dev_social_score,
                    'Min Social Score@%s Threshold' % threshold: self.min_social_score,
                    'Max Social Score@%s Threshold' % threshold: self.max_social_score,
                    'Mean Social Score@%s Threshold_Scatter' % threshold: self.social_score_dict[threshold]}}
        return output_dict
