import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
import os
import copy
import time
from tqdm import tqdm
from DataLoader import *
from Agent import Agent
from CollisionsPerAgent import CollisionsPerAgent
from AverageAgentPositionDerivatives import AverageAgentPositionDerivatives
from AgentClosestProximity import AgentClosestProximity
from ExtraTimeToGoal import ExtraTimeToGoal
from PathIrregularity import PathIrregularity
from SocialScore import SocialScore
from ExtraTimeToGoal import ExtraTimeToGoal
from CongestionScore import CongestionScore
from PopulationDensity import PopulationDensity
from SimilarityScores import SimilarityScores
from PersonalSpace import PersonalSpace
from ScenarioArea import ScenarioArea

# holds all the classes for different evaluation metrics
evaluation_metrics_dict = {
    'CollisionsPerAgent': CollisionsPerAgent,
    'ExtraTimeToGoal': ExtraTimeToGoal,
    'AgentClosestProximity': AgentClosestProximity,
    'PathIrregularity': PathIrregularity,
    'AverageAgentPositionDerivatives': AverageAgentPositionDerivatives,
    'SocialScore': SocialScore,
    'CongestionScore': CongestionScore,
    'PopulationDensity': PopulationDensity,
    'SimilarityScores': SimilarityScores,
    'PersonalSpace': PersonalSpace,
    'ScenarioArea': ScenarioArea
}

# how many timesteps from the beginning to ignore
ignored_time_steps = 10

def create_data_dict_from_directory(dir, recursive=False):
    data_dict = {}
    entries = ['path', 'data_format', 'metrics']
    files = os.listdir(dir)
    for filename in files:
        path = os.path.join(dir, filename)
        if os.path.isdir(path) and recursive:
            data_dict = {**data_dict, **create_data_dict_from_directory(path)}
        if filename.endswith(".txt"):
            job_name = filename.split(".")[0]
            data_format = 'SIM' #ETH
            metrics = ['PopulationDensity',
                       'CollisionsPerAgent',
                       'AverageAgentPositionDerivatives',
                       'ExtraTimeToGoal',
                       'AgentClosestProximity',
                       'PathIrregularity',
                       #'SocialScore',
                       'CongestionScore'] #,
                       #'SimilarityScores',#
                       #'PersonalSpace',#
                       #'ScenarioArea']
            data_dict[job_name] = dict(zip(entries, [path, data_format, metrics]))
    return data_dict

def create_batch_data_dict(batch_data_dict, search_directory):
    for file in os.listdir(search_directory):
        batch_data_dict[file] = create_data_dict_from_directory(os.path.join(search_directory, file))

# Each folder within self.central_data_path is considered a batch
# Each file within each batch will have .csv files generated to compare them
#   specify batch_search_directory and central_data_path
class Evaluator():
    def __init__(self, name):
        self.name = name
        self.agents = dict()
        #self.central_data_path = "../results"
        self.central_data_path = self.batch_search_directory = "datasets/processed_20_population_density/0.3/CADRL"#"datasets/processed_20_full_table/ZARA2/RAW"
        self.batch_data_dict = dict()
        create_batch_data_dict(self.batch_data_dict, self.batch_search_directory)
        self.total_batches = len(self.batch_data_dict.keys())
        self.evaluation_metrics = dict()
        self.populate_evaluation_metrics()
        self.df_dict = dict()

    def populate_evaluation_metrics(self):
        for batch in self.batch_data_dict:
            self.evaluation_metrics[batch] = []
            all_paths = copy.deepcopy(self.batch_data_dict[batch])
            for job in all_paths:
                if not os.path.isfile(self.batch_data_dict[batch][job]['path']):
                    print("Invalid path for: %s", self.data_dict[batch][job]['path'])
                    del self.batch_data_dict[batch][job]
                    continue
                job_metrics = []
                data_format = self.batch_data_dict[batch][job]['data_format']
                for metric in self.batch_data_dict[batch][job]['metrics']:
                    new_metric = evaluation_metrics_dict[metric]
                    job_metrics.append(new_metric(data_format))
                self.evaluation_metrics[batch].append(job_metrics)

    def parse_into_state(self, data, data_format):
        switch = {
            'ETH': data * format_dict[data_format]['data_type_scale'],
            'SIM': data * format_dict[data_format]['data_type_scale']
        }
        return switch.get(data_format, "Data format not recognized")

    def evaluate_data(self):
        for ii, batch in enumerate(self.batch_data_dict):
            print("[Batch %s]: %d / %d" % (batch, ii + 1, self.total_batches))
            data_dict = self.batch_data_dict[batch]
            for job_id, job in enumerate(data_dict):
                self.agents = {}
                path, format = data_dict[job]['path'], data_dict[job]['data_format']
                data_loader = DataLoader(path, format)
                data_scale = format_dict[format]['data_type_scale']
                capture_freq = format_dict[format]['capture_freq']
                prev_time = 0
                curr_time = 0
                time_frame_done = False
                normalized_time_step = 0
                curr_agents = {}
                for _ in tqdm(range(data_loader.total_time_steps + 1)):
                    if data_loader.done:
                        for evaluation_metric in self.evaluation_metrics[batch][job_id]:
                            evaluation_metric.evaluate(self.agents, curr_agents, prev_time)
                        break
                    data = data_loader.step_data()
                    time_step, agent_id, x, y = self.parse_into_state(data, format)
                    curr_time = time_step
                    if((curr_time != prev_time and _ != 0) or _ == data_loader.total_time_steps):
                        if normalized_time_step > ignored_time_steps:
                            for evaluation_metric in self.evaluation_metrics[batch][job_id]:
                                evaluation_metric.evaluate(self.agents, curr_agents, prev_time)
                        curr_agents = {}
                        normalized_time_step += 1
                    if agent_id not in self.agents:
                        self.agents[agent_id] = Agent(agent_id, [x, y], curr_time, capture_freq)
                    else:
                        self.agents[agent_id].update_state([x, y], curr_time)
                    curr_agents[agent_id] = self.agents[agent_id]
                    prev_time = curr_time
                self.write_stats(job_id, job, batch)
                self.write_batch_stats(batch)
            self.df_dict = {}

    def add_agent(self, agent):
        return

    #will write useful averages for each batch
    def write_batch_stats(self, batch):
        batch_pd = pd.read_csv(os.path.join(self.central_data_path, batch) + ".csv")
        stat_values = []
        n = len(batch_pd.iloc[:,0].to_numpy())
        for col in list(batch_pd):
            values = batch_pd.loc[:, col].to_numpy()
            if "scatter" in col.lower():
                stat_values.append({0: 0})
            elif "max" in col.lower():
                stat_values.append(np.max(values))
            elif "min" in col.lower():
                stat_values.append(np.min(values))
            elif "mean" in col.lower() or "total" in col.lower():
                stat_values.append(np.mean(values))
            elif "job" in col.lower():
                stat_values.append("Batch Statistics")
            elif "stddev" in col.lower():
                stat_values.append(np.sqrt(np.sum(np.square(values)))/float(n))
            else:
                stat_values.append(-1)
        stat_value_dict = dict(zip(list(batch_pd), stat_values))
        batch_pd = batch_pd.append(stat_value_dict, ignore_index=True)
        batch_pd.to_csv(os.path.join(self.central_data_path, batch) + ".csv", index=False)

    def write_stats(self, job_id, job, batch):
        if 'job' not in self.df_dict:
            self.df_dict['job'] = []
        path = self.batch_data_dict[batch][job]['path']
        self.df_dict['job'].append(job)
        if 'Dataset Path' not in self.df_dict:
            self.df_dict['Dataset Path'] = [path]
        else:
            self.df_dict['Dataset Path'].append(path)
        folder_path = path[::-1][path[::-1].index('/'):][::-1]
        if not os.path.isdir(self.central_data_path):
            os.makedirs(self.central_data_path)
        for evaluation_metric in self.evaluation_metrics[batch][job_id]:
            eval_dict = evaluation_metric.write_out(self.agents)
            for metric in eval_dict:
                if metric not in self.df_dict:
                    self.df_dict[metric] = [eval_dict[metric]]
                else:
                    self.df_dict[metric].append(eval_dict[metric])
        job_data = pd.DataFrame.from_dict(self.df_dict)
        job_data.to_csv(os.path.join(self.central_data_path, batch) + ".csv", index=False)

    def create_data_dict_from_xml(self, data_path):
        with open(data_path) as f:
            data_dict = xmltodict.parse(f.read())
        print(data_dict)

def main():
    evaluator = Evaluator('Eval')
    evaluator.evaluate_data()

if __name__ == "__main__":
    main()


