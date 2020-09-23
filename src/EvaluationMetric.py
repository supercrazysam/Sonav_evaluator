import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xmltodict
from DataLoader import *
import os


class EvaluationMetric():
    def __init__(self, data_format):
        self.data_format = format_dict[data_format]['format']
        self.state_space = format_dict[data_format]['state_space']


    def evaluate(self, agents, curr_agents, time_stamp):

        raise NotImplementedError


    def format_data(self):

        raise NotImplementedError

    def write_out(self):

        raise NotImplementedError