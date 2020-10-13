import pandas as pd
import numpy as np
import os

# holds all the different formats for datasets. Capture_freq is 1/fps
format_dict = {
    'ETH': {'format': ['time_step', 'agent_id', 'x', 'y'], 'delimiter': '\t', 'data_type': [float, float, float, float],
            'state_space': ['x', 'y'], 'data_type_scale': [1.0/2.5/10.0, 1, 1, 1], 'capture_freq': .4},
    'SIM': {'format': ['time_step', 'agent_id', 'x', 'y'], 'delimiter': '\t', 'data_type': [float, float, float, float],
            'state_space': ['x', 'y'], 'data_type_scale': [1.0/10.0/10.0, 1, 1, 1], 'capture_freq': .1}
}


class DataLoader():
    def __init__(self, path, data_format):
        if(os.path.exists(path)):
            self.path = path
        else:
            print("File " + path + " does not exist!")
            return
        self.data_format = format_dict[data_format]
        self.data_types = self.data_format['data_type']
        self.format_order = self.data_format['format']
        self.delimiter = self.data_format['delimiter']
        self.load_data_into_df()
        self.time_step = -1
        self.done = False
        self.total_time_steps = self.get_total_time_steps()

    def get_total_time_steps(self):
        return self.df.shape[0]


    def load_data_into_df(self):
        f = open(self.path, "r")
        data_dict = {key:[] for key in self.format_order}
        for line in f.readlines():
            for ii, data_value in enumerate(line.split(self.delimiter)):
                data_dict[self.format_order[ii]].append(self.data_types[ii](data_value))
        self.df = pd.DataFrame(data_dict, columns = data_dict.keys())

    def step_data(self):
        self.time_step += 1
        if self.time_step >= self.df.shape[0] - 1:
            self.done = True
        return self.df.iloc[self.time_step, :]



def main():
    dloader = DataLoader("../data/eth/train/biwi_hotel_train.txt", 'ETH')

if __name__ == "__main__":
    main()
    print(1)
