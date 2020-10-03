import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import ast
from scipy.stats import norm
import operator

np.seterr(divide='ignore', invalid='ignore')
def autolabel(xs, ys):
    for x,y in zip(xs,ys):
        if type(y) == float:
            try:
                decimal_places = len(str(y).split(".")[1])
                #string = "{:."+str(decimal_places)+"f}"
                string = "{:.5f}"
                #print(string)
                label = string.format(y)
            except IndexError:
                label = "{:.3f}".format(y)
        else:
            label = y
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     fontsize=7, 
                     ha='center') # horizontal alignment can be left, right or center

class Visualizer():
    def __init__(self, all_data_path, output_path):
        self.all_data_path = all_data_path
        self.output_path = output_path
        if not os.path.isfile(self.all_data_path):
            print("All Data Path does not exist!")
            return
        self.all_data = pd.read_csv(self.all_data_path)

        self.scenario_name = all_data_path.split("/")[-1].split(".")[0]

    def visualize_all_jobs(self):
        for metric in self.all_data:
            print("metric")
            print(metric)
            if "_Scatter" not in metric and "_Time" not in metric:


                if ("job" in metric) or ("Max" in metric) or ("Min" in metric) or ("ID" in metric) or ("StdDev" in metric) or ("Num Agents Batch" in metric) or ("Furthest" in metric) or ("Dataset Path" in metric) or (("Mean" not in metric) and ("Closest Proximity" in metric) ): continue

                self.all_data.sort_values(metric, ascending=False, inplace=True)
                x_axis = self.all_data['job']
                y_axis = self.all_data[metric]

                std_metric = metric
                std_metric = std_metric.replace("Mean ","StdDev ")

                fig = plt.figure()
                plt.xticks(rotation=90)
                plt.title(self.scenario_name+"   "+metric)
                #plt.xlabel(metric)
                plt.xlabel("Population Density (agents / m^2)")
                plt.ylabel(metric)
                
                
                
                file_name = metric.replace(" ", "_")
                file_name = file_name.replace("/", "_per_")
                file_path = "%s" % (self.output_path)
                if not os.path.isdir(file_path):
                    os.makedirs(file_path)




                try:
                    print(x_axis)
                    print(y_axis)
                    
                    
                    if ("Collision" in metric) or ("Arrival" in metric) : raise TypeError

                    elif ("Population Density" in metric):         plt.ylim(0.2,0.8)
                    elif ("Num Agents per Timestep" in metric):    plt.ylim(0,70)
                    
                    elif ("Agent velocity" in metric):             plt.ylim(0.2,1.5)
                    elif ("Agent Accel" in metric):                plt.ylim(0,0.02) #plt.ylim(0,0.15)
                    elif ("Agent Jerk" in metric):                 plt.ylim(0,0.0075)#plt.ylim(0,0.02)
                    elif ("Agent Energy" in metric):               plt.ylim(0,1) #plt.ylim(0,7)
                    
                    elif ("Extra Time to Goal" in metric):         plt.ylim(0,5)
                    elif ("Dynamic Time Warping" in metric):       plt.ylim(0,100)
                    elif ("Time Congested below" in metric):       plt.ylim(0,100)
                    elif ("Average Speeds Slower than" in metric): plt.ylim(0,100,)
                    
                    elif ("Frechet Distance" in metric):           plt.ylim(0,2)
                    elif ("Path Irregularity" in metric):          plt.ylim(500,1300)#plt.ylim(0,10)
                    
                    elif ("Path Efficiency" in metric):
                        plt.ylim(0.60,1.05)
                        plt.yticks(np.arange(0.6,1.05, 0.05))
                    elif ("Time Efficiency" in metric):            plt.ylim(0.60,1)

                    elif ("Personal Space" in metric):             plt.ylim(0,50)
                    elif ("Closest Proximity" in metric):          plt.ylim(0,1) #plt.ylim(0,50)


                    
                    elif ("Social Score" in metric) and ("0.2" in metric):               plt.ylim(-0.01,0)
                    elif ("Social Score" in metric) and ("0.2" in metric):               plt.ylim(-0.01,0)
                    elif ("Social Score" in metric) and ("0.2" in metric):               plt.ylim(-0.01,0)
                    elif ("X" in metric):                          plt.ylim(0,20)
                    elif ("Y" in metric):                          plt.ylim(0,20)
                    else:
                        plt.ylim(-200,200,1000)

                    std    =  self.all_data[std_metric]

                    x_axis_normal_graph = x_axis.to_numpy()
                    y_axis_normal_graph = y_axis.to_numpy()
                    std_normal_graph    = std.to_numpy()
                    #std_normal_graph[std_normal_graph ==0 ] = 0.1 #eplison

                    print("X"*20)
                    print(x_axis_normal_graph)
                    print(y_axis_normal_graph)
                    print(std_normal_graph)

                    #zip list together and sort by algorithm name
                    zipped_list = sorted(list(zip(x_axis_normal_graph,y_axis_normal_graph,std_normal_graph)), key = operator.itemgetter(0))

                    '''Bar chart with standard deviation I bar '''
                    for algorithm_name ,data_y,data_std in zipped_list:
                        ###################################################
                        ###################################################

                        if "Percentage" in metric: temp_graph_plot_min       =  0

                        plt.xlim(0,0.65) #actual range to show on graph, adjust to fit legend
                        plt.xticks(np.arange(0.1,0.55, 0.05))  #show x tick number
                        bar = plt.bar(float( algorithm_name.split("=")[1])  , data_y , alpha=0.2,width=0.05)
                        plt.errorbar( float( algorithm_name.split("=")[1])  , data_y , yerr=data_std, alpha=0.5 , label=algorithm_name, capsize=10)
                        
       
                    plt.legend(loc="upper right", prop={'size': 6}, framealpha=0.5)
                    #autolabel(x_axis, y_axis)



                except:
                    print(sys.exc_info())
                    
                    x_axis_normal_graph = x_axis.to_numpy()
                    y_axis_normal_graph = y_axis.to_numpy()

                    if    ("Arrival Rate" in metric):
                        plt.ylim(0.7,1.2) #actual range to show on graph, adjust to fit legend
                        
                    elif  ("Collision Rate" in metric):
                        plt.ylim(0,0.1) #actual range to show on graph, adjust to fit legend

                    elif ("Congestion" in metric):
                        plt.ylim(0,0.1)
                        plt.yticks(np.arange(0,0.15, 0.05))
                        y_axis_normal_graph/=100


                    #std_normal_graph[std_normal_graph ==0 ] = 0.1 #eplison

                    print("X"*20)
                    print(x_axis_normal_graph)
                    print(y_axis_normal_graph)

                    #zip list together and sort by algorithm name
                    zipped_list = sorted(list(zip(x_axis_normal_graph,y_axis_normal_graph)), key = operator.itemgetter(0))

                    '''Bar chart with standard deviation I bar '''
                    for algorithm_name ,data_y in zipped_list:
                        ###################################################
                        ###################################################

                        if "Percentage" in metric: temp_graph_plot_min       =  0

                        plt.xlim(0,0.65) #actual range to show on graph, adjust to fit legend
                        plt.xticks(np.arange(0.1,0.55, 0.05))  #show x tick number
                        print("K"*30)
                        print(algorithm_name.split("=")[1])
                        bar = plt.bar(float( algorithm_name.split("=")[1]) , data_y , alpha=0.2,width=0.05 , label=algorithm_name)   
       
                    plt.legend(loc="upper right", prop={'size': 6}, framealpha=0.5)
                    print("XXXX!@$@$E")
                    print(x_axis)
                    print(y_axis)
                    #autolabel(x_axis, y_axis)
                    
                
                
                name = file_path + "/%s%s" % (file_name, ".png")
                fig.savefig(name, bbox_inches="tight")
                plt.close(fig)

def main():
    #specify source and destination directories for csv batch files
##    data_dict = {"../results/test.csv":"../results/figures/test",
##                 "../results/train.csv":"../results/figures/train",
##                 "../results/val.csv":"../results/figures/val"}
    
##    data_dict = {"datasets/processed_20_full_table/ETH/GA3CCADRL/test.csv":"results/ETH/GA3CCADRL",
##                 "datasets/processed_20_full_table/ETH/CADRL/test.csv"    :"results/ETH/CADRL",
##                 "datasets/processed_20_full_table/ETH/RVO/test.csv"      :"results/ETH/EVO" }


#######WAS USING THIS
##    data_dict = {"extracted/ETH.csv"       :"results/ETH",
##                 "extracted/HOTEL.csv"     :"results/HOTEL",
##                 "extracted/UNIV.csv"      :"results/UNIV", 
##                 "extracted/ZARA1.csv"     :"results/ZARA1",
##                 "extracted/ZARA2.csv"     :"results/ZARA2",
##                 "extracted/ALL.csv"     :"results/ALL"   }

    data_dict = {"extracted_population/GA3CCADRL.csv"       :"results/population/density/GA3CCADRL",
                 "extracted_population/CADRL.csv"           :"results/population/density/CADRL",
                 "extracted_population/RVO.csv"             :"results/population/density/RVO"   }

    for data_path in data_dict:
        print(data_path)
        visualizer = Visualizer(data_path, data_dict[data_path])
        visualizer.visualize_all_jobs()


if __name__ == "__main__":
    main()
