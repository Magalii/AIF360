import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier

import sys 
sys.path.append('../parent_aif360/')
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.metrics.classification_metric import ClassificationMetric
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric

#import from within datasetCompExperiment
from datasetCompExperiment.dataset_custom.data_custom_preproc import load_custom_compas, load_custom_adult
from datasetCompExperiment.dataset_custom.student_dataset import StudentDataset
from datasetCompExperiment.expe_meta_helper import run_expe_meta
from datasetCompExperiment.expe_meta_helper import plot_result
from datasetCompExperiment.dataset_custom.new_preproc_function import load_preproc_data_student

"""
    This code has been written by Magali Legast and Lisa Koutsoviti Koumeri
"""

np.random.seed(12345)

#################################
## GENERAL EXPERIMENT SETTINGS ##
#################################

TRAIN = True #Whether the models will be (re)trained and metrics (re)computed (true) or not (false)
PLOT = True #Whether to generate plot(s) of the results (true) or not (false)
PLOT_STYLE = 'FILLED_STDEV' #'SIMPLE_PLOT', 'ERROR_BAR' or 'FILLED_STDEV'
#'SIMPLE_PLOT' for no representation of standard deviation, 'ERROR_BAR' for standard dev. represented as error bars, 'FILLED_STDEV' for standard representation as filled areas
SAVE_PLOT = True #Whether or not the plot(s) will be saved (as a pdf in 'Result/Plots')
SHOW_PLOT = True #Whether or not the plot(s) will be displayed once the computation is done
TYPE = 'sr' #choice fairness metric to be considered as the fairness constraint. 'sr' for Statistical Rate, 'fdr' for False Discovery Rate. Only 'sr' has been used in our experiment, but fdr is also supported by the MetaFairClassifier class

path_datasets = 'datasetCompExperiment/DatasetsRaw/'
path_result = 'datasetCompExperiment/Results/'

####################
## DATASET CHOICE ##
####################

# Uncomment here the dataset to be used for training and/or for which the results should be displayed
# 'data_name' is used in the name of files to be saved or retrieved from disk (pickle of results and plot files)

#data_orig = load_custom_adult(path=path_datasets) #Adult with sensitive attributes 'sex' and 'race'
#data_name = 'adult'
#data_orig = load_custom_compas(path=path_datasets) #COMPAS with sensitive attributes 'sex' and 'race'
#data_name = 'compas'
# Note : for Adult and COMPAS, we performed the experiment using a train sample size of 1000 during in-processing bias-mitigation (see line 106 in file aif360/algorithms/inprocessing/celisMeta/General.py)

data_orig = StudentDataset() #Standard preproc -> 'age' is kept as numerical, sensitive attribute is 'sex'
data_name = 'student'
#data_orig = load_preproc_data_student() #Custom preproc -> age is binarized, sensitive attributes are 'sex' and 'age'
#data_name = 'student_bin17'
# Note : Value of sample size for in-processing bias mitigation shouldn't exceed 454, which is the train set size (see line 106 in file aif360/algorithms/inprocessing/celisMeta/General.py)

#######################
## LAUNCH EXPERIMENT ##
#######################

if TRAIN :
  #Run experiment and save results as pickle file
  print("\n--- experiment ---\n")
  taus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  #taus =  [0.0, 0.3, 0.8]
  results = run_expe_meta(data_orig, 10, taus, [TYPE])
  print(results)

  with open(path_result+'expe_comp_'+data_name+'_'+TYPE+'.pkl', 'wb') as f :
      pickle.dump(results, f)

if PLOT :
  #Retrieve results from pickle file and plot them
  file_res_student = path_result+'expe_comp_'+data_name+'_'+TYPE+'.pkl'
  file_plot = path_result + '/Plots/plot_'+data_name
  plot_result(file_res_student,plot_style=PLOT_STYLE, save = SAVE_PLOT, plot_file = file_plot, display = SHOW_PLOT)
