#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

"""
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score, classification_report, make_scorer, accuracy_score, precision_recall_fscore_support as score
from sklearn.metrics import pair_confusion_matrix
"""
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier

import sys 
sys.path.append('../parent_aif360/')
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.metrics.classification_metric import ClassificationMetric
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric

#import from within fairMetricsProject
from fairMetricsProject.dataset_custom.data_custom_preproc import load_custom_compas, load_custom_adult
from fairMetricsProject.dataset_custom.student_dataset import StudentDataset
from fairMetricsProject.expe_meta_helper import run_expe_meta
from fairMetricsProject.expe_meta_helper import plot_result
from fairMetricsProject.dataset_custom.new_preproc_function import load_preproc_data_student


np.random.seed(12345)

TRAIN = True
PLOT = True
PLOT_STYLE = 'FILLED_STDEV' #'SIMPLE_PLOT', 'FILLED_STDEV' or 'ERROR_BAR'
SAVE_PLOT = True
SHOW_PLOT = True
TYPE = 'sr'

################
## Experiment ##
################
#print("\n--- data ---\n")
path_datasets = 'fairMetricsProject/DatasetsRaw/'
path_result = 'fairMetricsProject/Results/'

data_orig = load_custom_adult(path=path_datasets)
data_name = 'adult'
#data_orig = load_custom_compas(path=path_datasets)
#data_name = 'compas'
#data_orig = StudentDataset() #Standard preproc -> age numerical ###WARNING### Value of sample size for bias mitigation shouldn't exceed 649 (total dataset size)
#data_name = 'student'
#data_orig = load_preproc_data_student() #Custom preproc -> age binary ###WARNING### Value of sample size for bias mitigation shouldn't exceed 649 (total dataset size)
#data_name = 'student_bin17_649'

if TRAIN :
  print("\n--- experiment ---\n")
  taus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  #taus =  [0.0, 0.3, 0.8]
  results = run_expe_meta(data_orig, 10, taus, [TYPE])
  print(results)

  with open(path_result+'expe_comp_'+data_name+'_'+TYPE+'.pkl', 'wb') as f :
      pickle.dump(results, f)

if PLOT :
  file_res_student = path_result+'expe_comp_'+data_name+'_'+TYPE+'.pkl'
  file_plot = path_result + '/Plots/plot_'+data_name
  plot_result(file_res_student,plot_style=PLOT_STYLE, save = SAVE_PLOT, plot_file = file_plot, display = SHOW_PLOT)

#print(data_orig)
