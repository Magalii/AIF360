from ucimlrepo import fetch_ucirepo 
import pandas as pd
import dill
import pickle
import joblib
#try:
 #   import cPickle as pickle #cPickle is like pickle, but written in C and faster
#except ModuleNotFoundError:
import pickle

import sys 
sys.path.append('../parent_aif360')
datasets = 'fairMetricsProject/DatasetsRaw/'

doPrint = True

# fetch dataset 
student_performance = fetch_ucirepo(id=320)
#uci dataset.data contains the different matrices composing the dataset. Each of these matrices is a pandas Dataframe.
#ids : Dataframe of ID columns
#features : Dataframe of feature columns
#targets : Dataframe of target columns
#original : Dataframe consisting of all IDs, features, and targets
#headers : List of all variable names/headers

#dataset. metadata contains metadata information about the dataset 


  
# data (as pandas dataframes) 
X = student_performance.data.features
y = student_performance.data.targets
all = student_performance.data.original

#student_performance.data.features.to_pickle(datasets+"student_features_df")
#student_performance.data.targets.to_pickle(datasets+"student_targets_df")
#student_performance.data.original.to_pickle(datasets+"student_df")

#TODO figure out how to save student_performance to pickle/dill. The type of object can't be save as is
"""
with open(datasets+"student_uci_ds.pkl",'wb') as f :
    pickle.dump(
        student_performance,
        f)
dill.dump(
        student_performance,
        open(datasets+"student_uci_ds.pickle","wb"))
"""
#df = pd.read_pickle(file_name)

if(doPrint):
    # metadata
    print("\n ####################### \n Metadata \n ####################### \n ")
    #print(student_performance.metadata)
    # access metadata
    print(student_performance.metadata.uci_id)
    print(student_performance.metadata.num_instances)
    print(student_performance.metadata.additional_info.summary)
    
    # variable information 
    print("\n ####################### \n Variables \n ####################### \n ")
    print(student_performance.variables) 

    #targets
    print("\n ####################### \n Targets \n ####################### \n ")
    print(y)

    #data
    print("\n ####################### \n Data \n ####################### \n ")
    print(student_performance.data)
