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

from fairMetricsProject.dataset_custom.student_dataset import StudentDataset

doPrint = False

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

def select_attr(df, attr, cond) :
    df_cond = df[df[attr] == cond]
    return df_cond

print("shape total : "+str(all.shape))

print("--- information age ---")
for age in range(15,23) :
    df_age = select_attr(all,'age',age)
    print("age "+str(age)+" :")
    print("shape :" + str(df_age.shape))
    print("success :"+ str(df_age[df_age['G3'] >= 10].shape))
    print("failure :"+ str(df_age[df_age['G3'] < 10].shape))
    print()

df_young = all[all['age'] <= 18]
df_old = all[all['age'] > 18]
print("shape young :" + str(df_young.shape))
print("success :"+ str(df_young[df_young['G3'] >= 10].shape))
print("failure :"+ str(df_young[df_young['G3'] < 10].shape))
print("shape old :" + str(df_old.shape))
print("success :"+ str(df_old[df_old['G3'] >= 10].shape))
print("failure :"+ str(df_old[df_old['G3'] < 10].shape))


print("--- information sex ---")
df_M = select_attr(all,'sex','M')
print("shape M :" + str(df_M.shape))
print("success :"+ str(df_M[df_M['G3'] >= 10].shape))
print("failure :"+ str(df_M[df_M['G3'] < 10].shape))
print()
df_F = select_attr(all,'sex','F')
print("shape F :" + str(df_F.shape))
print("success :"+ str(df_F[df_F['G3'] >= 10].shape))
print("failure :"+ str(df_F[df_F['G3'] < 10].shape))





#student_performance.data.features.to_pickle(datasets+"student_features_df")
#student_performance.data.targets.to_pickle(datasets+"student_targets_df")
#student_performance.data.original.to_pickle(datasets+"student_df")

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
