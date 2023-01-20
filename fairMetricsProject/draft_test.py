import sys 
sys.path.append('../parent_aif360')

#import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
#from tqdm import tqdm
from scipy.stats import multivariate_normal

#from aif360.metrics import BinaryLabelDatasetMetric
#from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
#from aif360.algorithms.inprocessing import MetaFairClassifier

dataset_orig = load_preproc_data_adult()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

min_max_scaler = MaxAbsScaler() #scale each feature value so they are within [-1, 1] 
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)#get the features with scaled values, but I don't understand what the "fit" part does
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

#print("dataset_orig_train: \n" + str(dataset_orig_train)) #complete well formed dataset
#print("dataset_orig_train.features :\n" + str(dataset_orig_train.features)) #matrix of features values
#print("dataset_orig_test :\n" + str(dataset_orig_test))
#print("dataset_orig_test.features :\n" + str(dataset_orig_test.features))

### explore "fit" function ###
#MetaFairClassifier(tau=0, sensitive_attr="sex", type="fdr").fit(dataset_orig_train)
dataset = dataset_orig_train
sensitive_attr = "sex"
x_train = dataset.features
y_train = np.where(dataset.labels.flatten() == dataset.favorable_label,
                           1, -1)
sens_idx = dataset.protected_attribute_names.index(sensitive_attr)
# dataset_orig_train.protected_attribute_names = ['sex', 'race']
# sens_idx = 0 because sex is at position 0 in the array of possible sensitive attributes
#print("dataset_orig_train.protected_attribute_names :\n" + str(dataset_orig_train.protected_attribute_names))
#print("sens_idx :" + str(sens_idx))

#print("dataset.protected_attributes :\n" + str(dataset.protected_attributes))

#x_control_train is boolean array indicating in which positions elements belong to priviledged group
x_control_train = np.where(
                np.isin(dataset.protected_attributes[:, sens_idx],
                        dataset.privileged_protected_attributes[sens_idx]),
                1, 0)

#print("x_control_train :\n" + str(x_control_train[0:200]))

### explore getModel function ###
X = x_train
y = y_train
sens = x_control_train

train = np.c_[X, y, sens]
mean = np.mean(train, axis=0)
cov = np.cov(train, rowvar=False)

dist = multivariate_normal(mean, cov, allow_singular=True,
                                   seed=None)
#print("dist :\n" + str(dist)) # <scipy.stats._multivariate.multivariate_normal_frozen object at 0x7ff99ed366d0>

'''
n = 7
array_n = [0] * n
print("array_n :\n" + str(array_n))
'''

a = np.arange(0.8/0.01, step=10) *0.01
b = (a + 0.01) / 0.8
c = np.minimum(b, 1)
d = np.c_[a, b]
print("a :\n" + str(a))
print("b :\n" + str(b))
print("c :\n" + str(c))
print("d :\n" + str(d))
#print(" :\n" + str())