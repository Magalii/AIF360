import pandas as pd

from aif360.datasets import StandardDataset
#from ucimlrepo import fetch_ucirepo 

import sys 
sys.path.append('../parent_aif360')
datasets = 'fairMetricsProject/DatasetsRaw/'

df_student_features = pd.read_pickle(datasets+"student_features_df")
df_student_targets = pd.read_pickle(datasets+"student_targets_df")
df_student_all = pd.read_pickle(datasets+"student_df")

default_mappings = {
    'label_maps': [{1.0: 'pass', 0.0: 'fail'}],  #TODO check if it works
    'protected_attribute_maps': [{1.0: 'female', 0.0: 'male'}] #'sex'
                                 #{blablabla}] #'age', numerical from 15 to 22 -> TODO see if it has been binarized
}

class StudentDataset(StandardDataset):
    """Student Performance UCI dataset
        uci repo dataset with id=320
    See https://archive.ics.uci.edu/dataset/320/student+performance
    or http://fairnessdata.dei.unipd.it
    """

    def __init__(self, label_name='G3',
                 favorable_classes= (lambda n: n>=10), #lambda function #TODO makes sure it works
                 protected_attribute_names=['sex'], #TODO add age after binarizing it
                 privileged_classes=[['M']],
                 instance_weights_name=None,
                 categorical_features=['school','address','Pstatus', 'Mjob', 'Fjob', 'guardian', 'famsize', 'reason','schoolsup','famsup','activities','paid','internet','nursery','higher','romantic'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings):
        #See :obj:`StandardDataset` for a description of the arguments.

        #example for Compas
        #filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'compas', 'compas-scores-two-years.csv')
        
        filepath = 'fairMetricsProject/DatasetsRaw/' #TODO make it more portable by using os.path.dirname

        try:
            df = pd.read_pickle(filepath)
        except IOError as err:
            try :
                from ucimlrepo import fetch_ucirepo 
                student_performance = fetch_ucirepo(id=320)
                df = student_performance.data.original
            except ModuleNotFoundError :
                print("ModuleNotFoundError: {}".format(err))
                print("") #TODO make it cleaner with good message
                sys.exit(1)

        #print("StudentDataset: df loaded \n",df,"\n")

        super(StudentDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata)
