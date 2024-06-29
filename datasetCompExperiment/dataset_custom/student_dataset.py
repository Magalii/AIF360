import pandas as pd

from aif360.datasets import StandardDataset

import sys 
sys.path.append('../parent_aif360')
datasets = 'datasetCompExperiment/DatasetsRaw/'

default_mappings = {
    'label_maps': [{1.0: 'pass', 0.0: 'fail'}],
    'protected_attribute_maps': [{1.0: 'male', 0.0: 'female'}]
}
# default dataset has only 'sex' as protected attribute. To study 'age', you should load the dataset with "load_preproc_data_student"

class StudentDataset(StandardDataset):
    """Student Performance UCI dataset (Portuguese subject) : Written by Magali Legast to be compatible with aif360 format
        uci repo dataset with id=320
        See https://archive.ics.uci.edu/dataset/320/student+performance
        or http://fairnessdata.dei.unipd.it
    """

    def __init__(self, label_name='G3',
                 favorable_classes= (lambda n: n>=10),
                 protected_attribute_names=['sex'],
                 privileged_classes=[['M']],
                 instance_weights_name=None,
                 categorical_features=['school','address','Pstatus', 'Mjob', 'Fjob', 'guardian', 'famsize', 'reason','schoolsup','famsup','activities','paid','internet','nursery','higher','romantic'],
                 features_to_keep=[], features_to_drop=[], #['G1','G2'] #G1 and G2 are previous grades. Removing those attributes increases the difficulty of the prediction task
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings):
        #See :obj:`StandardDataset` for a description of the arguments

        filepath = 'datasetCompExperiment/DatasetsRaw/student_df'

        try:
            df = pd.read_pickle(filepath)
        except IOError as err:
            try :
                from ucimlrepo import fetch_ucirepo 
                student_performance = fetch_ucirepo(id=320)
                df = student_performance.data.original
            except ModuleNotFoundError :
                print("ModuleNotFoundError: {}".format(err))
                sys.exit(1)

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
