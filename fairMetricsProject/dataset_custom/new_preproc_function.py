from fairMetricsProject.dataset_custom.student_dataset import StudentDataset


"""
    Functions to load datasets using aif360 standard way of loading datasets
    #TODO those functions are meant to be added to aif360 later on
"""


def load_preproc_data_student(protected_attributes=None, sub_samp=False, balance=False):
    def custom_preprocessing(df):
        #TODO transformer les yes et no en bouleen numérique

        def group_age(x):
            if x <=18: #Original version : x <=18
                return 1.0
            else :
                return 0.0
        df['age'] = df['age'].apply(lambda x:  group_age(x))
        return df
    
    D_features = ['sex', 'age'] if protected_attributes is None else protected_attributes

    # privileged classes
    all_privileged_classes = {"sex": [1.0], "age": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'M', 0.0: 'F'},
                                    "age": {1.0: '<=18', 0.0: '>18'}}    

    metadata_preproc = {'label_maps': [{1.0: 'pass', 0.0: 'fail'}], 
                        'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]}

    return StudentDataset(
        protected_attribute_names=D_features,
        privileged_classes = [['M'],[1.0]],
        custom_preprocessing=custom_preprocessing,
        metadata=metadata_preproc)
        #privileged_classes = [all_privileged_classes[x] for x in D_features],
