from fairMetricsProject.dataset_custom.student_dataset import StudentDataset


"""
    Functions to load datasets using aif360 standard way of loading datasets
    #TODO those functions are meant to be added to aif360 later on
"""


def load_preproc_data_student(protected_attributes=None, sub_samp=False, balance=False):
    #This preprocessing only considers sex as the protected_attributes
    def custom_preprocessing(df):
        #transformer les yes et no en bouleen numérique
        return df

    all_privileged_classes = {"sex": [1.0]}

    metadata = {
    'label_maps': [{1.0: 'pass', 0.0: 'fail'}], 
    'protected_attribute_maps': [{1: 'female', 0: 'male'}] #'sex'
                                 #{blablabla}] #'age', numerical from 15 to 22 -> TODO see if it has been binarized
    }
    # privileged classes
    return StudentDataset()
        #privileged_classes = all_privileged_classes["sex"],
        #custom_preprocessing = custom_preprocessing)
        
        #metadata=metadata)