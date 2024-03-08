# Documentation of Classes

class Dataset(ABC):
    """Abstract base class for datasets."""

    Methods:
    - def copy(self, deepcopy=False):
        Convenience method to return a copy of this dataset.
        Args:
            deepcopy (bool, optional): :func:`~copy.deepcopy` this dataset if
                `True`, shallow copy otherwise.
        Returns:
            Dataset: A new dataset with fields copied from this object and
            metadata set accordingly.
    - def export_dataset(self): @abstractmethod
        Save this Dataset to disk.
    - def split(self, num_or_size_splits, shuffle=False):
        Split this dataset into multiple partitions.
        Args:
            num_or_size_splits (array or int): If `num_or_size_splits` is an
                int, *k*, the value is the number of equal-sized folds to make
                (if *k* does not evenly divide the dataset these folds are
                approximately equal-sized). If `num_or_size_splits` is an array
                of type int, the values are taken as the indices at which to
                split the dataset. If the values are floats (< 1.), they are
                considered to be fractional proportions of the dataset at which
                to split.
            shuffle (bool, optional): Randomly shuffle the dataset before
                splitting.
        Returns:
            list(Dataset): Splits. Contains *k* or `len(num_or_size_splits) + 1`
            datasets depending on `num_or_size_splits`.

class StructuredDataset(Dataset):
    """Base class for all structured datasets.

    A StructuredDataset requires data to be stored in :obj:`numpy.ndarray`
    objects with :obj:`~numpy.dtype` as :obj:`~numpy.float64`.

    Attributes:
        features (numpy.ndarray): Dataset features for each instance.
        labels (numpy.ndarray): Generic label corresponding to each instance
            (could be ground-truth, predicted, cluster assignments, etc.).
        scores (numpy.ndarray): Probability score associated with each label.
            Same shape as `labels`. Only valid for binary labels (this includes
            one-hot categorical labels as well).
        protected_attributes (numpy.ndarray): A subset of `features` for which
            fairness is desired.
        feature_names (list(str)): Names describing each dataset feature.
        label_names (list(str)): Names describing each label.
        protected_attribute_names (list(str)): A subset of `feature_names`
            corresponding to `protected_attributes`.
        privileged_protected_attributes (list(numpy.ndarray)): A subset of
            protected attribute values which are considered privileged from a
            fairness perspective.
        unprivileged_protected_attributes (list(numpy.ndarray)): The remaining
            possible protected attribute values which are not included in
            `privileged_protected_attributes`.
        instance_names (list(str)): Indentifiers for each instance. Sequential
            integers by default.
        instance_weights (numpy.ndarray):  Weighting for each instance. All
            equal (ones) by default. Pursuant to standard practice in social
            science data, 1 means one person or entity. These weights are hence
            person or entity multipliers (see:
            https://www.ibm.com/support/knowledgecenter/en/SS3RA7_15.0.0/com.ibm.spss.modeler.help/netezza_decisiontrees_weights.htm)
            These weights *may not* be normalized to sum to 1 across the entire
            dataset, rather the nominal (default) weight of each entity/record
            in the data is 1. This is similar in spirit to the person weight in
            census microdata samples.
            https://www.census.gov/programs-surveys/acs/technical-documentation/pums/about.html
        ignore_fields (set(str)): Attribute names to ignore when doing equality
            comparisons. Always at least contains `'metadata'`.
        metadata (dict): Details about the creation of this dataset.
    
    Methods :
