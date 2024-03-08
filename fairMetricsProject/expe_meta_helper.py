import pickle
import pandas as pd
import itertools
import matplotlib.pyplot as plt


from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.metrics.classification_metric import ClassificationMetric
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from sklearn.preprocessing import MaxAbsScaler

tau_def = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mitig_def = ['sr', 'fdr']

#def run_metaclassifier_crossval(folds, dataset_orig, Rs = [], metadata = [], mitigation_types = ['sr', 'fdr']):
def run_expe_meta(dataset_orig, folds, tau_range = tau_def, mitig_types = ['sr', 'fdr']):
  """
  folds  : int, number of fold
  dataset_orig : aif360 dataset type result of load_preproc_data_[name]
  Rs : List with strings
  metadata : our own metadata, similar to aif360 #TODO check if it is the same
  mitigation_types : List
  """
  #results = {}
  #results['info'] = {'dataset' : dataset_orig.metadata}
  hp_results = {} #hyperparameters ? #key : mitigation_type
  prot_attr_map = {} #dictionnary key : s_attr = dataset_orig.protected_attribute_names

  # zip() allows to loop over more than one values (loops once of the ith elements for all the zipped objects)
  for s_attr, priv, unpriv in zip(dataset_orig.protected_attribute_names, dataset_orig.privileged_protected_attributes, dataset_orig.unprivileged_protected_attributes):
    prot_attr_map[s_attr] = {} #keys are 'Privileged' and 'Unprivileged'
    prot_attr_map[s_attr]['Privileged'] = priv
    prot_attr_map[s_attr]['Unprivileged'] = unpriv

  for mitigation_type in mitig_types:
    print()
    print('---------------')
    print('mitigation type', mitigation_type)
    hp_results[mitigation_type] = {} #key : sensitive attribute

    for s_attr in dataset_orig.protected_attribute_names:
      print(' ')
      print('protected feature', s_attr)
      print('Folds:', end = '\n')
      privileged_groups = [{s_attr: prot_attr_map[s_attr]['Privileged'][0]}]
      unprivileged_groups = [{s_attr: prot_attr_map[s_attr]['Unprivileged'][0]}]

      hp_results[mitigation_type][s_attr] = {} #key is string 'Fold'+str(fold)

      for fold in range(0,folds):
        print(str(fold)+' ', end = '\n')
        hp_results[mitigation_type][s_attr]['Fold'+str(fold)] = {}

        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed = 100*fold)

        #RandomForestClassifier used as a baseline
        #clf = RandomForestClassifier()
        #clf.fit(dataset_orig_train.convert_to_dataframe()[0].iloc[:,:-1], dataset_orig_train.convert_to_dataframe()[0].iloc[:,-1])
        #score() measures the accuracy
        #acc_base_score = clf.score(dataset_orig_test.convert_to_dataframe()[0].iloc[:,:-1], dataset_orig_test.convert_to_dataframe()[0].iloc[:,-1])
        #acc_base_score should be int

        # scaling is not necessary here since everything is between 0-1 already. they do it in the jupyter notebook so...
        min_max_scaler = MaxAbsScaler()
        dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
        dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)

        metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups)

        #mesures for each metrics before training 

        # demographic parity
        dpratio_train = min(metric_orig_train.disparate_impact(), 1/metric_orig_train.disparate_impact()) # pretrain
        dpratio_test = min(metric_orig_test.disparate_impact(), 1/metric_orig_test.disparate_impact()) # pretrain
        
        for tau in tau_range:

            hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)] = {}

            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['RandomForrest accuracy'] = acc_base_score
            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['DP GT train'] = dpratio_train
            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['DP GT test'] = dpratio_test
            
            biased_model = MetaFairClassifier(tau=tau, sensitive_attr=s_attr, type=mitigation_type).fit(dataset_orig_train)
            dataset_bias_test = biased_model.predict(dataset_orig_test)

            # compute CDD on predictions with tau0
            
            classified_metric_bias_test = ClassificationMetric(dataset_orig_test, dataset_bias_test,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
            
            TPR = classified_metric_bias_test.true_positive_rate()
            TNR = classified_metric_bias_test.true_negative_rate()
            bal_acc_bias_test = 0.5*(TPR+TNR)

            #dpratio = min(classified_metric_bias_test.disparate_impact(), 1/classified_metric_bias_test.disparate_impact())
            dpratio = classified_metric_bias_test.disparate_impact()

            fdr = classified_metric_bias_test.false_discovery_rate_ratio()
            #fdr = min(fdr, 1/fdr)

            #Accuracy
            hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['accuracy'] = classified_metric_bias_test.accuracy()
            #ratios metrics
            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['FDR'] = fdr
            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['DP_ratio'] = dpratio
            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['error_rate_ratio'] = classified_metric_bias_test.error_rate_ratio()
            #difference metrics
            hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['Stat. parity'] = classified_metric_bias_test.statistical_parity_difference()
            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['Eq. odds'] = classified_metric_bias_test.equalized_odds_difference()
            hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['Eq. opportunity'] = classified_metric_bias_test.equal_opportunity_difference()

            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['avg_odds_diff'] = classified_metric_bias_test.average_abs_odds_difference()

            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)][''] = classified_metric_bias_test.

  #results['data'] = hp_results

  return hp_results

def plot_result(data_file, title, save = False, plot_file = None, display = True) :
    """
    file_name : file containing a dictionnary as returned by run_exp_meta
    """
    fd = open(data_file, 'rb')
    loaded_dict = pickle.load(fd)
    #loaded_dict = loaded_dict['data'] #

    #Variables that will be replaced by the last corresponding key found in the unput dict
    valid_mitig = 'no valid mitig type in input data'
    valid_s_attr = 'no valid sensitive attribute in input data'
    valid_metric = 'no valid metric in input data'

    results_dict = {}
    for mitig in loaded_dict.keys(): #mitigation_type
       results_dict[mitig] = {}
       valid_mitig = mitig
       for s_attr in loaded_dict[mitig].keys() : #sensitive attribute
          results_dict[mitig][s_attr] = {}
          valid_s_attr = s_attr
          for metric in pd.DataFrame(loaded_dict[mitig][s_attr]['Fold1']).index:
            results_dict[mitig][s_attr][metric] = pd.DataFrame(index=loaded_dict[mitig][s_attr].keys(), columns=loaded_dict[mitig][s_attr]['Fold1'].keys())
            valid_metric = metric
            for fold in loaded_dict[mitig][s_attr].keys():
                #print('')
                #print(fold, end = ' - ')
                for tau in loaded_dict[mitig][s_attr][fold].keys():
                    #print(tau, ':_', cases['compas_guess']['sr']['sex'][fold][tau]['Metaclass. accuracy'], end = ' ')
                    results_dict[mitig][s_attr][metric].loc[fold, tau] = loaded_dict[mitig][s_attr][fold][tau][metric]
                
    #Initialize empty dataset where the values will be stored.
    metrics_list = results_dict[mitig][s_attr].keys()
    final_results = pd.DataFrame(columns=metrics_list)
    #final_results_tau0 = pd.DataFrame(columns=cols) #TODO Why a different one for tau0 ?

    case_study = data_file
    means = {}

    tickers = [combo for combo in itertools.product(results_dict.keys(), results_dict[valid_mitig].keys())]

    for ticker, i in zip(tickers, range(len(tickers))):

        means_per_tau = pd.DataFrame(index=results_dict[valid_mitig][valid_s_attr][valid_metric].columns,
                                    columns = results_dict[valid_mitig][valid_s_attr].keys())

        for metric in metrics_list:
            #print(metric, all_metrics[valid_mitig][valid_s_attr][metric].mean(axis=0), end = '')
            means_per_tau[metric] = results_dict[ticker[0]][ticker[1]][metric].mean(axis=0)

        means[ticker] = means_per_tau

        #accuracies, statistical_rates, fdr, error_rate_ratio = means_per_tau['accuracy'].values, means_per_tau['DP'].values, means_per_tau['FDR'].values, means_per_tau['error_rate_ratio']
        all_tau = [float(tau[-3:]) for tau in means_per_tau.index.values]

        for metric in metrics_list :
           plt.plot(all_tau,means_per_tau[metric],label = str(metric),linestyle="--",marker="o")
        """
        plt.plot(all_tau, accuracies, label = 'Metaclassifier accuracy', linestyle="--",marker="o")
        plt.plot(all_tau, statistical_rates, label = 'Demographic parity', linestyle="--",marker="o", c='grey')
        plt.plot(all_tau, fdr, label = 'False Discovery Rate', linestyle="--",marker="o", c='green')
        plt.plot(all_tau, error_rate_ratio, label = 'error_rate_ratio', linestyle="--",marker="o", c='salmon')
        """
        plt.xlabel(r'$\tau$', size=14)
        plt.tick_params(labelsize = 'large',which='major')
        #plt.ylim(bottom=0,top=1)
        #plt.title('Protected att.:'+ticker[1]+' with '+ticker[0]+' constraint', fontsize=14)
        plt.title(title)
        plt.legend(prop={'size':14})
        plt.grid(visible=True)

        if(save) :
            plt.savefig(plot_file+'_'+ticker[1]+"_.pdf", format="pdf", bbox_inches="tight")
        #plt.suptitle('Accuracy and metrics for '+case_study+' with '+str(folds)+'fold cross-validation', y=0.95, fontsize='xx-large', fontweight='bold')
        if(display) :
            plt.show()