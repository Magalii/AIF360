import pickle
import pandas as pd
import itertools
import matplotlib.pyplot as plt


from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.metrics.classification_metric import ClassificationMetric
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score

tau_def = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mitig_def = ['sr', 'fdr']

def run_expe_meta(dataset_orig, folds, tau_range = tau_def, mitig_types = ['sr', 'fdr']):
  """
  Compute several models using the MetaFairClassifier meta-algorithm using dataset_orig divided in train and split as well as the fairness and performance metric of each model
  Args :
    dataset_orig : aif360 dataset type, result of load_preproc_data_[name]
    folds  : int, number of fold
    tau_range : List, all value of tau (minimal value allowed for fairness measure) to be considered
    mitigation_types : List, fairness metric(s) that should be considered as for the fairness constraint. 'sr' for Statistical Rate, 'fdr' for False Discovery Rate
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
            #F1 score : sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
            f1 = f1_score(dataset_orig_test.labels,dataset_bias_test.labels)
            hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['F1 score'] = f1
            #ratios metrics
            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['FDR'] = fdr
            hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['DP_ratio'] = dpratio
            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['error_rate_ratio'] = classified_metric_bias_test.error_rate_ratio()
            #difference metrics
            hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['Stat. parity'] = classified_metric_bias_test.statistical_parity_difference()
            hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['Eq. odds'] = classified_metric_bias_test.equalized_odds_difference()
            hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['Eq. opportunity'] = classified_metric_bias_test.equal_opportunity_difference()
            hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['Consistency'] = classified_metric_bias_test.consistency()

            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)]['avg_odds_diff'] = classified_metric_bias_test.average_abs_odds_difference()

            #hp_results[mitigation_type][s_attr]['Fold'+str(fold)]['Tau'+str(tau)][''] = classified_metric_bias_test.

  #results['data'] = hp_results

  return hp_results


def plot_result(data_file, plot_style = 'SIMPLE_PLOT', save = False, plot_file = None, display = True) :
    """
    file_name : file containing a dictionnary as returned by run_exp_meta
    plot_style : 'SIMPLE_PLOT', 'FILLED_STDEV' or 'ERROR_BAR'
                'SIMPLE_PLOT' for no display of standard deviation
                'FILLED_STDEV' for display of standard deviation as colored area arround the curve
                'ERROR_BAR' for display of standard deviation as error bars
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
    print(results_dict[mitig][s_attr]['Eq. opportunity'])

    means = {}
    std = {}

    tickers = [combo for combo in itertools.product(results_dict.keys(), results_dict[valid_mitig].keys())]

    for ticker, i in zip(tickers, range(len(tickers))):

        means_per_tau = pd.DataFrame(index=results_dict[valid_mitig][valid_s_attr][valid_metric].columns,
                                    columns = results_dict[valid_mitig][valid_s_attr].keys())
        std_per_tau = pd.DataFrame(index=results_dict[valid_mitig][valid_s_attr][valid_metric].columns,
                                    columns = results_dict[valid_mitig][valid_s_attr].keys())

        for metric in metrics_list:
            #print(metric, all_metrics[valid_mitig][valid_s_attr][metric].mean(axis=0), end = '')
            means_per_tau[metric] = results_dict[ticker[0]][ticker[1]][metric].mean(axis=0)
            std_per_tau[metric] = results_dict[ticker[0]][ticker[1]][metric].std(ddof=0) #ddof = 0 takes the actual std_dev normalized over N instead of the corrected sample standard deviation that uses N − 1

        means[ticker] = means_per_tau
        std[ticker] = std_per_tau

        #accuracies, statistical_rates, fdr, error_rate_ratio = means_per_tau['accuracy'].values, means_per_tau['DP'].values, means_per_tau['FDR'].values, means_per_tau['error_rate_ratio']
        all_tau = [float(tau[-3:]) for tau in means_per_tau.index.values]

        fig, ax = plt.subplots() # (figsize=(8, 4))
        ax.hlines(0,0,1,colors='black')

        # Use these blocks instead of the following one to automate plotting
        """
        #for metric in metrics_list :
          #if metric != "DP_ratio":
          #  plt.plot(all_tau,means_per_tau[metric],label = str(metric),linestyle="--",marker="o")
        plt.style.use('tableau-colorblind10')
        """
        """
        for metric in metrics_list :
          if metric != "DP_ratio":
            ax.fill_between(all_tau, means_per_tau[metric] - std_per_tau[metric], means_per_tau[metric] + std_per_tau[metric], color='#888888', alpha=0.4)
            ax.fill_between(all_tau, means_per_tau[metric] - 2*std_per_tau[metric], means_per_tau[metric] + 2*std_per_tau[metric], color='#888888', alpha=0.2)
        """
        
        if plot_style == 'ERROR_BAR':
          ax.errorbar(all_tau, means_per_tau['Consistency'], yerr=std_per_tau['Consistency'], capsize=4, capthick=1.5, label = 'Consistency', linestyle="--",marker="^", color="#006BA4")#Cerulean/Blue
          ax.errorbar(all_tau, means_per_tau['F1 score'], yerr=std_per_tau['F1 score'], capsize=4, capthick=1.5, label = 'F1', linestyle="--",marker="o", color='#595959')##Dark gray
          ax.errorbar(all_tau, means_per_tau['accuracy'], yerr=std_per_tau['accuracy'], capsize=4, capthick=1.5, label = 'Accuracy', linestyle="--",marker="o", color='#ABABAB')
          ax.errorbar(all_tau, means_per_tau['Eq. odds'], yerr=std_per_tau['Eq. odds'], capsize=4, capthick=1.5, label = 'Eq. Odds', linestyle="--",marker="X", c='#C85200')#Tenne/Dark orange
          ax.errorbar(all_tau, means_per_tau['Eq. opportunity'], yerr=std_per_tau['Eq. opportunity'], capsize=4, capthick=1.5, label = 'Eq. Opp', linestyle="--",marker="d", c="#FF800E")#Pumpkin/Bright orange
          ax.errorbar(all_tau, means_per_tau['Stat. parity'], yerr=std_per_tau['Stat. parity'], capsize=4,capthick=1.5, label = 'SR', linestyle="--",marker="s", c="#A2C8EC")#Seil/Light blue
        else :
          #Mean values
          ax.plot(all_tau, means_per_tau['Consistency'], label = 'Consistency', linestyle="--",marker="^", color="#006BA4")#Cerulean/Blue
          ax.plot(all_tau, means_per_tau['accuracy'], label = 'Accuracy', linestyle="--",marker="o", color='#ABABAB')##Dark gray
          ax.plot(all_tau, means_per_tau['F1 score'], label = 'F1 score', linestyle="--",marker="o", color='#595959')##Dark gray
          #ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
          ax.plot(all_tau, means_per_tau['Eq. odds'], label = 'Eq. Odds', linestyle="--",marker="X", c='#C85200')#Tenne/Dark orange
          ax.plot(all_tau, means_per_tau['Eq. opportunity'], label = 'Eq. Opp', linestyle="--",marker="d", c="#FF800E")#Pumpkin/Bright orange
          ax.plot(all_tau, means_per_tau['Stat. parity'], label = 'SR', linestyle="--",marker="s", c="#A2C8EC")#Seil/Light blue
          
          #'#FFBC79' Light orange/Mac and chesse '#898989'#Suva Grey/Light Gray '#ABABAB'#Dark gray '#595959'#Mortar/Darker Grey
          #https://stackoverflow.com/questions/74830439/list-of-color-names-for-matplotlib-style-tableau-colorblind10
          
          if plot_style == 'FILLED_STDEV':
            #Shade for std values
            ax.fill_between(all_tau, means_per_tau['Consistency'] - std_per_tau['Consistency'], means_per_tau['Consistency'] + std_per_tau['Consistency'], edgecolor = None, facecolor='#006BA4', alpha=0.4)
            ax.fill_between(all_tau, means_per_tau['accuracy'] - std_per_tau['accuracy'], means_per_tau['accuracy'] + std_per_tau['accuracy'], edgecolor = None, facecolor='#ABABAB', alpha=0.4)
            #ax.fill_between(all_tau, means_per_tau['accuracy'] - 2*std_per_tau['accuracy'], means_per_tau['accuracy'] + 2*std_per_tau['accuracy'], edgecolor = None, facecolor='#ABABAB', alpha=0.2)
            ax.fill_between(all_tau, means_per_tau['F1 score'] - std_per_tau['F1 score'], means_per_tau['F1 score'] + std_per_tau['F1 score'], edgecolor = None, facecolor='#595959', alpha=0.4)
            ax.fill_between(all_tau, means_per_tau['Eq. odds'] - std_per_tau['Eq. odds'], means_per_tau['Eq. odds'] + std_per_tau['Eq. odds'], edgecolor = None, facecolor='#C85200', alpha=0.4)
            ax.fill_between(all_tau, means_per_tau['Eq. opportunity'] - std_per_tau['Eq. opportunity'], means_per_tau['Eq. opportunity'] + std_per_tau['Eq. opportunity'], edgecolor = None, facecolor='#FF800E', alpha=0.4)
            ax.fill_between(all_tau, means_per_tau['Stat. parity'] - std_per_tau['Stat. parity'], means_per_tau['Stat. parity'] + std_per_tau['Stat. parity'], edgecolor = None, facecolor='#A2C8EC', alpha=0.4)

        ax.tick_params(labelsize = 'large',which='major')
        ax.set_ylim([-0.2,1.0])
        ax.set_xlabel(r'$\tau$', size=14)
        ax.grid(visible=True)
        #ax.legend(loc='best')
        ax.legend(prop={'size':10}, loc='upper right',  bbox_to_anchor=(1, 0.87))
        #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend
        #Good position dor legends : Adult: loc='upper right', bbox_to_anchor=(1, 0.78) #COMPAS: loc = 'center', bbox_to_anchor=(0.42, 0.47) #Student (sex and age): loc='upper right', bbox_to_anchor=(1, 0.87)
        #plt.title('Protected att.:'+ticker[1]+' with '+ticker[0]+' constraint', fontsize=14)
        #plt.suptitle('Accuracy and metrics for '+case_study+' with '+str(folds)+'fold cross-validation', y=0.95, fontsize='xx-large', fontweight='bold')

        if(save) :
            plt.savefig(plot_file+'_'+ticker[1]+'_'+ticker[0]+'_'+plot_style+".pdf", format="pdf", bbox_inches="tight") # dpi=1000) #dpi changes image quality
        if(display) :
            
            plt.show()