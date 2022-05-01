from math import ceil
import sys
sys.path.append("..")
import os
from naml import main
import json
import csv


TSF_default_parameters = {"min_interval":3, "n_estimators":200, "n_jobs":1,"random_state":1}
SupervisedTimeSeriesForest_default_parameters = {"n_estimators":200,"n_jobs":1,"random_state":1}
HIVECOTEV1_default_parameters = {"stc_params":None, "tsf_params":None, "rise_params":None, "cboss_params":None, "verbose":0, "n_jobs":1, "random_state":1}
KNN_default_parameters = {"n_neighbors":1, "weights":"uniform", "distance":"dtw", "distance_params":None}
RISE_default_parameters = {'n_estimators':200,'random_state':1,'min_interval':16,'acf_lag':100,'acf_min_values':4, 'n_jobs' : 1 }
EE_default_parameters = {"distance_measures":['dtw','ddtw','wdtw','wddtw','msm'], "proportion_of_param_options":1.0, "proportion_train_in_param_finding":1.0, "proportion_train_for_test":1.0, "n_jobs":1, "random_state":1, "verbose":0}
BOSSE_default_parameters = {'n_parameter_samples':250,'threshold':0.92,'max_ensemble_size':500,'max_win_len_prop':1,'time_limit':0.0,'min_window':3,'max_window':10, 'random_state':1, "n_jobs":1}
IndividualBOSS_default_parameters = {"window_size":10, "word_length":8, "norm":False, "alphabet_size":4, "save_words":True, "n_jobs":1, "random_state":1}
ContractableBOSS_default_parameters = {"n_parameter_samples":250, "max_ensemble_size":50, "max_win_len_prop":1, "min_window":10, "save_train_predictions":False, "n_jobs":1, "random_state":1}
IndividualTDE_default_parameters = {"window_size":10, "word_length":8, "norm":False, "levels":1, "igb":False, "alphabet_size":4, "bigrams":True, "dim_threshold":0.85, "max_dims":20, "n_jobs":1, "random_state":1}
WEASEL_default_parameters = {'anova':True,'bigrams':True,'binning_strategy':'information-gain','window_inc':4,'chi2_threshold':-1,'random_state':1}
TDE_default_parameters = {"n_parameter_samples":250, "max_ensemble_size":50, "max_win_len_prop":1, "min_window":10, "randomly_selected_params":50, "bigrams":None, "dim_threshold":0.85, "max_dims":20, "time_limit_in_minutes":0.0, "save_train_predictions":False, "n_jobs":1, "random_state":1}
SHAPELET_default_parameters={"n_shapelet_samples":10000, "max_shapelets":None, "max_shapelet_length":None, "estimator":None, "transform_limit_in_minutes":0, "time_limit_in_minutes":0, "save_transformed_data":False, "n_jobs":1, "batch_size":100, "random_state" : 1}
MUSE_default_parameters = {"anova":True, "bigrams":True, "window_inc":2, "p_threshold":0.05, "use_first_order_differences":False, "n_jobs":1, "random_state":1}

classifier_dict = {
    'TimeSeriesForestClassifier' : TSF_default_parameters,
    'SupervisedTimeSeriesForest' : SupervisedTimeSeriesForest_default_parameters,
    'KNeighborsTimeSeriesClassifier' : KNN_default_parameters,
    'HIVECOTEV1' : HIVECOTEV1_default_parameters,
    'RandomIntervalSpectralEnsemble' : RISE_default_parameters,
    'ElasticEnsemble' : EE_default_parameters,
    'IndividualBOSS' : IndividualBOSS_default_parameters,
    'ContractableBOSS' : ContractableBOSS_default_parameters,
    'BOSSEnsemble' : BOSSE_default_parameters,
    'TemporalDictionaryEnsemble' : TDE_default_parameters,
    'IndividualTDE' : IndividualTDE_default_parameters,
    'WEASEL' : WEASEL_default_parameters,
    'ShapeletTransformClassifier' : SHAPELET_default_parameters,
    'MUSE' : MUSE_default_parameters
}


# def process_param(input, i, j):
#     if input == None: return None
#     elif input == False: return False
#     elif input == True: return True
#     else:
#         try:
#             inputAsFloat = float(input)
#             if inputAsFloat.is_integer():
#                 return int(input) + j
#             else:
#                 return inputAsFloat * pow(0.99, i)
#         except ValueError:
#             return input

active = True
classifier = input("Enter classifier name \n")
parameters = classifier_dict[classifier]
new_params = []
file_set = set()
for property in parameters:
    if property == "random_state":
        continue
    elif property == "n_jobs":
        continue
    elif isinstance(parameters[property], int):
        new_params.append(property)
        for i in range(15):
            new_params.append(int(parameters[property]) + max(1, ceil( (i+1) * parameters[property] * 0.1)))
            print(new_params)
    elif isinstance(parameters[property], float):
        new_params.append(property)
        for i in range(15):
            new_params.append(parameters[property] * pow(0.98, i))
            print(new_params)
    else:
        continue
for value in new_params:
    if isinstance(value, str):
        active_prop = value
        continue
    base_val = parameters[active_prop]
    parameters[active_prop] = value
    file_name = "{0}-{1}-{2}.json".format(classifier, active_prop, value)
    job = {
        "name" : file_name,
        "filepath" : "/home/NAML/NAML/naml_backend/naml_django/naml/csvpkl.pickle",
        "logging" : True,
        "target_col": "event",
        "n_splits":8,
        "jobs": [
            {
            "classifier": "ColumnEnsembleClassifier",
            "ensembleInfo":[
                {
                    "classifier": classifier,
                    "parameters": parameters,
                    "columnNum": 0
                },
                {
                    "classifier": classifier,
                    "parameters": parameters,
                    "columnNum": 1
                },
                {
                    "classifier": classifier,
                    "parameters": parameters,
                    "columnNum": 2
                },
                {
                    "classifier": classifier,
                    "parameters": parameters,
                    "columnNum": 3
                },
                {
                    "classifier": classifier,
                    "parameters": parameters,
                    "columnNum": 4
                },
                {
                    "classifier": classifier,
                    "parameters": parameters,
                    "columnNum": 5
                },
                {
                    "classifier": classifier,
                    "parameters": parameters,
                    "columnNum": 6
                },
                {
                    "classifier": classifier,
                    "parameters": parameters,
                    "columnNum": 7
                }
                ]
            }
        ]
    }

    with open(file_name, 'w') as outfile:
        json.dump(job, outfile)
    parameters[active_prop] = base_val
    file_path = "/home/NAML/NAML/naml_backend/naml_django/naml/configFiles/" + file_name
    file_set.add(file_path)

for file_path in file_set:
    main(file_path)