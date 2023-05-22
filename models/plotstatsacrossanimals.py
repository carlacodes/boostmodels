import sklearn.metrics
# from rpy2.robjects import pandas2ri
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

import shap
import matplotlib
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold
# scaler = MinMaxScaler()
import os
import xgboost as xgb
import matplotlib.pyplot as plt
# import rpy2.robjects.numpy2ri
import matplotlib.colors as mcolors
import sklearn
from sklearn.model_selection import train_test_split
from helpers.behaviouralhelpersformodels import *
from helpers.calculate_stats import *

def run_stats_calc(df, ferrets, stats_dict, pitch_param = 'control_trial'):

    df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    df_catchnoncorrection = df[(df['catchTrial'] == 1)]
    count = int(0)
    stats_dict[pitch_param] = {}
    stats_dict[pitch_param]['hits'] = {}
    stats_dict[pitch_param]['false_alarms'] = {}
    stats_dict[pitch_param]['correct_rejections'] = {}
    talkers = [1,2]
    for ferret in ferrets:

        selected_ferret = df_noncatchnoncorrection[df_noncatchnoncorrection['ferret'] == count]
        selected_ferret_catch = df_catchnoncorrection[df_catchnoncorrection['ferret'] == count]
        for talker in talkers:
            selected_ferret = selected_ferret[selected_ferret['talker'] == talker]
            selected_ferret_catch = selected_ferret_catch[selected_ferret_catch['talker'] == talker]

            stats_dict[pitch_param]['hits'][talker][ferret] = np.mean(selected_ferret['hit'])
            stats_dict[pitch_param]['false_alarms'][talker][ferret] = np.mean(selected_ferret['falsealarm'])
            stats_dict[pitch_param]['correct_rejections'][talker][ferret] = np.mean(selected_ferret_catch['response'] == 3)
        count += 1
    stats_dict_all = {}
    stats_dict_all[1] ={}
    stats_dict_all[2] ={}

    stats_dict_all[1][pitch_param]= {}
    stats_dict_all[2][pitch_param]= {}

    for talker in talkers:

        hits = np.mean(df_noncatchnoncorrection['hit'])
        false_alarms = np.mean(df_noncatchnoncorrection['falsealarm'])
        correct_rejections = np.mean(df_catchnoncorrection['response'] == 3)

        stats_dict_all[pitch_param][talker]['hits'] = hits
        stats_dict_all[pitch_param][talker]['false_alarms'] = false_alarms
        stats_dict_all[pitch_param][talker]['correct_rejections'] = correct_rejections

    return stats_dict_all, stats_dict

def plot_stats(stats_dict_all_combined, stats_dict_combined):

    #generate bar plots
    stats = pd.DataFrame.from_dict(stats_dict_all_combined)
    x = np.arange(len(stats_dict_all_combined))  # the label locations
    #plot bar plots

    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in stats_dict_all_combined.items():
        print(measurement)
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement['hits'], width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    plt.ylim(0, 1)
    plt.ylabel('Proportion of hits')
    plt.title('Proportion of hits across all ferrets')
    plt.legend()
    plt.show()





    #get proportion of hits and false alarms for the dataframe
    fig, ax = plt.subplots()
    sns.barplot(x='ferret', y='hits', hue='type', data=stats, ax=ax)

    #get proportion of hits and false alarms for the dataframe




if __name__ == '__main__':
    stats_dict = {}
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    df = behaviouralhelperscg.get_stats_df(ferrets=ferrets, startdate='04-01-2016', finishdate='01-03-2023')

    pitch_type_list = ['control_trial', 'inter_trial_roving', 'intra_trial_roving']
    stats_dict_all_combined = {}
    stats_dict_combined = {}
    for pitch in pitch_type_list:
        stats_dict_all, stats_dict = run_stats_calc(df, ferrets, stats_dict, pitch_param=pitch)
        #append to dataframe
        stats_dict_all_combined[pitch]= stats_dict_all[pitch]
        stats_dict_combined[pitch] = stats_dict[pitch]

    plot_stats(stats_dict_all_combined, stats_dict_combined)

