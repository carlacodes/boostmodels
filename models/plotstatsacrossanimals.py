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
    # stats_dict[pitch_param] = {}
    # stats_dict[pitch_param]['hits'] = {}
    # stats_dict[pitch_param]['false_alarms'] = {}
    # stats_dict[pitch_param]['correct_rejections'] = {}
    talkers = [1,2]
    stats_dict[1] = {}
    stats_dict[2] = {}
    stats_dict[1][pitch_param] = {}
    stats_dict[2][pitch_param] = {}

    stats_dict[1][pitch_param]['hits'] = {}
    stats_dict[1][pitch_param]['false_alarms']= {}
    stats_dict[1][pitch_param]['correct_rejections']= {}

    stats_dict[2][pitch_param]['hits'] ={}
    stats_dict[2][pitch_param]['false_alarms'] = {}
    stats_dict[2][pitch_param]['correct_rejections'] = {}
    count = 0
    for ferret in ferrets:

        selected_ferret = df_noncatchnoncorrection[df_noncatchnoncorrection['ferret'] == count]
        selected_ferret_catch = df_catchnoncorrection[df_catchnoncorrection['ferret'] == count]
        for talker in talkers:
            selected_ferret_talker = selected_ferret[selected_ferret['talker'] == talker]
            selected_ferret_catch_talker = selected_ferret_catch[selected_ferret_catch['talker'] == talker]

            stats_dict[talker][pitch_param]['hits'][ferret] = np.mean(selected_ferret_talker['hit'])
            stats_dict[talker][pitch_param]['false_alarms'][ferret] = np.mean(selected_ferret_talker['falsealarm'])
            stats_dict[talker][pitch_param]['correct_rejections'][ferret] = np.mean(selected_ferret_catch_talker['response'] ==3)
        count += 1
    stats_dict_all = {}
    stats_dict_all[1] ={}
    stats_dict_all[2] ={}

    stats_dict_all[1][pitch_param]= {}
    stats_dict_all[2][pitch_param]= {}

    for talker in talkers:
        df_noncatchnoncorrection_talker = df_noncatchnoncorrection[df_noncatchnoncorrection['talker'] == talker]
        df_catchnoncorrection_talker = df_catchnoncorrection[df_catchnoncorrection['talker'] == talker]
        hits = np.mean(df_noncatchnoncorrection_talker['hit'])
        false_alarms = np.mean(df_noncatchnoncorrection_talker['falsealarm'])
        correct_rejections = np.mean(df_catchnoncorrection_talker['response'] == 3)

        stats_dict_all[talker][pitch_param]['hits'] = hits
        stats_dict_all[talker][pitch_param]['false_alarms'] = false_alarms
        stats_dict_all[talker][pitch_param]['correct_rejections'] = correct_rejections

    return stats_dict_all, stats_dict

def plot_stats(stats_dict_all_combined, stats_dict_combined):

    #generate bar plots
    stats = pd.DataFrame.from_dict(stats_dict_all_combined)
    x = np.arange(len(stats_dict_all_combined))  # the label locations
    #plot bar plots

    width = 0.15  # the width of the bars
    multiplier = 0
    #
    # fig, ax = plt.subplots(layout='constrained')
    #
    # for attribute, measurement in stats_dict_all_combined.items():
    #     for talker, measurement in measurement.items():
    #         print(measurement)
    #         offset = width * multiplier
    #         rects = ax.bar(offset, measurement['hits'], width, label=talker)
    #         # ax.bar_label(rects, padding=3)
    #         multiplier += 1
    #
    # plt.ylim(0, 1)
    # plt.ylabel('Proportion of hits')
    # plt.title('Proportion of hits across all ferrets')
    # plt.legend()
    # plt.show()
    width = 0.25  # the width of the bars
    multiplier = 0
    gap_width = 0.2  # Width of the gap between series

    fig, ax = plt.subplots(layout='constrained')

    color_map = plt.cm.get_cmap('tab10')  # Choose a colormap

    for attribute, measurement in stats_dict_all_combined.items():
        for talker, measurement_data in measurement.items():
            print(measurement_data)
            if multiplier < 3:
                offset = width * multiplier
            else:
                offset = (gap_width) + (width * multiplier)  # Add gap offset for the second series

            color = color_map(
                np.where(np.array(list(measurement.keys())) == talker)[0][0])  # Assign color based on label
            rects = ax.bar(offset, measurement_data['hits'], width, label=talker, color=color)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 'x', 'v', 's', 'p']
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute][talker]['hits'].items():
                #add jitter to offset
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax.scatter(offset_jitter, ferret_data, color=color, marker=marker_list[count], edgecolors='black')
                count += 1

            multiplier += 1

    plt.ylim(0, 1)
    plt.legend(['control F0', 'inter F0', 'intra F0'])
    plt.ylabel('Proportion of hits')
    plt.title('Proportion of hits across all ferrets')
    plt.show()

    fig, ax = plt.subplots(layout='constrained')

    color_map = plt.cm.get_cmap('tab10')  # Choose a colormap

    for attribute, measurement in stats_dict_all_combined.items():
        for talker, measurement_data in measurement.items():
            print(measurement_data)
            if multiplier < 3:
                offset = width * multiplier
            else:
                offset = (gap_width) + (width * multiplier)  # Add gap offset for the second series

            color = color_map(
                np.where(np.array(list(measurement.keys())) == talker)[0][0])  # Assign color based on label
            rects = ax.bar(offset, measurement_data['false_alarms'], width, label=talker, color=color)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 'x', 'v', 's', 'p']
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute][talker]['false_alarms'].items():
                #add jitter to offset
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax.scatter(offset_jitter, ferret_data, color=color, marker=marker_list[count], edgecolors='black')
                count += 1

            multiplier += 1

    plt.ylim(0, 1)
    plt.legend()
    plt.ylabel('Proportion of hits')
    plt.title('Proportion of hits across all ferrets')
    plt.show()


    fig, ax = plt.subplots(layout='constrained')

    color_map = plt.cm.get_cmap('tab10')  # Choose a colormap

    for attribute, measurement in stats_dict_all_combined.items():
        for talker, measurement_data in measurement.items():
            print(measurement_data)
            if multiplier < 3:
                offset = width * multiplier
            else:
                offset = (gap_width) + (width * multiplier)  # Add gap offset for the second series

            color = color_map(
                np.where(np.array(list(measurement.keys())) == talker)[0][0])  # Assign color based on label
            rects = ax.bar(offset, measurement_data['correct_rejections'], width, label=talker, color=color)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 'x', 'v', 's', 'p']
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute][talker]['correct_rejections'].items():
                #add jitter to offset
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax.scatter(offset_jitter, ferret_data, color=color, marker=marker_list[count], edgecolors='black')
                count += 1

            multiplier += 1

    plt.ylim(0, 1)
    plt.legend()
    plt.ylabel('Proportion of correct rejections')
    plt.title('Proportion of correct rejections across all ferrets')
    plt.show()







    #get proportion of hits and false alarms for the dataframe
    fig, ax = plt.subplots()





if __name__ == '__main__':
    stats_dict = {}
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    df = behaviouralhelperscg.get_stats_df(ferrets=ferrets, startdate='04-01-2016', finishdate='01-03-2023')

    pitch_type_list = ['control_trial', 'inter_trial_roving', 'intra_trial_roving']
    stats_dict_all_combined = {}
    stats_dict_combined = {}
    stats_dict_all_combined[1] = {}
    stats_dict_all_combined[2] = {}
    stats_dict_combined[1] = {}
    stats_dict_combined[2] = {}



    for pitch in pitch_type_list:
        stats_dict_all, stats_dict = run_stats_calc(df, ferrets, stats_dict, pitch_param=pitch)
        #append to dataframe

        stats_dict_all_combined[1][pitch] = stats_dict_all[1][pitch]
        stats_dict_all_combined[2][pitch] = stats_dict_all[2][pitch]

        stats_dict_combined[1][pitch] = stats_dict[1][pitch]
        stats_dict_combined[2][pitch] = stats_dict[2][pitch]

    plot_stats(stats_dict_all_combined, stats_dict_combined)

