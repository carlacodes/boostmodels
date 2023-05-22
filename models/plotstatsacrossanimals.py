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

def run_stats_calc(stats_dict, pitch_param = 'control_trial'):
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']

    df = behaviouralhelperscg.get_stats_df(ferrets=ferrets, startdate='04-01-2020',

                                                             finishdate='01-03-2023')
    df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    count = 0

    stats_dict[pitch_param]['hits'] = {}
    stats_dict[pitch_param]['false_alarms'] = {}
    stats_dict[pitch_param]['correct_rejections'] = {}
    for ferret in ferrets:

        selected_ferret = df_noncatchnoncorrection[df_noncatchnoncorrection['ferret'] == count]
        selected_ferret_catch = df_catchnoncorrection[df_catchnoncorrection['ferret'] == count]

        stats_dict[pitch_param]['hits'][ferret] = np.mean(selected_ferret['hit'])
        stats_dict[pitch_param]['false_alarms'][ferret] = np.mean(selected_ferret['falsealarm'])
        stats_dict[pitch_param]['correct_rejections'][ferret] = np.mean(selected_ferret_catch['response'] == 3)
        count += 1

    hits = np.mean(df_noncatchnoncorrection['hit'])
    false_alarms = np.mean(df_noncatchnoncorrection['falsealarm'])
    correct_rejections = np.mean(df_catchnoncorrection['response'] == 3)
    stats = CalculateStats.get_stats(df)


    #get proportion of hits and false alarms for the dataframe
    fig, ax = plt.subplots()
    sns.barplot(x='ferret', y='hits', hue='type', data=stats, ax=ax)

    #get proportion of hits and false alarms for the dataframe




if __name__ == '__main__':
    stats_dict = {}

    run_stats_calc(stats_dict, pitch_param='inter_trial_roving')