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
from helpers.behaviouralhelpersformodels import *\


def main():
    ferrets = ['F2105_Clove', 'F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni']

    df = behaviouralhelperscg.get_stats_df(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-03-2023')

    #get proportion of hits and false alarms for the dataframe




if __name__ == '__main__':
    main()