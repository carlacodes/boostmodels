import sklearn.metrics
import seaborn as sns
from instruments.io.BehaviourIO import BehaviourDataSet
from sklearn.inspection import permutation_importance
from instruments.behaviouralAnalysis import reactionTimeAnalysis  # outputbehaviordf
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import shap
import matplotlib
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn
from sklearn.model_selection import train_test_split
from helpers.behaviouralhelpersformodels import *
# os.environ['R_HOME'] = 'C:/PROGRA~1/R/R-41~1.3/bin/x64/R'

def cli_reaction_time(path=None,
                      output=None,
                      ferrets=None,
                      startdate=None,
                      finishdate=None):
    if output is None:
        output = behaviourOutput

    if path is None:
        path = behaviouralDataPath

    dataSet = BehaviourDataSet(filepath=path,
                               startDate=startdate,
                               finishDate=finishdate,
                               ferrets=ferrets,
                               outDir=output)

    allData = dataSet._load()
    # for ferret in ferrets:
    ferret = ferrets
    ferrData = allData.loc[allData.ferretname == ferret]
    # if ferret == 'F1702_Zola':
    #     ferrData = ferrData.loc[(ferrData.dates != '2021-10-04 10:25:00')]

    ferretFigs = reactionTimeAnalysis(ferrData)
    dataSet._save(figs=ferretFigs, file_name='reaction_times_{}_{}_{}.pdf'.format(ferret, startdate, finishdate))




# editing to extract different vars from df
def plotpredictedversusactual(predictedrelease, dfuse):
    fig, ax = plt.subplots()
    ax.scatter(dfuse['realRelReleaseTimes'], predictedrelease, alpha=0.5)
    ax.set_xlabel('Actual Release Time')
    ax.set_ylabel('Predicted Release Time (s)')
    ax.set_title('Predicted vs. Actual Release Time (s)')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    plt.show()
    fig, ax = plt.subplots()
    ax.scatter(dfuse['realRelReleaseTimes'], dfuse['realRelReleaseTimes'] - predictedrelease, alpha=0.5)
    ax.set_xlabel('Actual Release Time (s)')
    ax.set_ylabel('Actual - Predicted Release Time (s)')
    ax.set_title('Actual - Predicted Release Time (s)')
    ax.plot([0, 1], [0, 0], transform=ax.transAxes)
    plt.show()


def plotpredictedversusactualcorrectresponse(predictedcorrectresp, dfcat_use):
    fig, ax = plt.subplots()
    ax.scatter(dfcat_use['correctresp'], predictedcorrectresp, alpha=0.5)
    ax.set_xlabel('Actual Correct Response')
    ax.set_ylabel('Predicted Correct Response')
    ax.set_title('Predicted vs. Actual Correct Response')
    ax.plot([0, 1], [0, 0], transform=ax.transAxes)
    plt.show()
    # np round down to the nearest integer
    cm = sklearn.metrics.confusion_matrix(dfcat_use['correctresp'], np.round(predictedcorrectresp))
    sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=['Incorrect', 'Correct']).plot()
    accuracy = sklearn.metrics.accuracy_score(dfcat_use['correctresp'], np.round(predictedcorrectresp))
    plt.show()
    print(accuracy)


def runxgboostreleasetimes(df_use):
    col = 'realRelReleaseTimes'
    dfx = df_use.loc[:, df_use.columns != col]
    # remove ferret as possible feature
    col = 'ferret'

    dfx = dfx.loc[:, dfx.columns != col]
    dfx['pitchoftarg'] = dfx['pitchoftarg'].astype('category')
    dfx['side'] = dfx['side'].astype('category')
    dfx['talker'] = dfx['talker'].astype('category')
    dfx['stepval'] = dfx['stepval'].astype('category')
    dfx['pitchofprecur'] = dfx['pitchofprecur'].astype('category')
    dfx['AM'] = dfx['AM'].astype('category')
    dfx['DaysSinceStart'] = dfx['DaysSinceStart'].astype('category')
    dfx['precur_and_targ_same'] = dfx['precur_and_targ_same'].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(dfx, df_use['realRelReleaseTimes'], test_size=0.2,
                                                        random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    xg_reg = xgb.XGBRegressor(tree_method='gpu_hist', colsample_bytree=0.3,
                              learning_rate=0.1,
                              max_depth=10, alpha=10, n_estimators=10, enable_categorical=True)
    xg_reg.fit(X_train, y_train)
    ypred = xg_reg.predict(X_test)
    xgb.plot_importance(xg_reg)
    plt.show()

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    results = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)

    mse = mean_squared_error(ypred, y_test)
    print("MSE: %.2f" % (mse))
    print("negative MSE: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(X_train)
    shap.summary_plot(shap_values, X_train)
    plt.show()
    return xg_reg, ypred, y_test, results



def objective(trial, X, y):
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1),
        "alpha": trial.suggest_float("alpha", 1, 20),
        "is_unbalanced": trial.suggest_categorical("is_unbalanced", [True]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=10),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=2),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=2),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 20, step=1),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(objective="binary", random_state=123, **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "binary_logloss")
            ],  # Add a pruning callback
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = sklearn.metrics.log_loss(y_test, preds)

    return np.mean(cv_scores)


def run_optuna_study_falsealarm(dataframe, y, ferret_as_feature = False):
    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")\
    if ferret_as_feature:
        df_to_use = dataframe[
            ["cosinesim", "ferret", "pitchofprecur", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
             "falsealarm", "pastcorrectresp", "temporalsim", "pastcatchtrial", "trialNum", "targTimes", ]]
    else:
        df_to_use = dataframe[
            ["cosinesim", "pitchofprecur", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
             "falsealarm", "pastcorrectresp", "temporalsim", "pastcatchtrial", "trialNum", "targTimes", ]]


    col = 'falsealarm'
    X = df_to_use.loc[:, df_to_use.columns != col]
    X = X.to_numpy()
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=1000)
    print("Number of finished trials: ", len(study.trials))
    print(f"\tBest value of binary log loss: {study.best_value:.5f}")
    print(f"\tBest params:")

    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")
    return study


def run_optuna_study_correctresponse(dataframe, y):
    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    df_to_use = dataframe[
        ["cosinesim", "pitchofprecur", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
         "falsealarm", "pastcorrectresp", "temporalsim", "pastcatchtrial", "trialNum", "targTimes", ]]

    col = 'correctesp'
    X = df_to_use.loc[:, df_to_use.columns != col]
    X = X.to_numpy()
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=1000)
    print("Number of finished trials: ", len(study.trials))
    print(f"\tBest value of binary log loss: {study.best_value:.5f}")
    print(f"\tBest params:")

    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")
    return study


def runlgbfaornotwithoptuna(dataframe, paramsinput, ferret_as_feature = False):
    if ferret_as_feature:
        df_to_use = dataframe[
            ["targTimes", "ferret", "trialNum",  "pitchofprecur", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
             "falsealarm", "temporalsim", "pastcorrectresp", "pastcatchtrial", "cosinesim",]]
    else:
        df_to_use = dataframe[
            ["targTimes", "trialNum", "pitchofprecur", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
             "falsealarm", "temporalsim", "pastcorrectresp", "pastcatchtrial", "cosinesim", ]]

    col = 'falsealarm'
    dfx = df_to_use.loc[:, df_to_use.columns != col]
    # remove ferret as possible feature
    X_train, X_test, y_train, y_test = train_test_split(dfx, df_to_use['falsealarm'], test_size=0.2, random_state=123)
    print(X_train.shape)
    print(X_test.shape)
    # ran optuna study 06/03/2022 to find best params, balanced accuracy 57%, accuracy 63%
    # paramsinput = {'colsample_bytree': 0.7024634011442671, 'alpha': 15.7349076305505, 'is_unbalanced': True,
    #                'n_estimators': 6900, 'learning_rate': 0.3579458041084967, 'num_leaves': 1790, 'max_depth': 4,
    #                'min_data_in_leaf': 200, 'lambda_l1': 0, 'lambda_l2': 24, 'min_gain_to_split': 2.34923345270416,
    #                'bagging_fraction': 0.9, 'bagging_freq': 12, 'feature_fraction': 0.9}

    xg_reg = lgb.LGBMClassifier(objective="binary", random_state=123,
                                **paramsinput)

    xg_reg.fit(X_train, y_train, eval_metric="cross_entropy_lambda", verbose=1000)
    ypred = xg_reg.predict_proba(X_test)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    results = cross_val_score(xg_reg, X_test, y_test, scoring='accuracy', cv=kfold)
    bal_accuracy = cross_val_score(xg_reg, X_test, y_test, scoring='balanced_accuracy', cv=kfold)
    print("Accuracy: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    print('Balanced Accuracy: %.2f%%' % (np.mean(bal_accuracy) * 100.0))

    plotfalsealarmmodel(xg_reg, ypred, y_test, results, X_train, y_train, X_test, bal_accuracy, dfx)

    return xg_reg, ypred, y_test, results, X_train, y_train, X_test, bal_accuracy, dfx

def plotfalsealarmmodel(xg_reg, ypred, y_test, results, X_train, y_train, X_test, bal_accuracy, dfx):
    shap_values1 = shap.TreeExplainer(xg_reg).shap_values(X_train)
    plt.subplots(figsize=(25, 25))

    custom_colors = ['slategray',  'hotpink', "yellow"]  # Add more colors as needed
    # cmapcustom = matplotlib.colors.ListedColormap(custom_colors)
    cmapcustom = mcolors.LinearSegmentedColormap.from_list('my_custom_cmap', custom_colors, N=1000)
    custom_colors_summary = ['slategray', 'hotpink',]  # Add more colors as needed
    cmapsummary = matplotlib.colors.ListedColormap(custom_colors_summary)


    feature_importances = np.abs(shap_values1).sum(axis=0)

    # Calculate the cumulative sum of feature importances
    cumulative_importances_list = []
    for shap_values in shap_values1:
        feature_importances = np.abs(shap_values).sum(axis=0)
        cumulative_importances = np.cumsum(feature_importances)
        cumulative_importances_list.append(cumulative_importances)

    # Calculate the combined cumulative sum of feature importances
    cumulative_importances_combined = np.sum(cumulative_importances_list, axis=0)
    feature_labels = dfx.columns
    # Plot the elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(feature_labels, cumulative_importances_combined, marker='o', color = 'slategray')
    plt.xlabel('Features')
    plt.ylabel('Cumulative Feature Importance')
    plt.title('Elbow Plot of Cumulative Feature Importance for False Alarm Model')
    plt.xticks(rotation=45, ha='right')  # rotate x-axis labels for better readability
    plt.savefig('D:/behavmodelfigs/fa_or_not_model/elbowplot.png', dpi=500, bbox_inches='tight')
    plt.show()



    shap.summary_plot(shap_values1, dfx, show = False, color=cmapsummary)
    fig, ax = plt.gcf(), plt.gca()
    plt.title('Ranked list of features over their \n impact in predicting a false alarm', fontsize = 18)
    # Get the plot's Patch objects
    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
    labels[11] = 'time to target presentation'
    labels[10] = 'pitch of precursor'
    labels[9] = 'trial number'
    labels[8] = 'side of audio presentation'
    labels[7] = 'talker'
    labels[6] = 'past trial was catch'
    labels[5] = 'past trial was correct'
    labels[4] = 'intra_trial roving'
    labels[3] = 'AM session'
    labels[2] = 'cosine similarity'
    labels[1] = 'Day since start of week'
    labels[0] = 'temporal similarity'
    ax.set_yticklabels(labels)
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/fa_or_not_model/ranked_features.png', dpi=1000, bbox_inches = "tight")
    plt.show()

    shap.dependence_plot("pitchofprecur", shap_values1[0], X_train)  #
    plt.show()

    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=1000,
                                    random_state=123, n_jobs=2)
    importances = result.importances_mean



    #partial dependency plots
    explainer = shap.Explainer(xg_reg, dfx)
    shap_values2 = explainer(X_train)
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot the scatter plot with the colormap
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "intra_trial_roving"], cmap=cmapcustom)
    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(left=-10, right=0.5)

    shap.plots.scatter(shap_values2[:, "pitchofprecur"], color=shap_values2[:, "intra_trial_roving"], show=False,
                                 cmap=cmapcustom)
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Intra-trial roving", fontsize=12)


    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Pitch of the precursor word \n versus impact in false alarm probability', fontsize=18)
    plt.xticks([1,2,3,4,5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'], fontsize=15)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Pitch of precursor word', fontsize=16)
    plt.savefig('D:/behavmodelfigs/fa_or_not_model/pitchofprecurintratrialrove.png', dpi=500)
    plt.show()


    shap.plots.scatter(shap_values2[:, "targTimes"], color=shap_values2[:, "trialNum"], show=False, cmap =cmapcustom)
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Target Presentation Time", fontsize=12)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Target presentation time \n versus impact in false alarm probability', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Trial number', fontsize=16)
    plt.savefig('D:/behavmodelfigs/fa_or_not_model/targtimescolouredbytrialnumber.png', dpi=1000)
    plt.show()

    shap.plots.scatter(shap_values2[:, "targTimes"], color=shap_values2[:, "pitchofprecur"], show=False, cmap = cmapcustom)
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_yticks([1, 2, 3,4, 5])
    cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
    cb_ax.set_ylabel("Pitch of precursor", fontsize=12)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Target presentation \n versus impact in predicting a false alarm', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Target presentation time', fontsize=16)
    plt.savefig('D:/behavmodelfigs/fa_or_not_model/trialtime_colouredbyprecur.png', dpi=1000)
    plt.show()


    return xg_reg, ypred, y_test, results, shap_values1, X_train, y_train, bal_accuracy, shap_values2


def runlgbfaornot(dataframe):
    df_to_use = dataframe[
        ["cosinesim", "pitchofprecur", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
         "falsealarm", "pastcorrectresp", "pastcatchtrial", "trialNum", "targTimes", ]]

    col = 'falsealarm'
    dfx = df_to_use.loc[:, df_to_use.columns != col]
    # remove ferret as possible feature

    X_train, X_test, y_train, y_test = train_test_split(dfx, df_to_use['falsealarm'], test_size=0.2, random_state=123)
    print(X_train.shape)
    print(X_test.shape)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test)
    params2 = {"n_estimators": 9300,
               "is_unbalanced": True,
               "colsample_bytree": 0.8163174226131737,
               "alpha": 4.971464509571637,
               "learning_rate": 0.2744671988597753,
               "num_leaves": 530,
               "max_depth": 15,
               "min_data_in_leaf": 400,
               "lambda_l1": 2,
               "lambda_l2": 44,
               "min_gain_to_split": 0.008680941888662716,
               "bagging_fraction": 0.9,
               "bagging_freq": 1,
               "feature_fraction": 0.6000000000000001}

    xg_reg = lgb.LGBMClassifier(objective="binary", random_state=123,
                                **params2)

    xg_reg.fit(X_train, y_train, eval_metric="cross_entropy_lambda", verbose=1000)
    ypred = xg_reg.predict_proba(X_test)

    kfold = KFold(n_splits=3, shuffle=True, random_state=123)
    results = cross_val_score(xg_reg, X_test, y_test, scoring='accuracy', cv=kfold)
    bal_accuracy = cross_val_score(xg_reg, X_test, y_test, scoring='balanced_accuracy', cv=kfold)
    print("Accuracy: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    print('Balanced Accuracy: %.2f%%' % (np.mean(bal_accuracy) * 100.0))

    shap_values1 = shap.TreeExplainer(xg_reg).shap_values(dfx)
    plt.subplots(figsize=(25, 25))
    shap.summary_plot(shap_values1, dfx, show=False)
    fig, ax = plt.gcf(), plt.gca()
    plt.title('Ranked list of features over their \n impact in predicting a false alarm', fontsize = 18)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
    labels[11] = 'time to target presentation'
    labels[10] = 'pitch of precursor'
    labels[9] = 'trial number'
    labels[8] = 'past trial was catch'
    labels[7] = 'side of audio presentation'
    labels[6] = 'talker'
    labels[5] = 'past trial was correct'
    labels[4] = 'AM'
    labels[3] = 'intra-trial roving'
    labels[2] = 'day since start of experiment week'
    labels[1] = 'cosine similarity'
    labels[0] = 'temporal similarity'
    ax.set_yticklabels(labels)
    fig.tight_layout()


    plt.savefig('D:/behavmodelfigs/fa_or_not_model/ranked_features.png', dpi=1000, bbox_inches = "tight")
    plt.show()

    shap.dependence_plot("pitchofprecur", shap_values1[0], dfx)  #
    plt.show()
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=10,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(20, 15))
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T)
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/permutation_importance.png', dpi=500)
    plt.show()
    explainer = shap.Explainer(xg_reg, dfx)
    shap_values2 = explainer(X_train)
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "intra_trial_roving"])
    fig.tight_layout()
    plt.tight_layout()
    plt.subplots_adjust(left=-10, right=0.5)

    plt.show()
    shap.plots.scatter(shap_values2[:, "pitchofprecur"], color=shap_values2[:, "talker"])
    plt.show()

    shap.plots.scatter(shap_values2[:, "pitchofprecur"], color=shap_values2[:, "intra_trial_roving"], show=False)
    plt.show()

    shap.plots.scatter(shap_values2[:, "intra_trial_roving"], color=shap_values2[:, "talker"])
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "talker"], show=False)
    plt.title('False alarm model - trial number as a function of SHAP values, coloured by talker')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "intra_trial_roving"], show=False)
    plt.title('False alarm model - SHAP values as a function of cosine similarity \n, coloured by intra trial roving')
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/cosinesimdepenencyplot.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "intra_trial_roving"], color=shap_values2[:, "cosinesim"], show=False)
    plt.savefig('D:/behavmodelfigs/intratrialrovingcosinecolor.png', dpi=500)

    plt.show()
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "pitchoftarg"], show=False)
    plt.title('False alarm model - trial number as a function of SHAP values, coloured by pitch of target')
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/trialnumcosinecolor.png', dpi=500)
    plt.show()

    fig, ax = plt.subplots(figsize=(18, 15))
    shap.plots.scatter(shap_values2[:, "targTimes"], color=shap_values2[:, "cosinesim"], show=False)
    plt.title('shap values for FA model as a function of target times, coloured by cosine similarity')
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/targtimescosinecolor.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "targTimes"], show=False)
    plt.title('Cosine Similarity as a function of SHAP values, coloured by targTimes')
    plt.savefig('D:/behavmodelfigs/cosinesimtargtimes.png', dpi=500)
    plt.show()

    return xg_reg, ypred, y_test, results, shap_values1, X_train, y_train, bal_accuracy, shap_values2


def runfalsealarmpipeline(ferrets, optimization = False, ferret_as_feature = False ):
    resultingfa_df = behaviouralhelperscg.get_false_alarm_behavdata(ferrets=ferrets, startdate='04-01-2020',
                                                                    finishdate='01-03-2023')
    len_of_data_male = {}
    len_of_data_female = {}
    len_of_data_female_intra = {}
    len_of_data_female_inter = {}
    len_of_data_male_intra = {}
    len_of_data_male_inter = {}
    for i in range(0, len(ferrets)):
        noncorrectiondata = resultingfa_df[resultingfa_df['correctionTrial'] == 0]
        noncorrectiondata = noncorrectiondata[noncorrectiondata['currAtten'] == 0]

        len_of_data_male[ferrets[i]] = len(noncorrectiondata[(noncorrectiondata['ferret'] == i) & (noncorrectiondata['talker'] == 2.0) & (noncorrectiondata['control_trial'] == 1)])
        len_of_data_female[ferrets[i]] = len(noncorrectiondata[(noncorrectiondata['ferret'] == i) & (noncorrectiondata['talker'] == 1.0) & (noncorrectiondata['control_trial'] == 1)])

        interdata = noncorrectiondata[noncorrectiondata['inter_trial_roving'] == 1]
        intradata = noncorrectiondata[noncorrectiondata['intra_trial_roving'] == 1]
        len_of_data_female_intra[ferrets[i]] = len(intradata[(intradata['ferret'] == i) & (intradata['talker'] == 1.0)])
        len_of_data_female_inter[ferrets[i]] = len(interdata[(interdata['ferret'] == i) & (interdata['talker'] == 1.0)])

        len_of_data_male_inter[ferrets[i]] = len(interdata[(interdata['ferret'] == i) & (interdata['talker'] == 2.0)])
        len_of_data_male_intra[ferrets[i]] = len(intradata[(intradata['ferret'] == i ) & (intradata['talker'] == 2.0)])


    filepath = Path('D:/dfformixedmodels/falsealarmmodel_dfuse.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)



    if optimization == False:
        #load the saved params
        params = np.load('optuna_results/falsealarm_optunaparams.npy', allow_pickle=True).item()
    else:
        study = run_optuna_study_falsealarm(resultingfa_df, resultingfa_df['falsealarm'].to_numpy(), ferret_as_feature = ferret_as_feature)
        print(study.best_params)
        params = study.best_params
        np.save('optuna_results/falsealarm_optunaparams.npy', study.best_params)

    resultingfa_df.to_csv(filepath)
    xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runlgbfaornotwithoptuna(
        resultingfa_df, params, ferret_as_feature = ferret_as_feature)
    return xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2


def run_reaction_time_fa_pipleine_female(ferrets):
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    df_use = resultingdf.loc[:, resultingdf.columns != 'ferret']
    df_use = df_use.loc[df_use['intra_trial_roving'] == 0]
    df_use = df_use.loc[df_use['talker'] == 1]
    df_use = df_use.loc[:, df_use.columns != 'targTimes']
    df_use = df_use.loc[:, df_use.columns != 'stepval']
    df_use = df_use.loc[:, df_use.columns != 'side']
    df_use = df_use.loc[:, df_use.columns != 'AM']
    df_use = df_use.loc[:, df_use.columns != 'distractor_or_fa']

    df_use = df_use.loc[:, df_use.columns != 'realRelReleaseTimes']

    col = 'centreRelease'
    dfx = df_use.loc[:, df_use.columns != col]
    # remove ferret as possible feature
    col = 'ferret'
    col2 = ['target', 'startResponseTime', 'distractors', 'recBlock', 'lickRelease2', 'lickReleaseCount',
            'PitchShiftMat', 'attenOrder', 'dDurs', 'tempAttens', 'currAttenList', 'attenList', 'fName', 'Level',
            'dates', 'ferretname', 'noiseType', 'noiseFile']
    dfx = dfx.loc[:, dfx.columns != col]
    # for name in col2:
    #     dfx = dfx.loc[:, dfx.columns != name]
    for column in dfx.columns:
        if column == 'AM' or column == 'side':
            pass
        elif column.isnumeric() == False:
            dfx = dfx.loc[:, dfx.columns != column]
        elif column.isnumeric():
            pass

    X_train, X_test, y_train, y_test = train_test_split(dfx, df_use['centreRelease'], test_size=0.2,
                                                        random_state=123)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test)

    param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    # bst = xgb.train(param, dtrain, num_round, evallist)
    xg_reg = lgb.LGBMRegressor(colsample_bytree=0.3, learning_rate=0.1,
                               max_depth=10, alpha=10, n_estimators=10, verbose=1)

    xg_reg.fit(X_train, y_train, eval_metric='neg_mean_squared_error', verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
    plt.title('feature importances for the LGBM release times model (both hits and false alarms)')
    plt.show()
    kfold = KFold(n_splits=10)
    results = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
    mse_test = mean_squared_error(ypred, y_test)
    print('mse test for female talker model: ', mse_test)
    print('mse train for female talker model: ', results.mean())

    shap_values1 = shap.TreeExplainer(xg_reg).shap_values(X_train)
    fig, ax = plt.subplots(figsize=(15, 65))
    shap.summary_plot(shap_values1, X_train, show=False)
    plt.title('Ranked list of features over their \n impact in predicting reaction time, female talker', fontsize=18)
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/ranked_features_rxntimealarmhitmodel.png', dpi=500)
    plt.show()

    return resultingdf


def run_reaction_time_fa_pipleine_male(ferrets):
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    df_use = resultingdf.loc[:, resultingdf.columns != 'ferret']
    df_use = df_use.loc[df_use['intra_trial_roving'] == 0]
    df_use = df_use.loc[df_use['talker'] == 2]
    df_use = df_use.loc[:, df_use.columns != 'targTimes']
    df_use = df_use.loc[:, df_use.columns != 'stepval']
    df_use = df_use.loc[:, df_use.columns != 'side']
    df_use = df_use.loc[:, df_use.columns != 'AM']

    df_use = df_use.loc[:, df_use.columns != 'distractor_or_fa']
    df_use = df_use.loc[:, df_use.columns != 'realRelReleaseTimes']

    col = 'centreRelease'
    dfx = df_use.loc[:, df_use.columns != col]
    col = 'ferret'
    col2 = ['target', 'startResponseTime', 'distractors', 'recBlock', 'lickRelease2', 'lickReleaseCount',
            'PitchShiftMat', 'attenOrder', 'dDurs', 'tempAttens', 'currAttenList', 'attenList', 'fName', 'Level',
            'dates', 'ferretname', 'noiseType', 'noiseFile']
    dfx = dfx.loc[:, dfx.columns != col]

    for column in dfx.columns:
        if column == 'AM' or column == 'side':
            pass
        elif column.isnumeric() == False:
            dfx = dfx.loc[:, dfx.columns != column]
        elif column.isnumeric():
            pass

    X_train, X_test, y_train, y_test = train_test_split(dfx, df_use['centreRelease'], test_size=0.2,
                                                        random_state=123)

    param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    xg_reg = lgb.LGBMRegressor(colsample_bytree=0.3, learning_rate=0.1,
                               max_depth=10, alpha=10, n_estimators=10, verbose=1)

    xg_reg.fit(X_train, y_train, eval_metric='neg_mean_squared_error', verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
    plt.title('feature importances for the LGBM release times model (both hits and false alarms)')
    plt.show()
    kfold = KFold(n_splits=10)
    results = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
    mse_test = mean_squared_error(ypred, y_test)
    print('mse test: ', mse_test)
    print('mse train: ', results.mean())

    shap_values1 = shap.TreeExplainer(xg_reg).shap_values(X_train)
    fig, ax = plt.subplots(figsize=(15, 65))
    shap.summary_plot(shap_values1, X_train, show=False)
    plt.title('Ranked list of features over their \n impact in predicting reaction time, male talker', fontsize=18)
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/ranked_features_rxntimealarmhitmodel_male.png', dpi=500)
    plt.show()

    return resultingdf


def plot_correct_response_byside(ferrets):
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    df_use = resultingdf
    df_use = df_use.loc[df_use['intra_trial_roving'] == 0]

    # plot the proportion of correct responses by side

    df_left = df_use.loc[df_use['side'] == 0]
    df_right = df_use.loc[df_use['side'] == 1]
    ax, fig = plt.subplots()
    plt.bar(['left', 'right'], [df_left['correct'].mean(), df_right['correct'].mean()])
    plt.title('Proportion of correct responses by side registered by sensors, \n irrespective of talker and ferret')
    plt.ylabel('proportion of correct responses')
    plt.xlabel('side of the auditory stimulus')
    plt.ylim(0, 1)
    plt.savefig('D:/behavmodelfigs/proportion_correct_responses_by_side.png', dpi=500)

    plt.show()
    df_left_by_ferret = {}
    df_right_by_ferret = {}

    # now plot by ferret ID
    ferrets = [0, 1, 2, 3, 4]
    for ferret in ferrets:
        df_left_test = df_use.loc[df_use['side'] == 0]
        df_right_test = df_use.loc[df_use['side'] == 1]
        df_left_by_ferret[ferret] = df_left_test.loc[df_left_test['ferret'] == ferret]
        df_right_by_ferret[ferret] = df_right_test.loc[df_right_test['ferret'] == ferret]

    ax, fig = plt.subplots(figsize=(10, 12))
    plt.bar(['left - F1702', 'right - F1702', 'left - 1815', 'right - F1815', 'left - F1803', 'right-F1803', 'left - F2002',
             'right- F2002', 'left - F2105', 'right - F2105'], [df_left_by_ferret[0]['correct'].mean(), df_right_by_ferret[0]['correct'].mean(),
                            df_left_by_ferret[1]['correct'].mean(), df_right_by_ferret[1]['correct'].mean(),
                            df_left_by_ferret[2]['correct'].mean(), df_right_by_ferret[2]['correct'].mean(),
                            df_left_by_ferret[3]['correct'].mean(), df_right_by_ferret[3]['correct'].mean(),
                            df_left_by_ferret[4]['correct'].mean(), df_right_by_ferret[4]['correct'].mean()])

    plt.title('Proportion of correct responses by side registered by sensors, \n  irrespective of talker, by ferret ID',
              fontsize=15)
    plt.xticks(rotation=45, fontsize=12)  # rotate the x axis labels
    plt.ylim(0, 1)

    plt.ylabel('proportion of correct responses', fontsize=13)
    plt.savefig('D:/behavmodelfigs/proportion_correct_responses_by_side_by_ferret.png', dpi=1000)
    plt.show()
    return df_left, df_right


def plot_reaction_times_intra(ferrets):
    # plot the reaction times by animal
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    df_use = resultingdf

    df_left_by_ferret = {}
    df_female = df_use.loc[df_use['talker'] == 1]
    df_female_control = df_female.loc[df_female['control_trial'] == 1]

    df_female = df_use.loc[df_use['talker'] == 1]
    df_female_rove = df_female.loc[df_female['intra_trial_roving'] == 1]

    df_male = df_use.loc[df_use['talker'] == 2]
    df_male_control = df_male.loc[df_male['control_trial'] == 1]
    df_male_rove = df_male.loc[df_male['intra_trial_roving'] == 1]

    # now plot generally by all ferrets
    ax, fig = plt.subplots(figsize=(10, 12))
    sns.distplot(df_use['realRelReleaseTimes'], hist=True, kde=False, color='blue',
                 hist_kws={'edgecolor': 'black'})
    plt.title('Distribution of reaction times, \n irrespective of talker and ferret', fontsize=15)
    plt.show()

    # now plot by talker, showing reaction times
    sns.distplot(df_female_control['realRelReleaseTimes'], color='blue', label='control F0')
    sns.distplot(df_female_rove['realRelReleaseTimes'], color='red', label='intra-roved F0')
    plt.title('Reaction times for the female talker, \n irrespective of ferret', fontsize=15)
    plt.legend(fontsize=10)
    plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_female.png', dpi=500)
    plt.show()

    sns.distplot(df_male_control['realRelReleaseTimes'], color='green', label='control F0')
    sns.distplot(df_male_rove['realRelReleaseTimes'], color='orange', label='intra-roved F0')
    plt.title('Reaction times for the male talker, \n irrespective of ferret', fontsize=15)
    plt.legend(fontsize=10)
    plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_male.png', dpi=500)
    plt.show()

    df_by_ferret = {}
    df_by_ferret_f_control = {}
    df_by_ferret_f_rove = {}
    df_by_ferret_m_control = {}
    df_by_ferret_m_rove = {}

    # now plot by ferret ID
    ferrets = [0, 1, 2, 3, 4]
    for ferret in ferrets:
        df_by_ferret_f_control[ferret] = df_female_control.loc[df_female_control['ferret'] == ferret]
        df_by_ferret_f_rove[ferret] = df_female_rove.loc[df_female_rove['ferret'] == ferret]
        df_by_ferret_m_control[ferret] = df_male_rove.loc[df_male_rove['ferret'] == ferret]
        df_by_ferret_m_rove[ferret] = df_male_control.loc[df_male_control['ferret'] == ferret]
    ferret_labels = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    for ferret in ferrets:
        sns.distplot(df_by_ferret_f_control[ferret]['realRelReleaseTimes'], color='blue', label='control F0, female')
        sns.distplot(df_by_ferret_f_rove[ferret]['realRelReleaseTimes'], color='red', label='intra-trial roved F0, female')
        sns.distplot(df_by_ferret_m_control[ferret]['realRelReleaseTimes'], color='green', label='control F0, male')
        sns.distplot(df_by_ferret_m_rove[ferret]['realRelReleaseTimes'], color='orange', label='intra-trial roved F0, male')
        plt.title('Reaction times for ferret ID ' + str(ferret_labels[ferret]), fontsize=15)
        plt.legend(fontsize=10)
        plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
        plt.savefig('D:/behavmodelfigs/reaction_times_by_ferret_' + str(ferret_labels[ferret]) + '.png', dpi=1000)
        plt.show()

    return df_by_ferret

    #
def plot_reaction_times_interandintra(ferrets):
    # plot the reaction times by animal
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    df_use = resultingdf

    df_left_by_ferret = {}
    df_female = df_use.loc[df_use['talker'] == 1]
    df_female_control = df_female.loc[df_female['control_trial'] == 1]

    df_female = df_use.loc[(df_use['talker'] == 1) | (df_use['talker'] == 3) | (df_use['talker'] == 5)]
    df_female_rove = df_female.loc[df_female['inter_trial_roving'] == 1]
    df_female_rove_intra = df_female.loc[df_female['intra_trial_roving'] == 1]


    df_male = df_use.loc[(df_use['talker'] == 2) | (df_use['talker'] == 8) | (df_use['talker'] == 13)]
    df_male_control = df_male.loc[df_male['control_trial'] == 1]
    df_male_rove = df_male.loc[df_male['inter_trial_roving'] == 1]
    df_male_rove_intra = df_male.loc[df_male['intra_trial_roving'] == 1]

    # now plot generally by all ferrets
    ax, fig = plt.subplots(figsize=(10, 12))
    sns.distplot(df_use['realRelReleaseTimes'], hist=True, kde=False, color='blue',
                 hist_kws={'edgecolor': 'black'})
    plt.title('Distribution of reaction times, \n irrespective of talker and ferret', fontsize=15)
    plt.show()

    # now plot by talker, showing reaction times
    sns.distplot(df_female_control['realRelReleaseTimes'], color='blue', label='control F0')
    sns.distplot(df_female_rove['realRelReleaseTimes'], color='red', label='inter-roved F0')
    sns.distplot(df_female_rove_intra['realRelReleaseTimes'], color='darkmagenta', label='intra-roved F0')

    plt.title('Reaction times for the female talker, \n irrespective of ferret', fontsize=15)
    plt.legend(fontsize=10)
    plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_female_interandintra.png', dpi=500)

    plt.show()

    sns.distplot(df_male_control['realRelReleaseTimes'], color='green', label='control F0')
    sns.distplot(df_male_rove['realRelReleaseTimes'], color='orange', label='inter-roved F0')
    sns.distplot(df_male_rove_intra['realRelReleaseTimes'], color='red', label='intra-roved F0')

    plt.title('Reaction times for the male talker, \n irrespective of ferret', fontsize=15)
    plt.legend(fontsize=10)
    plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_male_interandintra.png', dpi=500)
    plt.show()

    df_by_ferret = {}
    df_by_ferret_f_control = {}
    df_by_ferret_f_rove = {}
    df_by_ferret_m_control = {}
    df_by_ferret_m_rove = {}
    df_by_ferret_f_rove_intra={}
    df_by_ferret_m_rove_intra = {}
    # now plot by ferret ID
    ferrets = [0, 1, 2, 3]
    for ferret in ferrets:
        df_by_ferret_f_control[ferret] = df_female_control.loc[df_female_control['ferret'] == ferret]
        df_by_ferret_f_rove[ferret] = df_female_rove.loc[df_female_rove['ferret'] == ferret]
        df_by_ferret_f_rove_intra[ferret] = df_female_rove_intra.loc[df_female_rove_intra['ferret'] == ferret]


        df_by_ferret_m_control[ferret] = df_male_rove.loc[df_male_rove['ferret'] == ferret]
        df_by_ferret_m_rove[ferret] = df_male_control.loc[df_male_control['ferret'] == ferret]
        df_by_ferret_m_rove_intra[ferret] = df_male_rove_intra.loc[df_male_rove_intra['ferret'] == ferret]

    ferret_labels = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni']
    for ferret in ferrets:
        sns.distplot(df_by_ferret_f_control[ferret]['realRelReleaseTimes'], color='blue', label='control F0, female')
        sns.distplot(df_by_ferret_f_rove[ferret]['realRelReleaseTimes'], color='red', label='inter-roved F0, female')
        sns.distplot(df_by_ferret_m_control[ferret]['realRelReleaseTimes'], color='green', label='control F0, male')
        sns.distplot(df_by_ferret_m_rove[ferret]['realRelReleaseTimes'], color='orange', label='inter-roved F0, male')
        sns.distplot(df_by_ferret_f_rove_intra[ferret]['realRelReleaseTimes'], color = 'darkmagenta', label = 'intra-roved F0, female')
        sns.distplot(df_by_ferret_m_rove_intra[ferret]['realRelReleaseTimes'], color = 'orangered', label = 'intra-roved F0, male')


        plt.title('Reaction times for ferret ID ' + str(ferret_labels[ferret]), fontsize=15)
        plt.legend(fontsize=10)
        plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
        plt.savefig('D:/behavmodelfigs/reaction_times_by_ferret_inter' + str(ferret_labels[ferret]) + '.png', dpi=500)
        plt.show()

    return df_by_ferret

if __name__ == '__main__':
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'] #'F2105_Clove'
    # df_by_ferretdict = plot_reaction_times(ferrets)
    # #
    # plot_reaction_times_interandintra(ferrets)
    #
    # df_left, df_right = plot_correct_response_byside(ferrets)
    # #
    # # test_df = run_reaction_time_fa_pipleine(ferrets)
    # #
    # # test_df2 = run_reaction_time_fa_pipleine_male(ferrets)
    #
    xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runfalsealarmpipeline(ferrets, optimization= False, ferret_as_feature = True)
    #

    # plot_reaction_times_intra(ferrets)


    # col3 = 'pitchofprecur'
    # dfx = dfx.loc[:, dfx.columns != col3]

    # col3 = 'stepval'
    # dfx = dfx.loc[:, dfx.columns != col3]

