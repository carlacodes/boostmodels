import sklearn.metrics
# from rpy2.robjects import pandas2ri
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import matplotlib.font_manager as fm

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

def get_axis_limits(ax, scale=1):
    return ax.get_xlim()[0] * scale, (ax.get_ylim()[1] * scale)
def run_optuna_study_releasetimes(X, y):
    study = optuna.create_study(direction="minimize", study_name="LGBM regressor")
    func = lambda trial: objective_releasetimes(trial, X, y)
    study.optimize(func, n_trials=1000)
    print("Number of finished trials: ", len(study.trials))
    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")
    return study

def objective_releasetimes(trial, X, y):
    # param_grid = {
    #     # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    # #     colsample_bytree = 0.3, learning_rate = 0.1,
    # # max_depth = 10, alpha = 10, n_estimators = 10, random_state = 42, verbose = 1
    #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.6),
    #     "alpha": trial.suggest_float("alpha", 5, 15),
    #     "n_estimators": trial.suggest_int("n_estimators", 2, 100, step=2),
    #     "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    #     "max_depth": trial.suggest_int("max_depth", 5, 20),
    #     "bagging_fraction": trial.suggest_float(
    #         "bagging_fraction", 0.1, 0.95, step=0.1
    #     ),
    #     "bagging_freq": trial.suggest_int("bagging_freq", 0, 30, step=1),
    # }
    param_grid = {
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "alpha": trial.suggest_float("alpha", 5, 15),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0, step=0.1),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 30),
        "lambda": trial.suggest_float("lambda", 0.0, 0.5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "subsample_for_bin": trial.suggest_int("subsample_for_bin", 100, 10000, step=100),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMRegressor(random_state=123, **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "l2")
            ],  # Add a pruning callback
        )
        preds = model.predict(X_test)
        cv_scores[idx] = sklearn.metrics.mean_squared_error(y_test, preds)

    return np.mean(cv_scores)
def runlgbreleasetimes_for_a_ferret(data, paramsinput=None, ferret=1, ferret_name='F1815_Cruella'
                                    ):
    data = data[data['ferret'] == ferret]
    col = 'realRelReleaseTimes'
    dfx = data.loc[:, data.columns != col]
    col = 'ferret'
    dfx = dfx.loc[:, dfx.columns != col]

    X_train, X_test, y_train, y_test = train_test_split(dfx, data['realRelReleaseTimes'], test_size=0.2,
                                                        random_state=123)

    xg_reg = lgb.LGBMRegressor(random_state=123, verbose=1, **paramsinput)
    xg_reg.fit(X_train, y_train, verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
    plt.title('feature importances for the LGBM Correct Release Times model for ferret ' + ferret_name)
    plt.show()

    kfold = KFold(n_splits=10)
    mse_train = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)

    mse_test = mean_squared_error(ypred, y_test)
    mse_test = cross_val_score(xg_reg, X_test, y_test, scoring='neg_mean_squared_error', cv=kfold)
    print("MSE on test: %.4f" % (mse_test) + ferret_name)
    print("negative MSE training: %.2f%%" % (np.mean(mse_train) * 100.0))
    print(mse_train)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(dfx)

    # cumulative_importances_list = []
    # for shap_values in shap_values1:
    #     feature_importances = np.abs(shap_values).sum(axis=0)
    #     cumulative_importances = np.cumsum(feature_importances)
    #     cumulative_importances_list.append(cumulative_importances)

    # Calculate the combined cumulative sum of feature importances
    # cumulative_importances_combined = np.sum(np.abs(shap_values), axis=0)
    # feature_labels = dfx.columns
    # # Plot the elbow plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(feature_labels, cumulative_importances_combined, marker='o', color = 'slategray')
    # plt.xlabel('Features')
    # plt.ylabel('Cumulative Feature Importance')
    # plt.title('Elbow Plot of Cumulative Feature Importance for False Alarm Model')
    # plt.xticks(rotation=45, ha='right')  # rotate x-axis labels for better readability
    # plt.savefig('D:/behavmodelfigs/fa_or_not_model/elbowplot.png', dpi=500, bbox_inches='tight')
    # plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    # title kwargs still does nothing so need this workaround for summary plots




    shap.summary_plot(shap_values, dfx, show=False)
    fig, ax = plt.gcf(), plt.gca()
    plt.title('Features over their impact \n on reaction time for ' + ferret_name)
    plt.xlabel('SHAP value (impact on model output) on reaction time' + ferret_name)

    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
    trainandtestaccuracy ={
        'ferret': ferret_name,
        'mse_test': mse_test,
        'mse_train': mse_train,
        'mean_mse_train': np.mean(mse_train),
    }
    np.save('metrics/modelmse' + ferret_name + '.npy', trainandtestaccuracy)
    # labels[11] = 'distance to sensor'
    # labels[10] = 'target F0'
    # labels[9] = 'trial number'
    # labels[8] = 'precursor = target F0'
    # labels[7] = 'male talker'
    # labels[6] = 'time until target'
    # labels[5] = 'target F0 - precursor F0'
    # labels[4] = 'day of week'
    # labels[3] = 'precursor F0'
    # labels[2] = 'past trial was catch'
    # labels[1] = 'trial took place in AM'
    # labels[0] = 'past trial was correct'

    ax.set_yticklabels(labels)
    plt.savefig('figs/shap_summary_plot_correct_release_times_' + ferret_name + '.png', dpi=300, bbox_inches='tight')

    plt.show()

    shap.dependence_plot("timeToTarget", shap_values, dfx)  #

    explainer = shap.Explainer(xg_reg, X_train)
    shap_values2 = explainer(X_train)
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "precur_and_targ_same"])
    fig.tight_layout()

    plt.subplots_adjust(left=-10, right=0.5)

    plt.show()
    shap.plots.scatter(shap_values2[:, "pitchoftarg"], color=shap_values2[:, "talker"])
    plt.title('Reaction Time Model')
    plt.show()
    # logthe release times
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "talker"],
                       title='Correct Responses - Reaction Time Model SHAP response \n vs. trial number for' + ferret_name)

    return xg_reg, ypred, y_test, mse_test



def runlgbreleasetimes(X, y, paramsinput=None, ferret_as_feature = False, one_ferret=False, ferrets=None):


    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                        random_state=42)

    # param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
    # param['nthread'] = 4
    # param['eval_metric'] = 'auc'

    from pathlib import Path
    if ferret_as_feature:
        if one_ferret:

            fig_savedir = Path('figs/correctrxntimemodel/ferret_as_feature/' + ferrets)
            if fig_savedir.exists():
                pass
            else:
                fig_savedir.mkdir(parents=True, exist_ok=True)
        else:
            fig_savedir = Path('../figs/correctrxntimemodel/ferret_as_feature')
    else:
        if one_ferret:

            fig_savedir = Path('figs/correctrxntimemodel/'+ ferrets)
            if fig_savedir.exists():
                pass
            else:
                fig_savedir.mkdir(parents=True, exist_ok=True)
        else:
            fig_savedir = Path('../figs/correctrxntimemodel/')

    xg_reg = lgb.LGBMRegressor(random_state=42, verbose=1, **paramsinput)
    # xg_reg = lgb.LGBMRegressor( colsample_bytree=0.3, learning_rate=0.1,
    #                           max_depth=10, alpha=10, n_estimators=10, random_state=42, verbose=1)


    xg_reg.fit(X_train, y_train, verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
    if one_ferret:
        plt.title('feature importances for the LGBM Correct Release Times model for' + ferrets)
    else:
        plt.title('feature importances for the LGBM Correct Release Times model')
    plt.show()

    kfold = KFold(n_splits=10)
    results = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
    mse_train = mean_squared_error(ypred, y_test)

    mse = mean_squared_error(ypred, y_test)
    print("MSE on test: %.4f" % (mse))
    print("negative MSE training: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(X)
    fig, ax = plt.subplots(figsize=(15, 15))
    # title kwargs still does nothing so need this workaround for summary plots
    cmapname = "viridis"



    feature_importances = np.abs(shap_values).sum(axis=0)
    sorted_indices = np.argsort(feature_importances)

    sorted_indices = sorted_indices[::-1]
    feature_importances = feature_importances[sorted_indices]
    feature_labels = X.columns[sorted_indices]
    cumulative_importances = np.cumsum(feature_importances)


    # Plot the elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(feature_labels, cumulative_importances, marker='o', color = 'cyan')
    plt.xlabel('Features')
    plt.ylabel('Cumulative Feature Importance')
    if one_ferret:
        plt.title('Elbow Plot of Cumulative Feature Importance \n for the Reaction Time Model \n for ' + ferrets, fontsize = 20)
    else:
        plt.title('Elbow Plot of Cumulative Feature Importance \n for the Reaction Time Model', fontsize = 20)
    plt.xticks(rotation=45, ha='right')  # rotate x-axis labels for better readability
    plt.savefig(fig_savedir / 'elbowplot.png', dpi=500, bbox_inches='tight')
    plt.show()

    shap.summary_plot(shap_values, X, show=False, cmap = matplotlib.colormaps[cmapname])
    fig, ax = plt.gcf(), plt.gca()
    # if one_ferret:
    #     plt.title('Features over impact in reaction time for ' + ferrets)
    # else:
    #     plt.title('Features over impact in reaction time')
    plt.xlabel('SHAP value (impact on model output) on reaction time')

    import matplotlib.image as mpimg
    fig.set_size_inches(9, 15)
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=18)
    ax.set_ylabel('Features', fontsize=18)
    plt.savefig(fig_savedir / 'shapsummaryplot_allanimals2.png', dpi=300, bbox_inches='tight')

    plt.show()


    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=100,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T, color = 'cyan')
    if one_ferret:
        ax.set_title('Permutation Importances of the \n Reaction Time Model for ' + ferrets, fontsize = 13)
    else:
        ax.set_title('Permutation Importances of the \n Reaction Time Model', fontsize = 13)
    plt.xlabel('Permutation Importance')
    fig.tight_layout()
    plt.savefig(fig_savedir / 'permutation_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    shap.dependence_plot("time to target", shap_values, X)  #
    explainer = shap.Explainer(xg_reg, X)
    shap_values2 = explainer(X_train)
    # shap.plots.scatter(shap_values2[:, "side of audio"], color=shap_values2[:, "ferret ID"], show=True, cmap = matplotlib.colormaps[cmapname])
    fig, ax = plt.subplots(figsize=(10,10))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "precursor = target F0"], show=False, cmap = matplotlib.colormaps[cmapname])
    plt.xticks([1,2], labels = ['male', 'female'])
    if one_ferret:
        plt.title('Talker versus impact \n on reaction time for ' + ferrets, fontsize=18)
    plt.savefig(fig_savedir / 'talker_vs_precursorequaltargF0.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "target F0"], show=False,
                       cmap=matplotlib.colormaps[cmapname])
    plt.xticks([1, 2], labels=['male', 'female'])
    if one_ferret:
        plt.title('Talker versus impact \n on reaction time for ' + ferrets, fontsize=18)
    plt.savefig(fig_savedir / 'talker_vs_targetF0.png', dpi=300, bbox_inches='tight')


    shap.plots.scatter(shap_values2[:, "time to target"], color=shap_values2[:, "trial number"], show=False, cmap = matplotlib.colormaps[cmapname])
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Trial number", fontsize=12)
    plt.ylabel('SHAP value', fontsize=10)
    if one_ferret:
        plt.title('Target presentation time  versus impact \n on reacton time for ' + ferrets, fontsize=18)
    else:
        plt.title('Target presentation time versus impact on reacton time', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Target presentation time', fontsize=16)
    plt.savefig(fig_savedir /'targtimescolouredbytrialnumber.png', dpi=300)
    plt.show()

    shap.plots.scatter(shap_values2[:, "precursor = target F0"], color=shap_values2[:, "talker"], show=False, cmap = matplotlib.colormaps[cmapname])
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("talker", fontsize=12)
    plt.ylabel('SHAP value', fontsize=10)
    if one_ferret:
        plt.title('Precursor = target F0 \n over reacton time impact for ' + ferrets, fontsize=18)
    else:
        plt.title('Precursor = target F0 over reaction time impact', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Precursor = target F0', fontsize=16)
    plt.savefig(fig_savedir /'pitchofprecur_equals_target_colouredbytalker.png', dpi=1000)
    plt.show()

    shap.plots.scatter(shap_values2[:, "target F0"], color=shap_values2[:, "precursor = target F0"], show=False, cmap = matplotlib.colormaps[cmapname])
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    # cb_ax.set_yticks([1, 2, 3,4, 5])
    # cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
    cb_ax.set_ylabel("Pitch of precursor = target", fontsize=12)
    plt.ylabel('SHAP value', fontsize=10)
    if one_ferret:
        plt.title('target F0 versus impact \n in predicted reacton time for ' + ferrets, fontsize=18)
    else:
        plt.title('target F0 versus impact \n in predicted reacton time', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('target F0', fontsize=16)
    plt.xticks([1,2,3,4,5], labels=['109', '124', '144 ', '191', '251'], fontsize=15)
    plt.savefig( fig_savedir /'pitchoftargcolouredbyprecur.png', dpi=1000)
    plt.show()

    if one_ferret == False:
        shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "time to target"], show=False, cmap = matplotlib.colormaps[cmapname])
        fig, ax = plt.gcf(), plt.gca()
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar parameters
        cb_ax.tick_params(labelsize=15)
        cb_ax.set_yticks([1, 2, 3,4, 5])
        # cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
        cb_ax.set_ylabel("Target presentation time ", fontsize=12)
        plt.ylabel('SHAP value', fontsize=10)
        if one_ferret:
            plt.title('Ferret \n versus impact in predicted reacton time for' + ferrets[0], fontsize=18)
        else:
            plt.title('Ferret versus impact on reaction time', fontsize=18)
        plt.ylabel('SHAP value', fontsize=16)
        plt.xlabel('Ferret', fontsize=16)
        # plt.xticks([1,2,3,4,5], labels=['109', '124', '144 ', '191', '251'], fontsize=15)
        plt.xticks([0,1,2,3,4], labels=['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'], fontsize=15)
        #rotate xtick labels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.savefig( fig_savedir /'ferretcolouredbytargtimes.png', dpi=1000)
        plt.show()
        if ferret_as_feature:
            shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "precursor = target F0"], show=False,
                               cmap=matplotlib.colormaps[cmapname])
            fig, ax = plt.gcf(), plt.gca()
            # Get colorbar
            cb_ax = fig.axes[1]
            # Modifying color bar parameters
            cb_ax.tick_params(labelsize=15)
            cb_ax.set_yticks([0,1])
            # cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
            plt.ylabel('SHAP value', fontsize=10)
            if one_ferret:
                plt.title('Ferret versus impact \n in predicted reaction time for ' + ferrets, fontsize=18)
            else:
                plt.title('Ferret versus impact in predicted reaction time', fontsize=18)
            plt.ylabel('SHAP value', fontsize=16)
            plt.xlabel('Ferret', fontsize=16)
            # plt.xticks([1,2,3,4,5], labels=['109', '124', '144 ', '191', '251'], fontsize=15)
            plt.xticks([0, 1, 2, 3, 4], labels=['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'],
                       fontsize=15)
            # rotate xtick labels:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.savefig(fig_savedir / 'ferretcolouredbyintratrialroving.png', dpi=1000)
            plt.show()

            shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "side of audio"], show=False,
                               cmap=matplotlib.colormaps[cmapname])
            fig, ax = plt.gcf(), plt.gca()
            # Get colorbar
            cb_ax = fig.axes[1]
            # Modifying color bar parameters
            cb_ax.tick_params(labelsize=15)
            cb_ax.set_yticks([0, 1])
            cb_ax.set_yticklabels(['Left', 'Right'])
            # cb_ax.set_ylabel("Target presentation time ", fontsize=12)
            plt.ylabel('SHAP value', fontsize=10)
            if one_ferret:
                plt.title('Ferret \n versus impact in predicted reacton time for' + ferrets[0], fontsize=18)
            else:
                plt.title('Ferret versus impact on reaction time', fontsize=18)
            plt.ylabel('SHAP value', fontsize=16)
            plt.xlabel('Ferret', fontsize=16)
            # plt.xticks([1,2,3,4,5], labels=['109', '124', '144 ', '191', '251'], fontsize=15)
            plt.xticks([0, 1, 2, 3, 4],
                       labels=['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'],
                       fontsize=15)
            # rotate xtick labels:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.savefig(fig_savedir / 'ferretcolouredbysideofaudio.png', dpi=1000)
            plt.show()

        shap.plots.scatter(shap_values2[:, "side of audio"], color=shap_values2[:, "ferret ID"], show=False,
                           cmap=matplotlib.colormaps[cmapname])
        fig, ax = plt.gcf(), plt.gca()
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar parameters
        cb_ax.tick_params(labelsize=15)
        cb_ax.set_yticks([0, 1, 2, 3, 4])
        cb_ax.set_yticklabels(['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'])
        cb_ax.set_ylabel("ferret ", fontsize=12)
        # plt.xticks([0, 1, 2, 3, 4], labels=['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'],
        #            fontsize=15)
        plt.ylabel('SHAP value', fontsize=10)

        plt.title('Side of audio versus impact on reaction time', fontsize=18)

        # plt.ylabel('SHAP value', fontsize=16)
        plt.xlabel('side', fontsize=16)
        plt.xticks([0,1], labels=['left', 'right'], fontsize=15)

        # rotate xtick labels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.savefig(fig_savedir / 'sidecolouredbyferret.png', dpi=1000)
        plt.show()

    if one_ferret == False:
        mosaic = ['A', 'B', 'C'], ['D', 'B', 'E']
        ferret_id_only = ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']

        fig = plt.figure(figsize=(20, 10))
        ax_dict = fig.subplot_mosaic(mosaic)

        # Plot the elbow plot
        ax_dict['A'].plot(feature_labels, cumulative_importances, marker='o', color='cyan')
        ax_dict['A'].set_xlabel('Features')
        ax_dict['A'].set_ylabel('Cumulative Feature Importance')
        ax_dict['A'].set_title('Elbow Plot of Cumulative Feature Importance for Rxn Time Prediction')
        ax_dict['A'].set_xticklabels(feature_labels, rotation=45, ha='right')  # rotate x-axis labels for better readability

        # rotate x-axis labels for better readability
        summary_img = mpimg.imread(fig_savedir / 'shapsummaryplot_allanimals2.png')
        ax_dict['B'].imshow(summary_img, aspect='auto', )
        ax_dict['B'].axis('off')  # Turn off axis ticks and labels
        ax_dict['B'].set_title('Ranked list of features over their \n impact on reaction time', fontsize=13)


        ax_dict['D'].barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T, color='cyan')
        ax_dict['D'].set_title("Permutation importances on reaction time")
        ax_dict['D'].set_xlabel("Permutation importance")


        shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "target F0"], ax=ax_dict['E'],
                           cmap=matplotlib.colormaps[cmapname], show=False)
        fig, ax = plt.gcf(), plt.gca()
        cb_ax = fig.axes[5]
        # Modifying color bar parameters
        cb_ax.tick_params(labelsize=15)
        cb_ax.set_ylabel("target F0 (Hz)", fontsize=12)
        cb_ax.set_yticks([1,2,3,4,5])
        cb_ax.set_yticklabels(['109', '124', '144 ', '191', '251'])
        ax_dict['E'].set_ylabel('SHAP value', fontsize=10)
        ax_dict['E'].set_title('Talker versus impact on reaction time', fontsize=13)
        ax_dict['E'].set_xlabel('Talker', fontsize=16)
        ax_dict['E'].set_xticks([1,2])
        ax_dict['E'].set_xticklabels(['Male', 'Female'], rotation=45, ha='right')

        shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "target F0"], ax=ax_dict['C'],
                           cmap = matplotlib.colormaps[cmapname], show=False)
        fig, ax = plt.gcf(), plt.gca()
        cb_ax = fig.axes[7]
        cb_ax.set_yticks([1, 2, 3,4, 5])
        cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
        cb_ax.tick_params(labelsize=15)
        cb_ax.set_ylabel("target F0 (Hz)", fontsize=10)

        # Modifying color bar parameters
        cb_ax.tick_params(labelsize=15)
        ax_dict['C'].set_ylabel('SHAP value', fontsize=10)
        ax_dict['C'].set_xlabel('Ferret ID', fontsize=16)
        ax_dict['C'].set_xticks([0, 1, 2, 3, 4])
        ax_dict['C'].set_title('Ferret ID versus impact on reaction time', fontsize=13)

        ax_dict['C'].set_xticklabels(ferret_id_only, rotation=45, ha='right')
        # ax_dict['C'].set_title('Ferret ID and precursor = target F0 versus SHAP value on miss probability', fontsize=18)
        #remove padding outside the figures
        font_props = fm.FontProperties(weight='bold', size=17)

        ax_dict['A'].annotate('a)', xy=get_axis_limits(ax_dict['A']), xytext=(-0.1, ax_dict['A'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props, zorder=10)
        ax_dict['B'].annotate('b)', xy=get_axis_limits(ax_dict['B']), xytext=(-0.1, ax_dict['B'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
        ax_dict['C'].annotate('c)', xy=get_axis_limits(ax_dict['C']), xytext=(-0.1, ax_dict['C'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
        ax_dict['D'].annotate('d)', xy=get_axis_limits(ax_dict['D']), xytext=(-0.1, ax_dict['D'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
        ax_dict['E'].annotate('e)', xy=get_axis_limits(ax_dict['E']), xytext=(-0.1, ax_dict['E'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)


        plt.tight_layout()
        plt.savefig(fig_savedir / 'big_summary_plot.png', dpi=500, bbox_inches="tight")
        plt.savefig(fig_savedir / 'big_summary_plot.pdf', dpi=500, bbox_inches="tight")
        plt.show()

    return xg_reg, ypred, y_test, results

def extract_release_times_data(ferrets):
    df = behaviouralhelperscg.get_df_behav(ferrets=ferrets, includefaandmiss=False, startdate='04-01-2020', finishdate='01-03-2023')
    #switch talker values so 1 is 2, and 2 is 1 simultaneously
    df['talker'] = df['talker'].replace({1: 2, 2: 1})
    #
    # df_intra = df[df['intra_trial_roving'] == 1]
    # df_inter = df[df['inter_trial_roving'] == 1]
    # df_control = df[df['control_trial'] == 1]
    #
    # if len(df_intra) > len(df_inter)*1.2:
    #     df_intra = df_intra.sample(n=len(df_inter), random_state=123)
    # elif len(df_inter) > len(df_intra)*1.2:
    #     df_inter = df_inter.sample(n=len(df_intra), random_state=123)
    #
    # if len(df_control) > len(df_intra)*1.2:
    #     df_control = df_control.sample(n=len(df_intra), random_state=123)
    # elif len(df_control) > len(df_inter)*1.2:
    #     df_control = df_control.sample(n=len(df_inter), random_state=123)
    #

    df_pitchtargsame = df[df['precur_and_targ_same'] == 1]
    df_pitchtargdiff = df[df['precur_and_targ_same'] == 0]
    if len(df_pitchtargsame) > len(df_pitchtargdiff)*1.2:
        df_pitchtargsame = df_pitchtargsame.sample(n=len(df_pitchtargdiff), random_state=123)
    elif len(df_pitchtargdiff) > len(df_pitchtargsame)*1.2:
        df_pitchtargdiff = df_pitchtargdiff.sample(n=len(df_pitchtargsame), random_state=123)
    df = pd.concat([df_pitchtargsame, df_pitchtargdiff], axis = 0)
    # df = pd.concat([df_intra, df_inter, df_control], axis = 0)
    dfuse = df[[ "pitchoftarg", "pastcatchtrial", "trialNum", "talker", "side", "precur_and_targ_same",
                "timeToTarget",
                "realRelReleaseTimes", "ferret", "pastcorrectresp"]]
    labels = ['target F0', 'past trial was catch', 'trial number', 'talker', 'side of audio', 'precursor = target F0', 'time to target', 'realRelReleaseTimes', 'ferret ID', 'past trial was correct']
    dfuse = dfuse.rename(columns=dict(zip(dfuse.columns, labels)))
    return dfuse


def run_correctrxntime_model(ferrets, optimization = False, ferret_as_feature = False ):
    df_use = extract_release_times_data(ferrets)
    col = 'realRelReleaseTimes'
    dfx = df_use.loc[:, df_use.columns != col]

    # remove ferret as possible feature
    if ferret_as_feature == False:
        col2 = 'ferret ID'
        dfx = dfx.loc[:, dfx.columns != col2]
        if optimization == False:
            best_params = np.load('../optuna_results/best_paramsreleastimemodel_allferrets.npy', allow_pickle=True).item()
        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params
            np.save('../optuna_results/best_paramsreleastimemodel_allferrets.npy', best_params)
    else:
        dfx = dfx
        if optimization == False:
            best_params = np.load('../optuna_results/best_paramsreleastimemodel_allferrets_ferretasfeature.npy', allow_pickle=True).item()
        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params
            np.save('../optuna_results/best_paramsreleastimemodel_allferrets_ferretasfeature.npy', best_params)




    xg_reg, ypred, y_test, results = runlgbreleasetimes(dfx, df_use[col], paramsinput=best_params, ferret_as_feature=ferret_as_feature, ferrets = ferrets)



def run_correctrxntime_model_for_a_ferret(ferrets, optimization = False, ferret_as_feature = False ):
    df_use = extract_release_times_data(ferrets)
    col = 'realRelReleaseTimes'
    dfx = df_use.loc[:, df_use.columns != col]


    # remove ferret as possible feature
    if ferret_as_feature == False:
        col2 = 'ferret ID'
        dfx = dfx.loc[:, dfx.columns != col2]
        if optimization == False:
            best_params = np.load('D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel_'+ ferrets[0]+ '.npy', allow_pickle=True).item()
        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params
            np.save('D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel_'+ ferrets[0]+ '.npy', best_params)
    else:
        dfx = dfx
        if optimization == False:
            best_params = np.load('D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel_ferretasfeature_'+ '.npy', allow_pickle=True).item()
        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params

            np.save('D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel_ferretasfeature_'+ '.npy', best_params)
    xg_reg, ypred, y_test, results = runlgbreleasetimes(dfx, df_use[col], paramsinput=best_params, ferret_as_feature=ferret_as_feature, one_ferret=True, ferrets=ferrets[0])


def main():
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']  # , 'F2105_Clove']
    # ferrets = ['F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    run_correctrxntime_model(ferrets, optimization = False, ferret_as_feature=True)
    #
    # for ferret in ferrets:
    #     run_correctrxntime_model_for_a_ferret([ferret], optimization=False, ferret_as_feature=False)



if __name__ == '__main__':
    main()