import sklearn.metrics
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import shap
import matplotlib
import seaborn as sns
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn
from sklearn.model_selection import train_test_split
from helpers.behaviouralhelpersformodels import * \
from pathlib import Path
def shap_summary_plot(
        shap_values2,
        feature_labels,
        ax=None,
        cmap = "viridis",
    ):
    ''' visualize the SHAP summary plot
    :param shap_values2: SHAP values
    :param feature_labels: feature labels
    :param ax: axis
    :param cmap: color map
    :return: SHAP summary plot
    '''
    plt.rcParams['font.family'] = 'sans-serif'
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    plt.sca(ax)
    shap.plots.beeswarm(shap_values2, show=False, color=cmap)
def get_axis_limits(ax, scale=1):
    return ax.get_xlim()[0] * scale, (ax.get_ylim()[1] * scale)
def run_optuna_study_releasetimes(X, y):
    '''
    :param X: training data
    :param y: target variable
    :return: the optuna study results
    '''
    study = optuna.create_study(direction="minimize", study_name="LGBM regressor")
    func = lambda trial: objective_releasetimes(trial, X, y)
    study.optimize(func, n_trials=1000)
    print("Number of finished trials: ", len(study.trials))
    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")
    return study

def objective_releasetimes(trial, X, y):
    '''Run an optimisation study for the correct release times model
    :param trial: optuna trial
    :param X: training data
    :param y: target variable
    :return: mean cross validation score'''

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
    ''' this function runs the LGBM model for the correct release times for a single ferret, then returns the model and calls the visualisation functions implementing SHAP.
    :param data: dataframe containing the data
    :param paramsinput: parameters for the LGBM model
    :param ferret: ferret ID
    :param ferret_name: ferret name
    :return: model, ypred, y_test, mse_test'''

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
    kfold = KFold(n_splits=5)
    mse_train = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)

    mse_test = mean_squared_error(ypred, y_test)
    mse_test = cross_val_score(xg_reg, X_test, y_test, scoring='neg_mean_squared_error', cv=kfold)
    r2_test = cross_val_score(xg_reg, X_test, y_test, scoring='r2', cv=kfold)
    print("MSE on test: %.4f" % (mse_test) + ferret_name)
    print("negative MSE training: %.4f" % (np.mean(mse_train)))
    print(mse_train)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(dfx)

    fig, ax = plt.subplots(figsize=(15, 15))
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
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "talker"],
                       title='Correct Responses - Reaction Time Model SHAP response \n vs. trial number for' + ferret_name)

    return xg_reg, ypred, y_test, mse_test


def runlgbreleasetimes(X, y, paramsinput=None, ferret_as_feature = False, one_ferret=False, ferrets=None, noise_floor = False):
    ''' run the LGBM model for the correct release times for all ferrets, then returns the model and calls the visualisation functions implementing SHAP.
    :param X: dataframe containing the TRAIN data
    :param y: dataframe containing the TEST data
    :param paramsinput: parameters for the LGBM model
    :param ferret_as_feature: whether to include ferret ID as a feature
    :param one_ferret: whether to run the model for one ferret or all ferrets
    :param ferrets: ferret ID
    :param noise_floor: whether to include the noise floor
    :return: model, ypred, y_test, mse_test
    '''

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                        random_state=42)
    if ferret_as_feature:
        if one_ferret:
            fig_savedir = Path('figs/correctrxntimemodel/ferret_as_feature/' + ferrets)
        else:
            fig_savedir = Path('../figs/correctrxntimemodel/ferret_as_feature')
    else:
        if one_ferret:
            fig_savedir = Path('figs/correctrxntimemodel/'+ ferrets)
        else:
            fig_savedir = Path('../figs/correctrxntimemodel/')
    if noise_floor == True:
        fig_savedir = fig_savedir / 'noise_floor'

    if fig_savedir.exists():
        pass
    else:
        fig_savedir.mkdir(parents=True, exist_ok=True)

    xg_reg = lgb.LGBMRegressor(random_state=42, verbose=1, **paramsinput)
    xg_reg.fit(X_train, y_train, verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
    if one_ferret:
        plt.title('feature importances for the LGBM Correct Release Times model for' + ferrets)
    else:
        plt.title('feature importances for the LGBM Correct Release Times model')
    plt.show()

    kfold = KFold(n_splits=5)
    results = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
    results_mae = cross_val_score(xg_reg, X_train, y_train, scoring='neg_median_absolute_error', cv=kfold)
    results_r2 = cross_val_score(xg_reg, X_train, y_train, scoring='r2', cv=kfold)
    # mse_train = mean_squared_error(ypred, y_test)

    mse_test = cross_val_score(xg_reg, X_test, y_test, scoring='neg_mean_squared_error', cv=kfold)
    mae_test  = cross_val_score(xg_reg, X_test, y_test, scoring='neg_median_absolute_error', cv=kfold)
    r2_test = cross_val_score(xg_reg, X_test, y_test, scoring='r2', cv=kfold)
    print("MSE on test: %.4f" % (np.mean(mse_test)))
    print("negative MSE training: %.2f%%" % (np.mean(results) * 100.0))
    print('r2 on test: %.4f' % (np.mean(r2_test)))
    print('r2 on training: %.4f' % (np.mean(results_r2)))
    mae = median_absolute_error(ypred, y_test)
    print("MAE on test: %.4f" % (np.mean(mae_test)))
    print("negative MAE training: %.2f%%" % (np.mean(results_mae) * 100.0))

    #export all scoring results
    trainandtestaccuracy ={
        'mse_test': mse_test,
        'mse_train': results,
        'mean_mse_train': np.mean(results),
        'mean_mse_test': np.mean(mse_test),
        'mae_test': mae_test,
        'mae_train': results_mae,
        'mean_mae_train': np.mean(results_mae),
        'mean_mae_test': np.mean(mae_test),
        'r2_test': r2_test,
        'r2_train': results_r2,
        'mean_r2_train': np.mean(results_r2),
        'mean_r2_test': np.mean(r2_test),
    }
    #savedictionary to csv
    trainandtestaccuracy = pd.DataFrame(trainandtestaccuracy)
    trainandtestaccuracy.to_csv('D:\mixedeffectmodelsbehavioural/metrics/correct_rxn_time_modelmse.csv')
    # np.savetxt('D:\mixedeffectmodelsbehavioural\metrics/correct_rxn_time_modelmse.csv', trainandtestaccuracy, delimiter=',', fmt='%s')
    print(results)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(X)
    fig, ax = plt.subplots(figsize=(15, 15))
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
    plt.xlabel('SHAP value (impact on model output) on reaction time')
    fig.set_size_inches(9, 15)
    #get the color bar
    colorbar = fig.axes[1]
    #change the font size of the color bar
    colorbar.tick_params(labelsize=30)
    #change the label of the color bar
    colorbar.set_ylabel(None)
    #increase the y tick label size
    ax.tick_params(axis='y', which='major', labelsize=25, rotation = 45)
    ax.tick_params(axis='x', which='major', labelsize=25)
    ax.set_xlabel(None)
    plt.savefig(fig_savedir / 'shapsummaryplot_allanimals2.png', dpi=300, bbox_inches='tight')
    plt.show()

    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=100,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T, color = 'cyan')
    if one_ferret:
        ax.set_title('Permutation Importances of the \n Reaction Time Model for ' + ferrets, fontsize = 20)
    else:
        ax.set_title('Permutation Importances of the \n Reaction Time Model', fontsize = 20)
    plt.xlabel('Permutation Importance')
    fig.tight_layout()
    plt.savefig(fig_savedir / 'permutation_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    shap.dependence_plot("target time", shap_values, X)  #
    explainer = shap.Explainer(xg_reg, X)
    shap_values2 = explainer(X_train)

    fig, ax = plt.subplots(figsize=(10,10))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "precur. = targ. F0"], show=False, cmap = matplotlib.colormaps[cmapname])
    plt.xticks([1,2], labels = ['male', 'female'])
    if one_ferret:
        plt.title('Talker versus impact \n on reaction time for ' + ferrets, fontsize=20)
    plt.savefig(fig_savedir / 'talker_vs_precursorequaltargF0.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots()
    custom_colors = ['dodgerblue',  'green', "limegreen"]  # Add more colors as needed
    cmapcustom = mcolors.LinearSegmentedColormap.from_list('my_custom_cmap', custom_colors, N=1000)
    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "precur. = targ. F0"], show=False, ax=ax, cmap = cmapcustom)
    colorbar_scatter = fig.axes[1]
    colorbar_scatter.set_yticks([0,1])
    colorbar_scatter.set_yticklabels(['False', 'True'], fontsize=18)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=20, rotation=45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    ax.set_ylabel('Influence on reaction time', fontsize=18)
    plt.title('Mean SHAP value over ferret ID', fontsize=20)
    plt.savefig(fig_savedir /'ferretIDbyprecurequaltargF0.png', dpi=500, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "target time"], show=False, ax=ax, cmap = 'viridis')
    colorbar_scatter = fig.axes[1]
    colorbar_scatter.set_ylabel('time to target (s)', fontsize=18)
    # colorbar_scatter.set_yticks([0,1])
    # colorbar_scatter.set_yticklabels(['False', 'True'], fontsize=18)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    ax.set_ylabel('Impact on reaction time', fontsize=18)
    plt.title('Time to target presentation', fontsize = 20)
    plt.savefig(fig_savedir /'ferretIDbytimetotarget.png', dpi=500, bbox_inches='tight')
    plt.show()


    ferret_ids = shap_values2[:, "ferret ID"].data
    precursor_values = shap_values2[:, "precur. = targ. F0"].data
    shap_values = shap_values2[:, "ferret ID"].values

    # Create a DataFrame with the necessary data
    data_df = pd.DataFrame({
        "ferret ID": ferret_ids,
        "precur. = targ. F0": precursor_values,
        "SHAP value": shap_values
    })

    custom_colors = ['dodgerblue', 'green', 'limegreen']
    cmapcustom = mcolors.LinearSegmentedColormap.from_list('my_custom_cmap', custom_colors, N=1000)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="ferret ID", y="SHAP value", hue="precur. = targ. F0", data=data_df, split=True, inner="quart",
                   palette=custom_colors, ax=ax)

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    ax.set_ylabel('Impact on reaction time', fontsize=20)  # Corrected y-label
    handles, labels = ax.get_legend_handles_labels()
    labels_new = ['false', 'true']
    ax.legend(handles=handles[0:], labels=labels_new, title="precur. = targ. F0", fontsize=14, title_fontsize=16)

    ax.set_title("precur. = targ. F0",  fontsize = 20)
    plt.savefig(fig_savedir / 'ferretIDbyprecurequaltargF0_violin.png', dpi=500, bbox_inches='tight')
    plt.show()

    ferret_ids = shap_values2[:, "ferret ID"].data
    talker_values = shap_values2[:, "talker"].data
    shap_values = shap_values2[:, "ferret ID"].values

    # Create a DataFrame with the necessary data
    data_df = pd.DataFrame({
        "ferret ID": ferret_ids,
        "talker": talker_values,
        "SHAP value": shap_values
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="ferret ID", y="SHAP value", hue="talker", data=data_df, split=True, inner="quart",
                   palette=custom_colors, ax=ax)

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    ax.set_ylabel('Impact on reaction time', fontsize=18)  # Corrected y-label

    # plt.title('Mean SHAP value over ferret ID', fontsize=18)

    # Optionally add a legend
    ax.legend(title="talker", fontsize=14, title_fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    labels = ['male', 'female']
    ax.legend(handles=handles[0:], labels=labels[0:], title="talker", fontsize=14, title_fontsize=16)
    ax.set_title('Talker', fontsize = 20)

    plt.savefig(fig_savedir / 'ferretIDbytalker_violin.png', dpi=500, bbox_inches='tight')
    plt.show()

    ferret_ids = shap_values2[:, "ferret ID"].data
    side_values = shap_values2[:, "side of audio"].data
    shap_values = shap_values2[:, "ferret ID"].values

    # Create a DataFrame with the necessary data
    data_df = pd.DataFrame({
        "ferret ID": ferret_ids,
        "side": side_values,
        "SHAP value": shap_values
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="ferret ID", y="SHAP value", hue="side", data=data_df, split=True, inner="quart",
                   palette=custom_colors, ax=ax)

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    ax.set_ylabel('Impact on reaction time', fontsize=18)  # Corrected y-label


    ax.legend(title="side of audio", fontsize=14, title_fontsize=16)
    #change legend labels
    handles, labels = ax.get_legend_handles_labels()
    labels = ['left', 'right']
    ax.legend(handles=handles[0:], labels=labels[0:], title="side of audio", fontsize=14, title_fontsize=16)
    ax.set_title('Side of audio presentation', fontsize = 20)


    plt.savefig(fig_savedir / 'ferretIDbysideofaudio_violin.png', dpi=500, bbox_inches='tight')
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "target F0"], show=False,
                       cmap=matplotlib.colormaps[cmapname])
    plt.xticks([1, 2], labels=['male', 'female'])
    if one_ferret:
        plt.title('Talker versus impact \n on reaction time for ' + ferrets, fontsize=20)
    plt.savefig(fig_savedir / 'talker_vs_targetF0.png', dpi=300, bbox_inches='tight')


    shap.plots.scatter(shap_values2[:, "target time"], color=shap_values2[:, "trial no."], show=False, cmap = matplotlib.colormaps[cmapname])
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Trial number", fontsize=12)
    plt.ylabel('SHAP value', fontsize=18)
    if one_ferret:
        plt.title('Target presentation time  versus impact \n on reacton time for ' + ferrets, fontsize=20)
    else:
        plt.title('Target presentation time versus impact on reacton time', fontsize=20)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Target presentation time', fontsize=16)
    plt.savefig(fig_savedir /'targtimescolouredbytrialnumber.png', dpi=300)
    plt.show()

    shap.plots.scatter(shap_values2[:, "precur. = targ. F0"], color=shap_values2[:, "talker"], show=False, cmap = matplotlib.colormaps[cmapname])
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("talker", fontsize=12)
    plt.ylabel('SHAP value', fontsize=18)
    if one_ferret:
        plt.title('Precursor = target F0 \n over reacton time impact for ' + ferrets, fontsize=20)
    else:
        plt.title('Precursor = target F0 over reaction time impact', fontsize=20)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel("precur. = targ. F0", fontsize=16)
    plt.savefig(fig_savedir /'pitchofprecur_equals_target_colouredbytalker.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "target F0"], color=shap_values2[:, "precur. = targ. F0"], show=False, cmap = matplotlib.colormaps[cmapname])
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Pitch of precursor = target", fontsize=12)
    plt.ylabel('SHAP value', fontsize=18)
    if one_ferret:
        plt.title('target F0 versus impact \n in predicted reacton time for ' + ferrets, fontsize=20)
    else:
        plt.title('target F0 versus impact \n in predicted reacton time', fontsize=20)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xlabel('target F0', fontsize=18)
    plt.xticks([1,2,3,4,5], labels=['109', '124', '144 ', '191', '251'], fontsize=15)
    plt.savefig( fig_savedir /'pitchoftargcolouredbyprecur.png', dpi=1000)
    plt.show()

    if one_ferret == False:
        shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "target time"], show=False, cmap = matplotlib.colormaps[cmapname])
        fig, ax = plt.gcf(), plt.gca()
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar parameters
        cb_ax.tick_params(labelsize=15)
        cb_ax.set_yticks([1, 2, 3,4, 5])
        # cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
        cb_ax.set_ylabel("Target presentation time ", fontsize=12)
        plt.ylabel('SHAP value', fontsize=18)
        if one_ferret:
            plt.title('Ferret \n versus impact in predicted reacton time for' + ferrets[0], fontsize=18)
        else:
            plt.title('Ferret versus impact on reaction time', fontsize=18)
        plt.ylabel('SHAP value', fontsize=18)
        plt.xlabel('Ferret', fontsize=18)
        # plt.xticks([1,2,3,4,5], labels=['109', '124', '144 ', '191', '251'], fontsize=15)
        plt.xticks([0,1,2,3,4], labels=['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'], fontsize=15)
        #rotate xtick labels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.savefig( fig_savedir /'ferretcolouredbytargtimes.png', dpi=1000)
        plt.show()

        if ferret_as_feature:
            shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "precur. = targ. F0"], show=False,
                               cmap=matplotlib.colormaps[cmapname])
            fig, ax = plt.gcf(), plt.gca()
            # Get colorbar
            cb_ax = fig.axes[1]
            # Modifying color bar parameters
            cb_ax.tick_params(labelsize=15)
            cb_ax.set_yticks([0,1])
            # cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
            plt.ylabel('SHAP value', fontsize=18)
            if one_ferret:
                plt.title('Ferret versus impact \n in predicted reaction time for ' + ferrets, fontsize=18)
            else:
                plt.title('Ferret versus impact in predicted reaction time', fontsize=18)
            plt.ylabel('SHAP value', fontsize=18)
            plt.xlabel('Ferret', fontsize=18)
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
            plt.ylabel('SHAP value', fontsize=18)
            if one_ferret:
                plt.title('Ferret \n versus impact in predicted reacton time for' + ferrets[0], fontsize=18)
            else:
                plt.title('Ferret versus impact on reaction time', fontsize=18)
            plt.ylabel('SHAP value', fontsize=18)
            plt.xlabel('Ferret', fontsize=18)
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
        plt.ylabel('SHAP value', fontsize=18)

        plt.title('Side of audio versus impact on reaction time', fontsize=18)

        # plt.ylabel('SHAP value', fontsize=18)
        plt.xlabel('side', fontsize=18)
        plt.xticks([0,1], labels=['left', 'right'], fontsize=15)
        # rotate xtick labels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.savefig(fig_savedir / 'sidecolouredbyferret.png', dpi=1000)
        plt.show()

    if one_ferret == False:
        text_width_pt = 419.67816  # Replace with your value
        # Convert the text width from points to inches
        text_width_inches = text_width_pt / 72.27
        mosaic = ['A', 'B'], ['D', 'B'], ['C', 'E']
        ferret_id_only = ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']

        fig = plt.figure(figsize=((text_width_inches / 2) * 4, text_width_inches * 4))
        ax_dict = fig.subplot_mosaic(mosaic)
        # Plot the elbow plot
        ax_dict['A'].plot(feature_labels, cumulative_importances, marker='o', color='cyan')
        ax_dict['A'].set_xlabel('Features', fontsize=18)
        ax_dict['A'].set_ylabel('Cumulative Feature Importance', fontsize=18)
        ax_dict['A'].set_xticklabels(feature_labels, rotation=20, ha='right')  # rotate x-axis labels for better readability


        axmini = ax_dict['B']
        shap_summary_plot(shap_values2, feature_labels, show_plots=False, ax=axmini, cmap=cmapcustom)
        ax_dict['B'].set_yticklabels(np.flip(feature_labels), fontsize=12, rotation=45, fontfamily='sans-serif')
        ax_dict['B'].set_xlabel('Impact on rxn time', fontsize=12)
        # ax_dict['B'].set_xticks([-1, -0.5, 0, 0.5, 1])
        cb_ax = fig.axes[5]
        cb_ax.tick_params(labelsize=8)
        cb_ax.set_ylabel('Value', fontsize=8, fontfamily='sans-serif')
        ax_dict['D'].barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T, color='cyan')
        # ax_dict['D'].set_title("Permutation importances on reaction time")
        ax_dict['D'].set_xlabel("Permutation importance", fontsize = 18)


        shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "target F0"], ax=ax_dict['E'],
                           cmap=matplotlib.colormaps[cmapname], show=False)
        fig, ax = plt.gcf(), plt.gca()
        cb_ax = fig.axes[6]
        # Modifying color bar parameters
        cb_ax.tick_params(labelsize=8)
        cb_ax.set_ylabel("target F0 (Hz)", fontsize=8)
        cb_ax.set_yticks([1,2,3,4,5])
        cb_ax.set_yticklabels(['109', '124', '144 ', '191', '251'])
        cb_ax.set_yticklabels(['109', '124', '144 ', '191', '251'])
        ax_dict['E'].set_ylabel('Impact on reaction time', fontsize=18)
        ax_dict['E'].set_xlabel('Talker', fontsize=18)
        ax_dict['E'].set_xticks([1,2])
        ax_dict['E'].set_xticklabels(['Male', 'Female'], rotation=45, ha='right')

        shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "target F0"], ax=ax_dict['C'],
                           cmap = matplotlib.colormaps[cmapname], show=False)
        fig, ax = plt.gcf(), plt.gca()
        cb_ax = fig.axes[8]
        cb_ax.set_yticks([1, 2, 3,4, 5])
        cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
        cb_ax.tick_params(labelsize=8)
        cb_ax.set_ylabel("target F0 (Hz)", fontsize=8)

        # Modifying color bar parameters
        ax_dict['C'].set_ylabel('Impact on reaction time', fontsize=18)
        ax_dict['C'].set_xlabel('Ferret ID', fontsize=18)
        ax_dict['C'].set_xticks([0, 1, 2, 3, 4])
        ax_dict['C'].set_xticklabels(ferret_id_only, rotation=45, ha='right')


        for key, ax in ax_dict.items():
            if key == 'B':
                ax.tick_params(axis='y', which='major', labelsize=5.7)
                ax.tick_params(axis='x', which='major', labelsize=6)
                ax.tick_params(axis='both', which='minor', labelsize=6)
                ax.xaxis.label.set_size(8)
                ax.yaxis.label.set_size(8)
            else:
                ax.tick_params(axis='y', which='major', labelsize=6)
                ax.tick_params(axis='x', which='major', labelsize=6)
                ax.tick_params(axis='both', which='minor', labelsize=6)
                ax.xaxis.label.set_size(8)
                ax.yaxis.label.set_size(8)
            for text in [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
                text.set_fontfamily('sans-serif')
                text.set_color('black')

        plt.subplots_adjust(wspace=0.35, hspace=0.5)
        plt.savefig(fig_savedir / 'big_summary_plot_1606.png', dpi=500, bbox_inches="tight")
        plt.savefig(fig_savedir / 'big_summary_plot_1606.pdf', dpi=500, bbox_inches="tight")
        plt.show()

    return xg_reg, ypred, y_test, results

def extract_release_times_data(ferrets):
    ''' this function extracts the release times data from the behavioural data
    :param ferrets: list of ferrets to include
    :return df: dataframe with the release times data'''

    df = behaviouralhelperscg.get_df_behav(ferrets=ferrets, includefaandmiss=False, startdate='04-01-2020', finishdate='01-03-2023')
    #switch talker values so 1 is 2, and 2 is 1 simultaneously
    df['talker'] = df['talker'].replace({1: 2, 2: 1})
    df_pitchtargsame = df[df['precur_and_targ_same'] == 1]
    df_pitchtargdiff = df[df['precur_and_targ_same'] == 0]
    if len(df_pitchtargsame) > len(df_pitchtargdiff)*1.2:
        df_pitchtargsame = df_pitchtargsame.sample(n=len(df_pitchtargdiff), random_state=123)
    elif len(df_pitchtargdiff) > len(df_pitchtargsame)*1.2:
        df_pitchtargdiff = df_pitchtargdiff.sample(n=len(df_pitchtargsame), random_state=123)
    df = pd.concat([df_pitchtargsame, df_pitchtargdiff], axis = 0)
    dfuse = df[[ "pitchoftarg", "pastcatchtrial", "trialNum", "talker", "side", "precur_and_targ_same",
                "timeToTarget",
                "realRelReleaseTimes", "ferret", "pastcorrectresp"]]
    labels = ['target F0', 'past trial catch', 'trial no.', 'talker', 'side of audio', "precur. = targ. F0", 'target time', 'realRelReleaseTimes', 'ferret ID', 'past resp. correct']
    dfuse = dfuse.rename(columns=dict(zip(dfuse.columns, labels)))

    #plot the proportion of trials that have target F0 = precursor F0 for each subset of target F0
    targ_1 = dfuse[dfuse['target F0'] == 1]
    targ_1_precur_and_targ_same = targ_1[targ_1["precur. = targ. F0"] == 1]
    targ_1_precur_and_targ_diff = targ_1[targ_1["precur. = targ. F0"] == 0]
    targ_2 = dfuse[dfuse['target F0'] == 2]
    targ_2_precur_and_targ_same = targ_2[targ_2["precur. = targ. F0"] == 1]
    targ_2_precur_and_targ_diff = targ_2[targ_2["precur. = targ. F0"] == 0]

    targ_3 = dfuse[dfuse['target F0'] == 3]
    targ_3_precur_and_targ_same = targ_3[targ_3["precur. = targ. F0"] == 1]
    targ_3_precur_and_targ_diff = targ_3[targ_3["precur. = targ. F0"] == 0]

    targ_4 = dfuse[dfuse['target F0'] == 4]
    targ_4_precur_and_targ_same = targ_4[targ_4["precur. = targ. F0"] == 1]
    targ_4_precur_and_targ_diff = targ_4[targ_4["precur. = targ. F0"] == 0]
    targ_5 = dfuse[dfuse['target F0'] == 5]
    targ_5_precur_and_targ_same = targ_5[targ_5["precur. = targ. F0"] == 1]
    targ_5_precur_and_targ_diff = targ_5[targ_5["precur. = targ. F0"] == 0]

    #plot the proportion of trials that have target F0 = precursor F0 for each subset of target F0
    fig, ax = plt.subplots()
    sns.barplot(x='target F0', y="precur. = targ. F0", data=dfuse, palette='Set2')
    plt.title('Proportion of trials with precursor = target F0 \n for each target F0', fontsize=18)
    plt.show()

    fig,ax = plt.subplots()
    sns.scatterplot(data=dfuse, x='target F0', y="precur. = targ. F0", hue='ferret ID', palette='Set2')
    plt.show()
    return dfuse

def run_mixed_effects_model_correctrxntime(df):
    '''this function runs a mixed effects model on the reaction time data to later compare with the LGBM model
    :param df: dataframe with the reaction time data
    :return result: mixed effects model results'''

    ferrets = ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']
    for col in df.columns:
        if col == "precur. = targ. F0":
            #rename this column to make it easier to work with
            df.rename(columns={col: 'precursor_equals_target_F0'}, inplace=True)
        else:

            df.rename(columns={col: col.replace(" ", "_")}, inplace=True)
    for col in df.columns:
            df.rename(columns={col: col.replace(".", "_")}, inplace=True)
    equation = 'realRelReleaseTimes ~target_F0 + past_trial_catch + trial_no_ + talker + side_of_audio + precursor_equals_target_F0 + target_time + past_resp__correct'

    df['past_resp__correct'] = df['past_resp__correct'].astype('category')
    df['side_of_audio'] = df['side_of_audio'].astype('category')
    df['side_of_audio'] = df['side_of_audio'].replace({0: 'Left', 1: 'Right'})
    df['talker'] = df['talker'].astype('category')
    df['precursor_equals_target_F0'] = df['precursor_equals_target_F0'].astype('category')
    df['target_F0'] = df['target_F0'].astype('category')
    df['talker'] = df['talker'].replace({1: 'Male', 2: 'Female'})
    df['target_F0'] = df['target_F0'].replace({1: '109 Hz', 2: '124 Hz', 3: '144 Hz', 4: '191 Hz', 5: '251 Hz'})
    df['ferret_ID'] = df['ferret_ID'].astype('category')
    df['past_trial_catch'] = df['past_trial_catch'].astype('category')

    df = df.dropna()
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    train_mse = []
    test_mse = []
    train_mae = []
    test_mae = []
    test_r2 = []
    train_r2 = []
    coefficients = []
    p_values = []
    std_error = []
    std_error_re = []
    random_effects_df = pd.DataFrame()
    for train_index, test_index in kf.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]
        model = smf.mixedlm(equation, train, groups=train["ferret_ID"])
        result = model.fit()
        random_effects = result.random_effects
        random_effects_2 = pd.DataFrame()
        for i, ferret in enumerate(ferrets):
            try:
                random_effects_2[ferret] = random_effects[i].values
            except:
                continue

        print(result.summary())
        params = result.params
        coefficients.append(params)
        random_effects_df = pd.concat([random_effects_df, random_effects_2])
        p_values.append(result.pvalues)
        std_error.append(result.bse)
        std_error_re.append(result.bse_re)

        var_resid = result.scale
        var_random_effect = float(result.cov_re.iloc[0])
        var_fixed_effect = result.predict(df).var()

        total_var = var_fixed_effect + var_random_effect + var_resid
        marginal_r2 = var_fixed_effect / total_var
        conditional_r2 = (var_fixed_effect + var_random_effect) / total_var

        print("marginal R2: {:.3f}".format(marginal_r2))
        print("conditional R2: {:.3f}".format(conditional_r2))
        #calculate the mean squared error

        ypred_train = result.predict(train)
        y_train = train['realRelReleaseTimes']
        mse_train = mean_squared_error(y_train, ypred_train)
        mae_train = median_absolute_error(y_train, ypred_train)
        r2_train = r2_score(y_train, ypred_train)
        train_r2.append(r2_train)
        train_mse.append(mse_train)
        train_mae.append(mae_train)
        print(mse_train)

        #calculate the r2
        ypred = result.predict(test)
        y_test = test['realRelReleaseTimes']
        mse = mean_squared_error(y_test, ypred)
        r2 = r2_score(y_test, ypred)
        test_r2.append(r2)
        test_mse.append(mse)
        #calculate the median absolute error
        mae = median_absolute_error(y_test, ypred)
        test_mae.append(mae)
        print(mae)
        print(mse)

    coefficients_df = pd.DataFrame(coefficients).mean()
    p_values_df = pd.DataFrame(p_values).mean()
    std_error_df = pd.DataFrame(std_error).mean()
    std_error_re_df = pd.DataFrame(std_error_re).mean()

    # combine into one dataframe
    result_coefficients = pd.concat([coefficients_df, p_values_df, std_error_df], axis=1, keys=['coefficients', 'p_values', 'std_error'])
    fig, ax = plt.subplots()
    # sort the coefficients by their mean value
    result_coefficients.index = result_coefficients.index.str.replace('Group Var', 'Ferret')
    result_coefficients = result_coefficients.sort_values(by='coefficients', ascending=False)

    ax.bar(result_coefficients.index, result_coefficients['coefficients'], color = 'cyan')
    ax.errorbar(result_coefficients.index, result_coefficients['coefficients'], yerr=result_coefficients['std_error'], fmt='none', ecolor='black', elinewidth=1, capsize=2)

    for i in range(len(result_coefficients)):
        if result_coefficients['p_values'][i] < 0.05:
            ax.text(i, 0.00, '*', fontsize=20)
    ax.set_xlabel('Features')
    ax.set_ylabel('Mean Coefficient')
    plt.xticks(rotation=45, ha='right')
    ax.set_title('Mean Coefficient for Each Feature, Correct Reaction Time Model')
    plt.savefig('D:\mixedeffectmodelsbehavioural\models/mixedeffects_csvs/correctrxntimemodel_mean_coefficients.png', dpi=500, bbox_inches='tight')
    plt.show()

    #calculate the mean accuracy
    print(np.mean(train_mse))
    print(np.mean(test_mse))
    print(np.mean(train_mae))
    print(np.mean(test_mae))
    print(np.mean(train_r2))
    print(np.mean(test_r2))
    mean_coefficients = pd.DataFrame(coefficients).mean()
    mean_coefficients = pd.concat([mean_coefficients, p_values_df, std_error_df], axis=1,
                                  keys=['coefficients', 'p_values', 'std_error'])
    print(mean_coefficients)
    mean_coefficients.to_csv('mixedeffects_csvs/correctrxntimemodel_mean_coefficients.csv')

    mean_random_effects = random_effects_df.mean(axis=0)
    print(mean_random_effects)
    big_df = pd.concat([mean_coefficients, mean_random_effects], axis=0)
    mean_random_effects.to_csv('mixedeffects_csvs//correctrxntimemodel__effects.csv')

    #make a results dictionary
    results = {'train_mse': train_mse, 'test_mse': test_mse, 'train_mae': train_mae, 'test_mae': test_mae, 'train_r2': train_r2, 'test_r2': test_r2,
                  'mean_train_mse': np.mean(train_mse), 'mean_test_mse': np.mean(test_mse), 'mean_train_mae': np.mean(train_mae), 'mean_test_mae': np.mean(test_mae), 'mean_train_r2': np.mean(train_r2), 'mean_test_r2': np.mean(test_r2)}
    #make results into a dataframe
    result = pd.DataFrame.from_dict(results)
    result.to_csv(f"D:\mixedeffectmodelsbehavioural\models/mixedeffects_csvs/correctrxntimemodel_mixed_effect_results.csv")
    return result
def run_correctrxntime_model(ferrets, optimization = False, ferret_as_feature = False, noise_floor = False):
    '''this function runs the LGBM model on the reaction time data
    :param ferrets: list of ferrets to include
    :param optimization: whether to run the optimization or not
    :param ferret_as_feature: whether to include ferret as a feature or not
    :param noise_floor: whether to shuffle the release times column 100 times to see if the model is predicting the noise floor
    :return: None'''

    df_use = extract_release_times_data(ferrets)
    #export to csv
    df_use.to_csv('D:\dfformixedmodels\correctrxntime_data.csv')
    df_use2 = df_use.copy()

    if noise_floor == True:
        #shuffle the realRelReleaseTimes column 100 times
        for i in range(1000):
            df_use2['realRelReleaseTimes'] = np.random.permutation(df_use2['realRelReleaseTimes'])
        #compare the columns
        releasetimecolumn = df_use['realRelReleaseTimes']
        releasetimecolumn2 = df_use2['realRelReleaseTimes']

        #check if they are identical
        print(np.array_equal(releasetimecolumn, releasetimecolumn2))

        talker_column = df_use['talker']
        talker_column2 = df_use2['talker']
        print(np.array_equal(talker_column, talker_column2))
        #figure out which fraction of the data is the sam
        same_list = []
        for i in range(len(talker_column)):
            if releasetimecolumn.values[i] == releasetimecolumn2.values[i]:
                same_list.append(1)
            else:
                same_list.append(0)
        #get the ratio of the same values
        print(sum(same_list)/len(same_list))

        df_use = df_use2

    run_mixed_effects_model_correctrxntime(df_use2)
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
            best_params = {
                "colsample_bytree": 0.46168728494506456,
                "alpha": 8.758272905706946,
                "n_estimators": 82,
                "learning_rate": 0.2165288044507529,
                "max_depth": 18,
                "bagging_fraction": 0.7000000000000001,
                "bagging_freq": 0
            }
        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params
            np.save('../optuna_results/best_paramsreleastimemodel_allferrets_ferretasfeature.npy', best_params)

    xg_reg, ypred, y_test, results = runlgbreleasetimes(dfx, df_use[col], paramsinput=best_params, ferret_as_feature=ferret_as_feature, ferrets = ferrets, noise_floor=noise_floor)



def run_correctrxntime_model_for_a_ferret(ferrets, optimization = False, ferret_as_feature = False ):
    '''this function runs the LGBM model on the reaction time data for a single ferret
    :param ferrets: list of ferrets to include
    :param optimization: whether to run the optimization or not
    :param ferret_as_feature: whether to include ferret as a feature or not
    :return: xg_reg: the trained model, ypred: the predicted values, y_test: the true values, results: the results of the model
    '''
    df_use = extract_release_times_data(ferrets)
    col = 'realRelReleaseTimes'
    dfx = df_use.loc[:, df_use.columns != col]


    # remove ferret as possible feature
    if ferret_as_feature == False:
        col2 = 'ferret ID'
        dfx = dfx.loc[:, dfx.columns != col2]
        if optimization == False:
            best_params = np.load('D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel2308_'+ ferrets[0]+ '.npy', allow_pickle=True).item()
        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params
            np.save('D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel2308_'+ ferrets[0]+ '.npy', best_params)
    else:
        dfx = dfx
        if optimization == False:
            # best_params = np.load('D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel_ferretasfeature2308_'+ '.npy', allow_pickle=True).item()
            best_params = {'colsample_bytree': 0.9984483617911889, 'alpha': 10.545892165925359, 'n_estimators': 120,
                       'learning_rate': 0.2585298848712121, 'max_depth': 20, 'bagging_fraction': 1.0,
                       'bagging_freq': 23, 'lambda': 0.19538105338084405, 'subsample': 0.8958044434304789,
                       'min_child_samples': 20, 'min_child_weight': 9.474782393947127, 'gamma': 0.1571174215092159,
                       'subsample_for_bin': 6200}

        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params

            np.save('D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel_ferretasfeature2308_'+ '.npy', best_params)
    xg_reg, ypred, y_test, results = runlgbreleasetimes(dfx, df_use[col], paramsinput=best_params, ferret_as_feature=ferret_as_feature, one_ferret=True, ferrets=ferrets[0])


def main():
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']  # , 'F2105_Clove']
    run_correctrxntime_model(ferrets, optimization = False, ferret_as_feature=True, noise_floor=False)
    #
    # for ferret in ferrets:
    #     run_correctrxntime_model_for_a_ferret([ferret], optimization=False, ferret_as_feature=False)



if __name__ == '__main__':
    main()