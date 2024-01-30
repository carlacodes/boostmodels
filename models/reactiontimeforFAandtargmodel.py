import npyx
import numpy as np
import sklearn.metrics
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from helpers.embedworddurations import *
from scipy.stats import spearmanr
import seaborn as sns
import shap
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from helpers.behaviouralhelpersformodels import *
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import matplotlib.cm as cm
import librosa
import librosa.display
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, median_absolute_error, r2_score


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
    '''objective function for the lightgbm model for the absolute release times
    :param trial: the trial
    :param X: the features
    :param y: the labels
    :return: the mean mse'''
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        #     colsample_bytree = 0.3, learning_rate = 0.1,
        # max_depth = 10, alpha = 10, n_estimators = 10, random_state = 42, verbose = 1
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.6),
        "alpha": trial.suggest_float("alpha", 5, 15),
        "n_estimators": trial.suggest_int("n_estimators", 2, 100, step=2),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.1, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 30, step=1),
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
    '''run the lightgbm model for the absolute release times for a ferret
    :param data: the data

    :param paramsinput: the parameters for the model
    :param ferret: the ferret number
    :param ferret_name: the ferret name
    :return: the model, the predictions, the test labels, the test mse
    '''

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
    mae_train = cross_val_score(xg_reg, X_train, y_train, scoring='neg_median_absolute_error', cv=kfold)

    mse_test = mean_squared_error(ypred, y_test)
    mse_test = cross_val_score(xg_reg, X_test, y_test, scoring='neg_mean_squared_error', cv=kfold)
    #
    results = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
    results_mae = cross_val_score(xg_reg, X_train, y_train, scoring='neg_median_absolute_error', cv=kfold)
    results_r2 = cross_val_score(xg_reg, X_train, y_train, scoring='r2', cv=kfold)

    mse_test = cross_val_score(xg_reg, X_test, y_test, scoring='neg_mean_squared_error', cv=kfold)
    mae_test = cross_val_score(xg_reg, X_test, y_test, scoring='neg_median_absolute_error', cv=kfold)
    r2_test = cross_val_score(xg_reg, X_test, y_test, scoring='r2', cv=kfold)
    print("MSE on test: %.4f" % (np.mean(mse_test)))
    print("negative MSE training: %.2f%%" % (np.mean(results) * 100.0))
    print('r2 on test: %.4f' % (np.mean(r2_test)))
    print('r2 on training: %.4f' % (np.mean(results_r2)))
    mae = median_absolute_error(ypred, y_test)
    print("MAE on test: %.4f" % (np.mean(mae_test)))
    print("negative MAE training: %.2f%%" % (np.mean(results_mae) * 100.0))

    # export all scoring results
    trainandtestaccuracy = {
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
    # savedictionary to csv
    trainandtestaccuracy = pd.DataFrame(trainandtestaccuracy)
    # np.savetxt(f'D:\mixedeffectmodelsbehavioural\metrics/absolute_rxn_model_resultsummary_talker_{talker}.csv', trainandtestaccuracy, delimiter=',', fmt='%s')


    print("MSE on test: %.4f" % (mse_test) + ferret_name)
    print("negative MSE training: %.4f" % (mse_train) + ferret_name)

    print('MAE on test: %.4f' % (mae_train) + ferret_name)
    print('MAE on test: %.4f' % (mae_train) + ferret_name)

    shap_values = shap.TreeExplainer(xg_reg).shap_values(dfx)

    # Calculate the combined cumulative sum of feature importances
    cumulative_importances_combined = np.sum(np.abs(shap_values), axis=0)
    feature_labels = dfx.columns
    # Plot the elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(feature_labels, cumulative_importances_combined, marker='o', color='slategray')
    plt.xlabel('Features')
    plt.ylabel('Cumulative Feature Importance')
    plt.title('Elbow Plot of Cumulative Feature Importance for False Alarm Model')
    plt.xticks(rotation=45, ha='right')  # rotate x-axis labels for better readability
    plt.savefig('D:/behavmodelfigs/fa_or_not_model/elbowplot.png', dpi=500, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    # title kwargs still does nothing so need this workaround for summary plots

    shap.summary_plot(shap_values, dfx, show=False)
    fig, ax = plt.gcf(), plt.gca()
    plt.title('Ranked list of features over their impact in predicting reaction time for' + ferret_name)
    plt.xlabel('SHAP value (impact on model output) on reaction time' + ferret_name)
    plt.savefig('figs/shap_summary_plot_correct_release_times_' + ferret_name + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    trainandtestaccuracy = {
        'ferret': ferret_name,
        'mse_test': mse_test,
        'mse_train': mse_train,
        'mean_mse_train': np.mean(mse_train),
    }
    np.save('metrics/modelmse' + ferret_name + '.npy', trainandtestaccuracy)

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


def runlgbreleasetimes(X, y, paramsinput=None, ferret_as_feature=False, one_ferret=False, ferrets=None, talker=1, noise_floor=False, bootstrap_words = True):
    '''run the lightgbm model for the absolute release times
    :param X: the features
    :param y: the labels
    :param paramsinput: the parameters for the model
    :param ferret_as_feature: whether to include the ferret as a feature
    :param one_ferret: whether to run the model for one ferret or all ferrets
    :param ferrets: the ferret name
    :param talker: the talker number
    :param noise_floor: whether to include the noise floor as a feature
    :param bootstrap_words: whether to include the bootstrap words as a feature
    :return: the model, the predictions, the test labels, the test mse

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    from pathlib import Path
    if ferret_as_feature:
        if one_ferret:

            fig_savedir = Path('figs/absolutereleasemodel/ferret_as_feature/' + ferrets + 'talker' + str(talker))
            if fig_savedir.exists():
                pass
            else:
                fig_savedir.mkdir(parents=True, exist_ok=True)
        else:
            fig_savedir = Path('figs/absolutereleasemodel/ferret_as_feature' + 'talker' + str(talker))
            if fig_savedir.exists():
                pass
            else:
                fig_savedir.mkdir(parents=True, exist_ok=True)
    else:
        if one_ferret:

            fig_savedir = Path('figs/absolutereleasemodel/' + ferrets + 'talker' + str(talker))

        else:
            fig_savedir = Path('figs/absolutereleasemodel/' + 'talker' + str(talker))
    if noise_floor == True:
        fig_savedir = fig_savedir / 'with_noise_floor'
    if bootstrap_words == False:
        fig_savedir = fig_savedir / 'with_no_bootstrap_words'
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
    female_word_labels = ['instruments', 'when a', 'sailor', 'in a small', 'craft', 'faces', 'of the might',
                          'of the vast', 'atlantic', 'ocean', 'today', 'he takes', 'the same', 'risks',
                          'that generations', 'took', 'before', 'him', 'but', 'in contrast', 'them', 'he can meet',
                          'any', 'emergency', 'that comes', 'his way', 'confidence', 'that stems', 'profound', 'trust',
                          'advance', 'of science', 'boats', 'stronger', 'more stable', 'protecting', 'against',
                          'and du', 'exposure', 'tools and', 'more ah', 'accurate', 'the more', 'reliable',
                          'helping in', 'normal weather', 'and conditions', 'food', 'and drink', 'of better',
                          'researched', 'than easier', 'to cook', 'than ever', 'before']
    male_word_labels = ['instruments', 'when a', 'sailor', 'in a', 'small', 'craft', 'faces', 'the might', 'of the',
                        'vast', 'atlantic', 'ocean', 'today', 'he', 'takes', 'the same', 'risks', 'that generations',
                        'took', 'before him', 'but', 'in contrast', 'to them', 'he', 'can meet', 'any', 'emergency',
                        'that comes', 'his way', 'with a', 'confidence', 'that stems', 'from', 'profound', 'trust',
                        'in the', 'advances', 'of science', 'boats', 'as stronger', 'and more', 'stable', 'protecting',
                        'against', 'undue', 'exposure', 'tools', 'and', 'accurate', 'and more', 'reliable', 'helping',
                        'in all', 'weather', 'and']

    kfold = KFold(n_splits=5)
    results = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
    results_mae = cross_val_score(xg_reg, X_train, y_train, scoring='neg_median_absolute_error', cv=kfold)
    results_r2 = cross_val_score(xg_reg, X_train, y_train, scoring='r2', cv=kfold)

    mse_test = cross_val_score(xg_reg, X_test, y_test, scoring='neg_mean_squared_error', cv=kfold)
    mae_test = cross_val_score(xg_reg, X_test, y_test, scoring='neg_median_absolute_error', cv=kfold)
    r2_test = cross_val_score(xg_reg, X_test, y_test, scoring='r2', cv=kfold)
    print("MSE on test: %.4f" % (np.mean(mse_test)))
    print("negative MSE training: %.4f" % (np.mean(results)))
    print('r2 on test: %.4f' % (np.mean(r2_test)))
    print('r2 on training: %.4f' % (np.mean(results_r2)))
    mae = median_absolute_error(ypred, y_test)
    print("MAE on test: %.4f" % (np.mean(mae_test)))
    print("negative MAE training: %.2f" % (np.mean(results_mae)))

    # export all scoring results
    trainandtestaccuracy = {
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


    # savedictionary to csv
    trainandtestaccuracy = pd.DataFrame(trainandtestaccuracy)
    trainandtestaccuracy.to_csv(f'D:\mixedeffectmodelsbehavioural\metrics/absolute_rxn_time_modelmetrics_talker_{talker}.csv')

    print(results)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(X)
    feature_importances = np.abs(shap_values).sum(axis=0)
    sorted_indices = np.argsort(feature_importances)

    sorted_indices = sorted_indices[::-1]
    feature_importances = feature_importances[sorted_indices]
    feature_labels = X.columns[sorted_indices]
    if talker == 1:
        female_word_labels = np.array(female_word_labels)

        feature_labels_words = female_word_labels[sorted_indices]
        talker_word = 'female'
        talker_color = 'purple'
        cmap_color = 'plasma'
    else:
        male_word_labels = np.array(male_word_labels)

        feature_labels_words = male_word_labels[sorted_indices]
        talker_word = 'male'
        talker_color = 'tomato'
        cmap_color = 'inferno'

    cumulative_importances = np.cumsum(feature_importances)

    talkerlist = ['female', 'male'
                  ]

    print('calculating permutation importance now')
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=100,
                                    random_state=123, n_jobs=2)

    print('finished calculating permutation importance now')
    sorted_idx = (result.importances.mean(axis=1)).argsort()
    if talker == 1:
        feature_labels_words_permutation = female_word_labels[sorted_idx]
    else:
        feature_labels_words_permutation = male_word_labels[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 18))
    ax.barh(feature_labels_words_permutation, result.importances[sorted_idx].mean(axis=1), color='cyan')
    permutation_importance_dataframe = pd.DataFrame(result.importances[sorted_idx].mean(axis=1), index=(feature_labels_words_permutation))
    #concatenate the arrays horizontally
    permutation_importance_array = np.concatenate(((feature_labels_words_permutation).reshape(-1,1), (sorted_idx).reshape(-1,1), result.importances[sorted_idx].mean(axis=1).reshape(-1,1)), axis=1)

    #save the dataframe
    np.save(fig_savedir / f'permutation_importance_dataframe_talker_{talker}.npy', permutation_importance_dataframe)
    np.save(fig_savedir / f'permutation_importance_array_talker_{talker}.npy', permutation_importance_array)

    if one_ferret:
        np.save(fig_savedir / 'permutation_importance_values.npy', result.importances[sorted_idx].mean(axis=1).T)
        np.save(fig_savedir / 'permutation_importance_labels.npy', feature_labels_words)

    plt.yticks(rotation=45, ha='right')
    # make font size smaller for y tick labels
    plt.yticks(fontsize=10)
    # add whitespace between y tick labels and plot
    plt.tight_layout()
    if one_ferret:
        ax.set_title(
            "Permutation importances on predicting the absolute release time for " + ferrets + ' talker' + talkerlist[
                talker - 1])
    else:
        ax.set_title(
            "Permutation importances on predicting the absolute release time, " + talkerlist[talker - 1] + 'talker')
    fig.tight_layout()
    plt.savefig(fig_savedir / 'permutation_importance.png', dpi=400, bbox_inches='tight')
    plt.show()

    # load the permutation importances by ferret and do a stacked bar plot based on the top 5 words:
    permutation_importance_dict = {}
    permutation_importance_labels_dict = {}
    if one_ferret == False:
        for ferret in ferrets:
            load_dir = Path('figs/absolutereleasemodel/' + ferret + 'talker' + str(talker))

            permutation_importance_values = np.load(load_dir / 'permutation_importance_values.npy')
            permutation_importance_labels = np.load(load_dir / 'permutation_importance_labels.npy')

            permutation_importance_values = np.flip(permutation_importance_values)
            top_words = permutation_importance_values[1:6]
            top_labels = permutation_importance_labels[1:6]
            # get the top 5 words

            permutation_importance_dict[ferret] = top_words
            permutation_importance_labels_dict[ferret] = top_labels

        all_labels = set().union(*[set(labels) for labels in permutation_importance_labels_dict.values()])

        # Plotting the stacked bar plot
        fig, ax = plt.subplots()
        width = 0.5

        positions = np.arange(len(all_labels))
        dirfemale = 'D:/Stimuli/19122022/FemaleSounds24k_addedPinkNoiseRevTargetdB.mat'
        dirmale = 'D:/Stimuli/19122022/MaleSounds24k_addedPinkNoiseRevTargetdB.mat'
        word_times, worddictionary_female = run_word_durations(dirfemale)
        word_times_male, worddict_male = run_word_durations_male(dirmale)

        # GET THE TOP 5 WORDS
        top_words = np.flip(X_test.columns[sorted_idx])[0:4]
        for top_word in top_words:
            top_word = top_word[4:]
            # replace the word with the word number
            top_word = int(top_word)
            # re-add it to the list
            top_words = np.append(top_words, top_word)
        # remove the word with the word number
        top_words = np.delete(top_words, [0, 1, 2, 3])

        mosaic = ['A', 'A', 'A', 'A', 'A'], ['D1', 'D1', 'D1', 'B', 'B'], ['D2', 'D2', 'D2', 'H', 'I'], ['D2', 'D2', 'D2', 'C',  'F', ], ['D2', 'D2', 'D2',  'J', 'K'],\
            ['D2', 'D2',  'D2',  'E','G']

        text_width_pt = 419.67816  # Replace with your value
        text_height_pt = 717.00946

        # Convert the text width from points to inches
        text_width_inches = text_width_pt / 72.27
        text_height_inches = text_height_pt / 72.27

        fig, ax = plt.subplots(figsize=(15,5))
        ax.plot(feature_labels, cumulative_importances, marker='o', color=talker_color)
        # ax_dict['A'].set_xlabel('Features', fontsize=15)
        ax.set_ylabel('Cumulative feature importance', fontsize=15)
        ax.set_title(
            'Elbow plot of cumulative feature importance in absolute reaction time model, ' + talker_word + ' talker',
            fontsize=15)
        ax.set_xticklabels(feature_labels_words, rotation=35, ha='right',fontsize=20)
        plt.savefig(os.path.join((fig_savedir), str(talker) + 'elobowplot_1606_noannotation.pdf'), dpi=500, bbox_inches='tight')
        plt.show()
        if talker == 1:
            color_list_bar = ['purple', 'crimson', 'darkorange', 'gold', 'burlywood']
        else:
            color_list_bar = ['red', 'chocolate', 'lightsalmon', 'peachpuff', 'orange']
        bottom = np.zeros(len(all_labels))
        fig, ax = plt.subplots(figsize=(15,5))
        for i, ferret in enumerate(permutation_importance_dict.keys()):
            values = np.zeros(len(all_labels))
            labels_ferret = permutation_importance_labels_dict[ferret]
            for j, label in enumerate(all_labels):
                if label in labels_ferret:
                    index = labels_ferret.tolist().index(label)
                    values[j] = permutation_importance_dict[ferret][index]
            ax.bar(positions, values, width, bottom=bottom, label=ferret, color=color_list_bar[i])
            bottom += values
        ax.set_ylabel('Permutation importance', fontsize=20)
        ax.set_title('Top 5 features for predicting absolute release time, ' + talkerlist[talker - 1] + ' talker',
                               fontsize=30)
        ax.set_xticks(np.arange(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=70, fontsize=20)
        ax.legend(fontsize=20)
        plt.savefig(os.path.join((fig_savedir), str(talker) + 'stackedbarplot_1606_noannotation.pdf'), dpi=500, bbox_inches='tight')
        plt.show()


        fig = plt.figure(figsize=(text_width_inches*3, (text_height_inches)*3))
        # fig = plt.figure(figsize=(20, 27))
        ax_dict = fig.subplot_mosaic(mosaic)
        # Plot the elbow plot
        ax_dict['A'].plot(feature_labels, cumulative_importances, marker='o', color=talker_color)
        # ax_dict['A'].set_xlabel('Features', fontsize=15)
        ax_dict['A'].set_ylabel('Cumulative feature importance', fontsize=15)
        ax_dict['A'].set_title(
            'Elbow plot of cumulative feature importance \n on absolute reaction time,' + talker_word + ' talker',
            fontsize=15)
        ax_dict['A'].set_xticklabels(feature_labels_words, rotation=35, ha='right',
                                     fontsize=10)  # rotate x-axis labels for better readability
        # rotate x-axis labels for better readability
        # summary_img = mpimg.imread(fig_savedir / 'shapsummaryplot_allanimals2.png')
        bottom = np.zeros(len(all_labels))

        for i, ferret in enumerate(permutation_importance_dict.keys()):
            values = np.zeros(len(all_labels))
            labels_ferret = permutation_importance_labels_dict[ferret]
            for j, label in enumerate(all_labels):
                if label in labels_ferret:
                    index = labels_ferret.tolist().index(label)
                    values[j] = permutation_importance_dict[ferret][index]
            ax_dict['B'].bar(positions, values, width, bottom=bottom, label=ferret, color=color_list_bar[i])
            bottom += values
        ax_dict['B'].set_ylabel('Permutation importance', fontsize=15)
        ax_dict['B'].set_title('Top 5 features for predicting \n absolute release time, ' + talkerlist[talker - 1] + ' talker',
                               fontsize=15)
        ax_dict['B'].set_xticks(np.arange(len(all_labels)))
        ax_dict['B'].set_xticklabels(all_labels, rotation=70, fontsize=10)
        ax_dict['B'].legend()


        data_perm_importance = (result.importances[sorted_idx].mean(axis=1).T)
        labels = np.flip(feature_labels_words_permutation)

        # Bar plot on the first axes
        ax_dict['D1'].barh(labels[0], data_perm_importance[-1], color=talker_color)
        ax_dict['D1'].set_title("Permutation importance features on absolute reaction time, " + talker_word + " talker",
                                fontsize=15)
        # ax_dict['D1'].set_xlabel("Permutation importance", fontsize=15)
        ax_dict['D1'].set_ylabel("Feature", fontsize=15)
        ax_dict['D1'].set_yticks(labels[0])

        # Bar plot on the second axes
        ax_dict['D2'].barh(labels[1:], data_perm_importance[1:], color=talker_color)
        # ax_dict['D2'].set_title("Permutation importance features on absolute reaction time, " + talker_word + " talker",
        #                         fontsize=15)
        ax_dict['D2'].set_xlabel("Permutation importance", fontsize=15)
        ax_dict['D2'].set_yticks(labels[1:])
        ax_dict['D2'].set_yticklabels(labels[1:], ha='right', fontsize=15)

        # zoom-in / limit the view to different portions of the data
        ax_dict['D1'].set_xlim(0, np.max(data_perm_importance) * 1.1)  # Adjust the limit based on your data
        ax_dict['D2'].set_xlim(0, 0.06)  # Adjust the limit based on your data

        # hide the spines between ax_dict['D1'] and ax_dict['D2']
        ax_dict['D1'].spines['right'].set_visible(False)
        ax_dict['D2'].spines['left'].set_visible(False)
        ax_dict['D1'].yaxis.tick_left()
        ax_dict['D1'].tick_params(labeltop='off')  # don't put tick labels at the top
        ax_dict['D2'].yaxis.tick_right()

        # Make the spacing between the two axes a bit smaller
        plt.subplots_adjust(wspace=0.15)

        d = 0.015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax_dict['D1'].transAxes, color='k', clip_on=False)
        ax_dict['D1'].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
        ax_dict['D1'].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

        kwargs.update(transform=ax_dict['D2'].transAxes)  # switch to the bottom axes
        ax_dict['D2'].plot((-d, d), (-d, +d), **kwargs)  # top-right diagonal
        ax_dict['D2'].plot((-d, d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        if talker == 1:
            worddict = worddictionary_female
        elif talker == 2:
            worddict = worddict_male
        # pxx, freq, t, cax = ax_dict['E'].specgram(worddict[int(top_words[1]) - 1].flatten(), Fs=24414.0625, mode = 'psd', cmap = cmap_color)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(worddict[int(top_words[1]) - 1].flatten())))
        cax = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax_dict['E'], sr=24414.0625, cmap=cmap_color)
        # remove colorbar
        # fig.colorbar(cax, ax=None)
        ax_dict['E'].set_title(f" '{feature_labels_words[1]}'")
        ax_dict['E'].set_xlabel('Time (s)')
        ax_dict['E'].set_xticks(np.round(np.arange(0.1, len(worddict[int(top_words[1]) - 1]) / 24414.0625, 0.1), 2))
        ax_dict['E'].set_xticklabels(np.round(np.arange(0.1, len(worddict[int(top_words[1]) - 1]) / 24414.0625, 0.1), 2),
                                     rotation=45, ha='right', fontsize=10)
        # plt.colorbar(cax, ax=None)
        # f, t, Sxx = scipy.signal.spectrogram(worddict[int(top_words[0]) - 1].flatten(), fs=24414.0625, window='hann')
        # cax = ax_dict['C'].pcolormesh(t, np.log10(f), Sxx, shading=cmap_color)
        #
        # pxx, freq, t, cax = ax_dict['C'].specgram(worddict[int(top_words[0]) - 1].flatten(), Fs=24414.0625, mode = 'psd', cmap = cmap_color)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(worddict[int(top_words[0]) - 1].flatten())))
        cax = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=24414.0625, ax=ax_dict['C'], cmap=cmap_color)
        ax_dict['C'].legend()
        ax_dict['C'].set_title(f" instruments")
        ax_dict['C'].set_xlabel('Time (s)')
        ax_dict['C'].set_xticks(np.round(np.arange(0.1, len(worddict[int(top_words[0]) - 1]) / 24414.0625, 0.1), 2))
        ax_dict['C'].set_xticklabels(np.round(np.arange(0.1, len(worddict[int(top_words[0]) - 1]) / 24414.0625, 0.1), 2),
                                     rotation=45, ha='right', fontsize=10)
        ax_dict['C'].set_ylabel('Frequency (Hz)')
        # plt.colorbar(cax, ax = ax_dict['C'])
        # pxx, freq, t, cax = ax_dict['F'].specgram(worddict[int(top_words[2]) - 1].flatten(), Fs=24414.0625, mode = 'psd', cmap = cmap_color)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(worddict[int(top_words[2]) - 1].flatten())))
        cax = librosa.display.specshow(D, sr=24414.0625, x_axis='time', y_axis='log', ax=ax_dict['F'], cmap=cmap_color)
        # fig.colorbar(cax, ax=None)
        ax_dict['F'].set_title(f"  '{feature_labels_words[2]}'")
        ax_dict['F'].set_xlabel('Time (s)')
        ax_dict['F'].set_xticks(np.round(np.arange(0.1, (len(worddict[int(top_words[2]) - 1]) / 24414.0625), 0.1), 2))
        ax_dict['F'].set_xticklabels(np.round(np.arange(0.1, len(worddict[int(top_words[2]) - 1]) / 24414.0625, 0.1), 2),
                                     rotation=45, ha='right', fontsize=10)
        # plt.colorbar(cax, ax = None)
        # take spectrogram of the word, log10 of frequency
        if talker == 2:
            data_scaled = scaledata(worddict[int(top_words[3]) - 1].flatten(), -7977, 7797)
        else:
            data_scaled = worddict[int(top_words[3]) - 1].flatten()
        ax_dict['G'].set_title(f" '{feature_labels_words[3]}'")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data_scaled)))
        # Plot the spectrogram on a logarithmic scale
        cax = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=24414.0625, ax=ax_dict['G'], cmap=cmap_color)
        ax_dict['G'].set_xlabel('Time (s)')
        ax_dict['G'].set_xticks(np.round(np.arange(0.1, len(data_scaled) / 24414.0625, 0.1), 2))
        ax_dict['G'].set_xticklabels(np.round(np.arange(0.1, len(data_scaled) / 24414.0625, 0.1), 2), rotation=45,
                                     ha='right', fontsize=10)
        # plt.colorbar(cax, ax=None )
        # ['H', 'I', 'J', 'K'],
        ax_dict['J'].fill_between(np.arange(len(np.abs(worddict[int(top_words[1]) - 1]))) / 24414.0625,
                                  (worddict[int(top_words[1]) - 1]).flatten(), color=talker_color, alpha=0.5)
        ax_dict['J'].set_title(f"'{feature_labels_words[1]}'")
        # ax_dict['J'].set_ylabel('Amplitude (a.u.)')
        ax_dict['H'].set_xlabel('Time (s)')
        ax_dict['H'].fill_between(np.arange(len(np.abs(worddict[0]))) / 24414.0625, (worddict[0]).flatten(),
                                  color=talker_color, alpha=0.5)
        ax_dict['H'].set_title(f" '{feature_labels_words[0]}'")
        ax_dict['H'].set_xlabel('Time (s)')
        ax_dict['I'].fill_between(np.arange(len(np.abs(worddict[int(top_words[2]) - 1]))) / 24414.0625,
                                  (worddict[int(top_words[2]) - 1]).flatten(), color=talker_color, alpha=0.5)
        ax_dict['I'].set_title(f"'{feature_labels_words[2]}'")
        ax_dict['I'].set_xlabel('Time (s)')
        ax_dict['K'].fill_between(np.arange(len(data_scaled)) / 24414.0625, (data_scaled.flatten()), color=talker_color,
                                  alpha=0.5)
        ax_dict['K'].set_title(f"'{feature_labels_words[3]}'")
        ax_dict['K'].set_xlabel('Time (s)')

        plt.subplots_adjust(wspace=0.33, hspace=0.53)

        plt.savefig(os.path.join((fig_savedir), str(talker) + '_talker_big_summary_plot_1606_noannotation.png'), dpi=500)
        plt.savefig(os.path.join((fig_savedir), str(talker) + '_talker_big_summary_plot_1606_noannotation.pdf'), dpi=500, bbox_inches='tight')
        fig.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(15,7))
        mosaic = [ 'H', 'I' ,'J', 'K'], [ 'C',  'F', 'E','G' ]
        ax_dict = fig.subplot_mosaic(mosaic)

        if talker == 1:
            worddict = worddictionary_female
        elif talker == 2:
            worddict = worddict_male
        # pxx, freq, t, cax = ax_dict['E'].specgram(worddict[int(top_words[1]) - 1].flatten(), Fs=24414.0625, mode = 'psd', cmap = cmap_color)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(worddict[int(top_words[1]) - 1].flatten())))
        cax = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax_dict['E'], sr=24414.0625, cmap=cmap_color)
        # remove colorbar
        ax_dict['E'].set_title(f" '{feature_labels_words[1]}'")
        ax_dict['E'].set_xlabel('Time (s)')
        ax_dict['E'].set_xticks(np.round(np.arange(0.1, len(worddict[int(top_words[1]) - 1]) / 24414.0625, 0.1), 2))
        ax_dict['E'].set_xticklabels(
            np.round(np.arange(0.1, len(worddict[int(top_words[1]) - 1]) / 24414.0625, 0.1), 2),
            rotation=45, ha='right', fontsize=10)

        D = librosa.amplitude_to_db(np.abs(librosa.stft(worddict[int(top_words[0]) - 1].flatten())))
        cax = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=24414.0625, ax=ax_dict['C'], cmap=cmap_color)
        ax_dict['C'].set_title(f" instruments")
        ax_dict['C'].set_xlabel('Time (s)')
        ax_dict['C'].set_xticks(np.round(np.arange(0.1, len(worddict[int(top_words[0]) - 1]) / 24414.0625, 0.1), 2))
        ax_dict['C'].set_xticklabels(
            np.round(np.arange(0.1, len(worddict[int(top_words[0]) - 1]) / 24414.0625, 0.1), 2),
            rotation=45, ha='right', fontsize=10)
        ax_dict['C'].set_ylabel('Frequency (Hz)')
        D = librosa.amplitude_to_db(np.abs(librosa.stft(worddict[int(top_words[2]) - 1].flatten())))
        cax = librosa.display.specshow(D, sr=24414.0625, x_axis='time', y_axis='log', ax=ax_dict['F'], cmap=cmap_color)
        ax_dict['F'].set_title(f"  '{feature_labels_words[2]}'")
        ax_dict['F'].set_xlabel('Time (s)')
        ax_dict['F'].set_xticks(np.round(np.arange(0.1, (len(worddict[int(top_words[2]) - 1]) / 24414.0625), 0.1), 2))
        ax_dict['F'].set_xticklabels(
            np.round(np.arange(0.1, len(worddict[int(top_words[2]) - 1]) / 24414.0625, 0.1), 2),
            rotation=45, ha='right', fontsize=10)
        if talker == 2:
            data_scaled = scaledata(worddict[int(top_words[3]) - 1].flatten(), -7977, 7797)
        else:
            data_scaled = worddict[int(top_words[3]) - 1].flatten()
        ax_dict['G'].set_title(f" '{feature_labels_words[3]}'")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data_scaled)))
        # Plot the spectrogram on a logarithmic scale
        cax = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=24414.0625, ax=ax_dict['G'], cmap=cmap_color)
        ax_dict['G'].set_xlabel('Time (s)')
        ax_dict['G'].set_xticks(np.round(np.arange(0.1, len(data_scaled) / 24414.0625, 0.1), 2))
        ax_dict['G'].set_xticklabels(np.round(np.arange(0.1, len(data_scaled) / 24414.0625, 0.1), 2), rotation=45,
                                     ha='right', fontsize=10)

        ax_dict['J'].fill_between(np.arange(len(np.abs(worddict[int(top_words[1]) - 1]))) / 24414.0625,
                                  (worddict[int(top_words[1]) - 1]).flatten(), color=talker_color, alpha=0.5)
        ax_dict['J'].set_title(f"'{feature_labels_words[1]}'")
        ax_dict['H'].set_xlabel('Time (s)')
        ax_dict['H'].fill_between(np.arange(len(np.abs(worddict[0]))) / 24414.0625, (worddict[0]).flatten(),
                                  color=talker_color, alpha=0.5)
        ax_dict['H'].set_title(f" '{feature_labels_words[0]}'")
        ax_dict['H'].set_xlabel('Time (s)')
        ax_dict['I'].fill_between(np.arange(len(np.abs(worddict[int(top_words[2]) - 1]))) / 24414.0625,
                                  (worddict[int(top_words[2]) - 1]).flatten(), color=talker_color, alpha=0.5)
        ax_dict['I'].set_title(f"'{feature_labels_words[2]}'")
        ax_dict['I'].set_xlabel('Time (s)')
        ax_dict['K'].fill_between(np.arange(len(data_scaled)) / 24414.0625, (data_scaled.flatten()), color=talker_color,
                                  alpha=0.5)
        ax_dict['K'].set_title(f"'{feature_labels_words[3]}'")
        ax_dict['K'].set_xlabel('Time (s)')
        plt.subplots_adjust(wspace=0.50, hspace=0.53)
        plt.savefig(os.path.join((fig_savedir), str(talker) + '_spectrograms_1606_noannotation.png'), dpi=500)
        plt.show()


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 15), gridspec_kw={'width_ratios': [4, 1]})
        # Plotting the majority of values on the first subplot
        ax2.barh(np.arange(0,len(data_perm_importance),1), data_perm_importance[:], color=talker_color)
        ax2.set_xlim(max(data_perm_importance[:-1])+0.002, data_perm_importance[-1] +0.05 )  # Adjust xlim for the outlier
        ax2.set_yticklabels([])

        ax1.barh(np.arange(0,len(data_perm_importance),1), data_perm_importance[:], color = talker_color)
        ax1.set_xlim(0, max(data_perm_importance[:-1])+0.002)  # Adjust xlim for the majority of values
        ax1.set_yticks(np.arange(0,len(data_perm_importance),1))
        ax1.set_yticklabels(np.flip(labels[:]), rotation = 45)

        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
        # ax1.set_xlabel('Value')
        fig.subplots_adjust(wspace=0)

        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        #remove ticks in the middle
        ax1.tick_params(right=False)
        ax2.tick_params(left=False)
        #add a break in the x-axis
        d = .015  # how big to make the diagonal lines in axes coordinates
        #plot the diagonal lines on the first subplot
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        #increase the x ticks font size
        ax1.tick_params(axis='x', labelsize=12)
        ax2.tick_params(axis='x', labelsize=12)

        fig.text(0.5, 0.00, 'Permutation Importance', ha='center', va='center', fontsize = 25)  # Add a common x-axis label

        # Display the chart
        plt.savefig(os.path.join((fig_savedir), str(talker) + '_permutationimportance_plot_2906_noannotation.png'), dpi=500, bbox_inches='tight')
        plt.show()
        if talker == 1:
            words_to_plot = [
            "he takes",
            "today",
            "of science",
            "but",
            "sailor",
            "any",
            "accurate",
            "in contrast",
            "researched",
            "boats",
            "took",
            "when a",
            "craft"
        ]
            words_to_plot = words_to_plot[::-1]
            feature_labels_words = feature_labels_words[::-1]

            fig, ax = plt.subplots(figsize=(15, 5))
            # do the same plot as above but with the words to plot
            # get the indices of the words to plot
            indices = []
            # need to reverse the feature labels words to match the permutation importance values

            for word in words_to_plot:
                index = np.where(feature_labels_words == word)[0][0]
                indices.append(index)
            indices = np.array(indices)
            indices = indices.flatten()

            # get the values of the words to plot
            values = data_perm_importance[indices]
            # get the labels of the words to plot
            labels = feature_labels_words[indices]
            data_for_plot = pd.DataFrame({'labels': labels, 'values': values})
            # data_for_plot = data_for_plot.sort_values(by='values', ascending=True)
            labels = data_for_plot['labels']
            values = data_for_plot['values']
            plt.xticks(fontsize = 15)
            plt.yticks(fontsize = 15)

            # plot the values
            ax.barh(labels, values, color=talker_color)
            ax.set_xlabel('Permutation importance', fontsize=20)
            ax.set_title('Permutation importance of words in absolute reaction time model, ' + talker_word + ' talker',
                            fontsize=20)
            plt.savefig(os.path.join((fig_savedir), str(talker) + 'permutationimportance_plot_frombehaviouralmodel_2906_noannotation.png'), dpi=500, bbox_inches='tight')
            plt.show()

    return xg_reg, ypred, y_test, results


def extract_releasedata_withdist(ferrets, talker=1, bootstrap_words = True):
    ''' This function extracts the data from the behavioural database and then subsamples the data so that the number of pitch conditions is realtively equal.
    :param ferrets: list of ferrets to include in the analysis
    :param talker: talker to include in the analysis
    :param bootstrap_words: whether to bootstrap the words or not
    :return: df_use: dataframe with the subsampled data'''
    df = behaviouralhelperscg.get_df_rxntimebydist(ferrets=ferrets, includefa=True, startdate='04-01-2020',
                                                   finishdate='01-03-2023', talker_param=talker)
    df_intra = df[df['intra_trial_roving'] == 1]
    df_inter = df[df['inter_trial_roving'] == 1]
    df_control = df[df['control_trial'] == 1]
    # subsample df_control so it is equal to the length of df_intra, maintain the column values
    # get names of all columns that start with lowercase dist
    dist_cols = [col for col in df_control.columns if col.startswith('dist') or col.startswith('centreRelease')]
    # get a dataframe with only these columns

    if len(df_intra) > len(df_inter) * 1.2:
        df_intra = df_intra.sample(n=len(df_inter), random_state=123)
    elif len(df_inter) > len(df_intra) * 1.2:
        df_inter = df_inter.sample(n=len(df_intra), random_state=123)

    if len(df_control) > len(df_intra) * 1.2:
        df_control = df_control.sample(n=len(df_intra), random_state=123)
    elif len(df_control) > len(df_inter) * 1.2:
        df_control = df_control.sample(n=len(df_inter), random_state=123)

    # then reconcatenate the three dfs
    df = pd.concat([df_intra, df_inter, df_control], axis=0)
    # get a dataframe with only the dist_cols and then combine with two other columns
    df_dist = df[dist_cols]
    #remove dist56 and dist57
    if 'distractorAtten' in df_dist.columns:
        df_dist = df_dist.drop(['distractorAtten'], axis=1)
    if 'distLvl' in df_dist.columns:
        df_dist = df_dist.drop(['distLvl'], axis=1)
    df_dist = df_dist.drop(['dist56', 'dist57', 'distractors'], axis=1)

    #subsample along the dist columns so the length corresponding to each word token column is the same length
    #get the counts of each column in the dataframe and then find the mean count

    # Calculate the mean count of each word (excluding 'dist1') in the dataframe
    if bootstrap_words == True:
        word_counts = df_dist.count(axis=0)
        # Determine the minimum count among the words (excluding 'dist1')
        min_count = word_counts.min()
        # List to hold subsampled DataFrames for each word
        subsampled_dfs = []

        # Iterate through each word (excluding 'dist1', and 'centreRelease')
        col_list = df_dist.columns.drop(['dist1', 'centreRelease']).to_list()
        for k in range(1, 20):
            col_list = np.flip(col_list)
            for col in col_list:
                word_df_notna = df_dist[df_dist[col].notna()]
                word_df_na = df_dist[df_dist[col].isna()]

                # Stratified subsampling based on the original frequencies
                if col == 'dist20' or col =='dist42'or col=='dist5':
                    print('higher freq word')
                    n_samples = int(min_count * 0.025)
                elif col =='dist32' or col =='dist2':
                    print('super high freq word')
                    continue
                else:
                    n_samples = int(min_count)
                word_df_notna_subsampled = word_df_notna.sample(n=n_samples, random_state=123, replace=True)
                word_df_na_subsampled = word_df_na.sample(n=n_samples, random_state=123, replace=True)

                # Append to list of DataFrames
                subsampled_dfs.append(pd.concat([word_df_notna_subsampled, word_df_na_subsampled], axis=0))

            # Concatenate the subsampled DataFrames for each word to create the final dataframe
        df_dist = pd.concat(subsampled_dfs, axis=0)

        female_word_labels = ['instruments', 'when a', 'sailor', 'in a small', 'craft', 'faces', 'of the might',
                              'of the vast', 'atlantic', 'ocean', 'today', 'he takes', 'the same', 'risks',
                              'that generations', 'took', 'before', 'him', 'but', 'in contrast', 'them', 'he can meet',
                              'any', 'emergency', 'that comes', 'his way', 'confidence', 'that stems', 'profound', 'trust',
                              'advance', 'of science', 'boats', 'stronger', 'more stable', 'protecting', 'against',
                              'and du', 'exposure', 'tools and', 'more ah', 'accurate', 'the more', 'reliable',
                              'helping in', 'normal weather', 'and conditions', 'food', 'and drink', 'of better',
                              'researched', 'than easier', 'to cook', 'than ever', 'before']
        male_word_labels = ['instruments', 'when a', 'sailor', 'in a', 'small', 'craft', 'faces', 'the might', 'of the',
                            'vast', 'atlantic', 'ocean', 'today', 'he', 'takes', 'the same', 'risks', 'that generations',
                            'took', 'before him', 'but', 'in contrast', 'to them', 'he', 'can meet', 'any', 'emergency',
                            'that comes', 'his way', 'with a', 'confidence', 'that stems', 'from', 'profound', 'trust',
                            'in the', 'advances', 'of science', 'boats', 'as stronger', 'and more', 'stable', 'protecting',
                            'against', 'undue', 'exposure', 'tools', 'and', 'accurate', 'and more', 'reliable', 'helping',
                            'in all', 'weather', 'and']

        fig, ax = plt.subplots(figsize=(30, 10))

        # Plot them in order in a bar plot from highest to lowest
        # Drop centrelrease
        df_dist_counts = df_dist.drop(['centreRelease'], axis=1)
        df_dist_counts = df_dist_counts.count(axis=0)

        # Sort the values as well as female_word_labels
        sorted_idx_distlabels = df_dist_counts.argsort()
        df_dist_counts = df_dist_counts.sort_values(ascending=False)
        if talker == 1:
            talker_color = 'purple'
        else:
            talker_color = 'orange'
        df_dist_counts.plot.bar(ax=ax, color = talker_color)
        ax.set_xlabel('Word', fontsize = 18)
        ax.set_ylabel('Number of occurrences', fontsize = 18)

        ax.set_xticks(np.arange(0, 55, 1))
        if talker == 1:
            ax.set_xticklabels(np.flip(np.array(female_word_labels)[sorted_idx_distlabels]), rotation=45, fontsize=15)
            plt.title('Distribution of bootstrapped words for female talker model', fontsize = 25)
        else:
            ax.set_xticklabels(np.flip(np.array(male_word_labels)[sorted_idx_distlabels]), rotation=45, fontsize=15)
            plt.title('Distribution of bootstrapped words for male talker model', fontsize = 25)

        plt.savefig(
            'D:\mixedeffectmodelsbehavioural\models/figs/absolutereleasemodel/distribution_non_nan_values_by_column_dfx_talker' + str(
                talker) + '.png')
        plt.show()

        fig, ax = plt.subplots(figsize=(30, 10))
        ax.barh(np.array(female_word_labels)[sorted_idx_distlabels][0], df_dist_counts[-1], color=talker_color)

        # ax_dict['D1'].set_xlabel("Permutation importance", fontsize=15)
        ax.set_ylabel("Feature", fontsize=15)
        ax.set_yticks(np.flip(np.array(female_word_labels)[sorted_idx_distlabels])[0])
        # ax_dict['D1'].set_yticklabels(labels[0], ha='right', fontsize=15)

        # Bar plot on the second axes
        ax.barh(np.array(female_word_labels)[sorted_idx_distlabels][1:], df_dist_counts[1:], color=talker_color)
        # ax_dict['D2'].set_title("Permutation importance features on absolute reaction time, " + talker_word + " talker",
        #                         fontsize=15)
        ax.set_xlabel("Permutation importance", fontsize=15)
        ax.set_yticks(np.array(female_word_labels)[sorted_idx_distlabels][1:])
        ax.set_yticklabels(np.array(female_word_labels)[sorted_idx_distlabels][1:], ha='right', fontsize=15)

        # zoom-in / limit the view to different portions of the data
        ax.set_xlim(0, np.max(df_dist_counts) * 1.1)  # Adjust the limit based on your data
        ax.set_xlim(0, 0.06)  # Adjust the limit based on your data

        # hide the spines between ax_dict['D1'] and ax_dict['D2']
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.tick_left()
        ax.tick_params(labeltop='off')  # don't put tick labels at the top
        ax.yaxis.tick_right()

        # Make the spacing between the two axes a bit smaller
        plt.subplots_adjust(wspace=0.15)
        if talker == 1:
            ax.set_xticklabels(np.flip(np.array(female_word_labels)[sorted_idx_distlabels]), rotation=45, fontsize=15)
            plt.title('Distribution of bootstrapped words for female talker model', fontsize=25)
        else:
            ax.set_xticklabels(np.flip(np.array(male_word_labels)[sorted_idx_distlabels]), rotation=45, fontsize=15)
            plt.title('Distribution of bootstrapped words for male talker model', fontsize=25)

        plt.savefig(
            'D:\mixedeffectmodelsbehavioural\models/figs/absolutereleasemodel/distribution_non_nan_values_by_column_dfx_talker_shrunk' + str(
                talker) + '.png')
        plt.show()

        fig, ax1 = plt.subplots(figsize=(10, 10))  # Adjust the figsize as needed

        # Bar plot on the first axes (left y-axis)
        bar_labels = np.array(female_word_labels)[sorted_idx_distlabels]
        bar_width = df_dist_counts[-1]  # Set bar width to the corresponding value

        ax1.barh(bar_labels[0], bar_width, color=talker_color)

        ax1.set_ylabel("Feature", fontsize=15)
        ax1.set_yticks([bar_labels[0]])
        ax1.set_yticklabels([bar_labels[0]], ha='right', fontsize=15)

        # Set the x-limits for the left axes to cover the space for dist1
        max_width = max(np.max(bar_width), np.max(df_dist_counts[1:]))
        ax1.set_xlim(0, max_width * 1.1)  # Adjust the limit based on your data

        # Create a second y-axis on the right side
        ax2 = ax1.twinx()

        # Bar plot on the second axes (right y-axis)
        bar_widths = df_dist_counts[1:]  # Set bar widths to the corresponding values
        ax2.barh(bar_labels[1:], bar_widths, color=talker_color)

        ax2.set_xlabel("Permutation importance", fontsize=15)
        ax2.set_yticks(bar_labels[1:])
        ax2.set_yticklabels(bar_labels[1:], ha='right', fontsize=15)

        # Set the x-limits for the right axes to cover the space for dist2 and others
        ax2.set_xlim(0, max_width * 1.1)  # Adjust the limit based on your data

        # Hide the x-axis tick labels and spines for both axes
        ax1.set_xticklabels([])  # Remove x-axis tick labels for ax1
        ax2.set_xticklabels([])  # Remove x-axis tick labels for ax2
        ax1.spines['top'].set_visible(False)  # Hide the top spine for ax1
        ax1.spines['bottom'].set_visible(False)  # Hide the bottom spine for ax1
        ax2.spines['top'].set_visible(False)  # Hide the top spine for ax2
        ax2.spines['bottom'].set_visible(False)  # Hide the bottom spine for ax2

        # Make the spacing between the two axes a bit smaller
        plt.subplots_adjust(wspace=0.15)

        if talker == 1:
            ax1.set_title('Distribution of bootstrapped words for female talker model', fontsize=25)
        else:
            ax1.set_title('Distribution of bootstrapped words for male talker model', fontsize=25)

        plt.tight_layout()  # Ensure all elements fit within the figure
        plt.savefig(
            'D:\mixedeffectmodelsbehavioural\models/figs/absolutereleasemodel/distribution_non_nan_values_by_column_dfx_talker_shrunk' + str(
                talker) + '.png')
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 15), gridspec_kw={'width_ratios': [4, 1]})

        # Plotting dist_counts on the first subplot
        dist_counts = df_dist_counts  # Replace this with your actual data

        # Bar plot on the first subplot
        ax1.barh(np.arange(0, len(dist_counts), 1), dist_counts[:], color=talker_color)
        ax1.set_xlim(0, max(dist_counts[1:]) + 0.002)  # Adjust xlim for the majority of values
        ax1.set_yticks(np.arange(0, len(dist_counts), 1))
        ax1.set_yticklabels(np.flip(bar_labels[:]), rotation=45)

        # Plotting dist_counts on the second subplot
        # You need to adapt the following lines for the correct data and limits
        ax2.barh(np.arange(0, len(dist_counts), 1), dist_counts[:], color=talker_color)
        ax2.set_xlim(max(dist_counts[1:]) + 0.002, max(dist_counts) + 0.05)  # Adjust xlim for the outlier
        ax2.set_yticklabels([])
        #make the x ticks in sciencetific notation
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.subplots_adjust(wspace=0)

        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        # Remove ticks in the middle
        ax1.tick_params(right=False)
        ax2.tick_params(left=False)

        # Add a break in the x-axis
        d = .015  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        #increase the font size of the x ticks
        ax1.tick_params(axis='x', labelsize=20)
        ax2.tick_params(axis='x', labelsize=20)


        # Add a common x-axis label
        fig.text(0.5, 0.00, 'Number of instances', ha='center', va='center', fontsize=25)
        if talker == 1:
            ax1.set_title('Distribution of bootstrapped words for female talker model', fontsize=25)
        else:
            ax1.set_title('Distribution of bootstrapped words for male talker model', fontsize=25)

        # Display the chart
        plt.savefig(os.path.join('D:\mixedeffectmodelsbehavioural/models/figs/absolutereleasemodel/', str(talker) + '_distcounts_plot.png'), dpi=500, bbox_inches='tight')
        plt.show()

    df_use = df_dist

    # drop the distractors column


    # df_use = df_use.rename(columns=dict(zip(df_use.columns, labels)))

    return df_use
#
#
# def run_correctrxntime_model(ferrets, optimization=False, ferret_as_feature=False, noise_floor = False):
#     df_use = extract_releasedata_withdist(ferrets)
#     col = 'realRelReleaseTimes'
#     dfx = df_use.loc[:, df_use.columns != col]
#
#     # remove ferret as possible feature
#     if ferret_as_feature == False:
#         col2 = 'ferret'
#         dfx = dfx.loc[:, dfx.columns != col2]
#         if optimization == False:
#             best_params = np.load('../optuna_results/best_paramsreleastimemodel_allferrets.npy',
#                                   allow_pickle=True).item()
#         else:
#             best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
#             best_params = best_study_results.best_params
#             np.save('../optuna_results/best_paramsreleastimemodel_allferrets.npy', best_params)
#     else:
#         dfx = dfx
#         if optimization == False:
#             best_params = np.load(
#                 'D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel_allferrets_ferretasfeature.npy',
#                 allow_pickle=True).item()
#         else:
#             best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
#             best_params = best_study_results.best_params
#             np.save(
#                 'D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel_allferrets_ferretasfeature.npy',
#                 best_params)
#
#     xg_reg, ypred, y_test, results = runlgbreleasetimes(dfx, df_use[col], paramsinput=best_params,
#                                                         ferret_as_feature=ferret_as_feature, noise_floor=noise_floor)

def run_mixed_effects_model_absrxntime(df, talker =1):
    '''run a mixed effects model on the absolute reaction time data
    df: dataframe with the absolute reaction time data
    talker: talker type, 1 is female, 2 is male
    returns: the model results'''

    #split the data into training and test set
    #relabel the labels by addding underscore for each label
    female_word_labels = ['release_time', 'instruments', 'when a', 'sailor', 'in a small', 'craft', 'faces', 'of the might',
                          'of the vast', 'atlantic', 'ocean', 'today', 'he takes', 'the same', 'risks',
                          'that generations', 'took', 'before', 'him', 'but', 'in contrast', 'them', 'he can meet',
                          'any', 'emergency', 'that comes', 'his way', 'confidence', 'that stems', 'profound', 'trust',
                          'advance', 'of science', 'boats', 'stronger', 'more stable', 'protecting', 'against',
                          'and du', 'exposure', 'tools and', 'more ah', 'accurate', 'the more', 'reliable',
                          'helping in', 'normal weather', 'and conditions', 'food', 'and drink', 'of better',
                          'researched', 'than easier', 'to cook', 'than ever', 'before']
    male_word_labels = ['release_time', 'instruments', 'when a', 'sailor', 'in a', 'small', 'craft', 'faces', 'the might', 'of the',
                        'vast', 'atlantic', 'ocean', 'today', 'he', 'takes', 'the same', 'risks', 'that generations',
                        'took', 'before him', 'but', 'in contrast', 'to them', 'he', 'can meet', 'any', 'emergency',
                        'that comes', 'his way', 'with a', 'confidence', 'that stems', '_from', 'profound', 'trust',
                        'in the', 'advances', 'of science', 'boats', 'as stronger', 'and more', 'stable', 'protecting',
                        'against', 'undue', 'exposure', 'tools', 'a_n_d', 'accurate', 'and more', 'reliable', 'helping',
                        'in all', 'weather', 'a_n_d']


    for i, col in enumerate(df.columns):
        if talker == 1:
            df.rename(columns={col: female_word_labels[i]}, inplace=True)
        else:
            df.rename(columns={col: male_word_labels[i]}, inplace=True)

    for col in df.columns:
        df.rename(columns={col: col.replace(" ", "_")}, inplace=True)

    #now define the equation
    if talker == 1:
        equation = 'release_time ~instruments + when_a + sailor + in_a_small + craft + faces + of_the_might + of_the_vast + atlantic + ocean + today + he_takes + the_same + risks + that_generations + took + before + him + but + in_contrast + them + he_can_meet + any + emergency + that_comes + his_way + confidence + that_stems + profound + trust + advance + of_science + boats + stronger + more_stable + protecting + against + and_du + exposure + tools_and + more_ah + accurate + the_more + reliable + helping_in + normal_weather + and_conditions + food + and_drink + of_better + researched + than_easier + to_cook + than_ever + before'
    else:
        equation = 'release_time ~instruments + when_a + sailor + in_a + small + craft + faces + the_might + of_the + vast + atlantic + ocean + today + he + takes + the_same + risks + that_generations + took + before_him + but + in_contrast + to_them + he + can_meet + any + emergency + that_comes + his_way + with_a + confidence + that_stems + _from + profound + trust + in_the + advances + of_science + boats + as_stronger + and_more + stable + protecting + against + undue + exposure + tools  + accurate + and_more + reliable + helping + in_all + weather + a_n_d'

    #not perfect but for mixed effects model, need to fill the missing rows with 0
    df = df.fillna(0)
    #drop the rows with missing values
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    train_mse = []
    test_mse = []
    train_mae = []
    test_mae = []
    train_r2 = []
    test_r2 = []
    coefficients = []
    p_values = []
    std_error = []
    for train_index, test_index in kf.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]

        # model = smf.mixedlm(equation, train)
        #make a linear m model without random effects
        model = smf.ols(equation, train)
        result = model.fit()
        print(result.summary())
        coefficients.append(result.params)


        #calculate the mean squared error
        ypred_train = result.predict(train)
        y_train = train['release_time']
        mse_train = mean_squared_error(y_train, ypred_train)
        mae_train = median_absolute_error(y_train, ypred_train)
        r2_train = r2_score(y_train, ypred_train)
        train_r2.append(r2_train)

        train_mse.append(mse_train)
        train_mae.append(mae_train)
        print(mse_train)


        ypred = result.predict(test)
        y_test = test['release_time']
        mse = mean_squared_error(y_test, ypred)
        r2_test = r2_score(y_test, ypred)
        test_mse.append(mse)
        test_r2.append(r2_test)
        #calculate the median absolute error
        mae = median_absolute_error(y_test, ypred)
        p_values.append(result.pvalues)
        std_error.append(result.bse)
        test_mae.append(mae)
        print(mae)

        print(mse)

    coefficients_df = pd.DataFrame(coefficients).mean()
    p_values_df = pd.DataFrame(p_values).mean()
    std_error_df = pd.DataFrame(std_error).mean()
    # combine into one dataframe
    if talker == 1:
        color_text = 'purple'
        talker_text = 'Female Talker'
    else:
        color_text = 'darkorange'
        talker_text = 'Male Talker'
    result_coefficients = pd.concat([coefficients_df, p_values_df, std_error_df], axis=1, keys=['coefficients', 'p_values', 'std_error'])
    #replace GroupVar with text Ferret
    fig, ax = plt.subplots(figsize = (20,5))
    # sort the coefficients by their mean value
    result_coefficients = result_coefficients.sort_values(by='coefficients', ascending=False)

    ax.bar(result_coefficients.index, result_coefficients['coefficients'], color=color_text)
    ax.errorbar(result_coefficients.index, result_coefficients['coefficients'], yerr=result_coefficients['std_error'], fmt='none', ecolor='black', elinewidth=1, capsize=2)

    # ax.set_xticklabels(result_coefficients['features'], rotation=45, ha='right')
    # if the mean p value is less than 0.05, then add a star to the bar plot
    for i in range(len(result_coefficients)):
        if result_coefficients['p_values'][i] < 0.05:
            ax.text(i, 0.00, '*', fontsize=20)
    ax.set_xlabel('Features')
    plt.yticks(fontsize=15)
    ax.set_ylabel('Mean Coefficient', fontsize = 20)
    plt.xticks(rotation=45, ha='right')
    ax.set_title(f'Mean Coefficient for Each Feature, Absolute Reaction Time Model, {talker_text}', fontsize = 20)
    plt.savefig(f'mixedeffects_csvs/mean_coefficients_absolute_rxn_time_talker_{talker}.png', dpi=500, bbox_inches='tight')
    plt.show()

    #calculate the mean accuracy
    print(np.mean(train_mse))
    print(np.mean(test_mse))
    print(np.mean(train_mae))
    print(np.mean(test_mae))
    mean_coefficients = pd.DataFrame(coefficients).mean()
    mean_coefficients = pd.concat([mean_coefficients, p_values_df, std_error_df], axis=1, keys=['coefficients', 'p_values', 'std_error'])
    print(mean_coefficients)
    mean_coefficients.to_csv(f"mixedeffects_csvs/absrxntimemodel_talker_{talker}_mean_coefficients.csv")

    results = {'train_mse': train_mse, 'test_mse': test_mse, 'train_mae': train_mae, 'test_mae': test_mae, 'train_r2': train_r2, 'test_r2': test_r2,
               'mean_train_mse': np.mean(train_mse), 'mean_test_mse': np.mean(test_mse), 'mean_train_mae': np.mean(train_mae), 'mean_test_mae': np.mean(test_mae),'mean_train_r2': np.mean(train_r2), 'mean_test_r2': np.mean(test_r2)}
    df_results = pd.DataFrame.from_dict(results)
    
    df_results.to_csv(f"mixedeffects_csvs/absrxntimemodel_talker_{talker}_mixed_effect_results.csv")

    
    return result
def predict_rxn_time_with_dist_model(ferrets, optimization=False, ferret_as_feature=False, talker=2, noise_floor = False, bootstrap_words = False):
    '''run a gradient-boosted regression tree model on the absolute reaction time data
    ferrets: list of ferrets to include in the model
    optimization: whether to run an optimization or not
    ferret_as_feature: whether to include ferret as a feature in the model
    talker: talker type, 1 is female, 2 is male
    noise_floor: whether to calculate a noise floor metric for the dataset
    bootstrap_words: whether to bootstrap the words in the dataset
    returns: the model results
    '''
    df_use = extract_releasedata_withdist(ferrets, talker=talker, bootstrap_words=bootstrap_words)
    df_use2 = df_use.copy()
    # run_mixed_effects_model_absrxntime(df_use2, talker = talker)
    col = 'centreRelease'

    if noise_floor == True:
        #shuffle the realRelReleaseTimes column 100 times
        for i in range(1000):
            df_use2['centreRelease'] = np.random.permutation(df_use2['centreRelease'])
        #compare the columns
        releasetimecolumn = df_use['centreRelease']
        releasetimecolumn2 = df_use2['centreRelease']

        #check if they are identical
        print(np.array_equal(releasetimecolumn, releasetimecolumn2))

        talker_column = df_use['dist10']
        talker_column2 = df_use2['dist10']
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

    dfx = df_use.loc[:, df_use.columns != col]
    if ferret_as_feature == False:
        col2 = 'ferret'
        dfx = dfx.loc[:, dfx.columns != col2]
    # count the frequencies of each time a value is not nan by column in dfx
    counts = dfx.count(axis=0)
    # get the minimum value in the counts
    min_count = min(counts)
    # get the column names of the columns that have the minimum value
    min_count_cols = counts[counts == min_count].index
    max_count = max(counts)
    max_count_cols = counts[counts == max_count].index

    print(dfx.count(axis=0))

    # remove ferret as possible feature
    if ferret_as_feature == False:
        col2 = 'ferret'
        dfx = dfx.loc[:, dfx.columns != col2]
        if optimization == False:
            best_params = np.load(
                'D:/mixedeffectmodelsbehavioural/optuna_results/best_paramsreleastime_dist_model_' + ferrets[
                    0] + str(talker) + '.npy', allow_pickle=True).item()

            if talker == 1:
                best_params = {
                    'colsample_bytree': 0.9984483617911889,
                    'alpha': 10.545892165925359,
                    'n_estimators': 120,
                    'learning_rate': 0.2585298848712121,
                    'max_depth': 20,
                    'bagging_fraction': 1.0,
                    'bagging_freq': 23,
                    'lambda': 0.19538105338084405,
                    'subsample': 0.8958044434304789,
                    'min_child_samples': 20,
                    'min_child_weight': 9.474782393947127,
                    'gamma': 0.1571174215092159,
                    'subsample_for_bin': 6200
                }

            elif talker == 2:
                best_params = {
                    'colsample_bytree': 0.5870762820095368,
                    'alpha': 10.840482953967314,
                    'n_estimators': 70,
                    'learning_rate': 0.18038495501541654,
                    'max_depth': 20,
                    'bagging_fraction': 0.9,
                    'bagging_freq': 30
                }


        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params
            np.save(
                'D:\mixedeffectmodelsbehavioural/optuna_results/best_paramsreleastime_dist_model_' + ferrets[0] + str(
                    talker) + '2007.npy', best_params)
    else:
        dfx = dfx
        if optimization == False:
            best_params = np.load(
                'D:\mixedeffectmodelsbehavioural/optuna_results/best_paramsreleastimemodel_dist_ferretasfeature_2805' + 'talker' + str(
                    talker) + '2007.npy', allow_pickle=True).item()
        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params
            np.save(
                'D:\mixedeffectmodelsbehavioural/optuna_results/best_paramsreleastimemodel_dist_ferretasfeature_2805' + 'talker' + str(
                    talker) + '2007.npy', best_params)
    if len(ferrets) == 1:
        one_ferret = True
        ferrets = ferrets[0]
    else:
        one_ferret = False



    xg_reg, ypred, y_test, results = runlgbreleasetimes(dfx, df_use[col], paramsinput=best_params,
                                                        ferret_as_feature=ferret_as_feature, one_ferret=one_ferret,
                                                        ferrets=ferrets, talker=talker, noise_floor=noise_floor, bootstrap_words = bootstrap_words)
    return results
def compare_bootstrap_permutation_test_results():
    '''compare the permutation importance values with and without subsampling by plotting, supplmental figure'''
    load_dir = 'D:\mixedeffectmodelsbehavioural\models/figs/absolutereleasemodel/'
    array_female_raw = np.load(load_dir+'/talker1/'+'with_no_bootstrap_words/permutation_importance_array_talker_1.npy')
    array_female_bootstrap = np.load(load_dir+'/talker1/'+'permutation_importance_array_talker_1.npy')
    array_male_raw = np.load(load_dir+'/talker2/'+'with_no_bootstrap_words/permutation_importance_array_talker_2.npy')
    array_male_bootstrap= np.load(load_dir+'/talker2/'+'permutation_importance_array_talker_2.npy')
    #rearrange array_male_bootstrap so it's in the same order  as array_male_raw
    words_raw = array_male_raw[:,0]
    words_bootstrap = array_male_bootstrap[:,0]

    words_raw_female = array_female_raw[:,0]
    words_bootstrap_female = array_female_bootstrap[:,0]

    #make a dictionary with the words and the permutation importance values
    #rearrange the words_bootstrap so that it's in the same order as words_raw
    array_male_bootstrap_reordered = []
    for word in words_raw:
        index = np.where(words_bootstrap == word)[0][0]
        array_male_bootstrap_reordered.append(array_male_bootstrap[index, :])
    array_male_bootstrap_reordered = np.array(array_male_bootstrap_reordered)


    array_female_bootstrap_reordered = []
    for word in words_raw_female:
        index = np.where(words_bootstrap_female == word)[0][0]
        array_female_bootstrap_reordered.append(array_female_bootstrap[index, :])
    array_female_bootstrap_reordered = np.array(array_female_bootstrap_reordered)


    #make a dataframe with the permutation importance values out of the two permutation importance arrays
    female_df = pd.DataFrame(array_female_raw[:,2].astype(float), columns=['permutation_importance'], index = array_female_raw[:,0] )
    female_df['permutation_importance_bootstrap'] = array_female_bootstrap_reordered[:,2].astype(float)
    male_df = pd.DataFrame(array_male_raw[:,2].astype(float), columns=['permutation_importance'], index = array_male_raw[:,0] )
    male_df['permuation_importance_bootstrap'] = array_male_bootstrap_reordered[:,2].astype(float)
    female_df.reset_index(inplace=True)
    male_df.reset_index(inplace=True)
    #remove instruments from both dataframes
    female_df = female_df[female_df.index != 'instruments']
    male_df = male_df[male_df.index != 'instruments']
    fig, ax = plt.subplots(figsize = (10,5))
    # plt.scatter(x=female_df['permutation_importance'], y=female_df['permutation_importance_bootstrap'], color='purple')
    sns.scatterplot(data = female_df, x='permutation_importance', y='permutation_importance_bootstrap', color='purple', label = 'female')
    sns.scatterplot(data= male_df, x='permutation_importance', y='permuation_importance_bootstrap', color='darkorange', label = 'male')
    #make the xticks rounded
    #get the spearmans correlation
    spearman_corr_female, _ = spearmanr(female_df['permutation_importance'], female_df['permutation_importance_bootstrap'])
    spearman_corr_male, _ =spearmanr(male_df['permutation_importance'], male_df['permuation_importance_bootstrap'])


    ax.set_xlim(0, 0.03)
    ax.set_ylim(0, 0.04)
    #put spearman correlation in a text box
    plt.annotate(f"spearman's correlation female: {spearman_corr_female:.2f}, \n spearman's correlation male: {spearman_corr_male:.2f}",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=15, ha='left', va='top', color='black')
    ax.legend(fontsize=12, loc = 'lower right')
    ax.set_xlabel('Permutation importance \n without subsampling', fontsize = 18)
    ax.set_ylabel('Permutation importance \n  with subsampling', fontsize = 18)
    plt.title('Permutation Importances vs Permutation Importances with Subsampling', fontsize = 18)
    plt.savefig('D:\mixedeffectmodelsbehavioural\models/figs/absolutereleasemodel/permutation_importance_vs_permutation_importance_with_subsampling.png', bbox_inches='tight')
    plt.show()
    return




def main():
    # compare_bootstrap_permutation_test_results()
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    predict_rxn_time_with_dist_model(ferrets, optimization=False, ferret_as_feature=False, talker=1, noise_floor=False, bootstrap_words=True)
    # for ferret in ferrets:
    #     predict_rxn_time_with_dist_model([ferret], optimization=False, ferret_as_feature=False, talker = 1)
    #     predict_rxn_time_with_dist_model([ferret], optimization=False, ferret_as_feature=False, talker = 2)


if __name__ == '__main__':
    main()
