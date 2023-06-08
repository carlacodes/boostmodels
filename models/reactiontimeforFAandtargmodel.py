import npyx
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from helpers.embedworddurations import *
import shap
import matplotlib
import math
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
import matplotlib.font_manager as fm
import matplotlib.image as mpimg


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
    print("negative MSE training: %.4f" % (mse_train) + ferret_name)


    print(mse_train)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(dfx)


    # Calculate the combined cumulative sum of feature importances
    cumulative_importances_combined = np.sum(np.abs(shap_values), axis=0)
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

    fig, ax = plt.subplots(figsize=(15, 15))
    # title kwargs still does nothing so need this workaround for summary plots




    shap.summary_plot(shap_values, dfx, show=False)
    fig, ax = plt.gcf(), plt.gca()
    plt.title('Ranked list of features over their impact in predicting reaction time for' + ferret_name)
    plt.xlabel('SHAP value (impact on model output) on reaction time' + ferret_name)
    plt.savefig('figs/shap_summary_plot_correct_release_times_' + ferret_name + '.png', dpi=300, bbox_inches='tight')
    plt.show()


    trainandtestaccuracy ={
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



def runlgbreleasetimes(X, y, paramsinput=None, ferret_as_feature = False, one_ferret=False, ferrets=None, talker = 1):


    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                        random_state=42)


    from pathlib import Path
    if ferret_as_feature:
        if one_ferret:

            fig_savedir = Path('figs/absolutereleasemodel/ferret_as_feature/' + ferrets+'talker'+str(talker))
            if fig_savedir.exists():
                pass
            else:
                fig_savedir.mkdir(parents=True, exist_ok=True)
        else:
            fig_savedir = Path('figs/absolutereleasemodel/ferret_as_feature'+'talker'+str(talker))
            if fig_savedir.exists():
                pass
            else:
                fig_savedir.mkdir(parents=True, exist_ok=True)
    else:
        if one_ferret:

            fig_savedir = Path('figs/absolutereleasemodel/'+ ferrets +'talker'+str(talker))
            if fig_savedir.exists():
                pass
            else:
                fig_savedir.mkdir(parents=True, exist_ok=True)
        else:
            fig_savedir = Path('figs/absolutereleasemodel/'+'talker'+str(talker))
            if fig_savedir.exists():
                pass
            else:
                fig_savedir.mkdir(parents=True, exist_ok=True)

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
    female_word_labels = ['instruments', 'when a', 'sailor', 'in a small', 'craft', 'faces', 'of the might', 'of the vast', 'atlantic', 'ocean', 'today', 'he takes', 'the same', 'risks', 'that generations', 'took', 'before', 'him', 'but', 'in contrast', 'them', 'he can meet', 'any', 'emergency', 'that comes', 'his way', 'confidence', 'that stems', 'profound', 'trust', 'advance', 'of science', 'boats', 'stronger', 'more stable', 'protecting', 'against', 'and du', 'exposure', 'tools and', 'more ah', 'accurate', 'the more', 'reliable', 'helping in', 'normal weather', 'and conditions', 'food', 'and drink', 'of better', 'researched', 'than easier', 'to cook', 'than ever', 'before', 'rev. instruments', 'pink noise']
    male_word_labels = ['instruments', 'when a', 'sailor', 'in a', 'small', 'craft', 'faces', 'the might', 'of the', 'vast', 'atlantic', 'ocean', 'today', 'he', 'takes', 'the same', 'risks', 'that generations', 'took', 'before him', 'but', 'in contrast', 'to them', 'he', 'can meet', 'any', 'emergency', 'that comes', 'his way', 'with a', 'confidence', 'that stems', 'from', 'profound', 'trust', 'in the', 'advances', 'of science', 'boats', 'as stronger', 'and more', 'stable', 'protecting', 'against', 'undue', 'exposure', 'tools', 'and', 'accurate', 'and more', 'reliable', 'helping', 'in all', 'weather', 'and', 'rev. instruments', 'pink noise']

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
    # Plot the elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(feature_labels_words, cumulative_importances, marker='o', color = 'cyan')
    plt.xlabel('Features')
    plt.ylabel('Cumulative Feature Importance')
    if one_ferret:
        plt.title('Elbow Plot of Cumulative Feature Importance for Correct Reaction Time Model for' + ferrets + ' talker'+ talkerlist[talker -1], fontsize = 15)
    else:
        plt.title('Elbow Plot of Cumulative Feature Importance for Correct Reaction Time Model', fontsize = 20)
    plt.xticks(rotation=45, ha='right')  # rotate x-axis labels for better readability
    plt.savefig(fig_savedir / 'elbowplot.png', dpi=500, bbox_inches='tight')
    plt.show()

    shap.summary_plot(shap_values, X, show=False, cmap = cmap_color)

    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(25, 9)
    #get the labels
    labels = ax.get_yticklabels()
    labels = [item.get_text() for item in ax.get_yticklabels()]
    #set the labels
    ax.set_yticklabels(np.flip(feature_labels_words[0:20]))
    #change fontsize of xticks
    plt.xticks(rotation=45, ha='right', fontsize = 18)  # rotate x-axis labels for better readability

    if one_ferret:
        plt.title('Ranked list of features over their impact in predicting reaction time for' + ferrets+ ' talker' + talkerlist[talker -1])

    plt.xlabel('SHAP value (impact on model output) on reaction time', fontsize = 25)

    # ax.set_yticklabels(labels)
    plt.savefig(fig_savedir / 'shapsummaryplot_allanimals2.png', dpi=400, bbox_inches='tight')
    plt.show()

    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=100,
                                    random_state=123, n_jobs=2)


    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(8, 18))
    ax.barh(feature_labels_words, result.importances[sorted_idx].mean(axis=1).T, height = 0.3, color = 'cyan')
    #save the permutation importance values

    if one_ferret:
        np.save(fig_savedir / 'permutation_importance_values.npy', result.importances[sorted_idx].mean(axis=1).T)
        np.save(fig_savedir / 'permutation_importance_labels.npy', feature_labels_words)


    #rotate y -axis labels for better readability
    plt.yticks(rotation=45, ha='right')
    #make font size smaller for y tick labels
    plt.yticks(fontsize=10)
    #add whitespace between y tick labels and plot
    plt.tight_layout()
    if one_ferret:
        ax.set_title("Permutation importances on predicting the absolute release time for " + ferrets + ' talker' + talkerlist[talker -1])
    else:
        ax.set_title("Permutation importances on predicting the absolute release time, " + talkerlist[talker -1] +'talker')
    fig.tight_layout()
    plt.savefig(fig_savedir / 'permutation_importance.png', dpi=400, bbox_inches='tight')
    plt.show()

    #load the permutation importances by ferret and do a stacked bar plot based on the top 5 words:
    permutation_importance_dict = {}
    permutation_importance_labels_dict = {}
    for ferret in ferrets:
        load_dir = Path('figs/absolutereleasemodel/' + ferret + 'talker' + str(talker))

        permutation_importance_values  = np.load(load_dir / 'permutation_importance_values.npy')
        permutation_importance_labels = np.load(load_dir / 'permutation_importance_labels.npy')
        #put into dictionary
        #get the top 5 words
        #need permutation importances in descending order rather than ascending order to match labels
        permutation_importance_values = np.flip(permutation_importance_values)
        top_words = permutation_importance_values[1:6]
        top_labels = permutation_importance_labels[1:6]
        #get the top 5 words

        permutation_importance_dict[ferret] = top_words
        permutation_importance_labels_dict[ferret] = top_labels

    # common_labels = set.intersection(*[set(labels) for labels in permutation_importance_labels_dict.values()])

    # Plotting the stacked bar plot
    # Determine the unique set of all labels
    # Determine the unique set of all labels
    all_labels = set().union(*[set(labels) for labels in permutation_importance_labels_dict.values()])

    # Plotting the stacked bar plot
    fig, ax = plt.subplots()
    width = 0.5

    positions = np.arange(len(all_labels))


    # fig, ax = plt.subplots()
    # width = 0.5
    # values = []
    # labels = []
    # for i, ferret in enumerate(permutation_importance_dict.keys()):
    #
    #     #     positions = np.arange(len(values))
    #     #
    #     labels_ferrets = permutation_importance_labels_dict[ferret].tolist()
    #     common_labels = list(set(labels_ferrets) - (set(labels_ferrets) - set(labels)))
    #     positions = np.arange(len(common_labels))
    #     #if it is the first ferret, just plot the values
    #     if i == 0:
    #         values = permutation_importance_dict[ferret][0:len(common_labels)]
    #         labels = common_labels
    #         ax.bar(positions, values, width, label=ferret)
    #     else:
    #         #if it is not the first ferret, plot the values on top of the previous ferret
    #         values = permutation_importance_dict[ferret][0:len(common_labels)]
    #         labels = common_labels
    #         ax.bar(positions, values, width, label=ferret, bottom=np.sum(list(permutation_importance_dict.values())[:i], axis=0))
    #
    #     labels.append(labels_ferrets)
    #
    #
    # ax.set_ylabel('Importance')
    # ax.set_title('Top 5 Common Features by Ferret')
    # ax.set_xticks(positions)
    # ax.set_xticklabels(common_labels)
    # ax.legend()
    #
    # plt.show()
    #
    # # Plotting the stacked bar plot
    # fig, ax = plt.subplots()
    # width = 0.5
    #
    # for i, ferret in enumerate(permutation_importance_dict.keys()):
    #     values = permutation_importance_dict[ferret]
    #     labels = permutation_importance_labels_dict[ferret]
    #     positions = np.arange(len(values))
    #
    #     ax.bar(positions, values, width, label=ferret,
    #            bottom=np.sum(list(permutation_importance_dict.values())[:i], axis=0))
    #
    # ax.set_ylabel('Importance')
    # ax.set_title('Top 6 Features by Subject')
    # ax.set_xticks(positions)
    # ax.legend()
    #
    # plt.show()







    dirfemale = 'D:/Stimuli/19122022/FemaleSounds24k_addedPinkNoiseRevTargetdB.mat'
    dirmale = 'D:/Stimuli/19122022/MaleSounds24k_addedPinkNoiseRevTargetdB.mat'
    word_times, worddictionary_female= run_word_durations(dirfemale)
    word_times_male, worddict_male = run_word_durations_male(dirmale)
    # Load the male sounds dictionary

    #plot the spectrogram and amplittude waveform of each word
    #take the top 5 words from the permutation importance plot
    #GET THE TOP 5 WORDS
    top_words = X_test.columns[sorted_idx][0:4]
    for top_word in top_words:
        top_word = top_word[4:]
        #replace the word with the word number
        top_word = int(top_word)
        #re-add it to the list
        top_words = np.append(top_words, top_word)
    #remove the word with the word number
    top_words = np.delete(top_words, [0,1,2, 3])



    mosaic = ['A', 'A', 'A', 'A'],['D', 'D', 'D', 'D'] ,['D', 'D', 'D', 'D'], ['B', 'B', 'B', 'B'], ['H', 'I', 'J', 'K'], ['C', 'F', 'E', 'G']
    fig = plt.figure(figsize=(10, 35))
    ax_dict = fig.subplot_mosaic(mosaic)

    # Plot the elbow plot
    ax_dict['A'].plot(feature_labels, cumulative_importances, marker='o', color=talker_color)
    ax_dict['A'].set_xlabel('Features', fontsize = 15)
    ax_dict['A'].set_ylabel('Cumulative feature importance', fontsize = 15)
    ax_dict['A'].set_title('Elbow plot of cumulative feature importance \n on absolute reaction time,' + talker_word + ' talker', fontsize = 15)
    ax_dict['A'].set_xticklabels(feature_labels_words, rotation=45, ha='right', fontsize=12)  # rotate x-axis labels for better readability

    # rotate x-axis labels for better readability
    # summary_img = mpimg.imread(fig_savedir / 'shapsummaryplot_allanimals2.png')
    bottom = np.zeros(len(all_labels))
    import matplotlib.cm as cm

    if talker == 1:
        color_list_bar = ['purple', 'crimson', 'darkorange', 'gold', 'burlywood']
    else:
        color_list_bar = ['red', 'chocolate', 'lightsalmon', 'peachpuff', 'orange']

    # Your code here...

    my_cmap = cm.get_cmap(cmap_color)  # Choose a colormap

    for i, ferret in enumerate(permutation_importance_dict.keys()):
        values = np.zeros(len(all_labels))
        labels_ferret = permutation_importance_labels_dict[ferret]
        for j, label in enumerate(all_labels):
            if label in labels_ferret:
                index = labels_ferret.tolist().index(label)
                values[j] = permutation_importance_dict[ferret][index]

        ax_dict['B'].bar(positions, values, width, bottom=bottom, label=ferret, color = color_list_bar[i])

        bottom += values

    ax_dict['B'].set_ylabel('Permutation importance', fontsize = 15)
    ax_dict['B'].set_title('Top 5 features for predicting absolute release time, ' + talkerlist[talker -1] +' talker', fontsize = 15)
    ax_dict['B'].set_xticks(np.arange(len(all_labels)))
    ax_dict['B'].set_xticklabels(all_labels, rotation=90, fontsize = 15)
    ax_dict['B'].legend()

    # ax_dict['B'].subplots_adjust(bottom=0.3)  # Adjust the bottom margin to accommodate rotated xtick labels



    ax_dict['D'].barh(np.flip(feature_labels_words), result.importances[sorted_idx].mean(axis=1).T, color=talker_color)
    ax_dict['D'].set_title("Permutation importance features on absolute reaction time, " + talker_word + " talker ", fontsize = 15)
    ax_dict['D'].set_xlabel("Permutation importance", fontsize = 15)
    ax_dict['D'].set_ylabel("Feature", fontsize = 15)
    ax_dict['D'].set_yticks(np.flip(feature_labels_words))
    ax_dict['D'].set_yticklabels(np.flip(feature_labels_words), ha='right', fontsize=15)  # rotate x-axis labels for better readability
    # ax_dict['D'].set_yticklabels(np.flip(feature_labels_words), rotation=45, ha='right', fontsize=10)  # rotate x-axis labels for better readability


    # Plot spectrogram
    if talker == 1:
        worddict = worddictionary_female
    elif talker == 2:
        worddict = worddict_male



    pxx, freq, t, cax = ax_dict['E'].specgram(worddict[int(top_words[1]) - 1].flatten(), Fs=24414.0625, mode = 'magnitude', cmap = cmap_color)
    ax_dict['E'].set_title(f" '{feature_labels_words[1]}'")
    ax_dict['E'].set_xlabel('Time (s)')
    plt.colorbar(cax, ax = ax_dict['E'])


    # f, t, Sxx = scipy.signal.spectrogram(worddict[int(top_words[0]) - 1].flatten(), fs=24414.0625, window='hann')
    # cax = ax_dict['C'].pcolormesh(t, np.log10(f), Sxx, shading=cmap_color)
    #
    pxx, freq, t, cax = ax_dict['C'].specgram(worddict[int(top_words[0]) - 1].flatten(), Fs=24414.0625, mode = 'magnitude', cmap = cmap_color)
    ax_dict['C'].legend()
    ax_dict['C'].set_title(f" target word")
    ax_dict['C'].set_xlabel('Time (s)')
    ax_dict['C'].set_ylabel('Frequency (Hz)')

    # ax_dict['E'].set_ylabel('Frequency')
    plt.colorbar(cax, ax = ax_dict['E'])

    pxx, freq, t, cax = ax_dict['F'].specgram(worddict[int(top_words[2]) - 1].flatten(), Fs=24414.0625, mode = 'magnitude', cmap = cmap_color)
    ax_dict['F'].set_title(f"  '{feature_labels_words[2]}'")
    ax_dict['F'].set_xlabel('Time (s)')
    # ax_dict['F'].set_ylabel('Frequency')
    plt.colorbar(cax, ax = ax_dict['F'])



    #take spectrogram of the word, log10 of frequency
    if talker == 2:
        data_scaled = scaledata(worddict[int(top_words[3]) - 1].flatten(), -7977, 7797)
    else:
        data_scaled = worddict[int(top_words[3]) - 1].flatten()
    #
    # f, t, Sxx = scipy.signal.spectrogram(data_scaled, fs=24414.0625, window='hann')
    # cax = ax_dict['G'].pcolormesh(t, math.log10(f), Sxx, shading=cmap_color)
    pxx, freq, t, cax = ax_dict['G'].specgram(worddict[int(top_words[3]) - 1].flatten(), Fs=24414.0625, mode = 'magnitude', cmap = cmap_color)
    ax_dict['G'].set_title(f" '{feature_labels_words[3]}'")
    ax_dict['G'].set_xlabel('Time (s)')
    # ax_dict['G'].set_ylabel('Frequency (Hz)')
    plt.colorbar(cax, ax = ax_dict['G'], )

    # ['H', 'I', 'J', 'K'],
    ax_dict['J'].fill_between(np.arange(len(np.abs(worddict[int(top_words[1]) - 1]))) / 24414.0625, (worddict[int(top_words[1]) - 1]).flatten(), color=talker_color, alpha=0.5)
    ax_dict['J'].set_title(f"Amplitude waveform of \n '{feature_labels_words[1]}'")
    ax_dict['J'].set_ylabel('Amplitude (a.u.)')
    ax_dict['H'].set_xlabel('Time (s)')

    ax_dict['H'].fill_between(np.arange(len(np.abs(worddict[0]))) / 24414.0625, (worddict[0]).flatten(), color=talker_color, alpha=0.5)
    ax_dict['H'].set_title(f"Amplitude waveform of \n '{feature_labels_words[0]}'")
    ax_dict['H'].set_xlabel('Time (s)')

    ax_dict['I'].fill_between(np.arange(len(np.abs(worddict[int(top_words[2]) - 1]))) / 24414.0625, (worddict[int(top_words[2]) - 1]).flatten(), color=talker_color, alpha=0.5)
    ax_dict['I'].set_title(f"Amplitude waveform of \n '{feature_labels_words[2]}'")
    ax_dict['I'].set_xlabel('Time (s)')

    ax_dict['K'].fill_between(np.arange(len(data_scaled)) / 24414.0625, (data_scaled.flatten()), color=talker_color, alpha=0.5)
    ax_dict['K'].set_title(f"Amplitude waveform of \n '{feature_labels_words[3]}'")
    ax_dict['K'].set_xlabel('Time (s)')

    #remove padding outside the figures
    font_props = fm.FontProperties(weight='bold', size=17)

    ax_dict['A'].annotate('a)', xy=get_axis_limits(ax_dict['A']), xytext=(-0.1, ax_dict['A'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props, zorder=10)
    ax_dict['B'].annotate('c)', xy=get_axis_limits(ax_dict['B']), xytext=(-0.1, ax_dict['B'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    ax_dict['C'].annotate('h)', xy=get_axis_limits(ax_dict['C']), xytext=(-0.1, ax_dict['C'].title.get_position()[1]+0.15), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    ax_dict['D'].annotate('b)', xy=get_axis_limits(ax_dict['D']), xytext=(-0.1, ax_dict['D'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    ax_dict['E'].annotate('j)', xy=get_axis_limits(ax_dict['E']), xytext=(-0.1, ax_dict['E'].title.get_position()[1]+0.15), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    ax_dict['F'].annotate('i)', xy=get_axis_limits(ax_dict['F']), xytext=(-0.1, ax_dict['F'].title.get_position()[1]+0.15), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    ax_dict['G'].annotate('k)', xy=get_axis_limits(ax_dict['G']), xytext=(-0.1, ax_dict['G'].title.get_position()[1]+0.15), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    ax_dict['H'].annotate('d)', xy=get_axis_limits(ax_dict['H']), xytext=(-0.1, ax_dict['H'].title.get_position()[1]+0.15), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    ax_dict['I'].annotate('e)', xy=get_axis_limits(ax_dict['I']), xytext=(-0.1, ax_dict['I'].title.get_position()[1]+0.15), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    ax_dict['J'].annotate('f)', xy=get_axis_limits(ax_dict['J']), xytext=(-0.1, ax_dict['J'].title.get_position()[1]+0.15), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    ax_dict['K'].annotate('g)', xy=get_axis_limits(ax_dict['K']), xytext=(-0.1, ax_dict['K'].title.get_position()[1]+0.15), textcoords='axes fraction', fontproperties = font_props,zorder=10)


    plt.tight_layout()
    plt.savefig(os.path.join((fig_savedir) , str(talker) +'_talker_big_summary_plot.png'), dpi=500, bbox_inches="tight")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.9, hspace=0.55)

    plt.show()




    # fig, ax = plt.subplots()
    # ax.specgram(worddict[int(top_words[3]) - 1].flatten(), Fs=24414.0625)
    # #rescale -7977, 7977 the amplitude of worddict
    # data_scaled = scaledata(worddict[int(top_words[3]) - 1].flatten(), -7797, 7797)
    # ax.fill_between(np.arange(len(data_scaled)) / 24414.0625, np.abs(data_scaled.flatten()), color='red', alpha=0.5)
    # plt.show()


    return xg_reg, ypred, y_test, results

def extract_releasedata_withdist(ferrets, talker = 1):
    df = behaviouralhelperscg.get_df_rxntimebydist(ferrets=ferrets, includefa=True, startdate='04-01-2020', finishdate='01-03-2023', talker_param = talker)
    df_intra = df[df['intra_trial_roving'] == 1]
    df_inter = df[df['inter_trial_roving'] == 1]
    df_control = df[df['control_trial'] == 1]
    #subsample df_control so it is equal to the length of df_intra, maintain the column values
    #get names of all columns that start with lowercase dist
    dist_cols = [col for col in df_control.columns if col.startswith('dist')]
    #get a dataframe with only these columns

    if len(df_intra) > len(df_inter)*1.2:
        df_intra = df_intra.sample(n=len(df_inter), random_state=123)
    elif len(df_inter) > len(df_intra)*1.2:
        df_inter = df_inter.sample(n=len(df_intra), random_state=123)

    if len(df_control) > len(df_intra)*1.2:
        df_control = df_control.sample(n=len(df_intra), random_state=123)
    elif len(df_control) > len(df_inter)*1.2:
        df_control = df_control.sample(n=len(df_inter), random_state=123)

    #then reconcatenate the three dfs
    df = pd.concat([df_intra, df_inter, df_control], axis = 0)
    #get a dataframe with only the dist_cols and then combine with two other columns
    df_dist = df[dist_cols]
    # if talker == 1:
    #     labels = ['instruments', 'when a', 'sailor', 'in a small', 'craft', 'faces', 'of the might', 'of the vast', 'atlantic', 'ocean', 'today', 'he takes', 'the same', 'risks', 'that generations', 'took', 'before', 'him', 'but', 'in contrast', 'them', 'he can meet', 'any', 'emergency', 'that comes', 'his way', 'confidence', 'that stems', 'profound', 'trust', 'advance', 'of science', 'boats', 'stronger', 'more stable', 'protecting', 'against', 'and du', 'exposure', 'tools and', 'more ah', 'accurate', 'the more', 'reliable', 'helping in', 'normal weather', 'and conditions', 'food', 'and drink', 'of better', 'researched', 'than easier', 'to cook', 'than ever', 'before', 'rev. instruments', 'pink noise']
    # else:
    #     labels = ['instruments', 'when a', 'sailor', 'in a', 'small', 'craft', 'faces', 'the might', 'of the', 'vast', 'atlantic', 'ocean', 'today', 'he', 'takes', 'the same', 'risks', 'that generations', 'took', 'before him', 'but', 'in contrast', 'to them', 'he', 'can meet', 'any', 'emergency', 'that comes', 'his way', 'with a', 'confidence', 'that stems', 'from', 'profound', 'trust', 'in the', 'advances', 'of science', 'boats', 'as stronger', 'and more', 'stable', 'protecting', 'against', 'undue', 'exposure', 'tools', 'and', 'accurate', 'and more', 'reliable', 'helping', 'in all', 'weather', 'and', 'rev. instruments', 'pink noise']


    df_use = pd.concat([df_dist, df['centreRelease']], axis=1)
    #drop the distractors column
    df_use = df_use.drop(['distractors'], axis=1)
    if 'distractorAtten' in df_use.columns:
        df_use = df_use.drop(['distractorAtten'], axis=1)
    if 'distLvl' in df_use.columns:
        df_use = df_use.drop(['distLvl'], axis=1)

    # df_use = df_use.rename(columns=dict(zip(df_use.columns, labels)))

    return df_use


def run_correctrxntime_model(ferrets, optimization=False, ferret_as_feature=False):
    df_use = extract_releasedata_withdist(ferrets)
    col = 'realRelReleaseTimes'
    dfx = df_use.loc[:, df_use.columns != col]

    # remove ferret as possible feature
    if ferret_as_feature == False:
        col2 = 'ferret'
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
            best_params = np.load('D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel_allferrets_ferretasfeature.npy', allow_pickle=True).item()
        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params
            np.save('D:\mixedeffectmodelsbehavioural\optuna_results/best_paramsreleastimemodel_allferrets_ferretasfeature.npy', best_params)




    xg_reg, ypred, y_test, results = runlgbreleasetimes(dfx, df_use[col], paramsinput=best_params, ferret_as_feature=ferret_as_feature)



def predict_rxn_time_with_dist_model(ferrets, optimization = False, ferret_as_feature = False, talker = 2):
    df_use = extract_releasedata_withdist(ferrets, talker=talker)
    col = 'centreRelease'
    dfx = df_use.loc[:, df_use.columns != col]
    if ferret_as_feature == False:
        col2 = 'ferret'
        dfx = dfx.loc[:, dfx.columns != col2]
    #count the frequencies of each time a value is not nan by column in dfx
    #count the frequencies of each time a value is not nan by column in dfx
    counts = dfx.count(axis=0)
    #get the minimum value in the counts
    min_count = min(counts)
    #get the column names of the columns that have the minimum value
    min_count_cols = counts[counts == min_count].index
    max_count = max(counts)
    max_count_cols = counts[counts == max_count].index

    print(dfx.count(axis=0))



    # remove ferret as possible feature
    if ferret_as_feature == False:
        col2 = 'ferret'
        dfx = dfx.loc[:, dfx.columns != col2]
        if optimization == False:
            best_params = np.load('D:/mixedeffectmodelsbehavioural/optuna_results/best_paramsreleastime_dist_model_'+ ferrets[0]+'talker'+ str(talker)+'.npy', allow_pickle=True).item()
        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params
            np.save('D:\mixedeffectmodelsbehavioural/optuna_results/best_paramsreleastime_dist_model_'+ ferrets[0]+ str(talker)+'.npy', best_params)
    else:
        dfx = dfx
        if optimization == False:
            best_params = np.load('D:\mixedeffectmodelsbehavioural/optuna_results/best_paramsreleastimemodel_dist_ferretasfeature_2805'+'talker'+str(talker)+'.npy', allow_pickle=True).item()
        else:
            best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
            best_params = best_study_results.best_params
            np.save('D:\mixedeffectmodelsbehavioural/optuna_results/best_paramsreleastimemodel_dist_ferretasfeature_2805'+'talker'+str(talker)+ '.npy', best_params)
    if len(ferrets) ==1:
        one_ferret = True
        ferrets = ferrets[0]
    else:
        one_ferret = False
    xg_reg, ypred, y_test, results = runlgbreleasetimes(dfx, df_use[col], paramsinput=best_params, ferret_as_feature = ferret_as_feature, one_ferret=one_ferret, ferrets=ferrets, talker = talker)


def main():
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'] #, 'F2105_Clove']

    # ferrets = ['F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    predict_rxn_time_with_dist_model(ferrets, optimization = False, ferret_as_feature=True, talker = 1)
    #
    # for ferret in ferrets:
    #     predict_rxn_time_with_dist_model([ferret], optimization=False, ferret_as_feature=False, talker = 1)
    #     predict_rxn_time_with_dist_model([ferret], optimization=False, ferret_as_feature=False, talker = 2)




if __name__ == '__main__':
    main()