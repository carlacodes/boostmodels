import seaborn as sns
from sklearn.inspection import permutation_importance
from pathlib import Path
from sklearn.model_selection import cross_val_score
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import shap
import matplotlib
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from helpers.behaviouralhelpersformodels import *
import sklearn.metrics as metrics
import random

def shap_summary_plot(
        shap_values2,
        feature_labels,
        ax=None,
        cmap = "viridis",
        show_plots=False,
        savefig=False,
        savefig_path=None,
    ):
    plt.rcParams['font.family'] = 'sans-serif'

    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    plt.sca(ax)
    shap.plots.beeswarm(shap_values2, show=False, color=cmap)


def objective(trial, X, y):
    '''run the objective function for optuna
    :param trial: optuna trial object
    :param X: training data
    :param y: training labels
    :return: mean cross validation score
    '''
    param_grid = {
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.5),
        "num_leaves": trial.suggest_int("num_leaves", 20, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 20),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 20),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10),
        "max_bin": trial.suggest_int("max_bin", 100, 1000),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.1, 50),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(objective="binary", random_state=42, **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            verbose=False,
        )
        preds = model.predict_proba(X_test)[:, 1]
        cv_scores.append(metrics.log_loss(y_test, preds))

    return np.mean(cv_scores)


def get_axis_limits(ax, scale=1):
    return ax.get_xlim()[0] * scale, (ax.get_ylim()[1] * scale)

def run_optuna_study_correctresp(X, y):
    '''run the optuna study for the correct response model
    :param X: training data
    :param y: training labels
    :return: optuna study object
    '''
    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=1000)
    print("Number of finished trials: ", len(study.trials))
    print(f"\tBest value of - auc: {study.best_value:.5f}")
    print(f"\tBest params:")

    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")
    return study

def run_mixed_effects_model_correctresp(df):
    '''run the mixed effects model for the correct response model
    :param df: dataframe
    :return: mixed effects model object
    '''
    ferrets= ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']
    equation = 'misslist ~ talker + side + precur_and_targ_same + targTimes + pastcorrectresp + pastcatchtrial + pitchoftarg'
    #split the data into training and test set
    #drop the rows with missing values
    df = df.dropna()
    #calculate the probability of a miss when the targtimes >=5.5
    df['misslist'] = df['misslist'].astype(int)
    df['targTimes'] = df['targTimes'].astype(float)
    df['pastcorrectresp'] = df['pastcorrectresp'].astype('category')
    df['precur_and_targ_same'] = df['precur_and_targ_same'].astype('category')
    df['side'] = df['side'].astype('category')
    df['side'] = df['side'].replace({0: 'Left', 1: 'Right'})

    df['talker'] = df['talker'].astype('category')
    df['precur_and_targ_same'] = df['precur_and_targ_same'].astype('category')
    df['pastcatchtrial'] = df['pastcatchtrial'].astype('category')
    df['pitchoftarg'] = df['pitchoftarg'].astype('category')
    df['talker'] = df['talker'].replace({1: 'Male', 2: 'Female'})
    df['pitchoftarg'] = df['pitchoftarg'].replace({1: '109 Hz', 2: '124 Hz', 3: '144 Hz', 4: '191 Hz', 5: '251 Hz'})
    df['ferret'] = df['ferret'].astype('category')



    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    fold_index = 1
    train_acc = []
    test_acc = []
    coefficients = []
    p_values = []
    std_error = []
    std_error_re = []
    random_effects_df = pd.DataFrame()
    for train_index, test_index in kf.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]

        model = smf.mixedlm(equation, train, groups=train["ferret"])
        result = model.fit()
        print(result.summary())

        var_resid = result.scale
        var_random_effect = float(result.cov_re.iloc[0])
        var_fixed_effect = result.predict(df).var()
        # coefficients.append(result.params)

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

        total_var = var_fixed_effect + var_random_effect + var_resid
        marginal_r2 = var_fixed_effect / total_var
        conditional_r2 = (var_fixed_effect + var_random_effect) / total_var

        # Generate confusion matrix for train set
        y_pred_train = result.predict(train)
        y_pred_train = (y_pred_train > 0.5).astype(int)
        y_true_train = train['misslist'].to_numpy()
        confusion_matrix_train = confusion_matrix(y_true_train, y_pred_train)
        print(confusion_matrix_train)

        # Calculate balanced accuracy for train set
        balanced_accuracy_train = balanced_accuracy_score(y_true_train, y_pred_train)
        print(balanced_accuracy_train)
        train_acc.append(balanced_accuracy_train)

        # Export confusion matrix and balanced accuracy for train set
        np.savetxt(f"mixedeffects_csvs/correctresp_confusionmatrix_train_fold{fold_index}.csv", confusion_matrix_train,
                   delimiter=",")
        np.savetxt(f"mixedeffects_csvs/correctresp_balac_train_fold{fold_index}.csv", [balanced_accuracy_train],
                   delimiter=",")

        # Generate confusion matrix for test set
        y_pred = result.predict(test)
        y_pred = (y_pred > 0.5).astype(int)
        y_true = test['misslist'].to_numpy()
        confusion_matrix_test = confusion_matrix(y_true, y_pred)
        print(confusion_matrix_test)

        # Calculate balanced accuracy for test set
        balanced_accuracy_test = balanced_accuracy_score(y_true, y_pred)
        print(balanced_accuracy_test)
        test_acc.append(balanced_accuracy_test)

        # Export confusion matrix and balanced accuracy for test set
        np.savetxt(f"mixedeffects_csvs/correctresp_confusionmatrix_test_fold{fold_index}.csv", confusion_matrix_test,
                   delimiter=",")
        np.savetxt(f"mixedeffects_csvs/correctresp_balac_test_fold{fold_index}.csv", [balanced_accuracy_test],
                   delimiter=",")

        fold_index += 1  # Increment fold index
    #calculate the mean accuracy
    coefficients_df = pd.DataFrame(coefficients).mean()
    p_values_df = pd.DataFrame(p_values).mean()
    std_error_df = pd.DataFrame(std_error).mean()
    std_error_re_df = pd.DataFrame(std_error_re).mean()
    # combine into one dataframe
    result_coefficients = pd.concat([coefficients_df, p_values_df, std_error_df], axis=1, keys=['coefficients', 'p_values', 'std_error'])
    fig, ax = plt.subplots()
    # sort the coefficients by their mean value
    result_coefficients.index = result_coefficients.index.str.replace('Group Var', 'ferret ID')
    result_coefficients.index = result_coefficients.index.str.replace('targTimes', 'target presentation time')
    result_coefficients.index = result_coefficients.index.str.replace('side', 'audio side')
    result_coefficients.index = result_coefficients.index.str.replace('pitchoftarg', 'target pitch')
    result_coefficients.index = result_coefficients.index.str.replace('pastcatchtrial', 'past trial was catch')

    result_coefficients.index = result_coefficients.index.str.replace('pastcorrectresp', 'past response correct')
    result_coefficients.index = result_coefficients.index.str.replace('precur_and_targ_same', 'precursor = target F0')
    result_coefficients = result_coefficients.sort_values(by='coefficients', ascending=False)
    ax.bar(result_coefficients.index, result_coefficients['coefficients'], color = 'peru')
    ax.errorbar(result_coefficients.index, result_coefficients['coefficients'], yerr=result_coefficients['std_error'], fmt='none', ecolor='black', elinewidth=1, capsize=2)
    # ax.set_xticklabels(result_coefficients['features'], rotation=45, ha='right')
    # if the mean p value is less than 0.05, then add a star to the bar plot
    for i in range(len(result_coefficients)):
        if result_coefficients['p_values'][i] < 0.05:
            ax.text(i, 0.00, '*', fontsize=20)
    ax.set_xlabel('Features')
    ax.set_ylabel('Mean Coefficient')
    plt.xticks(rotation=45, ha='right')
    ax.set_title('Mean Coefficient for Each Feature, Miss Model')

    plt.savefig('mixedeffects_csvs/mean_coefficients_correct_response.png', dpi=500, bbox_inches='tight')
    plt.show()

    print(np.mean(train_acc))
    print(np.mean(test_acc))
    mean_coefficients = pd.DataFrame(coefficients).mean()
    mean_coefficients = pd.concat([mean_coefficients, p_values_df, std_error_df], axis=1, keys=['coefficients', 'p_values', 'std_error'])
    print(mean_coefficients)
    mean_coefficients.to_csv('mixedeffects_csvs/correctresp_mean_coefficients.csv')

    mean_random_effects = random_effects_df.mean(axis=0)
    print(mean_random_effects)
    big_df = pd.concat([mean_coefficients, mean_random_effects], axis=0)
    mean_random_effects.to_csv('mixedeffects_csvs/random_effects.csv')

    #export to dataframe
    np.savetxt(f"mixedeffects_csvs/correctresp_balac_train_mean.csv", [np.mean(train_acc)], delimiter=",")
    np.savetxt(f"mixedeffects_csvs/correctresp_balac_test_mean.csv", [np.mean(test_acc)], delimiter=",")
    return result

def runlgbcorrectrespornotwithoptuna(dataframe, paramsinput=None, optimization = False, ferret_as_feature=False, one_ferret = False, ferrets = None):
    '''run the lightgbm model for the correct response model
    :param dataframe: dataframe
    :param paramsinput: parameters for the model
    :param optimization: whether to run the optuna study or not
    :param ferret_as_feature: whether to include ferret as a feature or not
    :param one_ferret: whether to run the model on one ferret or not
    :param ferrets: which ferret(s) to run the model on
    :return: none
    '''
    if ferret_as_feature == True:
        df_to_use = dataframe[["trialNum", "misslist", "talker", "side", "precur_and_targ_same",
                               "targTimes", "pastcorrectresp",
                               "pastcatchtrial", "pitchoftarg", "ferret"]]
        labels = ['trial no.','misslist', 'talker', 'audio side', 'precur. = targ. F0','target time', 'past resp. correct', 'past trial catch',"target F0", 'ferret ID']

        run_mixed_effects_model_correctresp(df_to_use)
        df_to_use = df_to_use.rename(columns=dict(zip(df_to_use.columns, labels)))

        fig_dir = Path('D:/behavmodelfigs/correctresp_or_miss/ferret_as_feature')
        col = 'misslist'
        dfx = df_to_use.loc[:, df_to_use.columns != col]
        if optimization == False:
            # load the saved params
            paramsinput = np.load('../optuna_results/correctresponse_optunaparams_ferretasfeature_180124.npy', allow_pickle=True).item()
        else:
            study = run_optuna_study_correctresp(dfx.to_numpy(), df_to_use['misslist'].to_numpy())
            print(study.best_params)
            paramsinput = study.best_params
            np.save('../optuna_results/correctresponse_optunaparams_ferretasfeature_180124.npy', study.best_params)

    else:
        df_to_use = dataframe[["trialNum", "misslist", "talker", "side", "precur_and_targ_same",
                           "targTimes","pastcorrectresp",
                           "pastcatchtrial", "pitchoftarg"]]
        #
        labels = ['trial no.','misslist', 'talker', 'audio side', "precur. = targ. F0",'target time', 'past resp. correct', 'past trial catch', 'target F0']
        df_to_use = df_to_use.rename(columns=dict(zip(df_to_use.columns, labels)))

        fig_dir = Path('D:/behavmodelfigs/correctresp_or_miss/')
        col = 'misslist'
        dfx = df_to_use.loc[:, df_to_use.columns != col]
        if optimization == False:
            # load the saved params
            if one_ferret:
                paramsinput = np.load('../optuna_results/correctresponse_optunaparams'+ferrets+'_180124.npy', allow_pickle=True).item()
            else:
                paramsinput = np.load('../optuna_results/correctresponse_optunaparams_180124.npy', allow_pickle=True).item()
        else:
            study = run_optuna_study_correctresp(dfx.to_numpy(), df_to_use['misslist'].to_numpy())
            print(study.best_params)
            paramsinput = study.best_params
            if one_ferret:
                np.save('../optuna_results/correctresponse_optunaparams_180124'+ferrets+'.npy', study.best_params)
            else:
                np.save('../optuna_results/correctresponse_optunaparams_180124.npy', study.best_params)


    X_train, X_test, y_train, y_test = train_test_split(dfx, df_to_use['misslist'], test_size=0.2, random_state=123)
    print(X_train.shape)
    print(X_test.shape)

    if ferret_as_feature:
        if one_ferret:

            fig_savedir = Path('D:/behavmodelfigs/correctresp_or_miss//ferret_as_feature/' + ferrets[0])
            if fig_savedir.exists():
                pass
            else:
                fig_savedir.mkdir(parents=True, exist_ok=True)
        else:
            fig_savedir = Path('D:/behavmodelfigs/correctresp_or_miss//ferret_as_feature')
    else:
        if one_ferret:

            fig_savedir = Path('D:/behavmodelfigs/correctresp_or_miss/'+ ferrets[0])
            if fig_savedir.exists():
                pass
            else:
                fig_savedir.mkdir(parents=True, exist_ok=True)
        else:
            fig_savedir = Path('D:/behavmodelfigs/correctresp_or_miss//')

    xg_reg = lgb.LGBMClassifier(objective="binary", random_state=123, **paramsinput)
    xg_reg.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="binary_logloss",
        early_stopping_rounds=100,
    )
    ypred = xg_reg.predict_proba(X_test)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    results = cross_val_score(xg_reg, X_test, y_test, scoring='accuracy', cv=kfold)

    bal_accuracy_train = cross_val_score(xg_reg, X_train, y_train, scoring='balanced_accuracy', cv=kfold)
    bal_accuracy = cross_val_score(xg_reg, X_test, y_test, scoring='balanced_accuracy', cv=kfold)
    print("Bal. accuracy, test: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    print('Balanced Accuracy, train: %.2f%%' % (np.mean(bal_accuracy) * 100.0))
    shap_values1 = shap.TreeExplainer(xg_reg).shap_values(dfx)

    custom_colors = ['gold',  'peru', "purple"]  # Add more colors as needed
    cmapcustom = mcolors.LinearSegmentedColormap.from_list('my_custom_cmap', custom_colors, N=1000)
    custom_colors_summary = ['peru', 'gold',]  # Add more colors as needed
    cmapsummary = matplotlib.colors.ListedColormap(custom_colors_summary)
    #take the absolute sum of shap_values1 across the class types
    feature_importances = np.abs(shap_values1).sum(axis=1).sum(axis=0)
    sorted_indices = np.argsort(feature_importances)


    sorted_indices = sorted_indices[::-1]
    feature_importances = feature_importances[sorted_indices]
    feature_labels = dfx.columns[sorted_indices]
    cumulative_importances = np.cumsum(feature_importances)
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=100,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    explainer = shap.Explainer(xg_reg, X_train, feature_names=X_train.columns)
    shap_values2 = explainer(X_train)

    summary_plot_file = 'summary_plot.png'
    shap.plots.beeswarm(shap_values2,  show=False, color=cmapcustom)
    fig, ax = plt.gcf(), plt.gca()

    fig.set_size_inches(4, 12)
    ax.set_xlabel('Log(odds) miss', fontsize=36)
    colorbar = fig.axes[1]
    #change the font size of the color bar
    colorbar.tick_params(labelsize=30)
    #change the label of the color bar
    colorbar.set_ylabel(None)
    # ax.set_yticks(range(len(feature_labels)))
    #for some reason y ticks are set in reverse order
    ax.set_yticklabels(np.flip(feature_labels), fontsize=17, rotation = 45)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    #reinsert the legend_hanldes and labels
    ax.legend(legend_handles, ['Hit', 'Miss'], loc='upper right', fontsize=18)


    plt.savefig(summary_plot_file, dpi=500, bbox_inches='tight')

    plt.show()
    #make the y labels smaller
    #do an example force plot for the data point with the highest miss probability
    #get the index of the data point with the highest miss probability
    max_miss_prob = np.max(ypred[:,1])
    max_miss_prob_index = np.where(ypred[:,1] == max_miss_prob)[0][0]

    shap_values_for_max_miss_prob = shap_values2[max_miss_prob_index]
    #do a force plot for that data point
    explainer = shap.TreeExplainer(xg_reg)
    explanation = explainer.shap_values(dfx)  # Corrected this line
    max_miss_probe = np.max(ypred[:, 1])
    max_shap_index = explanation[1].sum(axis=1).argmax()

    cumulative_shap_values = explanation[1].sum(axis=1)

    # Find the indices of the top 10 instances with the highest cumulative Shapley values
    top_10_indices = cumulative_shap_values.argsort()[-10:][::-1]
    #find the indices where the log odds of miss is greater than 0.5
    log_odds_greater_than_05 = np.where(explanation[1].sum(axis=1) > 0.5)[0]
    #find the indices where the log odds for time to target presentation is greater than 0.5
    log_odds_greater_than_05_time = np.where(explanation[1][:, 4] > 0.5)[0]
    #find the indices where the log odds for time to target presentation is greater than 0.5


    # Create force plots for the top 10 instances
    for i in log_odds_greater_than_05_time:
        force_plot_top_10 = shap.force_plot(explainer.expected_value[1], explanation[1][i, :], dfx.iloc[i, :])
        shap.save_html(str(fig_savedir / f'forceplot_odds_greater_than_05_time_{i}.html'), force_plot_top_10)
        shap.save_html(str(fig_savedir / f'forceplot_odds_greater_than_05_time_{i}.svg'), force_plot_top_10)


    # Create a force plot for the data point with the maximum Shapley value
    force_plot_max = shap.force_plot(explainer.expected_value[1], explanation[1][max_shap_index, :],
                                     dfx.iloc[max_shap_index, :])
    shap.save_html(str(fig_savedir / 'forceplot_max_shap.html'), force_plot_max)

    force_plot = shap.force_plot(explainer.expected_value[1], explanation[1], dfx)
    shap.save_html(str(fig_savedir / 'forceplot.html'), force_plot)





    fig, ax = plt.subplots()
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "target F0"], cmap=cmapcustom, show=False)
    plt.xticks([1,2], labels = ['Male', 'Female'], fontsize=18)
    plt.show()


    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "precur. = targ. F0"], cmap=cmapcustom, show=True)
    shap.plots.scatter(shap_values2[:, "audio side"], color=shap_values2[:, "ferret ID"], cmap=cmapcustom, show=True)
    shap.plots.scatter(shap_values2[:,  "trial no."], color=shap_values2[:, "precur. = targ. F0"], cmap=cmapcustom, show=True)
    shap.plots.scatter(shap_values2[:, "target time"], color=shap_values2[:, "ferret ID"], cmap=cmapcustom, show=True)

    shap.plots.scatter(shap_values2[:, "target F0"], color=shap_values2[:, "precur. = targ. F0"], cmap=cmapcustom, show=True)


    fig, ax = plt.subplots()
    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "audio side"], cmap=cmapcustom,ax=ax,  show=False)
    ax.set_xticks([0,1,2,3,4])
    colorbar_scatter = fig.axes[1]
    colorbar_scatter.set_yticks([0,1])
    colorbar_scatter.set_yticklabels(['Left', 'Right'], fontsize=18)
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation = 45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    ax.set_ylabel('Log(odds) miss', fontsize=18)
    # plt.title('Mean SHAP value over ferret ID', fontsize=18)
    plt.savefig(fig_dir /'ferretID_vs_audioSide.png', dpi=500, bbox_inches='tight')
    plt.show()



    fig, ax = plt.subplots()
    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "target time"], cmap=cmapcustom,ax=ax,  show=False)
    ax.set_xticks([0,1,2,3,4])
    colorbar_scatter = fig.axes[1]
    colorbar_scatter.set_ylabel('Target presentation time (s)', fontsize=15)
    # colorbar_scatter.set_yticks([0,1])
    # colorbar_scatter.set_yticklabels(['Left', 'Right'], fontsize=18)
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    ax.set_ylabel('Log(odds) miss', fontsize=18)
    plt.title('Time to target presentation', fontsize=18)
    plt.savefig(fig_dir /'ferretidbytargettime.png', dpi=500, bbox_inches='tight')
    plt.show()

    ferret_ids = shap_values2[:, "ferret ID"].data
    side_values = shap_values2[:, "audio side"].data
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
    ax.set_ylabel('Log(odds) miss', fontsize=18)  # Corrected y-label

    # plt.title('Mean SHAP value over ferret ID', fontsize=18)

    # Optionally add a legend
    ax.legend(title="side of audio", fontsize=14, title_fontsize=16)
    # change legend labels
    handles, labels = ax.get_legend_handles_labels()
    labels = ['left', 'right']
    ax.legend(handles=handles[0:], labels=labels[0:], title="side of audio", fontsize=14, title_fontsize=16)
    ax.set_title('Side of audio presentation', fontsize=25)

    plt.savefig(fig_dir / 'ferretIDbysideofaudio_violin.png', dpi=500, bbox_inches='tight')
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
    ax.set_ylabel('Log(odds) miss', fontsize=18)  # Corrected y-label

    # plt.title('Mean SHAP value over ferret ID', fontsize=18)

    # Optionally add a legend
    ax.legend(title="talker", fontsize=14, title_fontsize=16)
    # change legend labels
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Male', 'Female']
    ax.legend(handles=handles[0:], labels=labels[0:], title="talker", fontsize=14, title_fontsize=16)
    ax.set_title('Talker type', fontsize=25)

    plt.savefig(fig_dir / 'ferretIDbytalkertype_violin.png', dpi=500, bbox_inches='tight')
    plt.show()


    text_width_pt = 419.67816
    # Replace with your value

    # Convert the text width from points to inches
    text_width_inches = text_width_pt / 72.27


    mosaic = ['A', 'B'], ['D', 'B'], ['C', 'E']
    ferret_id_only = ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']

    # fig = plt.figure(figsize=(20, 10))
    fig = plt.figure(figsize=((text_width_inches / 2) * 4, text_width_inches * 4))

    ax_dict = fig.subplot_mosaic(mosaic)

    # Plot the elbow plot
    ax_dict['A'].plot(feature_labels, cumulative_importances, marker='o', color='gold')
    # ax_dict['A'].set_xlabel('Features')
    ax_dict['A'].set_ylabel('Cumulative Feature Importance', fontsize = 18, fontfamily='sans-serif')
    # ax_dict['A'].set_title('Elbow plot for miss vs hit', fontsize=13)
    ax_dict['A'].set_xticklabels(feature_labels, rotation=20, ha='right', fontfamily='sans-serif')  # rotate x-axis labels for better readability

    # rotate x-axis labels for better readability
    # summary_img = mpimg.imread(summary_plot_file)
    # ax_dict['B'].imshow(summary_img, aspect='auto', )
    # ax_dict['B'].axis('off')  # Turn off axis ticks and labels
    # ax_dict['B'].set_title('Miss vs hit', fontsize=13)
    axmini = ax_dict['B']
    shap_summary_plot(shap_values2, feature_labels, show_plots=False, ax=axmini, cmap=cmapcustom)
    ax_dict['B'].set_yticklabels(np.flip(feature_labels), fontsize=12, rotation=45, fontfamily='sans-serif')
    ax_dict['B'].set_xlabel('Log(odds) miss', fontsize=12)
    ax_dict['B'].set_xticks([-1, -0.5, 0, 0.5, 1])
    cb_ax = fig.axes[5]
    cb_ax.tick_params(labelsize=8)
    cb_ax.set_ylabel('Value', fontsize=8, fontfamily='sans-serif')



    ax_dict['D'].barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T, color='peru')
    # ax_dict['D'].set_title("Permutation importances on predicting a miss")
    ax_dict['D'].set_xlabel("Permutation importance", fontsize=18, fontfamily='sans-serif')

    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "target F0"], ax=ax_dict['E'],
                       cmap=cmapcustom, show=False)
    fig, ax = plt.gcf(), plt.gca()
    cb_ax = fig.axes[6]
    # # Modifying color bar parameters
    # cb_ax.tick_params(labelsize=15)
    # cb_ax.set_ylabel("precursor = target F0 word", fontsize=15)
    ax_dict['E'].set_ylabel('Log(odds) miss', fontsize=18, fontfamily='sans-serif')
    # ax_dict['E'].set_title('Talker versus impact on miss probability', fontsize=18)
    cb_ax.set_yticks([1, 2, 3,4, 5])
    cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
    cb_ax.set_ylabel("target F0 (Hz)", fontsize=8,fontfamily='sans-serif')
    cb_ax.tick_params(labelsize=8)
    # ax_dict['E'].set_xlabel('Talker', fontsize=16)
    ax_dict['E'].set_xlabel('')

    ax_dict['E'].set_xticks([1,2])
    ax_dict['E'].set_xticklabels(['Male', 'Female'], fontsize=16, rotation = 45, ha='right')

    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "target F0"], ax=ax_dict['C'],
                       cmap=cmapcustom, show=False)
    fig, ax = plt.gcf(), plt.gca()
    # ax_dict['C'].set_title('Miss vs hit', fontsize=12)

    cb_ax = fig.axes[8]
    cb_ax.set_yticks([1, 2, 3,4, 5])
    cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
    cb_ax.tick_params(labelsize=8)
    cb_ax.set_ylabel("target F0 (Hz)", fontsize=8)
    cb_ax.tick_params(labelsize=8)
    ax_dict['C'].set_ylabel('Log(odds) miss', fontsize=18)
    # ax_dict['C'].set_xlabel('Ferret ID', fontsize=16)
    ax_dict['C'].set_xlabel('')
    ax_dict['C'].set_xticks([0, 1, 2, 3, 4])
    ax_dict['C'].set_xticklabels(ferret_id_only, fontsize=18, rotation = 45, ha='right')

    # ax_dict['C'].set_xticklabels(ferrets, fontsize=16)

    # ax_dict['C'].set_title('Ferret ID and precursor = target F0 versus SHAP value on miss probability', fontsize=18)
    #remove padding outside the figures
    #
    # ax_dict['A'].annotate('a)', xy=get_axis_limits(ax_dict['A']), xytext=(-0.1, ax_dict['A'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props, zorder=10)
    # ax_dict['B'].annotate('b)', xy=get_axis_limits(ax_dict['B']), xytext=(-0.1, ax_dict['B'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    # ax_dict['C'].annotate('c)', xy=get_axis_limits(ax_dict['C']), xytext=(-0.1, ax_dict['C'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    # ax_dict['D'].annotate('d)', xy=get_axis_limits(ax_dict['D']), xytext=(-0.1, ax_dict['D'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    # ax_dict['E'].annotate('e)', xy=get_axis_limits(ax_dict['E']), xytext=(-0.1, ax_dict['E'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)

    import matplotlib.transforms as mtransforms
    # for label, ax in ax_dict.items():
    #     # label physical distance to the left and up:
    #     trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    #     ax.text(0.0, 1.05, label, transform=ax.transAxes + trans,
    #             fontsize=25, va='bottom', weight = 'bold')

    # plt.tight_layout()

    # plt.suptitle('Target words: miss versus hit model', fontsize=25)
    # for ax in ax_dict.values():
    #     ax.tick_params(axis='both', which='major', labelsize=8)
    #     ax.tick_params(axis='both', which='minor', labelsize=8)
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
    # plt.tight_layout()

    # plt.tight_layout()
    plt.savefig(fig_dir / 'big_summary_plot_1606_noannotation.png', dpi=500, bbox_inches="tight")
    plt.savefig(fig_dir / 'big_summary_plot_1606_noannotation.pdf', dpi=500, bbox_inches="tight")

    plt.show()
    # Plot the scatter plot for trial number and precursor pitch
    fig, ax = plt.subplots()
    shap.plots.scatter(shap_values2[:,  "trial no."], color=shap_values2[:, "precur. = targ. F0"],
                       ax=ax, cmap=cmapcustom, show=False)
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("precur. = targ. F0", fontsize=14)
    cb_ax.set_yticks([0.25, 0.75])
    cb_ax.set_yticklabels(['True', 'False'])
    plt.title('Trial number', fontsize=18)
    plt.xlabel('Trial number', fontsize=15)
    plt.ylabel('Log(odds) miss', fontsize=15)
    plt.savefig(fig_dir / 'trialnum_vs_precurpitch.png', dpi=1000, bbox_inches="tight")
    plt.show()

    # Plot the scatter plot for side of audio presentation and precursor pitch
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.scatter(shap_values2[:, "audio side"], color=shap_values2[:, "precur. = targ. F0"],
                       ax=ax, cmap=cmapcustom, show=False)
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("precursor = target F0 word", fontsize=15)
    plt.xticks([0, 1], labels=['left', 'right'], fontsize=15)
    plt.ylabel('SHAP value', fontsize=16)
    plt.title('Pitch of the side of the booth versus impact in miss probability', fontsize=18)
    plt.xlabel('Side of audio presentation', fontsize=16)
    plt.savefig(fig_dir / 'side_vs_precurpitch.png', dpi=1000, bbox_inches="tight")
    plt.show()


    fig, ax = plt.subplots()
    shap.plots.scatter(shap_values2[:,  "trial no."], color=shap_values2[:, "precur. = targ. F0"], ax=ax, cmap = cmapcustom, show = False)
    fig, ax = plt.gcf(), plt.gca()
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    # cb_ax.set_yticks([1, 2, 3,4, 5])
    # cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
    cb_ax.set_ylabel("precur. = targ. F0", fontsize=15)
    plt.title('Trial number and its effect on the \n miss probability', fontsize = 18)
    plt.xlabel('Trial number', fontsize = 15)
    plt.ylabel('SHAP value', fontsize = 15)
    plt.savefig( fig_dir / 'trialnum_vs_precurpitch.png', dpi=1000, bbox_inches = "tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.scatter(shap_values2[:, "audio side"], color=shap_values2[:, "precur. = targ. F0"], ax=ax, cmap = cmapcustom, show = False)
    fig, ax = plt.gcf(), plt.gca()
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("precursor = target F0 word", fontsize=15)

    plt.xticks([0, 1 ], labels = ['left', 'right'], fontsize =15)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Pitch of the side of the booth \n versus impact in miss probability', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Side of audio presentation', fontsize=16)
    plt.savefig(fig_dir / 'side_vs_precurpitch.png', dpi=1000, bbox_inches = "tight")
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.scatter(shap_values2[:, "precur. = targ. F0"], color=shap_values2[:,  "trial no."], ax=ax, cmap = cmapcustom, show = False)
    fig, ax = plt.gcf(), plt.gca()
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("precursor = target F0 word", fontsize=15)

    plt.xticks([0, 1 ], labels = ['Precursor = target F0', ' Precursor ≠ target F0'], fontsize =15)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Precusor = target F0 \n versus impact in miss probability', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Side of audio presentation', fontsize=16)
    # plt.savefig(fig_dir / 'precursortargpitchintrialnumber.png', dpi=1000, bbox_inches = "tight")
    plt.show()
    #
    # shap.plots.scatter(shap_values2[:, "pitch of target"], color=shap_values2[:, "precur. = targ. F0"], show=False, cmap = cmapcustom)
    # fig, ax = plt.gcf(), plt.gca()
    # # Get colorbar
    # cb_ax = fig.axes[1]
    # # Modifying color bar parameters
    # cb_ax.tick_params(labelsize=15)
    # cb_ax.set_yticks([1, 2, 3,4, 5])
    # # cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
    # cb_ax.set_ylabel("precur. = targ. F0", fontsize=12)
    # # cb_ax.set_yticklabels( ['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'], fontsize=15)
    # plt.ylabel('SHAP value', fontsize=10)
    # plt.title('Pitch of target \n versus impact in miss probability', fontsize=18)
    # plt.ylabel('SHAP value', fontsize=16)
    # plt.xlabel('Pitch of target (Hz)', fontsize=16)
    # # plt.xticks([1,2,3,4,5], labels=['109', '124', '144 ', '191', '251'], fontsize=15)
    # plt.show()


    shap.plots.scatter(shap_values2[:, "precur. = targ. F0"], color=shap_values2[:, "talker"])
    plt.show()
    shap.plots.scatter(shap_values2[:,  "trial no."], color=shap_values2[:, "talker"], show=False)
    plt.title('trial number \n vs. SHAP value impact')
    plt.ylabel('SHAP value', fontsize=18)
    plt.show()

    shap.plots.scatter(shap_values2[:,  "trial no."], color=shap_values2[:, "target time"], show=False)
    plt.title('Trial number versus SHAP value for miss probability, \n colored by target presentation time', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xlabel('Trial number', fontsize=15)
    plt.show()

    shap.plots.scatter(shap_values2[:, "target time"], color=shap_values2[:,  "trial no."], show=False)
    plt.title('target presentation time versus SHAP value for miss probability, \n colored by trial number', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xlabel('Target presentation time', fontsize=15)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 35))
    shap.plots.scatter(shap_values2[:, "audio side"], color=shap_values2[:,  "trial no."], show=False)
    plt.title('SHAP values as a function of the side of the audio, \n coloured by the trial number', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xticks([0, 1], ['Left', 'Right'], fontsize=18)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 55))
    shap.plots.scatter(shap_values2[:, "precur. = targ. F0"], color=shap_values2[:,  "trial no."], show=False)
    plt.title('SHAP values as a function of the pitch of the target, \n coloured by the target presentation time',
              fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xlabel('Pitch of target', fontsize=12)
    # plt.xticks([1, 2, 3, 4, 5], ['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'], fontsize=18)
    plt.show()

    return xg_reg, ypred, y_test, results, shap_values1, X_train, y_train, bal_accuracy, shap_values2





def reservoir_sampling_dataframe(df, k):
    reservoir = []
    n = 0
    for index, row in df.iterrows():
        n += 1
        if len(reservoir) < k:
            reservoir.append(row.copy())
        else:
            # Randomly replace rows in the reservoir with decreasing probability
            replace_index = random.randint(0, n - 1)
            if replace_index < k:
                reservoir[replace_index] = row.copy()

    # Create a new DataFrame from the reservoir
    sampled_df = pd.DataFrame(reservoir)

    return sampled_df

def run_correct_responsepipeline(ferrets):
    resultingcr_df = behaviouralhelperscg.get_df_behav(ferrets=ferrets, includefaandmiss=False, includemissonly=True, startdate='04-01-2020',
                                  finishdate='03-01-2023')
    resultingcr_df['talker'] = resultingcr_df['talker'].replace({1: 2, 2: 1})
    filepath = Path('D:/dfformixedmodels/correctresponsemodel_dfuse.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    resultingcr_df.to_csv(filepath)



    df_intra = resultingcr_df[resultingcr_df['intra_trial_roving'] == 1]
    df_inter = resultingcr_df[resultingcr_df['inter_trial_roving'] == 1]
    df_control = resultingcr_df[resultingcr_df['control_trial'] == 1]

    if len(df_intra) > len(df_inter)*1.2:
        df_intra = df_intra.sample(n=len(df_inter), random_state=123)
    elif len(df_inter) > len(df_intra)*1.2:
        df_inter = df_inter.sample(n=len(df_intra), random_state=123)

    if len(df_control) > len(df_intra)*1.2:
        df_control = df_control.sample(n=len(df_intra), random_state=123)
    elif len(df_control) > len(df_inter)*1.2:
        df_control = df_control.sample(n=len(df_inter), random_state=123)

    resultingcr_df = pd.concat([df_control, df_intra, df_inter], axis=0)


    df_miss = resultingcr_df[resultingcr_df['misslist'] == 1]
    df_nomiss = resultingcr_df[resultingcr_df['misslist'] == 0]

    if len(df_nomiss) > len(df_miss)*1.2:
        df_nomiss = df_nomiss.sample(n=len(df_miss), random_state=123)
    elif len(df_miss) > len(df_nomiss)*1.2:
        df_miss = df_miss.sample(n=len(df_nomiss), random_state=123)

    resultingcr_df = pd.concat([df_nomiss, df_miss], axis=0)



    if len(ferrets) == 1:
        one_ferret = True
        ferret_as_feature = False
    else:
        one_ferret = False
        ferret_as_feature = True

    xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runlgbcorrectrespornotwithoptuna(
        resultingcr_df, optimization=False, ferret_as_feature = ferret_as_feature, one_ferret=one_ferret, ferrets=ferrets)
    return xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2




def run_models_for_all_or_one_ferret(run_individual_ferret_models):
    if run_individual_ferret_models:
        ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
        for ferret in ferrets:
            xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = run_correct_responsepipeline(
                ferrets = [ferret])
    else:
        xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = run_correct_responsepipeline(
            ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'])

if __name__ == '__main__':
    run_models_for_all_or_one_ferret(run_individual_ferret_models=False)
