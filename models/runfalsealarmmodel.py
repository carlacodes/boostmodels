from pathlib import Path
import lightgbm as lgb
import matplotlib
import matplotlib.colors as mcolors
import optuna
import shap
import matplotlib.image as mpimg
import sklearn
import sklearn.metrics
import xgboost as xgb
import pickle
from optuna.integration import LightGBMPruningCallback
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from helpers.behaviouralhelpersformodels import *
from instruments.behaviouralAnalysis import reactionTimeAnalysis  # outputbehaviordf
from instruments.io.BehaviourIO import BehaviourDataSet
import matplotlib.font_manager as fm
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


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

def get_axis_limits(ax, scale=1):
    return ax.get_xlim()[0] * scale, (ax.get_ylim()[1] * scale)

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
    # param_grid = {
    #     # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1),
    #     "alpha": trial.suggest_float("alpha", 1, 20),
    #     "is_unbalanced": trial.suggest_categorical("is_unbalanced", [True]),
    #     "n_estimators": trial.suggest_int("n_estimators", 100, 10000, step=100),
    #     "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
    #     "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=10),
    #     "max_depth": trial.suggest_int("max_depth", 3, 20),
    #     "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
    #     "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=2),
    #     "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=2),
    #     "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    #     "bagging_fraction": trial.suggest_float(
    #         "bagging_fraction", 0.2, 0.95, step=0.1
    #     ),
    #     "bagging_freq": trial.suggest_int("bagging_freq", 1, 20, step=1),
    #     "feature_fraction": trial.suggest_float(
    #         "feature_fraction", 0.2, 0.95, step=0.1
    #     ),
    # }
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

    # param_grid = {
    #     "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
    #     "num_leaves": trial.suggest_int("num_leaves", 20, 200),
    #     "max_depth": trial.suggest_int("max_depth", 5, 15),
    #     "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
    #     "subsample_for_bin": trial.suggest_int("subsample_for_bin", 20000, 300000, step=20000),
    #     "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
    #     "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
    #     "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    #     "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    #     "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    #     "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10),
    #     "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
    #     "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
    #     "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
    #     "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
    #     "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
    #     "max_bin": trial.suggest_int("max_bin", 100, 500),
    #     "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
    #     "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.1, 10),
    # }

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


def run_optuna_study_falsealarm(dataframe, y, ferret_as_feature=False):
    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    if ferret_as_feature:
        df_to_use = dataframe[
            ["pitchof0oflastword", "time_elapsed", "ferret", "trialNum", "talker", "side", "intra_trial_roving",
             "pastcorrectresp", "pastcatchtrial",
             "falsealarm"]]
        # dfuse = df[["pitchoftarg", "pastcatchtrial", "trialNum", "talker", "side", "precur_and_targ_same",
        #             "timeToTarget",
        #             "realRelReleaseTimes", "ferret", "pastcorrectresp"]]
        labels = ["F0", "time in trial", "ferret ID", "trial no.", "talker", "audio side",
                  "intra-F0 roving", "past resp. correct", "past trial catch", "falsealarm"]
        df_to_use = df_to_use.rename(columns=dict(zip(df_to_use.columns, labels)))
    else:
        df_to_use = dataframe[
            ["pitchof0oflastword", "time_elapsed", "trialNum", "talker", "side", "intra_trial_roving", "pastcorrectresp",
             "pastcatchtrial",
             "falsealarm"]]
        labels = ["F0", "time in trial", "trial no", "talker", "audio side", "intra-F0 roving",
                  "past resp. correct", "past trial catch", "falsealarm"]
        df_to_use = df_to_use.rename(columns=dict(zip(df_to_use.columns, labels)))

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
        ["cosinesim", "pitchof0oflastword", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
         "falsealarm", "pastcorrectresp", "temporalsim", "pastcatchtrial", "trialNum", "time_elapsed", ]]

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
def run_mixed_effects_model_falsealarm(df):
    equation = 'falsealarm ~ talker +time_since_trial_start+ trial_number + audio_side + intra_trial_F0_roving + past_response_correct + past_trial_was_catch + F0'

    #split the data into training and test set
    #drop the rows with missing values
    labels_mixed_effects = ["time_since_trial_start", "ferret_ID", "trial_number", "talker", "audio_side",
                            "intra_trial_F0_roving", "past_response_correct", "past_trial_was_catch", "falsealarm",
                            "F0"]

    df['past_response_correct'] = df['past_response_correct'].astype('category')
    df['audio_side'] = df['audio_side'].astype('category')
    df['audio_side'] = df['audio_side'].replace({0: 'Left', 1: 'Right'})

    df['talker'] = df['talker'].astype('category')

    df['past_trial_was_catch'] = df['past_trial_was_catch'].astype('category')
    df['F0'] = df['F0'].astype('category')
    df['talker'] = df['talker'].replace({1: 'Male', 2: 'Female'})
    df['F0'] = df['F0'].replace({1: '109 Hz', 2: '124 Hz', 3: '144 Hz', 4: '191 Hz', 5: '251 Hz'})
    df['ferret_ID'] = df['ferret_ID'].astype('category')
    df["intra_trial_F0_roving"] = df["intra_trial_F0_roving"].astype('category')


    df = df.dropna()
    df['talker'] = df['talker'].replace({1: 2, 2: 1})
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    fold_index = 1
    train_acc = []
    test_acc = []
    coefficients = []
    p_values   = []
    std_error = []
    std_error_re = []
    random_effects_df = pd.DataFrame()
    for train_index, test_index in kf.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]

        model = smf.mixedlm(equation, train, groups=train["ferret_ID"])
        result = model.fit()

        random_effects = result.random_effects
        #commbine all into one series
        # random_effects = pd.DataFrame(random_effects)
        random_effects_2 = pd.DataFrame()
        for i, ferret in enumerate(ferrets):
            try:
                random_effects_2[ferret] = random_effects[i].values
            except:
                continue

        #
        #flatten the random_effects
        print(random_effects)

        print(result.summary())

        var_resid = result.scale
        var_random_effect = float(result.cov_re.iloc[0])
        var_fixed_effect = result.predict(df).var()

        total_var = var_fixed_effect + var_random_effect + var_resid
        marginal_r2 = var_fixed_effect / total_var
        conditional_r2 = (var_fixed_effect + var_random_effect) / total_var
        params = result.params
        #combiune params and random effects into one series
        # params = pd.concat([params, random_effects_2.mean(axis=0)], axis=0)

        coefficients.append(params)
        random_effects_df = pd.concat([random_effects_df, random_effects_2])
        p_values.append(result.pvalues)
        std_error.append(result.bse)
        std_error_re.append(result.bse_re)

        # Generate confusion matrix for train set
        y_pred_train = result.predict(train)
        y_pred_train = (y_pred_train > 0.5).astype(int)
        y_true_train = train['falsealarm'].to_numpy()
        confusion_matrix_train = confusion_matrix(y_true_train, y_pred_train)
        print(confusion_matrix_train)

        # Calculate balanced accuracy for train set
        balanced_accuracy_train = balanced_accuracy_score(y_true_train, y_pred_train)
        print(balanced_accuracy_train)
        train_acc.append(balanced_accuracy_train)

        # Export confusion matrix and balanced accuracy for train set
        np.savetxt(f"mixedeffects_csvs/falsealarm_confusionmatrix_train_fold{fold_index}.csv", confusion_matrix_train,
                   delimiter=",")
        np.savetxt(f"mixedeffects_csvs/falsealarm_balac_train_fold{fold_index}.csv", [balanced_accuracy_train],
                   delimiter=",")

        # Generate confusion matrix for test set
        y_pred = result.predict(test)
        y_pred = (y_pred > 0.5).astype(int)
        y_true = test['falsealarm'].to_numpy()
        confusion_matrix_test = confusion_matrix(y_true, y_pred)
        print(confusion_matrix_test)


        # Calculate balanced accuracy for test set
        balanced_accuracy_test = balanced_accuracy_score(y_true, y_pred)
        print(balanced_accuracy_test)
        test_acc.append(balanced_accuracy_test)

        # Export confusion matrix and balanced accuracy for test set
        np.savetxt(f"mixedeffects_csvs/falsealarmp_confusionmatrix_test_fold{fold_index}.csv", confusion_matrix_test,
                   delimiter=",")
        np.savetxt(f"mixedeffects_csvs/falsealarm_balac_test_fold{fold_index}.csv", [balanced_accuracy_test],
                   delimiter=",")

        fold_index += 1  # Increment fold index
    #calculate the mean accuracy
    #plot the mean coefficients as a bar plot
    #make a dataframe of the coefficients, p-values, and features
    coefficients_df = pd.DataFrame(coefficients).mean()
    index = coefficients_df.index
    p_values_df = pd.DataFrame(p_values).mean()
    std_error_df = pd.DataFrame(std_error).mean()
    std_error_re_df = pd.DataFrame(std_error_re).mean()
    labels_mixed_effects_df = pd.DataFrame(labels_mixed_effects)
    #combine into one dataframe

    result_coefficients = pd.concat([coefficients_df, p_values_df, std_error_df], axis=1, keys=['coefficients', 'p_values', 'std_error'])
    fig, ax = plt.subplots()
    result_coefficients.index = result_coefficients.index.str.replace('Group Var', 'Ferret')

    #sort the coefficients by their mean value


    # ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']


    result_coefficients = result_coefficients.sort_values(by='coefficients', ascending=False)
    ax.bar(result_coefficients.index, result_coefficients['coefficients'])
    ax.errorbar(result_coefficients.index, result_coefficients['coefficients'], yerr=result_coefficients['std_error'], fmt='none', ecolor='black', elinewidth=1, capsize=2)

    # ax.set_xticklabels(result_coefficients['features'], rotation=45, ha='right')
    #if the mean p value is less than 0.05, then add a star to the bar plot
    for i in range(len(result_coefficients)):
        if result_coefficients['p_values'][i] < 0.05:
            ax.text(i, 0.00, '*', fontsize=20)
    ax.set_xlabel('Features')
    ax.set_ylabel('Mean Coefficient')
    plt.xticks(rotation=45, ha='right')
    ax.set_title('Mean Coefficient for Each Feature, False Alarm Model')
    plt.savefig('mixedeffects_csvs//fa_or_not_model_mean_coefficients.png', dpi=500, bbox_inches='tight')
    plt.show()

    #plot the mean coefficients as a bar plot

    # fig, ax = plt.subplots()
    # #sort the coefficients by their mean value
    # pd.DataFrame(coefficients).index
    # coefficients.index
    # ax.bar(labels_mixed_effects, pd.DataFrame(coefficients).mean())
    # ax.set_xticklabels(labels_mixed_effects, rotation=45, ha='right')
    # #if the mean p value is less than 0.05, then add a star to the bar plot
    # for i in range(len(p_values)):
    #     if p_values[i].mean() < 0.05:
    #         ax.text(i, 0.05, '*', fontsize=20)
    # ax.set_xlabel('Features')
    # ax.set_ylabel('Mean Coefficient')
    # ax.set_title('Mean Coefficient for each Feature')
    # plt.savefig('D:/behavmodelfigs/fa_or_not_model/mean_coefficients.png', dpi=500, bbox_inches='tight')
    # plt.show()

    print(np.mean(train_acc))
    print(np.mean(test_acc))
    mean_coefficients = pd.DataFrame(coefficients).mean()
    mean_coefficients = pd.concat([mean_coefficients, p_values_df, std_error_df], axis=1, keys=['coefficients', 'p_values', 'std_error'])
    print(mean_coefficients)
    mean_coefficients.to_csv('mixedeffects_csvs/falsealarm_mean_coefficients.csv')

    mean_random_effects = random_effects_df.mean(axis=0)
    print(mean_random_effects)
    big_df = pd.concat([mean_coefficients, mean_random_effects], axis=0)
    mean_random_effects.to_csv('mixedeffects_csvs/false_alarm_random_effects.csv')

    print(mean_coefficients)    #export
    np.savetxt(f"mixedeffects_csvs/falsealarm_balac_train_mean.csv", [np.mean(train_acc)], delimiter=",")
    np.savetxt(f"mixedeffects_csvs/falsealarmbalac_test_mean.csv", [np.mean(test_acc)], delimiter=",")
    return result


def runlgbfaornotwithoptuna(dataframe, paramsinput, ferret_as_feature=False, one_ferret = False, ferrets = None):
    if ferret_as_feature:
        df_to_use = dataframe[
            [ "time_elapsed", "ferret", "trialNum", "talker", "side", "intra_trial_roving",
             "pastcorrectresp", "pastcatchtrial",
             "falsealarm", "pitchof0oflastword"]]
        # dfuse = df[["pitchoftarg", "pastcatchtrial", "trialNum", "talker", "side", "precur_and_targ_same",
        #             "timeToTarget",
        #             "realRelReleaseTimes", "ferret", "pastcorrectresp"]]
        labels = ["time in trial", "ferret ID", "trial no.", "talker", "audio side",
                  "intra-F0 roving", "past resp. correct", "past trial catch", "falsealarm", "F0"]
        labels_mixed_effects = ["time_since_trial_start", "ferret_ID", "trial_number", "talker", "audio_side",
                  "intra_trial_F0_roving", "past_response_correct", "past_trial_was_catch", "falsealarm", "F0"]
        df_to_use2 = df_to_use.copy()
        df_to_use_mixed_effects = df_to_use2.rename(columns=dict(zip(df_to_use.columns, labels_mixed_effects)))

        df_to_use = df_to_use.rename(columns=dict(zip(df_to_use.columns, labels)))
        run_mixed_effects_model_falsealarm(df_to_use_mixed_effects)

    else:
        df_to_use = dataframe[
            ["time_elapsed", "trialNum", "talker", "side", "intra_trial_roving", "pastcorrectresp",
             "pastcatchtrial",
             "falsealarm", "pitchof0oflastword"]]
        labels = [ "time in trial", "trial number", "talker", "audio side", "intra-F0 roving",
                  "past resp. correct", "past trial was catch", "falsealarm", "F0"]
        df_to_use = df_to_use.rename(columns=dict(zip(df_to_use.columns, labels)))

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
    results_training = cross_val_score(xg_reg, X_train, y_train, scoring='balanced_accuracy', cv=kfold)
    print('Balanced Accuracy train: %.2f%%' % (np.mean(results_training) * 100.0))
    results = cross_val_score(xg_reg, X_test, y_test, scoring='accuracy', cv=kfold)
    bal_accuracy = cross_val_score(xg_reg, X_test, y_test, scoring='balanced_accuracy', cv=kfold)
    print("Accuracy: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    print('Balanced Accuracy: %.2f%%' % (np.mean(bal_accuracy) * 100.0))

    plotfalsealarmmodel(xg_reg, ypred, y_test, results, X_train, y_train, X_test, bal_accuracy, dfx,
                        ferret_as_feature=ferret_as_feature, ferrets = ferrets, one_ferret=one_ferret)

    return xg_reg, ypred, y_test, results, X_train, y_train, X_test, bal_accuracy, dfx


def plotfalsealarmmodel(xg_reg, ypred, y_test, results, X_train, y_train, X_test, bal_accuracy, dfx,
                        ferret_as_feature=False, ferrets = None, one_ferret=False):

    if ferret_as_feature:
        if one_ferret:
            fig_dir = Path('D:/behavmodelfigs/fa_or_not_model/ferret_as_feature/' + ferrets[0])
            if fig_dir.exists():
                pass
            else:
                fig_dir.mkdir(parents=True, exist_ok=True)
        else:
            fig_dir = Path('D:/behavmodelfigs/fa_or_not_model/ferret_as_feature')
    else:
        if one_ferret:
            fig_dir = Path('D:/behavmodelfigs/fa_or_not_model/'+ ferrets[0])
            if fig_dir.exists():
                pass
            else:
                fig_dir.mkdir(parents=True, exist_ok=True)
        else:
            fig_dir = Path('D:/behavmodelfigs/fa_or_not_model/')

    shap_values1 = shap.TreeExplainer(xg_reg).shap_values(dfx)
    #convert shap values to probability of false alarm
    # shap_values1 = 1/(1+np.exp(-shap_values1))
    explainer = shap.Explainer(xg_reg, X_train, feature_names=X_train.columns)
    shap_values2 = explainer(X_train)
    #convert shape values to probability of false alarm
    # shap_values2 = 1/(1+np.exp(-shap_values2))

    custom_colors = ['slategray', 'hotpink', "yellow"]  # Add more colors as needed
    cmapcustom = mcolors.LinearSegmentedColormap.from_list('my_custom_cmap', custom_colors, N=1000)
    custom_colors_summary = ['slategray', 'hotpink', ]  # Add more colors as needed
    cmapsummary = matplotlib.colors.ListedColormap(custom_colors_summary)


    feature_importances = np.abs(shap_values1).sum(axis=1).sum(axis=0)
    sorted_indices = np.argsort(feature_importances)
    sorted_indices = sorted_indices[::-1]
    feature_importances = feature_importances[sorted_indices]
    feature_labels = dfx.columns[sorted_indices]
    cumulative_importances = np.cumsum(feature_importances)
    # Calculate the combined cumulative sum of feature importances
    # cumulative_importances_combined = np.sum(cumulative_importances_list, axis=0)
    # feature_labels = dfx.columns
    # Plot the elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(feature_labels, cumulative_importances, marker='o', color='slategray')
    plt.xlabel('Features')
    plt.ylabel('Cumulative Feature Importance')
    plt.title('Elbow Plot of Cumulative Feature Importance for False Alarm Model')
    plt.xticks(rotation=45, ha='right')  # rotate x-axis labels for better readability
    plt.savefig(fig_dir / 'elbowplot14091409.png', dpi=500, bbox_inches='tight')
    plt.show()



    fig, ax = plt.subplots()
    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "time in trial"], show=False, ax=ax, cmap = cmapcustom)
    colorbar_scatter = fig.axes[1]
    # colorbar_scatter.set_yticks([0,1])
    # colorbar_scatter.set_yticklabels(['Left', 'Right'], fontsize=18)
    colorbar_scatter.set_ylabel('Time since trial start', fontsize=15)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    ax.set_ylabel('Log(odds) FA', fontsize=18)
    plt.title('Time since trial start', fontsize=18)
    plt.savefig(fig_dir /'ferretIDby_timesincestartoftrial1409.png', dpi=500, bbox_inches='tight')
    plt.show()


    #ferret x audio side and ferret x response time
    fig, ax = plt.subplots()
    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "audio side"], show=False, ax=ax, cmap = cmapcustom)
    colorbar_scatter = fig.axes[1]
    colorbar_scatter.set_yticks([0,1])
    colorbar_scatter.set_yticklabels(['Left', 'Right'], fontsize=18)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    ax.set_ylabel('Log(odds) FA', fontsize=18)
    # plt.title('Mean SHAP value over ferret ID', fontsize=18)
    plt.savefig(fig_dir /'ferretIDbysideofaudio1409.png', dpi=500, bbox_inches='tight')
    plt.show()






    # shap.summary_plot(shap_values1, dfx, show=False, color=cmapsummary)
    shap.plots.beeswarm(shap_values2, show=False,  color=cmapcustom)
    # shap.plots.bees(shap_values1, dfx, show=False, color=cmapsummary)
    fig, ax = plt.gcf(), plt.gca()
    # plt.title('Ranked list of features over their \n impact in predicting a false alarm', fontsize=18)
    # Get the plot's Patch objects
    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
    fig.set_size_inches(8, 12)
    ax.set_xlabel('Mean SHAP value', fontsize=18)
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(np.flip(feature_labels), fontsize=18, rotation = 45)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    #reinsert the legend_hanldes and labels
    ax.legend(legend_handles, ['Correct Rejection', 'False Alarm'], loc='upper right', fontsize=13)
    colorbar = fig.axes[1]
    #change the font size of the color bar
    colorbar.tick_params(labelsize=30)
    #change the label of the color bar
    colorbar.set_ylabel(None)
    # ax.set_xlabel('Log(odds) FA', fontsize=36)
    fig.tight_layout()
    plt.savefig(fig_dir / 'ranked_features1409.png', dpi=1000, bbox_inches="tight")
    plt.show()

    # calculate permutation importance
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=100,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T, color='slategray')
    ax.set_title("Permutation importance for the false alarm model")
    fig.tight_layout()
    plt.savefig(fig_dir / 'permutation_importance1409.png', dpi=500)
    plt.show()


    # partial dependency plots

    # Plot the scatter plot with the colormap
    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "intra-F0 roving"], cmap=cmapcustom)
    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "F0"], cmap=cmapcustom)
    shap.plots.scatter(shap_values2[:, "audio side"], color=shap_values2[:, "ferret ID"], cmap=cmapcustom)
    shap.plots.scatter(shap_values2[:, "F0"], color=shap_values2[:, "time in trial"], cmap=cmapcustom)
    shap.plots.scatter(shap_values2[:, "past resp. correct"], color=shap_values2[:, "ferret ID"], cmap=cmapcustom)



    fig, ax = plt.subplots(figsize=(5, 5))
    shap.plots.scatter(shap_values2[:, "F0"], color=shap_values2[:, "time in trial"], show= False,  ax=ax, cmap=cmapcustom)
    cax = fig.axes[1]
    cax.tick_params(labelsize=15)
    cax.set_ylabel("Time since start of trial", fontsize=12)
    plt.xlim(0.8, 5.2)
    plt.xticks([1, 2, 3, 4, 5], labels = ["109", "124", "144", "191", "251"])
    # shap.plots.scatter(shap_values2[:, "time since start of trial"], color=shap_values2[:, "F0"], show= True, ax =ax,  cmap=cmapcustom)
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
    ax.set_ylabel('Log(odds) FA', fontsize=18)  # Corrected y-label

    # plt.title('Mean SHAP value over ferret ID', fontsize=18)

    # Optionally add a legend
    ax.legend(title="side of audio", fontsize=14, title_fontsize=16)
    # change legend labels
    handles, labels = ax.get_legend_handles_labels()
    labels = ['left', 'right']
    ax.legend(handles=handles[0:], labels=labels[0:], title="side of audio", fontsize=14, title_fontsize=16)
    ax.set_title('Side of audio presentation', fontsize=25)

    plt.savefig(fig_dir / 'ferretIDbysideofaudio_violin1409.png', dpi=500, bbox_inches='tight')
    plt.show()

    intra_values = shap_values2[:, "intra-F0 roving"].data
    ferret_ids = shap_values2[:, "ferret ID"].data
    shap_values = shap_values2[:, "ferret ID"].values

    data_df = pd.DataFrame({
        "ferret ID": ferret_ids,
        "intra-trial roving": intra_values,
        "SHAP value": shap_values
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="ferret ID", y="SHAP value", hue="intra-trial roving", data=data_df, split=True, inner="quart",
                   palette=custom_colors, ax=ax)

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    ax.set_ylabel('Log(odds) FA', fontsize=18)  # Corrected y-label

    # plt.title('Mean SHAP value over ferret ID', fontsize=18)

    # Optionally add a legend
    ax.legend(title="side of audio", fontsize=14, title_fontsize=16)
    # change legend labels
    handles, labels = ax.get_legend_handles_labels()
    labels = ['False', 'True']
    ax.legend(handles=handles[0:], labels=labels[0:], title="intra-F0 roving", fontsize=14, title_fontsize=16)
    ax.set_title('Intra trial F0 roving', fontsize=25)

    plt.savefig(fig_dir / 'ferretIDbyINTRA_violin1409.png', dpi=500, bbox_inches='tight')
    plt.show()





    fig, ax = plt.subplots(figsize=(5, 5))
    shap.plots.scatter(shap_values2[:, "F0"], color=shap_values2[:, "time in trial"], show=False, ax=ax,
                       cmap=cmapcustom)
    cax = fig.axes[1]
    cax.tick_params(labelsize=15)
    cax.set_ylabel("Time since start of trial", fontsize=12)
    plt.title('Time since start of trial', fontsize=18)
    ax.set_ylabel('Log(odds) FA', fontsize=18)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['109', '124', '144', '191', '251'], fontsize=18, rotation=45)
    plt.savefig(fig_dir / 'F0bytimestart_supplemental1409.png', dpi=500, bbox_inches='tight')
    # plt.xlim(0.8, 5.2)
    # plt.xticks([1, 2, 3, 4, 5], labels = ["109", "124", "144", "191", "251"])
    # shap.plots.scatter(shap_values2[:, "time since start of trial"], color=shap_values2[:, "F0"], show= True, ax =ax,  cmap=cmapcustom)
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    shap.plots.scatter(shap_values2[:, "time in trial"], color=shap_values2[:, "F0"], show=False, ax=ax,
                       cmap=cmapcustom)
    cax = fig.axes[1]
    cax.tick_params(labelsize=15)
    cax.set_ylabel("F0", fontsize=12)
    cax.set_yticks([ 1, 2, 3, 4, 5])
    cax.set_yticklabels(['109', '124', '144', '191', '251'], fontsize=18, rotation=45)
    plt.title('F0', fontsize=18)
    ax.set_ylabel('Log(odds) FA', fontsize=18)
    ax.set_xticks([1, 2, 3, 4, 5])

    plt.savefig(fig_dir / 'timestartbyF0_supplemental1409.png', dpi=500, bbox_inches='tight')
    # plt.xlim(0.8, 5.2)
    # plt.xticks([1, 2, 3, 4, 5], labels = ["109", "124", "144", "191", "251"])
    # shap.plots.scatter(shap_values2[:, "time since start of trial"], color=shap_values2[:, "F0"], show= True, ax =ax,  cmap=cmapcustom)
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    shap.plots.scatter(shap_values2[:, "F0"], color=shap_values2[:, "trial no."], show=False, ax=ax,
                       cmap=cmapcustom)
    cax = fig.axes[1]
    cax.tick_params(labelsize=15)
    cax.set_ylabel("Trial number", fontsize=12)
    # cax.set_yticks([0, 1, 2, 3, 4])
    # cax.set_yticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    plt.title('Trial number', fontsize=18)
    ax.set_ylabel('Log(odds) FA', fontsize=18)

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['109', '124', '144', '191', '251'], fontsize=18, rotation=45)
    plt.savefig(fig_dir / 'F0bytrialnum_supplemental1409.png', dpi=500, bbox_inches='tight')
    # plt.xlim(0.8, 5.2)
    # plt.xticks([1, 2, 3, 4, 5], labels = ["109", "124", "144", "191", "251"])
    # shap.plots.scatter(shap_values2[:, "time since start of trial"], color=shap_values2[:, "F0"], show= True, ax =ax,  cmap=cmapcustom)
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "trial no."], show=False, ax=ax,
                       cmap=cmapcustom)
    cax = fig.axes[1]
    cax.tick_params(labelsize=15)
    cax.set_ylabel("Trial number", fontsize=12)
    # cax.set_yticks([0, 1, 2, 3, 4])
    # cax.set_yticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    plt.title('Trial number', fontsize=18)
    ax.set_ylabel('Log(odds) FA', fontsize=18)

    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    ax.set_xlabel('Ferret ID', fontsize=18)
    plt.savefig(fig_dir / 'FERRETIDbytrialnum_supplemental1409.png', dpi=500, bbox_inches='tight')
    # plt.xlim(0.8, 5.2)
    # plt.xticks([1, 2, 3, 4, 5], labels = ["109", "124", "144", "191", "251"])
    # shap.plots.scatter(shap_values2[:, "time since start of trial"], color=shap_values2[:, "F0"], show= True, ax =ax,  cmap=cmapcustom)
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))

    shap.plots.scatter(shap_values2[:, "intra-F0 roving"], color=shap_values2[:, "ferret ID"], ax=ax,
                       cmap=cmapcustom, show=False)

    fig, ax = plt.gcf(), plt.gca()
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Ferret ID", fontsize=12)
    cb_ax.set_yticks([0,1,2,3,4])
    cb_ax.set_yticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'])

    ax.set_ylabel('Log(odds) FA', fontsize=18)
    # ax_dict['E'].set_title('Intra-trial roving versus impact on false alarm probability', fontsize=13)
    ax.set_xticks([0,1])

    ax.set_xticklabels(['False', 'True'],  fontsize = 16, rotation=45, ha='right')
    ax.set_xlabel('')
    plt.title('Intra trial roving', fontsize=16)
    plt.savefig(fig_dir / 'intratrialrovingbyferretID_supplemental1409.png', dpi=500, bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(5, 5))
    shap.plots.scatter(shap_values2[:, "F0"], color=shap_values2[:, "ferret ID"], show=False, ax=ax,
                       cmap=cmapcustom)
    cax = fig.axes[1]
    cax.tick_params(labelsize=15)
    cax.set_ylabel("Ferret ID", fontsize=12)
    cax.set_yticks([0, 1, 2, 3, 4])
    cax.set_yticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=18, rotation=45)
    plt.title('Ferret ID', fontsize=18)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylabel('Log(odds) FA', fontsize=18)

    ax.set_xticklabels(['109', '124', '144', '191', '251'], fontsize=18, rotation=45)
    plt.savefig(fig_dir / 'F0byferretID_supplemental1409.png', dpi=500, bbox_inches='tight')
    # plt.xlim(0.8, 5.2)
    # plt.xticks([1, 2, 3, 4, 5], labels = ["109", "124", "144", "191", "251"])
    # shap.plots.scatter(shap_values2[:, "time since start of trial"], color=shap_values2[:, "F0"], show= True, ax =ax,  cmap=cmapcustom)
    plt.show()

    #F0 by talker violin plot, supp.
    F0s = shap_values2[:, "F0"].data
    talker_values = shap_values2[:, "talker"].data
    shap_values = shap_values2[:, "F0"].values

    # Create a DataFrame with the necessary data
    data_df = pd.DataFrame({
        "F0": F0s,
        "talker": talker_values,
        "SHAP value": shap_values
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="F0", y="SHAP value", hue="talker", data=data_df, split=True, inner="quart",
                   palette=custom_colors, ax=ax)

    ax.set_xticks([ 0, 1, 2, 3, 4])
    ax.set_xticklabels(['109', '124', '144', '191', '251'], fontsize=18, rotation=45)
    ax.set_xlabel('F0', fontsize=18)
    ax.set_ylabel('Log(odds) FA', fontsize=18)  # Corrected y-label


    # Optionally add a legend
    ax.legend(title="Talker", fontsize=14, title_fontsize=16)
    # change legend labels
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Female', 'Male']
    ax.legend(handles=handles[0:], labels=labels[0:], title="talker", fontsize=14, title_fontsize=16)
    ax.set_title('Talker type', fontsize=25)

    plt.savefig(fig_dir / 'F0byutalker_violin1409.png', dpi=500, bbox_inches='tight')
    plt.show()

    ferret_ids = shap_values2[:, "ferret ID"].data
    talker_values = shap_values2[:, "talker"].data
    shap_values = shap_values2[:, "ferret ID"].values
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
    ax.set_ylabel('Log(odds) FA', fontsize=18)  # Corrected y-label


    # Optionally add a legend
    ax.legend(title="side of audio", fontsize=14, title_fontsize=16)
    # change legend labels
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Female', 'Male']
    ax.legend(handles=handles[0:], labels=labels[0:], title="talker", fontsize=14, title_fontsize=16)
    ax.set_title('Talker type', fontsize=25)

    plt.savefig(fig_dir / 'ferretIDbyTALKER_violin1409.png', dpi=500, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    shap.plots.scatter(shap_values2[:, "intra-F0 roving"], color=shap_values2[:, "ferret ID"], show= False, ax =ax,  cmap=cmapcustom)
    cax = fig.axes[1]
    cax.tick_params(labelsize=15)
    cax.set_ylabel("Ferret ID", fontsize=12)
    # plt.xlim(0.8, 5.2)
    # plt.xticks([1, 2, 3, 4, 5], labels = ["109", "124", "144", "191", "251"])
    # shap.plots.scatter(shap_values2[:, "time since start of trial"], color=shap_values2[:, "F0"], show= True, ax =ax,  cmap=cmapcustom)
    plt.show()


    plt.tight_layout()
    plt.subplots_adjust(left=-10, right=0.5)
    shap.plots.scatter(shap_values2[:, "time in trial"], color=shap_values2[:, "ferret ID"], show= True,
                       cmap=cmapcustom)

    shap.plots.scatter(shap_values2[:, "trial no."], color=shap_values2[:, "ferret ID"], show= True,
                       cmap=cmapcustom)

    fig, ax = plt.subplots(figsize=(5, 5))
    shap.plots.scatter(shap_values2[:, "intra-F0 roving"], color=shap_values2[:, "F0"], show= False,
                       cmap=cmapcustom)
    # plt.xticks([0, 1], labels = ["No", "Yes"])
    fig, ax = plt.gcf(), plt.gca()
    cax = fig.axes[1]
    cax.tick_params(labelsize=15)
    cax.set_ylabel("F0", fontsize=12)
    cax.set_yticks([1, 2, 3, 4, 5])
    cax.set_yticklabels(["109", "124", "144", "191", "251"])
    plt.title('Intra-trial roving \n versus impact in false alarm probability', fontsize=18)
    plt.show()

    shap.plots.scatter(shap_values2[:, "F0"], color=shap_values2[:, "intra-F0 roving"], show=False,
                       cmap=cmapcustom)
    fig, ax = plt.gcf(), plt.gca()
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Intra-trial roving", fontsize=12)

    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Pitch of the precursor word \n versus impact in false alarm probability', fontsize=18)
    plt.xticks([1, 2, 3, 4, 5], labels=['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'], fontsize=15)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('precursor F0 word', fontsize=16)
    plt.savefig(fig_dir / 'precursor F0intratrialrove1409.png', dpi=500)
    plt.show()
    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "intra-F0 roving"], show=True, cmap=cmapcustom)
    shap.plots.scatter(shap_values2[:, "intra-F0 roving"], color=shap_values2[:, "F0"], show=True, cmap=cmapcustom)
    shap.plots.scatter(shap_values2[:, "intra-F0 roving"], color=shap_values2[:, "ferret ID"], show=True, cmap=cmapcustom)

    shap.plots.scatter(shap_values2[:, "time in trial"], color=shap_values2[:, "trial no."], show=False, cmap=cmapcustom)
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Trial number", fontsize=12)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Target presentation time \n versus impact in false alarm probability', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Target presentation time', fontsize=16)
    plt.savefig(fig_dir / 'time_elapsedcolouredbytrialnumber1409.png', dpi=1000)
    plt.show()


    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "trial no."], show=False, cmap=cmapcustom)
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Trial number", fontsize=12)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Ferret ID \n versus impact in false alarm probability', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Ferret ID', fontsize=16)
    plt.xticks([0, 1, 2, 3, 4], labels=['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], fontsize=15)
    plt.savefig(fig_dir / 'ferretIDcolouredbytrialnumber1409.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "time in trial"], color=shap_values2[:, "F0"], show=False,
                       cmap=cmapcustom)
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_yticks([1, 2, 3, 4, 5])
    cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
    cb_ax.set_ylabel("F0", fontsize=12)
    ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3,3.5, 4, 4.5, 5, 5.5, 6])
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Target presentation versus \n impact on false alarm probability', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Target presentation time', fontsize=16)
    plt.savefig(fig_dir / 'trialtime_colouredbyprecur1409.png', dpi=1000)
    plt.show()
    text_width_pt = 419.67816  # Replace with your value

    # Convert the text width from points to inches
    text_width_inches = text_width_pt / 72.27

    # Create the figure with the desired figsize
    mosaic = ['A', 'B'], ['D', 'B'], ['C', 'E']
    ferret_id_only = ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']

    # fig = plt.figure(figsize=(20, 10))
    fig = plt.figure(figsize=((text_width_inches / 2) * 4, text_width_inches * 4))

    ax_dict = fig.subplot_mosaic(mosaic)


    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(text_width_inches, text_width_inches),
    #                          gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 1], 'hspace': 0.2})
    # Plot the elbow plot
    ax_dict['A'].plot(feature_labels, cumulative_importances, marker='o', color='slategray')
    # ax_dict['A'].set_xlabel('Features', fontsize=18)
    ax_dict['A'].set_ylabel('Cumulative \n feature importance', fontsize=15)
    # ax_dict['A'].set_title('Elbow plot of cumulative feature importance for false alarm model', fontsize=13)
    #decrease fontsize of xtick labels
    ax_dict['A'].set_xticklabels(feature_labels, fontsize = 10, rotation=20, ha='right')  # rotate x-axis labels for better readability
    ax_dict['A'].tick_params(axis='x', which='major', labelsize=12)

    # rotate x-axis labels for better readability
    # summary_img = mpimg.imread(fig_dir / 'ranked_features1409.png')
    # ax_dict['B'].imshow(summary_img, aspect='auto', )
    # ax_dict['B'].axis('off')  # Turn off axis ticks and labels
    # ax_dict['B'].set_xlabel('Log(odds) FA', fontsize=18)
    axmini = ax_dict['B']
    shap_summary_plot(shap_values2, feature_labels, show_plots=False, ax=axmini, cmap=cmapcustom)
    ax_dict['B'].set_yticklabels(np.flip(feature_labels), fontsize=12, rotation=45, fontfamily='sans-serif')
    ax_dict['B'].set_xlabel('Log(odds) FA', fontsize=12)
    # ax_dict['B'].set_xticks([-1, -0.5, 0, 0.5, 1])
    cb_ax = fig.axes[5]
    cb_ax.tick_params(labelsize=8)
    cb_ax.set_ylabel('Value', fontsize=8, fontfamily='sans-serif')


    ax_dict['D'].barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T, color='slategray')

    ax_dict['D'].set_xlabel("Permutation importance", fontsize=18)
    ax_dict['D'].set_yticklabels((X_test.columns[sorted_idx]), fontsize=10, rotation=20, ha='right')

    data_df = pd.DataFrame({
        "ferret ID": ferret_ids,
        "intra-trial roving": intra_values,
        "SHAP value": shap_values
    })
    sns.violinplot(x="ferret ID", y="SHAP value", hue="intra-trial roving", data=data_df, split=True, inner="quart",
                   palette=custom_colors, ax=ax_dict['E'])

    ax_dict['E'].set_xticks([0, 1, 2, 3, 4])
    ax_dict['E'].set_xticklabels(['F1702', 'F1815', 'F1803', 'F2002', 'F2105'], rotation=45)
    ax_dict['E'].set_xlabel('Ferret ID', fontsize=18)
    ax_dict['E'].set_ylabel('Log(odds) FA', fontsize=18)  # Corrected y-label

    # plt.title('Mean SHAP value over ferret ID', fontsize=18)

    # Optionally add a legend
    # change legend labels
    handles, labels =  ax_dict['E'].get_legend_handles_labels()
    labels = ['False', 'True']
    ax_dict['E'].legend(handles=handles[0:], labels=labels[0:], title="", fontsize=4, title_fontsize=12)
    ax_dict['E'].set_xlabel('Intra trial F0 roving', fontsize=8)
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "F0"], show= False, ax =ax_dict['C'],  cmap=cmapcustom)

    fig, ax = plt.gcf(), plt.gca()
    cax = fig.axes[6]
    cax.tick_params(labelsize=8)
    cax.set_yticks([1, 2, 3, 4, 5])
    cax.set_yticklabels(['109', '124', '144', '191', '251'])
    cax.set_ylabel("F0 (Hz)", fontsize=8)

    ax_dict['C'].set_xlabel('talker', fontsize=18)
    ax_dict['C'].set_xticks([1,2])
    ax_dict['C'].set_xticklabels(['Female', 'Male'])

    ax_dict['C'].set_ylabel('Log(odds) FA', fontsize=18)



    # ax_dict['A'].annotate('A', xy=get_axis_limits(ax_dict['A']), xytext=(-0.05, ax_dict['A'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props, zorder=10)
    # ax_dict['B'].annotate('B', xy=get_axis_limits(ax_dict['B']), xytext=(-0.05, ax_dict['B'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    # ax_dict['C'].annotate('C', xy=get_axis_limits(ax_dict['C']), xytext=(-0.05, ax_dict['C'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    # ax_dict['D'].annotate('D', xy=get_axis_limits(ax_dict['D']), xytext=(-0.05, ax_dict['D'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    # ax_dict['E'].annotate('E', xy=get_axis_limits(ax_dict['E']), xytext=(-0.05, ax_dict['E'].title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    import matplotlib.transforms as mtransforms
    # for label, ax in ax_dict.items():
    #     # label physical distance to the left and up:
    #     trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    #     ax.text(0.0, 1.05, label, transform=ax.transAxes + trans,
    #             fontsize=25, va='bottom', weight = 'bold')

    # plt.tight_layout()
    # plt.suptitle('Non-target and target words: false alarm vs. no false alarm model', fontsize=25)
    # plt.tight_layout()
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


    plt.savefig(fig_dir / 'big_summary_plot_2_noannotations.png', dpi=500, bbox_inches="tight")
    plt.savefig(fig_dir / 'big_summary_plot_2_noannotations.pdf', dpi=500, bbox_inches="tight")
    plt.show()
    return xg_reg, ypred, y_test, results, shap_values1, X_train, y_train, bal_accuracy, shap_values2


def runlgbfaornot(dataframe):
    df_to_use = dataframe[
        ["cosinesim", "pitchof0oflastword", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
         "falsealarm", "pastcorrectresp", "pastcatchtrial", "trialNum", "time_elapsed", ]]

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

    kfold = KFold(n_splits=5, shuffle=True, random_state=123)
    results_training = cross_val_score(xg_reg, X_train, y_train, scoring='balanced_accuracy', cv=kfold)
    results = cross_val_score(xg_reg, X_test, y_test, scoring='accuracy', cv=kfold)
    bal_accuracy = cross_val_score(xg_reg, X_test, y_test, scoring='balanced_accuracy', cv=kfold)
    print("Accuracy on test set: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    print('Balanced Accuracy on test set: %.2f%%' % (np.mean(bal_accuracy) * 100.0))

    shap_values1 = shap.TreeExplainer(xg_reg).shap_values(dfx)
    explainer = shap.Explainer(xg_reg, dfx)
    shap_values2 = explainer(X_train)
    # shap.summary_plot(shap_values1, dfx, show=True, plot_type='dot')
    shap.plots.beeswarm(shap_values2)

    #export shap_values2 to pickle
    with open('D:/behavmodelfigs/fa_or_not_model/shap_values2.pkl', 'wb') as f:
        pickle.dump(shap_values2, f)

    plt.subplots(figsize=(25, 25))
    # shap.summary_plot(shap_values1, dfx, show=False)
    shap.plots.bar(shap_values2, show=False)
    fig, ax = plt.gcf(), plt.gca()
    plt.title('Ranked list of features over their \n impact in predicting a false alarm', fontsize=18)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
    labels[11] = 'time to target presentation'
    labels[10] = 'precursor F0'
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

    plt.savefig('D:/behavmodelfigs/fa_or_not_model/ranked_features14091409.png', dpi=1000, bbox_inches="tight")
    plt.show()

    shap.dependence_plot("F0", shap_values1[0], dfx)  #
    plt.show()
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=10,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(20, 15))
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T)
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/permutation_importance14091409.png', dpi=500)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "intra-F0 roving"])
    fig.tight_layout()
    plt.tight_layout()
    plt.subplots_adjust(left=-10, right=0.5)

    plt.show()
    shap.plots.scatter(shap_values2[:, "F0"], color=shap_values2[:, "talker"])
    plt.show()

    shap.plots.scatter(shap_values2[:, "ferret ID"], color=shap_values2[:, "intra-F0 roving"], show=False)
    plt.show()

    shap.plots.scatter(shap_values2[:, "intra-F0 roving"], color=shap_values2[:, "talker"])
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "talker"], show=False)
    plt.title('False alarm model - trial number as a function of SHAP values, coloured by talker')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "intra-F0 roving"], show=False)
    plt.title('False alarm model - SHAP values as a function of cosine similarity \n, coloured by intra trial roving')
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/cosinesimdepenencyplot14091409.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "intra-F0 roving"], color=shap_values2[:, "cosinesim"], show=False)
    plt.savefig('D:/behavmodelfigs/intratrialrovingcosinecolor14091409.png', dpi=500)

    plt.show()
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "pitchoftarg"], show=False)
    plt.title('False alarm model - trial number as a function of SHAP values, coloured by pitch of target')
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/trialnumcosinecolor14091409.png', dpi=500)
    plt.show()

    fig, ax = plt.subplots(figsize=(18, 15))
    shap.plots.scatter(shap_values2[:, "time_elapsed"], color=shap_values2[:, "cosinesim"], show=False)
    plt.title('shap values for FA model as a function of time since start of trial, coloured by cosine similarity')
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/time_elapsedcosinecolor14091409.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "time_elapsed"], show=False)
    plt.title('Cosine Similarity as a function of SHAP values, coloured by time_elapsed')
    plt.savefig('D:/behavmodelfigs/cosinesimtime_elapsed14091409.png', dpi=500)
    plt.show()

    return xg_reg, ypred, y_test, results, shap_values1, X_train, y_train, bal_accuracy, shap_values2


def runfalsealarmpipeline(ferrets, optimization=False, ferret_as_feature=False):
    resultingfa_df = behaviouralhelperscg.get_false_alarm_behavdata(ferrets=ferrets, startdate='04-01-2020',
                                                              finishdate='01-03-2023')
    #extract female talker
    # resultingfa_df = resultingfa_df[resultingfa_df['talker'] == 1.0]
    #get the min of thepitchof0oflastword and find which rows have that value
    # minpitch = np.min(resultingfa_df['pitchof0oflastword'].values)
    # minpitchrows = resultingfa_df[resultingfa_df['pitchof0oflastword'] == minpitch]
    # np.min(resultingfa_df['pitchof0oflastword'].values)

    if len(ferrets) == 1:
        one_ferret = True
    len_of_data_male = {}
    len_of_data_female = {}
    len_of_data_female_intra = {}
    len_of_data_female_inter = {}
    len_of_data_male_intra = {}
    len_of_data_male_inter = {}
    for i in range(0, len(ferrets)):
        noncorrectiondata = resultingfa_df[resultingfa_df['correctionTrial'] == 0]
        noncorrectiondata = noncorrectiondata[noncorrectiondata['currAtten'] == 0]

        len_of_data_male[ferrets[i]] = len(noncorrectiondata[(noncorrectiondata['ferret'] == i) & (
                noncorrectiondata['talker'] == 2.0) & (noncorrectiondata['control_trial'] == 1)])
        len_of_data_female[ferrets[i]] = len(noncorrectiondata[(noncorrectiondata['ferret'] == i) & (
                noncorrectiondata['talker'] == 1.0) & (noncorrectiondata['control_trial'] == 1)])

        interdata = noncorrectiondata[noncorrectiondata['inter_trial_roving'] == 1]
        intradata = noncorrectiondata[noncorrectiondata['intra_trial_roving'] == 1]
        len_of_data_female_intra[ferrets[i]] = len(intradata[(intradata['ferret'] == i) & (intradata['talker'] == 1.0)])
        len_of_data_female_inter[ferrets[i]] = len(interdata[(interdata['ferret'] == i) & (interdata['talker'] == 1.0)])

        len_of_data_male_inter[ferrets[i]] = len(interdata[(interdata['ferret'] == i) & (interdata['talker'] == 2.0)])
        len_of_data_male_intra[ferrets[i]] = len(intradata[(intradata['ferret'] == i) & (intradata['talker'] == 2.0)])
    df_intra = resultingfa_df[resultingfa_df['intra_trial_roving'] == 1]
    df_inter = resultingfa_df[resultingfa_df['inter_trial_roving'] == 1]
    df_control = resultingfa_df[resultingfa_df['control_trial'] == 1]
    # now we need to balance the data, if it's a fifth more than the other, we need to sample it down

    df_fa_intra = df_intra[df_intra['falsealarm'] == 1]
    df_nofa_intra = df_intra[df_intra['falsealarm'] == 0]

    df_fa_inter = df_inter[df_inter['falsealarm'] == 1]
    df_nofa_inter = df_inter[df_inter['falsealarm'] == 0]

    df_fa_control = df_control[df_control['falsealarm'] == 1]
    df_nofa_control = df_control[df_control['falsealarm'] == 0]



    if len(df_nofa_intra) > len(df_fa_intra) * 1.2:
        df_nofa_intra = df_nofa_intra.sample(n=len(df_fa_intra), random_state=123)
    elif len(df_fa_intra) > len(df_nofa_intra) * 1.2:
        df_fa_intra = df_fa_intra.sample(n=len(df_nofa_intra), random_state=123)

    if len(df_nofa_inter) > len(df_fa_inter) * 1.2:
        df_nofa_inter = df_nofa_inter.sample(n=len(df_fa_inter), random_state=123)
    elif len(df_fa_inter) > len(df_nofa_inter) * 1.2:
        df_fa_inter = df_fa_inter.sample(n=len(df_nofa_inter), random_state=123)

    if len(df_nofa_control) > len(df_fa_control) * 1.2:
        df_nofa_control = df_nofa_control.sample(n=len(df_fa_control), random_state=123)
    elif len(df_fa_control) > len(df_nofa_control) * 1.2:
        df_fa_control = df_fa_control.sample(n=len(df_nofa_control), random_state=123)


    #then reconcatenate the three dfs
    resultingfa_df = pd.concat([df_nofa_intra, df_fa_intra, df_nofa_inter, df_fa_inter, df_nofa_control, df_fa_control], axis = 0)

    #
    # df_intra = resultingfa_df[resultingfa_df['intra_trial_roving'] == 1]
    # df_inter = resultingfa_df[resultingfa_df['inter_trial_roving'] == 1]
    # df_control = resultingfa_df[resultingfa_df['control_trial'] == 1]
    # # now we need to balance the data, if it's a fifth more than the other, we need to sample it down
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
    # resultingfa_df = pd.concat([df_intra, df_inter, df_control], axis = 0)

    # subsample from the distribution of df_miss

    # find the middle point between the length of df



    # resultingfa_df = pd.concat([df_nofa, df_fa], axis=0)

    #find all the df rows where ferret == 0
    df_ferret0 = resultingfa_df[resultingfa_df['ferret'] == 0]
    filepath = Path('D:/dfformixedmodels/falsealarmmodel_dfuse.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if optimization == False:
        # load the saved params
        params = np.load('../optuna_results/falsealarm_optunaparams_1309_4.npy', allow_pickle=True).item()
    else:
        study = run_optuna_study_falsealarm(resultingfa_df, resultingfa_df['falsealarm'].to_numpy(),
                                            ferret_as_feature=ferret_as_feature)
        print(study.best_params)
        params = study.best_params
        np.save('../optuna_results/falsealarm_optunaparams_1309_4.npy', study.best_params)

    resultingfa_df.to_csv(filepath)

    if len(ferrets) == 1:
        one_ferret = True
    else:
        one_ferret = False
    xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runlgbfaornotwithoptuna(
        resultingfa_df, params, ferret_as_feature=ferret_as_feature, ferrets = ferrets,  one_ferret=one_ferret)
    return xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2


def run_reaction_time_fa_pipleine_female(ferrets):
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    df_use = resultingdf.loc[:, resultingdf.columns != 'ferret']
    df_use = df_use.loc[df_use['intra_trial_roving'] == 0]
    df_use = df_use.loc[df_use['talker'] == 1]
    df_use = df_use.loc[:, df_use.columns != 'time_elapsed']
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
    plt.savefig('D:/behavmodelfigs/ranked_features_rxntimealarmhitmodel14091409.png', dpi=500)
    plt.show()

    return resultingdf



def run_reaction_time_fa_pipleine_male(ferrets):
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2023')
    df_use = resultingdf.loc[:, resultingdf.columns != 'ferret']
    df_use = df_use.loc[df_use['intra_trial_roving'] == 0]
    df_use = df_use.loc[df_use['talker'] == 2]
    df_use = df_use.loc[:, df_use.columns != 'time_elapsed']
    df_use = df_use.loc[:, df_use.columns != 'stepval']
    df_use = df_use.loc[:, df_use.columns != 'side']
    df_use = df_use.loc[:, df_use.columns != 'AM']

    df_use = df_use.loc[:, df_use.columns != 'distractor_or_fa']
    df_use = df_use.loc[:, df_use.columns != 'realRelReleaseTimes']

    col = 'centreRelease'
    dfx = df_use.loc[:, df_use.columns != col]
    col = 'ferret'
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
    plt.savefig('D:/behavmodelfigs/ranked_features_rxntimealarmhitmodel_male14091409.png', dpi=500)
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
    plt.savefig('D:/behavmodelfigs/proportion_correct_responses_by_side14091409.png', dpi=500)

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
    plt.bar(
        ['left - F1702', 'right - F1702', 'left - 1815', 'right - F1815', 'left - F1803', 'right-F1803', 'left - F2002',
         'right- F2002', 'left - F2105', 'right - F2105'],
        [df_left_by_ferret[0]['correct'].mean(), df_right_by_ferret[0]['correct'].mean(),
         df_left_by_ferret[1]['correct'].mean(), df_right_by_ferret[1]['correct'].mean(),
         df_left_by_ferret[2]['correct'].mean(), df_right_by_ferret[2]['correct'].mean(),
         df_left_by_ferret[3]['correct'].mean(), df_right_by_ferret[3]['correct'].mean(),
         df_left_by_ferret[4]['correct'].mean(), df_right_by_ferret[4]['correct'].mean()])

    plt.title('Proportion of correct responses by side registered by sensors, \n  irrespective of talker, by ferret ID',
              fontsize=15)
    plt.xticks(rotation=45, fontsize=12)  # rotate the x axis labels
    plt.ylim(0, 1)

    plt.ylabel('proportion of correct responses', fontsize=13)
    plt.savefig('D:/behavmodelfigs/proportion_correct_responses_by_side_by_ferret14091409.png', dpi=1000)
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
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_female14091409.png', dpi=500)
    plt.show()

    sns.distplot(df_male_control['realRelReleaseTimes'], color='green', label='control F0')
    sns.distplot(df_male_rove['realRelReleaseTimes'], color='orange', label='intra-roved F0')
    plt.title('Reaction times for the male talker, \n irrespective of ferret', fontsize=15)
    plt.legend(fontsize=10)
    plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_male14091409.png', dpi=500)
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
        sns.distplot(df_by_ferret_f_rove[ferret]['realRelReleaseTimes'], color='red',
                     label='intra-trial roved F0, female')
        sns.distplot(df_by_ferret_m_control[ferret]['realRelReleaseTimes'], color='green', label='control F0, male')
        sns.distplot(df_by_ferret_m_rove[ferret]['realRelReleaseTimes'], color='orange',
                     label='intra-trial roved F0, male')
        plt.title('Reaction times for ferret ID ' + str(ferret_labels[ferret]), fontsize=15)
        plt.legend(fontsize=10)
        plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
        plt.savefig('D:/behavmodelfigs/reaction_times_by_ferret_' + str(ferret_labels[ferret]) + '14091409.png', dpi=1000)
        plt.show()

    return df_by_ferret

    #


def plot_reaction_times_interandintra(ferrets):
    # plot the reaction times by animal
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    resultingdf['lickReleasefromtrialstart'] = resultingdf['lickRelease'] - resultingdf['startTrialLick']
    resultingdf['responseTimereltostart'] = (resultingdf['responseTime'] / 24414.0625) - resultingdf['startTrialLick']


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
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_female_interandintra14091409.png', dpi=500)

    plt.show()

    sns.distplot(df_male_control['realRelReleaseTimes'], color='green', label='control F0')
    sns.distplot(df_male_rove['realRelReleaseTimes'], color='orange', label='inter-roved F0')
    sns.distplot(df_male_rove_intra['realRelReleaseTimes'], color='red', label='intra-roved F0')

    plt.title('Reaction times for the male talker, \n irrespective of ferret', fontsize=15)
    plt.legend(fontsize=10)
    plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_male_interandintra14091409.png', dpi=500)
    plt.show()

    df_by_ferret = {}
    df_by_ferret_f_control = {}
    df_by_ferret_f_rove = {}
    df_by_ferret_m_control = {}
    df_by_ferret_m_rove = {}
    df_by_ferret_f_rove_intra = {}
    df_by_ferret_m_rove_intra = {}
    # now plot by ferret ID
    df_by_ferret_total = {}
    ferrets = [0, 1, 2, 3, 4]
    for ferret in ferrets:
        df_by_ferret_f_control[ferret] = df_female_control.loc[df_female_control['ferret'] == ferret]
        df_by_ferret_f_rove[ferret] = df_female_rove.loc[df_female_rove['ferret'] == ferret]
        df_by_ferret_f_rove_intra[ferret] = df_female_rove_intra.loc[df_female_rove_intra['ferret'] == ferret]

        df_by_ferret_m_control[ferret] = df_male_rove.loc[df_male_rove['ferret'] == ferret]
        df_by_ferret_m_rove[ferret] = df_male_control.loc[df_male_control['ferret'] == ferret]
        df_by_ferret_m_rove_intra[ferret] = df_male_rove_intra.loc[df_male_rove_intra['ferret'] == ferret]
        df_by_ferret_total[ferret] = pd.concat([df_by_ferret_f_control[ferret], df_by_ferret_f_rove[ferret], df_by_ferret_f_rove_intra[ferret], df_by_ferret_m_control[ferret], df_by_ferret_m_rove[ferret], df_by_ferret_m_rove_intra[ferret]], axis = 0)

    ferret_labels = ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']
    mosaic = ['0', '1', '2', '3', '4'], ['0', '1', '2', '3', '4']
    fig = plt.figure(figsize=(20, 5))
    ax_dict = fig.subplot_mosaic(mosaic)
    for ferret in ferrets:
        sns.distplot(df_by_ferret_f_control[ferret]['realRelReleaseTimes'], color='blue', label='control F0, female', ax = ax_dict[str(ferret)])
        sns.distplot(df_by_ferret_f_rove[ferret]['realRelReleaseTimes'], color='red', label='inter-roved F0, female', ax = ax_dict[str(ferret)])
        sns.distplot(df_by_ferret_m_control[ferret]['realRelReleaseTimes'], color='green', label='control F0, male', ax = ax_dict[str(ferret)])
        sns.distplot(df_by_ferret_m_rove[ferret]['realRelReleaseTimes'], color='orange', label='inter-roved F0, male', ax = ax_dict[str(ferret)])
        sns.distplot(df_by_ferret_f_rove_intra[ferret]['realRelReleaseTimes'], color='darkmagenta',
                     label='intra-roved F0, female', ax = ax_dict[str(ferret)])
        sns.distplot(df_by_ferret_m_rove_intra[ferret]['realRelReleaseTimes'], color='orangered',
                     label='intra-roved F0, male', ax=ax_dict[str(ferret)])

        ax_dict[str(ferret)].set_title('Reaction times for ' + str(ferret_labels[ferret]), fontsize=12)
        if ferret == 0:
            ax_dict[str(ferret)].legend(fontsize=8)
        ax_dict[str(ferret)].set_xlabel('reaction time relative \n to target presentation (s)', fontsize=10)
    fig.tight_layout()

    font_props = fm.FontProperties(weight='bold', size=17)

    ax_dict['0'].annotate('A', xy=get_axis_limits(ax_dict['0']), xytext=(-0.1, ax_dict['0'].title.get_position()[1]+0.01), textcoords='axes fraction', fontproperties = font_props, zorder=1)
    ax_dict['1'].annotate('B', xy=get_axis_limits(ax_dict['1']), xytext=(-0.1, ax_dict['1'].title.get_position()[1]+0.01), textcoords='axes fraction', fontproperties = font_props,zorder=1)
    ax_dict['2'].annotate('C', xy=get_axis_limits(ax_dict['2']), xytext=(-0.1, ax_dict['2'].title.get_position()[1]+0.01), textcoords='axes fraction', fontproperties = font_props,zorder=1)
    ax_dict['3'].annotate('D', xy=get_axis_limits(ax_dict['3']), xytext=(-0.1, ax_dict['3'].title.get_position()[1]+0.01), textcoords='axes fraction', fontproperties = font_props,zorder=1)
    ax_dict['4'].annotate('E', xy=get_axis_limits(ax_dict['4']), xytext=(-0.1, ax_dict['4'].title.get_position()[1]+0.01), textcoords='axes fraction', fontproperties = font_props,zorder=1)


    # plt.savefig('D:/behavmodelfigs/reaction_times_by_ferret_interbigmosaic_280614091409.png', dpi=500)
    # plt.savefig('D:/behavmodelfigs/reaction_times_by_ferret_interbigmosaic_2806.pdf', dpi=500)

    plt.show()
    fig = plt.figure(figsize=(20, 5))
    ax_dict = fig.subplot_mosaic(mosaic)
    color_animal = ['blue', 'red', 'green', 'orange', 'darkmagenta', 'orangered']
    for ferret in ferrets:
        sns.distplot(df_by_ferret_total[ferret]['realRelReleaseTimes'], color=color_animal[ferret],
                     ax=ax_dict[str(ferret)])

        ax_dict[str(ferret)].set_title('Reaction times for ' + str(ferret_labels[ferret]), fontsize=12)
        # if ferret == 0:
        #     ax_dict[str(ferret)].legend(fontsize=8)
        ax_dict[str(ferret)].set_xlabel('reaction time relative \n to target presentation (s)', fontsize=10)
    fig.tight_layout()


    # ax_dict['0'].annotate('A', xy=get_axis_limits(ax_dict['0']),
    #                       xytext=(-0.1, ax_dict['0'].title.get_position()[1] + 0.01), textcoords='axes fraction',
    #                       fontproperties=font_props, zorder=1)
    # ax_dict['1'].annotate('B', xy=get_axis_limits(ax_dict['1']),
    #                       xytext=(-0.1, ax_dict['1'].title.get_position()[1] + 0.01), textcoords='axes fraction',
    #                       fontproperties=font_props, zorder=1)
    # ax_dict['2'].annotate('C', xy=get_axis_limits(ax_dict['2']),
    #                       xytext=(-0.1, ax_dict['2'].title.get_position()[1] + 0.01), textcoords='axes fraction',
    #                       fontproperties=font_props, zorder=1)
    # ax_dict['3'].annotate('D', xy=get_axis_limits(ax_dict['3']),
    #                       xytext=(-0.1, ax_dict['3'].title.get_position()[1] + 0.01), textcoords='axes fraction',
    #                       fontproperties=font_props, zorder=1)
    # ax_dict['4'].annotate('E', xy=get_axis_limits(ax_dict['4']),
    #                       xytext=(-0.1, ax_dict['4'].title.get_position()[1] + 0.01), textcoords='axes fraction',
    #                       fontproperties=font_props, zorder=1)

    plt.savefig('D:/behavmodelfigs/reaction_times_by_ferret_panel.png', dpi=500)
    plt.savefig('D:/behavmodelfigs/reaction_times_by_ferret_panel.png', dpi=500)

    plt.show()

    fig = plt.figure(figsize=(20, 5))
    #filter so response == 0 or 1
    ax_dict = fig.subplot_mosaic(mosaic)
    color_animal = ['blue', 'red', 'green', 'orange', 'darkmagenta', 'orangered']
    for ferret in ferrets:
        df_by_ferret_total[ferret] = df_by_ferret_total[ferret].loc[(df_by_ferret_total[ferret]['response'] == 1) | (df_by_ferret_total[ferret]['response'] == 0)]
        sns.distplot(df_by_ferret_total[ferret]['responseTimereltostart'], color=color_animal[ferret],
                     ax=ax_dict[str(ferret)])

        ax_dict[str(ferret)].set_title('response times for ' + str(ferret_labels[ferret]) +'\n, max response time:' + str(np.max(df_by_ferret_total[ferret]['responseTimereltostart'])), fontsize=12)
        # if ferret == 0:
        #     ax_dict[str(ferret)].legend(fontsize=8)
        ax_dict[str(ferret)].set_xlabel('response time relative \n to trial start lick (s)', fontsize=10)
    fig.tight_layout()

    plt.savefig('D:/behavmodelfigs/abs_response_times_by_ferret_panel.png', dpi=500)
    plt.savefig('D:/behavmodelfigs/abs_response_times_by_ferret_panel.png', dpi=500)

    plt.show()

    return df_by_ferret


def plot_reaction_times_interandintra_swarm(ferrets):
    # plot the reaction times by animal
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    #only get correct hit trials for the swarm plot
    df_use = resultingdf[(resultingdf['realRelReleaseTimes'] >= 0) & (resultingdf['realRelReleaseTimes'] <= 2)]

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
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_female_interandintra14091409.png', dpi=500)

    plt.show()

    sns.distplot(df_male_control['realRelReleaseTimes'], color='green', label='control F0')
    sns.distplot(df_male_rove['realRelReleaseTimes'], color='orange', label='inter-roved F0')
    sns.distplot(df_male_rove_intra['realRelReleaseTimes'], color='red', label='intra-roved F0')

    plt.title('Reaction times for the male talker, \n irrespective of ferret', fontsize=15)
    plt.legend(fontsize=10)
    plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_male_interandintra14091409.png', dpi=500)
    plt.show()

    df_by_ferret = {}
    df_by_ferret_f_control = {}
    df_by_ferret_f_rove = {}
    df_by_ferret_m_control = {}
    df_by_ferret_m_rove = {}
    df_by_ferret_f_rove_intra = {}
    df_by_ferret_m_rove_intra = {}
    # now plot by ferret ID
    ferrets = [0, 1, 2, 3, 4]
    dict_by_ferret = {}
    #break down by ferret and pitch of target
    for ferret in ferrets:
        dict_by_ferret[ferret]={}
        for pitch in [1, 2, 3, 4, 5]:
            df_ferret_inst= df_use[df_use['ferret'] == ferret]
            dict_by_ferret[ferret][pitch]={}
            dict_by_ferret[ferret][pitch] = df_ferret_inst[df_ferret_inst['pitchoftarg'] == pitch]
    for ferret in ferrets:
        df_by_ferret_f_control[ferret] = df_female_control.loc[df_female_control['ferret'] == ferret]
        df_by_ferret_f_rove[ferret] = df_female_rove.loc[df_female_rove['ferret'] == ferret]
        df_by_ferret_f_rove_intra[ferret] = df_female_rove_intra.loc[df_female_rove_intra['ferret'] == ferret]

        df_by_ferret_m_control[ferret] = df_male_rove.loc[df_male_rove['ferret'] == ferret]
        df_by_ferret_m_rove[ferret] = df_male_control.loc[df_male_control['ferret'] == ferret]
        df_by_ferret_m_rove_intra[ferret] = df_male_rove_intra.loc[df_male_rove_intra['ferret'] == ferret]

    ferret_labels = ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']

    mosaic = ['0', '1', '2'], [ '3', '4', '5']
    text_width_pt = 419.67816  # Replace with your value

    # Convert the text width from points to inches
    text_width_inches = text_width_pt / 72.27
    # fig, ((ax1, ax2)) = plt.subplots(2,1, layout='constrained',figsize=(0.75*text_width_inches,0.5*text_width_inches))
    # fig = plt.figure(figsize=(10, 10))
    fig = plt.figure(figsize = (1.5*text_width_inches,1.5*text_width_inches))
    ax_dict = fig.subplot_mosaic(mosaic)
    pitchlist = ['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz']
    colorlist = ['blue', 'red', 'darkmagenta', 'green', 'orange']
    for ferret in ferrets:
        count = 0
        for pitch in [1, 2, 3, 4, 5]:
            print(ferret)
            print(colorlist[count])
            print(pitch)
            sns.swarmplot(dict_by_ferret[ferret][pitch]['realRelReleaseTimes'], color=colorlist[count], label=pitchlist[count], ax = ax_dict[str(ferret)], alpha =0.5)
            count += 1

        ax_dict[str(ferret)].set_xlabel('')
        if ferret == 0:
            #put legend in last box

            handles, labels = ax_dict['0'].get_legend_handles_labels()
            ax_dict['5'].legend(handles, labels, fontsize=15, loc='upper center', ncol=1)

        if ferret == 0 or ferret == 3:
            ax_dict[str(ferret)].set_ylabel('reaction time (s)', fontsize=15)
        else:
            ax_dict[str(ferret)].set_ylabel('')

        ax_dict[str(ferret)].set_title(str(ferret_labels[ferret]), fontsize=18)
        ax_dict[str(ferret)].set_yticks([0,0.5, 1.0, 1.5, 2.0], fontsize = 18)
        ax_dict[str(ferret)].set_yticklabels([0,0.5, 1.0, 1.5, 2.0], fontsize = 18)


    fig.tight_layout()
    #have the legend go to axdict[5] but information is from other plot

    font_props = fm.FontProperties(weight='bold', size=18)

    # ax_dict['0'].annotate('A', xy=get_axis_limits(ax_dict['0']),
    #                       xytext=(-0.15, ax_dict['0'].title.get_position()[1] + 0.01), textcoords='axes fraction',
    #                       fontproperties=font_props, zorder=1)
    # ax_dict['1'].annotate('B', xy=get_axis_limits(ax_dict['1']),
    #                       xytext=(-0.15, ax_dict['1'].title.get_position()[1] + 0.01), textcoords='axes fraction',
    #                       fontproperties=font_props, zorder=1)
    # ax_dict['2'].annotate('C', xy=get_axis_limits(ax_dict['2']),
    #                       xytext=(-0.15, ax_dict['2'].title.get_position()[1] + 0.01), textcoords='axes fraction',
    #                       fontproperties=font_props, zorder=1)
    # ax_dict['3'].annotate('D', xy=get_axis_limits(ax_dict['3']),
    #                       xytext=(-0.15, ax_dict['3'].title.get_position()[1] + 0.01), textcoords='axes fraction',
    #                       fontproperties=font_props, zorder=1)
    # ax_dict['4'].annotate('E', xy=get_axis_limits(ax_dict['4']),
    #                       xytext=(-0.15, ax_dict['4'].title.get_position()[1] + 0.01), textcoords='axes fraction',
    #                       fontproperties=font_props, zorder=1)

    # #remove spines from ax_dict['5']
    ax_dict['5'].spines['right'].set_visible(False)
    ax_dict['5'].spines['top'].set_visible(False)
    ax_dict['5'].spines['left'].set_visible(False)
    ax_dict['5'].spines['bottom'].set_visible(False)
    ax_dict['5'].set_xticks([])
    ax_dict['5'].set_yticks([])



    plt.savefig('D:/behavmodelfigs/reaction_times_by_ferret_swarm_byF0_bigmosaic_noannotation14091409.png', dpi=500)
    plt.savefig('D:/behavmodelfigs/reaction_times_by_ferret_swarm_byF0_bigmosaic_noannotation.pdf', dpi=500)
    plt.show()

    # mosaic = ['0', '1', '2', '3', '4'], ['0', '1', '2', '3', '4']
    # fig = plt.figure(figsize=(20, 5))
    # ax_dict = fig.subplot_mosaic(mosaic)
    # for ferret in ferrets:
    #     sns.swarmplot(df_by_ferret_f_control[ferret]['realRelReleaseTimes'], color='blue', label='control F0, female', ax = ax_dict[str(ferret)], alpha=0.2)
    #     sns.swarmplot(df_by_ferret_f_rove[ferret]['realRelReleaseTimes'], color='red', label='inter-roved F0, female', ax = ax_dict[str(ferret)], alpha=0.2)
    #     sns.swarmplot(df_by_ferret_m_control[ferret]['realRelReleaseTimes'], color='green', label='control F0, male', ax = ax_dict[str(ferret)], alpha = 0.2)
    #     sns.swarmplot(df_by_ferret_m_rove[ferret]['realRelReleaseTimes'], color='orange', label='inter-roved F0, male', ax = ax_dict[str(ferret)], alpha = 0.2)
    #     sns.swarmplot(df_by_ferret_f_rove_intra[ferret]['realRelReleaseTimes'], color='darkmagenta',
    #                  label='intra-roved F0, female', ax = ax_dict[str(ferret)], alpha = 0.2)
    #     sns.swarmplot(df_by_ferret_m_rove_intra[ferret]['realRelReleaseTimes'], color='orangered',
    #                  label='intra-roved F0, male', ax=ax_dict[str(ferret)], alpha =0.2)
    #
    #     ax_dict[str(ferret)].set_title('Reaction times for ' + str(ferret_labels[ferret]), fontsize=12)
    #     if ferret == 0:
    #         ax_dict[str(ferret)].legend(fontsize=8)
    #     ax_dict[str(ferret)].set_xlabel('reaction time relative \n to target presentation (s)', fontsize=10)
    # fig.tight_layout()
    #
    # font_props = fm.FontProperties(weight='bold', size=17)
    #
    # ax_dict['0'].annotate('a)', xy=get_axis_limits(ax_dict['0']), xytext=(-0.1, ax_dict['0'].title.get_position()[1]+0.01), textcoords='axes fraction', fontproperties = font_props, zorder=1)
    # ax_dict['1'].annotate('b)', xy=get_axis_limits(ax_dict['1']), xytext=(-0.1, ax_dict['1'].title.get_position()[1]+0.01), textcoords='axes fraction', fontproperties = font_props,zorder=1)
    # ax_dict['2'].annotate('c)', xy=get_axis_limits(ax_dict['2']), xytext=(-0.1, ax_dict['2'].title.get_position()[1]+0.01), textcoords='axes fraction', fontproperties = font_props,zorder=1)
    # ax_dict['3'].annotate('d)', xy=get_axis_limits(ax_dict['3']), xytext=(-0.1, ax_dict['3'].title.get_position()[1]+0.01), textcoords='axes fraction', fontproperties = font_props,zorder=1)
    # ax_dict['4'].annotate('e)', xy=get_axis_limits(ax_dict['4']), xytext=(-0.1, ax_dict['4'].title.get_position()[1]+0.01), textcoords='axes fraction', fontproperties = font_props,zorder=1)
    #
    #
    # plt.savefig('D:/behavmodelfigs/reaction_times_by_ferret_swarm_interbigmosaic14091409.png', dpi=500)
    # plt.show()

    return df_by_ferret


def plot_reaction_times_interandintra_violin(ferrets):
    # plot the reaction times by animal
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    #only get correct hit trials for the swarm plot
    df_use = resultingdf[(resultingdf['realRelReleaseTimes'] >= 0) & (resultingdf['realRelReleaseTimes'] <= 2)]

    df_female = df_use.loc[(df_use['talker'] == 1) | (df_use['talker'] == 3) | (df_use['talker'] == 5)]
    df_female_control = df_female.loc[df_female['control_trial'] == 1]
    df_female_rove = df_female.loc[df_female['inter_trial_roving'] == 1]
    df_female_rove_intra = df_female.loc[df_female['intra_trial_roving'] == 1]

    df_male = df_use.loc[(df_use['talker'] == 2) | (df_use['talker'] == 8) | (df_use['talker'] == 13)]
    df_male_control = df_male.loc[df_male['control_trial'] == 1]
    df_male_rove = df_male.loc[df_male['inter_trial_roving'] == 1]
    df_male_rove_intra = df_male.loc[df_male['intra_trial_roving'] == 1]

    df_use['female_talker'] = df_use['talker'].isin([1, 3, 5]).astype(int)
    pitch_type_array = []
    custom_colors = ['slategray', 'hotpink', "yellow"]  # Add more colors as needed

    for i in range(0, len(df_use)):
        if df_use['inter_trial_roving'].iloc[i] == 1:
            pitch_type_array.append(1)
        elif df_use['intra_trial_roving'].iloc[i] == 1:
            pitch_type_array.append(2)
        else:
            pitch_type_array.append(0)
    df_use['roving_type'] = pitch_type_array





    roving_type = df_use['roving_type'].values
    talker_values = df_use["female_talker"].values
    release_values = df_use["realRelReleaseTimes"].values
    data_df = pd.DataFrame({
        "roving_type": roving_type,
        "talker": talker_values,
        "release_values": release_values
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="roving_type", y="release_values", hue="talker", data=data_df, split=True, inner="quart",
                   palette='Set2', ax=ax)
    max_release_time = np.max(df_use['realRelReleaseTimes'])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['control', 'inter', 'intra'], fontsize=18, rotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('Release time (s)', fontsize=18)  # Corrected y-label
    plt.ylim([0,2])
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Male', 'Female']
    ax.legend(handles=handles[0:], labels=labels[0:], title="talker", fontsize=14, title_fontsize=16)
    plt.savefig( 'D:/behavmodelfigs/rovingtypebytalker_violin14091409.png', dpi=500, bbox_inches='tight')
    plt.show()
    print('done')



def plot_reaction_times_interandintra_swarm(ferrets):
    # plot the reaction times by animal
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    #only get correct hit trials for the swarm plot
    df_use = resultingdf[(resultingdf['realRelReleaseTimes'] >= 0) & (resultingdf['realRelReleaseTimes'] <= 2)]

    df_female = df_use.loc[(df_use['talker'] == 1) | (df_use['talker'] == 3) | (df_use['talker'] == 5)]
    df_female_control = df_female.loc[df_female['control_trial'] == 1]
    df_female_rove = df_female.loc[df_female['inter_trial_roving'] == 1]
    df_female_rove_intra = df_female.loc[df_female['intra_trial_roving'] == 1]

    df_male = df_use.loc[(df_use['talker'] == 2) | (df_use['talker'] == 8) | (df_use['talker'] == 13)]
    df_male_control = df_male.loc[df_male['control_trial'] == 1]
    df_male_rove = df_male.loc[df_male['inter_trial_roving'] == 1]
    df_male_rove_intra = df_male.loc[df_male['intra_trial_roving'] == 1]

    # Set a manual palette with colors for female and male talkers
    palette = {'Female': 'blue', 'Male': 'red'}

    # now plot generally by all ferrets
    ax, fig = plt.subplots(figsize=(10, 12))
    sns.swarmplot(x=df_use['realRelReleaseTimes'], hue=df_use['talker'].map({1: 'Female', 2: 'Male', 3: 'Female', 5: 'Female', 8: 'Male', 13: 'Male'}), palette=palette)
    plt.title('Distribution of reaction times, \n irrespective of talker and ferret', fontsize=15)
    plt.show()

    # now plot by talker, showing reaction times
    data = [df_female_control['realRelReleaseTimes'], df_female_rove['realRelReleaseTimes'], df_female_rove_intra['realRelReleaseTimes']]
    labels = ['control F0', 'inter-roved F0', 'intra-roved F0']

    ax, fig = plt.subplots(figsize=(10, 8))
    sns.swarmplot(data=data, hue=df_female_control['talker'].map({1: 'Female', 3: 'Female', 5: 'Female'}), palette={'Female': 'blue'})
    sns.swarmplot(data=[df_male_control['realRelReleaseTimes'], df_male_rove['realRelReleaseTimes'], df_male_rove_intra['realRelReleaseTimes']], hue=df_male_control['talker'].map({2: 'Male', 8: 'Male', 13: 'Male'}), palette={'Male': 'red'})

    plt.title('Reaction times for the female and male talkers, \n irrespective of ferret', fontsize=15)
    plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.xticks(ticks=[0, 1, 2], labels=labels)
    plt.legend(title='Gender', title_fontsize=12, fontsize=10)
    plt.savefig('D:/behavmodelfigs/reaction_times_by_talker_female_male_interandintra14091409.png', dpi=500)
    plt.show()


    print('done')

if __name__ == '__main__':
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    # plot_reaction_times_interandintra_violin(ferrets)
    # plot_reaction_times_interandintra(ferrets)

    # plot_reaction_times_interandintra_swarm(ferrets)
    xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runfalsealarmpipeline(
        ferrets, optimization=False, ferret_as_feature=True)
    # ferrets = ['F2105_Clove']# 'F2105_Clove'
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
    # xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runfalsealarmpipeline(
    #     ferrets, optimization=False, ferret_as_feature=True)
    # for i in ferrets:
    #     xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runfalsealarmpipeline(
    #         [i], optimization=True, ferret_as_feature=False)


    #

    # plot_reaction_times_intra(ferrets)

    # col3 = 'pitchofprecur'
    # dfx = dfx.loc[:, dfx.columns != col3]

    # col3 = 'stepval'
    # dfx = dfx.loc[:, dfx.columns != col3]
