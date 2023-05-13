import sklearn.metrics
import seaborn as sns
from instruments.io.BehaviourIO import BehaviourDataSet
from sklearn.inspection import permutation_importance
from instruments.behaviouralAnalysis import reactionTimeAnalysis  # outputbehaviordf
from pathlib import Path
from sklearn.model_selection import cross_val_score
import shap
import matplotlib
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn
from sklearn.model_selection import train_test_split
from helpers.behaviouralhelpersformodels import *




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


def run_optuna_study_correctresp(X, y):
    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=1000)
    print("Number of finished trials: ", len(study.trials))
    print(f"\tBest value of binary log loss: {study.best_value:.5f}")
    print(f"\tBest params:")

    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")
    return study
def runlgbcorrectrespornotwithoptuna(dataframe, paramsinput=None, optimization = False, ferret_as_feature=False):
    if paramsinput is None:
        paramsinput = {'colsample_bytree': 0.4253436090928764, 'alpha': 18.500902749816458, 'is_unbalanced': True,
                       'n_estimators': 8700, 'learning_rate': 0.45629236152754304, 'num_leaves': 670, 'max_depth': 3,
                       'min_data_in_leaf': 200, 'lambda_l1': 4, 'lambda_l2': 64,
                       'min_gain_to_split': 0.016768866636887758, 'bagging_fraction': 0.9, 'bagging_freq': 3,
                       'feature_fraction': 0.2}

    if ferret_as_feature == True:
        df_to_use = dataframe[["pitchoftarg", "trialNum", "misslist", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                           "targTimes","pastcorrectresp",
                           "pastcatchtrial", "ferret"]]
        labels = ['pitch of target', 'trial number','misslist', 'pitch of precursor', 'talker', 'audio side', 'precursor = target pitch','target presentation time', 'past response was correct', 'past trial was catch', 'ferret ID']
        df_to_use = df_to_use.rename(columns=dict(zip(df_to_use.columns, labels)))

        fig_dir = Path('D:/behavmodelfigs/correctresp_or_miss/ferret_as_feature')
        col = 'misslist'
        dfx = df_to_use.loc[:, df_to_use.columns != col]
        if optimization == False:
            # load the saved params
            paramsinput = np.load('../optuna_results/correctresponse_optunaparams_ferretasfeature.npy', allow_pickle=True).item()
        else:
            study = run_optuna_study_correctresp(dfx.to_numpy(), df_to_use['misslist'].to_numpy())
            print(study.best_params)
            paramsinput = study.best_params
            np.save('../optuna_results/correctresponse_optunaparams_ferretasfeature.npy', study.best_params)

    else:
        df_to_use = dataframe[["pitchoftarg", "trialNum", "misslist", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                           "targTimes","pastcorrectresp",
                           "pastcatchtrial", ]]

        labels = ['pitch of target', 'trial number','misslist', 'pitch of precursor', 'talker', 'audio side', 'precursor = target pitch','target presentation time', 'past response was correct', 'past trial was catch']
        df_to_use = df_to_use.rename(columns=dict(zip(df_to_use.columns, labels)))

        fig_dir = Path('D:/behavmodelfigs/correctresp_or_miss/')
        col = 'misslist'
        dfx = df_to_use.loc[:, df_to_use.columns != col]
        if optimization == False:
            # load the saved params
            paramsinput = np.load('../optuna_results/correctresponse_optunaparams.npy', allow_pickle=True).item()
        else:
            study = run_optuna_study_correctresp(dfx.to_numpy(), df_to_use['misslist'].to_numpy())
            print(study.best_params)
            paramsinput = study.best_params
            np.save('../optuna_results/correctresponse_optunaparams.npy', study.best_params)



    X_train, X_test, y_train, y_test = train_test_split(dfx, df_to_use['misslist'], test_size=0.2, random_state=123)
    print(X_train.shape)
    print(X_test.shape)

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

    shap_values1 = shap.TreeExplainer(xg_reg).shap_values(X_train)


    custom_colors = ['gold',  'peru', "purple"]  # Add more colors as needed
    cmapcustom = mcolors.LinearSegmentedColormap.from_list('my_custom_cmap', custom_colors, N=1000)
    custom_colors_summary = ['peru', 'gold',]  # Add more colors as needed
    cmapsummary = matplotlib.colors.ListedColormap(custom_colors_summary)

    #elbow plot
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
    plt.plot(feature_labels, cumulative_importances_combined, marker='o', color = 'gold')
    plt.xlabel('Features')
    plt.ylabel('Cumulative Feature Importance')
    plt.title('Elbow Plot of Cumulative Feature Importance for Miss Model')
    plt.xticks(rotation=45, ha='right')  # rotate x-axis labels for better readability
    plt.savefig(fig_dir / 'elbowplot.png', dpi=500, bbox_inches='tight')
    plt.show()

    shap.summary_plot(shap_values1, X_train, show = False, color=cmapsummary)
    fig, ax = plt.gcf(), plt.gca()
    plt.title('Ranked list of features over their \n impact in predicting a miss', fontsize = 18)
    # Get the plot's Patch objects
    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
    labels[12] = 'side of audio presentation'
    labels[11] = 'trial number'
    labels[10] = 'pitch of precursor'
    labels[9] = 'target presentation time'
    labels[8] = 'pitch of target'
    labels[7] = 'session occured in the morning'
    labels[6] = 'cosine similarity'
    labels[5] = 'past trial was catch'
    labels[4] = 'precursor pitch = target pitch'
    labels[3] = 'past trial was correct'
    labels[2] = 'pitch change'
    labels[1] = 'Days since start of week'
    labels[0] = 'talker'
    # ax.set_yticklabels(labels)
    fig.tight_layout()
    plt.savefig(fig_dir / 'shap_summary_correctresp.png', dpi=1000, bbox_inches = "tight")


    shap.dependence_plot("pitchofprecur", shap_values1[0], X_train)  #
    plt.show()

    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=100,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T, color = 'peru')
    ax.set_title("Permutation importances on predicting a miss")
    fig.tight_layout()
    plt.savefig(fig_dir / 'permutation_importance.png', dpi=500)
    plt.show()


    explainer = shap.Explainer(xg_reg, X_train)
    shap_values2 = explainer(X_train)

    fig, ax = plt.subplots()
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "pitchofprecur"], ax=ax, cmap = cmapcustom, show = False)
    fig, ax = plt.gcf(), plt.gca()
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_yticks([1, 2, 3,4, 5])
    cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
    cb_ax.set_ylabel("Pitch of precursor word (Hz)", fontsize=15)
    plt.title('Trial number and its effet on the \n miss probability', fontsize = 18)
    plt.xlabel('Trial number', fontsize = 15)
    plt.ylabel('SHAP value', fontsize = 15)
    plt.savefig( fig_dir / 'trialnum_vs_precurpitch.png', dpi=1000, bbox_inches = "tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.scatter(shap_values2[:, "side"], color=shap_values2[:, "pitchofprecur"], ax=ax, cmap = cmapcustom, show = False)
    fig, ax = plt.gcf(), plt.gca()
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Pitch of precursor word", fontsize=15)

    plt.xticks([0, 1 ], labels = ['left', 'right'], fontsize =15)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Pitch of the side of the booth \n versus impact in miss probability', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Side of audio presenetation', fontsize=16)
    plt.show()

    shap.plots.scatter(shap_values2[:, "pitchoftarg"], color=shap_values2[:, "pitchofprecur"], show=False, cmap = cmapcustom)
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_yticks([1, 2, 3,4, 5])
    cb_ax.set_yticklabels(['109', '124', '144', '191', '251'])
    cb_ax.set_ylabel("Pitch of precursor", fontsize=12)
    cb_ax.set_yticklabels( ['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'], fontsize=15)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Pitch of target \n versus impact in miss probability', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Pitch of target (Hz)', fontsize=16)
    plt.xticks([1,2,3,4,5], labels=['109', '124', '144 ', '191', '251'], fontsize=15)
    plt.show()


    shap.plots.scatter(shap_values2[:, "precur_and_targ_same"], color=shap_values2[:, "talker"])
    plt.show()
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "talker"], show=False)
    plt.title('trial number \n vs. SHAP value impact')
    plt.ylabel('SHAP value', fontsize=18)
    plt.show()

    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "precur_and_targ_same"], show=False)
    plt.title('Cosine similarity \n vs. SHAP value impact')
    plt.ylabel('SHAP value', fontsize=18)
    # plt.savefig('D:/behavmodelfigs/correctrespmodel/cosinesimdepenencyplot.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "precur_and_targ_same"], color=shap_values2[:, "cosinesim"], show=False)

    plt.title('Intra trial roving \n versus SHAP value impact', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.show()

    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "targTimes"], show=False)
    plt.title('CR model - Trial number versus SHAP value, \n colored by target presentation time', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xlabel('Trial number', fontsize=15)
    plt.show()

    shap.plots.scatter(shap_values2[:, "targTimes"], color=shap_values2[:, "trialNum"], show=False)
    plt.title('CR model - Target times versus SHAP value, \n colored by trial number', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xlabel('Target presentation time', fontsize=15)
    plt.show()

    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "targTimes"], show=False)
    plt.title('Cosine Similarity as a function \n of SHAP values coloured by targTimes')
    plt.show()

    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "pitchoftarg"], show=False)
    plt.title('Cosine similarity as a function \n of SHAP values, coloured by the target pitch', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.show()
    fig, ax = plt.subplots(figsize=(15, 35))
    shap.plots.scatter(shap_values2[:, "side"], color=shap_values2[:, "trialNum"], show=False)
    plt.title('SHAP values as a function of the side of the audio, \n coloured by the trial number', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xticks([0, 1], ['Left', 'Right'], fontsize=18)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 55))
    shap.plots.scatter(shap_values2[:, "pitchoftarg"], color=shap_values2[:, "targTimes"], show=False)
    plt.title('SHAP values as a function of the pitch of the target, \n coloured by the target presentation time',
              fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xlabel('Pitch of target', fontsize=12)
    plt.xticks([1, 2, 3, 4, 5], ['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'], fontsize=18)
    plt.show()

    return xg_reg, ypred, y_test, results, shap_values1, X_train, y_train, bal_accuracy, shap_values2


def run_correct_responsepipeline(ferrets):
    resultingcr_df = behaviouralhelperscg.get_df_behav(ferrets=ferrets, includefaandmiss=False, includemissonly=True, startdate='04-01-2020',
                                  finishdate='03-01-2023')
    filepath = Path('D:/dfformixedmodels/correctresponsemodel_dfuse.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    resultingcr_df.to_csv(filepath)
    xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runlgbcorrectrespornotwithoptuna(
        resultingcr_df, optimization=False, ferret_as_feature = True)
    return xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2


def test_function(ferrets):
    resultingcr_df = behaviouralhelperscg.get_df_behav(ferrets=ferrets, includefaandmiss=False, includemissonly=True, startdate='04-01-2020',
                                  finishdate='03-01-2023')
    filepath = Path('D:/dfformixedmodels/correctresponsemodel_dfuse.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    resultingcr_df.to_csv(filepath)
    xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runlgbcorrectrespornotwithoptuna(
        resultingcr_df, optimization=False, ferret_as_feature = True)
    return xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2

if __name__ == '__main__':
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'] #'F2105_Clove'

    xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = run_correct_responsepipeline(ferrets)

