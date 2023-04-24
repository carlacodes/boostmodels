import sklearn.metrics
# from rpy2.robjects import pandas2ri
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
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

    # param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
    # param['nthread'] = 4
    # param['eval_metric'] = 'auc'

    xg_reg = lgb.LGBMRegressor(random_state=123, verbose=1, **paramsinput)
    xg_reg.fit(X_train, y_train, verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
    plt.title('feature importances for the LGBM Correct Release Times model for ferret ' + ferret_name)
    plt.show()

    kfold = KFold(n_splits=10)
    mse_train = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)

    mse_test = mean_squared_error(ypred, y_test)
    print("MSE on test: %.4f" % (mse_test) + ferret_name)
    print("negative MSE training: %.2f%%" % (np.mean(mse_train) * 100.0))
    print(mse_train)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(dfx)
    fig, ax = plt.subplots(figsize=(15, 15))
    # title kwargs still does nothing so need this workaround for summary plots
    shap.summary_plot(shap_values, dfx, show=False)
    fig, ax = plt.gcf(), plt.gca()
    plt.title('Ranked list of features over their impact in predicting reaction time for' + ferret_name)
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

    return xg_reg, ypred, y_test, results, mse_test



def runlgbreleasetimes(X, y, paramsinput=None):


    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                        random_state=42)

    # param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
    # param['nthread'] = 4
    # param['eval_metric'] = 'auc'

    xg_reg = lgb.LGBMRegressor(random_state=42, verbose=1, **paramsinput)
    # xg_reg = lgb.LGBMRegressor( colsample_bytree=0.3, learning_rate=0.1,
    #                           max_depth=10, alpha=10, n_estimators=10, random_state=42, verbose=1)


    xg_reg.fit(X_train, y_train, verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
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

    #introduce cmap
    custom_colors = ['lightcoral', "cyan", "orange"]  # Add more colors as needed
    # cmapcustom = matplotlib.colors.ListedColormap(custom_colors)
    cmapcustom = mcolors.LinearSegmentedColormap.from_list('my_custom_cmap', custom_colors, N=1000)
    custom_colors_summary = ['lightcoral', "cyan", "orange"]  # Add more colors as needed
    cmapsummary = matplotlib.colors.ListedColormap(custom_colors_summary)
    cmapname = "viridis"
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.summary_plot(shap_values, X, show=False, cmap = matplotlib.colormaps[cmapname])

    fig, ax = plt.gcf(), plt.gca()
    plt.title('Ranked list of features over their impact in predicting reaction time')
    plt.xlabel('SHAP value (impact on model output) on reaction time')
    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
    labels[11] = 'trial Number'
    labels[10] = 'time to target'
    labels[9] = 'side of audio presentation'
    labels[8] = 'pitch of target'
    labels[7] = 'pitch of precursor'
    labels[6] = 'day since start of week'
    labels[5] = 'trial took place in AM'
    labels[4] = 'past trial catch'
    labels[3] = 'male/female talker'
    labels[2] = 'precursor = target pitch'
    labels[1] = 'change in pitch value'
    labels[0] = 'past trial was correct'

    ax.set_yticklabels(labels)
    plt.savefig('figs/correctrxntimemodel/shapsummaryplot_allanimals2.png', dpi=1000, bbox_inches='tight')

    plt.show()

    shap.dependence_plot("timeToTarget", shap_values, X)  #
    explainer = shap.Explainer(xg_reg, X)
    shap_values2 = explainer(X_train)


    shap.plots.scatter(shap_values2[:, "timeToTarget"], color=shap_values2[:, "trialNum"], show=False, cmap = matplotlib.colormaps[cmapname])
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Trial number", fontsize=12)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Target presentation time \n versus impact in predicted reacton time', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Target presentation time', fontsize=16)
    plt.savefig('figs/correctrxntimemodel/targtimescolouredbytrialnumber.png', dpi=1000)
    plt.show()

    shap.plots.scatter(shap_values2[:, "pitchoftarg"], color=shap_values2[:, "pitchofprecur"], show=False, cmap = matplotlib.colormaps[cmapname])
    fig, ax = plt.gcf(), plt.gca()
    # Get colorbar
    cb_ax = fig.axes[1]
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Pitch of precursor", fontsize=12)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('Pitch of target \n versus impact in predicted reacton time', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Pitch of target', fontsize=16)
    plt.savefig('figs/correctrxntimemodel/pitchoftargcolouredbyprecur.png', dpi=1000)
    plt.show()

    return xg_reg, ypred, y_test, results

def extract_release_times_data(ferrets):
    df = behaviouralhelperscg.get_df_behav(ferrets=ferrets, includefaandmiss=False, startdate='04-01-2020', finishdate='01-03-2023')
    dfuse = df[["pitchoftarg", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                "timeToTarget", "DaysSinceStart", "AM",
                "realRelReleaseTimes", "ferret", "stepval", "pastcorrectresp", "pastcatchtrial", "trialNum"]]
    return dfuse



def run_correctrxntime_model(ferrets, optimization = False ):
    df_use = extract_release_times_data(ferrets)

    col = 'realRelReleaseTimes'
    dfx = df_use.loc[:, df_use.columns != col]

    # remove ferret as possible feature
    col2 = 'ferret'
    dfx = dfx.loc[:, dfx.columns != col2]
    # study_release_times = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
    if optimization == False:
        best_params = {'colsample_bytree': 0.49619263716341894,
                       'alpha': 8.537376181435246,
                       'n_estimators': 96,
                       'learning_rate': 0.17871472565344848,
                       'max_depth': 5,
                       'bagging_fraction': 0.7000000000000001,
                       'bagging_freq': 6}
        best_params = np.load('optuna_results/best_paramsreleastimemodel_allferrets.npy', allow_pickle=True).item()
    else:
        best_study_results = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
        best_params = best_study_results.best_params
        np.save('optuna_results/best_paramsreleastimemodel_allferrets.npy', best_params)

    xg_reg, ypred, y_test, results = runlgbreleasetimes(dfx, df_use[col], paramsinput=best_params)


def main():
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove'] #'F2105_Clove'
    run_correctrxntime_model(ferrets, optimization = False)


if __name__ == '__main__':
    main()