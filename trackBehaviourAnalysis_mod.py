import click
import instruments
from instruments.io.BehaviourIO import BehaviourDataSet, WeekBehaviourDataSet
from instruments.config import behaviouralDataPath, behaviourOutput
from instruments.behaviouralAnalysis import createWeekBehaviourFigs, reactionTimeAnalysis, outputbehaviordf
import math
from time import time

from scipy.stats import sem
import os
import numpy as np
from instruments.helpers.extract_helpers import extractAllFerretData
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pysr3.lme.models import L1LmeModelSR3
from pysr3.lme.problems import LMEProblem, LMEStratifiedShuffleSplit
import numpy as np
from pysr3.linear.models import LinearL1ModelSR3
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
import statistics as stats


# jules' extract behvioural data analysis
# @click.group()
# def cli():
#     pass
#
#
# @click.command(name='behaviour_week')
# @click.option('--path', '-p', type=click.Path(exists=True))
# @click.option('--output', '-o', type=click.Path(exists=False))
# @click.option('--ferrets', '-f', default=None)
# @click.option('--day', '-d', default=None)
def cli_behaviour_week(path=None,
                       output=None,
                       ferrets=None,
                       day=None):
    if output is None:
        output = behaviourOutput

    if path is None:
        path = behaviouralDataPath

    dataSet = WeekBehaviourDataSet(filepath=path,
                                   outDir=output,
                                   ferrets=ferrets,
                                   day=day)

    if ferrets is not None:
        ferrets = [ferrets]
    else:
        ferrets = [ferret for ferret in next(os.walk(db_path))[1] if ferret.startswith('F')]

    allData = dataSet._load()
    for ferret in ferrets:
        ferretFigs = createWeekBehaviourFigs(allData, ferret)
        dataSet._save(figs=ferretFigs)


#
# cli.add_command(cli_behaviour_week)


# @click.command(name='reaction_time')
# @click.option('--path', '-p', type=click.Path(exists=True))
# @click.option('--output', '-o', type=click.Path(exists=False))
# @click.option('--ferrets', '-f', default=None)
# @click.option('--startdate', '-sta', default=None)
# @click.option('--finishdate', '-sto', default=None)
def cli_reaction_time(path=None,
                      output=None,
                      ferrets=None,
                      startdate=None,
                      finishdate=None):
    if output is None:
        output = behaviourOutput

    if path is None:
        path = behaviouralDataPath

    #    if ferrets is not None:
    #        ferrets = [ferrets]
    #    else:
    #        ferrets = [ferret for ferret in next(os.walk(db_path))[1] if ferret.startswith('F')]

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


def get_df_behav(path=None,
                 output=None,
                 ferrets=None,
                 startdate=None,
                 finishdate=None):
    if output is None:
        output = behaviourOutput

    if path is None:
        path = behaviouralDataPath

    #    if ferrets is not None:
    #        ferrets = [ferrets]
    #    else:
    #        ferrets = [ferret for ferret in next(os.walk(db_path))[1] if ferret.startswith('F')]

    dataSet = BehaviourDataSet(filepath=path,
                               startDate=startdate,
                               finishDate=finishdate,
                               ferrets=ferrets,
                               outDir=output)
    allData, ferrets = extractAllFerretData(ferrets, path, startDate=startdate,
                                            finishDate=finishdate)

    # allData = dataSet._load()

    # for ferret in ferrets:
    ferret = ferrets
    # ferrData = allData.loc[allData.ferretname == ferret]
    # if ferret == 'F1702_Zola':
    #     ferrData = ferrData.loc[(ferrData.dates != '2021-10-04 10:25:00')]
    # newdata = allData[allData['catchTrial'].isin([0])]
    # newdata[newdata['response'].isin([0,1])]
    # allData = newdata
    fs = 24414.062500
    newdata = allData[(allData.response == 0) | (allData.response == 1)]  # | (allData.response == 7)
    newdata = newdata[(newdata.correctionTrial == 0)]  # | (allData.response == 7)

    # newdata = allData['absentTime'][0]
    newdata['targTimes'] = newdata['timeToTarget'] / fs

    newdata['centreRelease'] = newdata['lickRelease'] - newdata['startTrialLick']
    newdata['relReleaseTimes'] = newdata['centreRelease'] - newdata['targTimes']
    newdata['realRelReleaseTimes'] = newdata['relReleaseTimes'] - newdata['absentTime']

    pitchshiftmat = newdata['PitchShiftMat']
    precursorlist = newdata['distractors']
    talkerlist = newdata['talker']
    pitchoftarg = np.empty(len(pitchshiftmat))
    pitchofprecur = np.empty(len(pitchshiftmat))
    gradinpitch = np.empty(len(pitchshiftmat))
    gradinpitchprecur = np.empty(len(pitchshiftmat))
    timetotarglist = np.empty(len(pitchshiftmat))

    for i in range(0, len(pitchshiftmat)):

        chosentrial = pitchshiftmat.values[i]
        chosendisttrial = precursorlist.values[i]
        chosentalker = talkerlist.values[i]
        if chosentalker == 1:
            origF0 = 191
        else:
            origF0 = 124

        targpos = np.where(chosendisttrial == 1)
        try:
            pitchoftarg[i] = chosentrial[targpos[0] - 1]
            if chosentrial[targpos[0] - 1] == 1:
                pitchoftarg[i] = 191
            if chosentrial[targpos[0] - 1] == 2:
                pitchoftarg[i] = 124
            if chosentrial[targpos[0] - 1] == 3:
                pitchoftarg[i] = 144
            if chosentrial[targpos[0] - 1] == 4:
                pitchoftarg[i] = 191

            if chosentrial[targpos[0] - 1] == 5:
                pitchoftarg[i] = 251
            if chosentrial[targpos[0] - 1] == 6:
                pitchoftarg[i] = 332
            if chosentrial[targpos[0] - 1] == 7:
                pitchoftarg[i] = 367
            if chosentrial[targpos[0] - 1] == 8:
                pitchoftarg[i] = 144
            if chosentrial[targpos[0] - 1] == 9:
                pitchoftarg[i] = 191
            if chosentrial[targpos[0] - 1] == 10:
                pitchoftarg[i] = 251
            if chosentrial[targpos[0] - 1] == 11:
                pitchoftarg[i] = 332
            if chosentrial[targpos[0] - 1] == 12:
                pitchoftarg[i] = 367
            if chosentrial[targpos[0] - 1] == 13:
                pitchoftarg[i] = 109
            if chosentrial[targpos[0] - 1] == 14:
                pitchoftarg[i] = 109

            gradinpitch[i] = origF0 - pitchoftarg[i]

            if chosentrial[targpos[0] - 1] == 0:
                pitchoftarg[i] = origF0  # talkerlist.values[i]

            pitchofprecur[i] = chosentrial[targpos[0] - 2]
            if chosentrial[targpos[0] - 2] == 1:
                pitchofprecur[i] = 191
            if chosentrial[targpos[0] - 2] == 2:
                pitchofprecur[i] = 124
            if chosentrial[targpos[0] - 2] == 3:
                pitchofprecur[i] = 144
            if chosentrial[targpos[0] - 2] == 4:
                pitchofprecur[i] = 191

            if chosentrial[targpos[0] - 2] == 5:
                pitchofprecur[i] = 251
            if chosentrial[targpos[0] - 2] == 6:
                pitchofprecur[i] = 332
            if chosentrial[targpos[0] - 2] == 7:
                pitchofprecur[i] = 367
            if chosentrial[targpos[0] - 2] == 8:
                pitchofprecur[i] = 144
            if chosentrial[targpos[0] - 2] == 9:
                pitchofprecur[i] = 191
            if chosentrial[targpos[0] - 2] == 10:
                pitchofprecur[i] = 251
            if chosentrial[targpos[0] - 2] == 11:
                pitchofprecur[i] = 332
            if chosentrial[targpos[0] - 2] == 12:
                pitchofprecur[i] = 367
            if chosentrial[targpos[0] - 2] == 13:
                pitchofprecur[i] = 109
            if chosentrial[targpos[0] - 2] == 14:
                pitchofprecur[i] = 109
            if chosentrial[targpos[0] - 2] == 0:
                pitchofprecur[i] = origF0  # talkerlist.values[i]
            gradinpitchprecur[i] = origF0 - pitchofprecur[i]
        except:
            newdata.drop(i)
            continue
        # if not isinstance(chosentrial, (np.ndarray, np.generic)):
        #     if math.isnan(chosentrial):
        #         chosentrial = np.zeros(5)
        # try:
        #     chosentrial = chosentrial[chosentrial != 0]
        # except:
        #     continue
        # if chosentrial.size == 0:
        #     pitchoftarg[i] = 0
        #
        # else:
        #     try:
        #         pitchoftarg[i] = int(chosentrial[targpos])
        #         pitchofprecur[i] = chosentrial[targpos - 1]
        #     except:
        #         pitchoftarg[i] = 0
        #         pitchofprecur[i] = 0

    newdata['pitchoftarg'] = pitchoftarg.tolist()
    newdata['pitchofprecur'] = pitchofprecur.tolist()
    newdata['gradinpitch'] = gradinpitch.tolist()
    newdata['gradinpitchprecur'] = gradinpitchprecur.tolist()
    newdata['timeToTarget'] = newdata['timeToTarget'] / 24414.0625

    return newdata

    # ferretFigs = reactionTimeAnalysis(ferrData)
    # dataSet._save(figs=ferretFigs, file_name='reaction_times_{}_{}_{}.pdf'.format(ferret, startdate, finishdate))


# editing to extract different vars from df
# cli.add_command(cli_reaction_time)

if __name__ == '__main__':
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni']
    # for i, currFerr in enumerate(ferrets):
    #     print(i, currFerr)
    df = get_df_behav(ferrets=ferrets, startdate='04-01-2020', finishdate='27-01-2022')
    # cli_reaction_time(ferrets='F1702_Zola', startdate='04-01-2020', finishdate='04-01-2022')
    # data = sm.datasets.get_rdataset("Sitka", "MASS").data
    endog = df['realRelReleaseTimes']
    endog2 = df['realRelReleaseTimes'].to_numpy()

    exog = df[["ferret", "pitchoftarg", "pitchofprecur", "talker", "side", "gradinpitch", "gradinpitchprecur",
               "timeToTarget"]]
    # testing AIC with different exog vars
    exog_reduced = df[['pitchofprecur', 'side','gradinpitchprecur', 'timeToTarget']]

    exog2 = df[["ferret", "pitchoftarg", "pitchofprecur", "talker", "side", "gradinpitch", "gradinpitchprecur",
                "timeToTarget"]].to_numpy()
    # TODO: CLEAN CODE
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import RidgeCV

    X = df[["pitchoftarg", "pitchofprecur", "talker", "side", "gradinpitch", "gradinpitchprecur",
            "timeToTarget"]].to_numpy()

    y = endog2

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LassoCV




    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
    importance = np.abs(ridge.coef_)
    feature_names = np.array(df[["pitchoftarg", "pitchofprecur", "talker", "side", "gradinpitch", "gradinpitchprecur",
                                 "timeToTarget"]].columns)
    plt.bar(height=importance, x=feature_names)
    plt.title("Feature importances via coefficients")
    plt.show()


    lasso = LassoCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
    importance = np.abs(lasso.coef_)
    feature_names = np.array(df[["pitchoftarg", "pitchofprecur", "talker", "side", "gradinpitch", "gradinpitchprecur",
                                 "timeToTarget"]].columns)
    plt.bar(height=importance, x=feature_names)
    plt.title("Feature importances via coefficients lasso")
    plt.show()

    from sklearn.feature_selection import SequentialFeatureSelector

    tic_fwd = time()
    sfs_forward = SequentialFeatureSelector(
        ridge, n_features_to_select=4, direction="forward"
    ).fit(X, y)
    toc_fwd = time()

    tic_bwd = time()
    sfs_backward = SequentialFeatureSelector(
        ridge, n_features_to_select=4, direction="backward"
    ).fit(X, y)
    toc_bwd = time()

    print(
        "Features selected by forward sequential selection: "
        f"{feature_names[sfs_forward.get_support()]}"
    )
    print(f"Done in {toc_fwd - tic_fwd:.3f}s")
    print(
        "Features selected by backward sequential selection: "
        f"{feature_names[sfs_backward.get_support()]}"
    )
    print(f"Done in {toc_bwd - tic_bwd:.3f}s")

    # varianceofarray = np.var(exog2, axis=1)
    from scipy.stats import sem

    stderrorofrealrelreleasetime = sem(df["realRelReleaseTimes"])
    varianceofarray = np.ones((len(exog2))) * (
                stderrorofrealrelreleasetime * stderrorofrealrelreleasetime)  # np.ones((len(exog2)))*1e-8 #taken as the
    # varofvar=varianceofarray.var()
    exog2 = np.insert(exog2, 8, varianceofarray, axis=1)
    md = sm.MixedLM(endog, exog, groups=df["ferret"], exog_re=None)

    mdf = md.fit(reml=False)
    print(mdf.params)
    print(mdf.summary())
    print(mdf.aic)
    print(mdf.bic)

    md_reduced = sm.MixedLM(endog, exog_reduced, groups=df["ferret"], exog_re=None)
    mdf_reduced = md_reduced.fit(reml=False)
    print(mdf_reduced.summary())
    print(mdf_reduced.aic)
    print(mdf_reduced.bic)
    seed = 42
    # print(mdf.params)
    model = L1LmeModelSR3()
    # x is  a long array of dependent VARS
    # y is the independent VAR for your prediction

    # We're going to select features by varying the strength of the prior
    # and choosing the model_name that yields the best information criterion
    # on the validation set.
    params = {
        "lam": loguniform(1e-3, 1e3)
    }
    # We use standard functionality of sklearn to perform grid-search.
    # column_labels = ['group'] * 1 + ["fixed+random"] * 7 + ['variance'] * 1
    # row_labels = ['ferret'] + ['pitchoftarg', 'pitchofprecur', 'talker', 'side', 'gradinpitchprecur', 'gradinpitch', 'timeToTarget'] + ['variance']
    # df['variances'] = varianceofarray
    # problem = LMEProblem.from_dataframe(
    #     data=df,
    #     fixed_effects=['talker', 'side', 'pitchoftarg', 'pitchofprecur','gradinpitchprecur', 'gradinpitch', 'timeToTarget'],
    #     random_effects=[],
    #     groups='ferret',
    #     variance='variances',
    #     target='realRelReleaseTimes',
    #     must_include_fe=['talker'],
    #     must_include_re=[],
    # )

    # LMEProblem provides a very convenient representation
    # of the problem. See the documentation for more details.

    # It also can be converted to a more familiar representation
    # x, y, columns_labels = problem.to_x_y()
    # selector = RandomizedSearchCV(estimator=model,
    #                               param_distributions=params,
    #                               n_iter=5,  # number of points from parameters space to sample
    #                               # the class below implements CV-splits for LME models
    #                               cv=LMEStratifiedShuffleSplit(n_splits=2, test_size=0.5,
    #                                                            random_state=seed, columns_labels=columns_labels),
    #                               # The function below will evaluate the information criterion
    #                               # on the test-sets during cross-validation.
    #                               # use the function below to evaluate the information criterion
    #                               scoring=lambda clf, x, y: -clf.get_information_criterion(exog2, endog2,
    #                                                                                        columns_labels=columns_labels,
    #                                                                                        ic="vaida_aic"),
    #                               random_state=seed,
    #                               n_jobs=20
    #                               )
    # selector.fit(x, y, columns_labels=columns_labels)
    # best_model = selector.best_estimator_
    #
    # maybe_beta = best_model.coef_["beta"]
    # maybe_gamma = best_model.coef_["gamma"]
    # beta_coeffs_to_use = abs(maybe_beta) > 1e-2
    # gamma_coeffs_to_use = abs(maybe_gamma) > 1e-2
    # # returning a binary array of 0 1 true false to get the actually relevant values?? -- yes as in the sk learn function y_true is the first argument and y_pred is the second argument
    # ftn, ffp, ffn, ftp = confusion_matrix(true_parameters["beta"], abs(maybe_beta) > 1e-2).ravel()
    # rtn, rfp, rfn, rtp = confusion_matrix(true_parameters["gamma"], abs(maybe_gamma) > 1e-2).ravel()
    # print(
    #     f"The model_name found {ftp} out of {ftp + ffn} correct fixed features, and also chose {ffp} out of {ftn + ffn} extra irrelevant fixed features. \n"
    #     f"It also identified {rtp} out of {rtp + rfn} random effects correctly, and got {rfp} out of {rtn + rfn} non-present random effects. \n"
    #     f"The best sparsity parameter is {selector.best_params_}")
