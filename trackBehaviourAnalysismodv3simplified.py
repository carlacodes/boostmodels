import click
import instruments
from instruments.io.BehaviourIO import BehaviourDataSet, WeekBehaviourDataSet
from instruments.config import behaviouralDataPath, behaviourOutput
from instruments.behaviouralAnalysis import createWeekBehaviourFigs, reactionTimeAnalysis  # outputbehaviordf
import math
from time import time
from pymer4.models import Lmer
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from pymer4.models import Lmer

scaler = MinMaxScaler()

from scipy.stats import sem
import os

import matplotlib.pyplot as plt
import numpy as np
from instruments.helpers.extract_helpers import extractAllFerretData
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pysr3.lme.models import L1LmeModelSR3
from pysr3.lme.problems import LMEProblem, LMEStratifiedShuffleSplit
import numpy as np
import pymer4
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from pysr3.linear.models import LinearL1ModelSR3
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
import statistics as stats
import rpy2
from rpy2.robjects.packages import importr

# import R's "base" package
base = importr('base')
easystats = importr('easystats')
performance = importr('performance')


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
                 includefaandmiss=False,
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
    fs = 24414.062500
    if includefaandmiss is True:
        newdata = allData[(allData.response == 0) | (allData.response == 1) | (allData.response == 7)]
    else:
        newdata = allData[(allData.response == 0) | (allData.response == 1)]
        newdata = newdata[(newdata.catchTrial == 0)]

    newdata = newdata[(newdata.correctionTrial == 0)]  # | (allData.response == 7)
    newdata = newdata[(newdata.currAtten == 0)]  # | (allData.response == 7)

    # newdata = allData['absentTime'][0]
    newdata['targTimes'] = newdata['timeToTarget'] / fs

    newdata['centreRelease'] = newdata['lickRelease'] - newdata['startTrialLick']
    newdata['relReleaseTimes'] = newdata['centreRelease'] - newdata['targTimes']
    newdata['realRelReleaseTimes'] = newdata['relReleaseTimes'] - newdata['absentTime']

    distractors = newdata['distractors']
    talkermat = {}
    talkerlist = newdata['talker']

    for i0 in range(0, len(distractors)):
        talkermat[i0] = int(talkerlist.values[i0]) * np.ones(len(distractors.values[i0]))
    talkermat = pd.Series(talkermat, index=talkermat.keys())

    pitchshiftmat = newdata['PitchShiftMat']
    # if len(pitchshiftmat) == 0:
    #     pitchshiftmat = talkermat  # make array equivalent to size of pitch shift mat just like talker [3,3,3,3] # if this is inter trial roving then talker is the pitch shift

    # except:
    #     pitchshiftmat = talkermat  # make array equivalent to size of pitch shift mat just like talker [3,3,3,3] # if this is inter trial roving then talker is the pitch shift
    precursorlist = newdata['distractors']
    chosenresponse = newdata['response']
    pitchoftarg = np.empty(len(pitchshiftmat))
    pitchofprecur = np.empty(len(pitchshiftmat))
    stepval = np.empty(len(pitchshiftmat))
    gradinpitch = np.empty(len(pitchshiftmat))
    gradinpitchprecur = np.empty(len(pitchshiftmat))
    timetotarglist = np.empty(len(pitchshiftmat))
    precur_and_targ_same = np.empty(len(pitchshiftmat))
    talkerlist2 = np.empty(len(pitchshiftmat))
    correctresp = np.empty(shape=(0, 0))
    droplist = np.empty(shape=(0, 0))

    for i in range(0, len(talkerlist)):
        chosenresponseindex = chosenresponse.values[i]
        chosentrial = pitchshiftmat.values[i]
        if isinstance(chosentrial, float):
            chosentrial = talkermat.values[i]
        chosendisttrial = precursorlist.values[i]
        chosentalker = talkerlist.values[i]
        if chosentalker == 3:
            chosentalker = 1
        if chosentalker == 8:
            chosentalker = 2
        if chosentalker == 13:
            chosentalker = 2
        if chosentalker == 5:
            chosentalker = 1
        talkerlist2[i] = chosentalker

        if chosentalker == 1:
            origF0 = 191
        else:
            origF0 = 124

        targpos = np.where(chosendisttrial == 1)
        if chosenresponseindex == 0 or chosenresponseindex == 1:
            correctresp = np.append(correctresp, 1)
        else:
            correctresp = np.append(correctresp, 0)
        try:
            if chosentrial[targpos[0]] == 8.0:
                pitchoftarg[i] == 3.0
            else:
                pitchoftarg[i] = chosentrial[targpos[0]]

            if chosentrial[targpos[0] - 1] == 8.0:
                pitchofprecur[i] == 3.0
            else:
                pitchofprecur[i] = chosentrial[targpos[0] - 1]
                # 1 is 191, 2 is 124, 3 is 144hz female, 5 is 251, 8 is 144hz male, 13 is109hz male
                # pitchof targ 1 is 124hz male, pitchoftarg4 is 109Hz Male

            if chosentrial[targpos[0] - 1]== 3.0:
                stepval[i] = 1.0
            elif chosentrial[targpos[0] - 1] == 8.0:
                stepval[i] = 2.0
            elif chosentrial[targpos[0] - 1] == 13.0:
                stepval[i] = 1.0
            elif chosentrial[targpos[0]-1] == 5.0:
                stepval[i] = 1.0
            else:
                stepval[i] = 0.0


            if pitchoftarg[i] == pitchofprecur[i]:
                precur_and_targ_same[i] = 1
            else:
                precur_and_targ_same[i] = 0


        except:
            indexdrop = newdata.iloc[i].name
            newdata.drop(indexdrop, axis=0, inplace=True)
            droplist = np.append(droplist, i)
            continue
    # indexdrop = newdata.iloc[i].name
    # newdata=newdata.drop(indexdrop, axis=0, inplace=True)
    droplist = [int(x) for x in droplist]  # drop corrupted metdata trials
    # newdata = newdata.drop(droplist, axis=0, inplace=True)

    pitchoftarg = pitchoftarg[~np.isnan(pitchoftarg)]
    pitchoftarg = pitchoftarg.astype(int)
    pitchofprecur = pitchofprecur[~np.isnan(pitchofprecur)]
    gradinpitch = gradinpitch[~np.isnan(gradinpitch)]
    # gradinpitchprecur = gradinpitchprecur[~np.isnan(gradinpitchprecur)]
    correctresp = correctresp[~np.isnan(correctresp)]
    #
    # pitchoftarg2 = np.empty(shape=(0, 0))
    # gradinpitch2 = np.empty(shape=(0, 0))
    #
    # pitchofprecur2 = np.empty(shape=(0, 0))
    # gradinpitchprecur2 = np.empty(shape=(0, 0))

    # for i in range(0, len(pitchofprecur)):
    #     if pitchofprecur[i] > 1:
    #         pitchofprecur2 = np.append(pitchofprecur2, pitchofprecur[i])
    #         gradinpitchprecur2 = np.append(gradinpitchprecur2, gradinpitchprecur[i])
    #
    # for i in range(0, len(pitchoftarg)):
    #     if pitchoftarg[i] > 1:
    #         pitchoftarg2 = np.append(pitchoftarg2, pitchoftarg[i])
    #         gradinpitch2 = np.append(gradinpitch2, gradinpitch[i])

    # pitchoftarg2 = pitchoftarg2 / np.linalg.norm(
    #     pitchoftarg2)  # np.array(scaler.fit_transform(pitchoftarg2.reshape(-1, 1)))
    # pitchofprecur2 = pitchofprecur2 / np.linalg.norm(
    #     pitchofprecur2)  # np.array(scaler.fit_transform(pitchofprecur2.reshape(-1, 1)))
    # gradinpitch2 = gradinpitch2 / np.linalg.norm(
    #     gradinpitch2)  # np.array(scaler.fit_transform(gradinpitch2.reshape(-1, 1)))
    # gradinpitchprecur2 = gradinpitchprecur2 / np.linalg.norm(
    #     gradinpitchprecur2)  # np.array(scaler.fit_transform(gradinpitchprecur2.reshape(-1, 1)))
    # pitchofprecur2 = np.array(scaler.fit_transform(pitchofprecur2.reshape(-1, 1)))
    # gradinpitch2 =  np.array(scaler.fit_transform(pitchofprecur2.reshape(-1, 1)))
    # gradinpitchprecur2 = np.array(scaler.fit_transform(gradinpitchprecur2.reshape(-1, 1)))

    # df_normalized = pd.DataFrame(x_scaled)
    pitchoftarg = np.delete(pitchoftarg, droplist)
    talkerlist2 = np.delete(talkerlist2, droplist)
    stepval = np.delete(stepval, droplist)

    newdata['pitchoftarg'] = pitchoftarg.tolist()

    # if droplist.size > 0:
    pitchofprecur = np.delete(pitchofprecur, droplist)
    newdata['pitchofprecur'] = pitchofprecur.tolist()

    correctresp = np.delete(correctresp, droplist)
    precur_and_targ_same = np.delete(precur_and_targ_same, droplist)
    correctresp = correctresp.astype(int)
    newdata['correctresp'] = correctresp.tolist()
    newdata['talker'] = talkerlist2.tolist()
    newdata['stepval'] = stepval.tolist()
    precur_and_targ_same = precur_and_targ_same.astype(int)
    newdata['precur_and_targ_same'] = precur_and_targ_same.tolist()
    newdata['timeToTarget'] = newdata['timeToTarget'] / 24414.0625
    newdata['AM'] = newdata['AM'].astype(int)
    newdata['talker'] = newdata['talker'] - 1
    # optionvector=[1 3 5];, male optionvector=[2 8 13]
    # only look at v2 pitches from recent experiments
    newdata = newdata[(newdata.pitchoftarg == 1) | (newdata.pitchoftarg == 2) | (newdata.pitchoftarg == 3) | (
            newdata.pitchoftarg == 5) | (newdata.pitchoftarg == 8) | (newdata.pitchoftarg == 13)]

    return newdata

    # ferretFigs = reactionTimeAnalysis(ferrData)
    # dataSet._save(figs=ferretFigs, file_name='reaction_times_{}_{}_{}.pdf'.format(ferret, startdate, finishdate))


# editing to extract different vars from df
# cli.add_command(cli_reaction_time)

if __name__ == '__main__':
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni']
    # for i, currFerr in enumerate(ferrets):
    #     print(i, currFerr)
    df = get_df_behav(ferrets=ferrets, includefaandmiss=False, startdate='04-01-2020', finishdate='01-10-2022')

    dfuse = df[["pitchoftarg", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                "timeToTarget", "DaysSinceStart", "AM",
                "realRelReleaseTimes", "ferret", "stepval", "pitchofprecur"]]
    endog2 = df['realRelReleaseTimes'].to_numpy()
    X = df[["pitchoftarg", "pitchofprecur", "talker", "side",
            "timeToTarget", "DaysSinceStart", "AM"]].to_numpy()

    y = endog2
    modelreg = Lmer(
        "realRelReleaseTimes ~ talker*(pitchoftarg)+ talker*(stepval)+ side + timeToTarget + DaysSinceStart + AM  + (1|ferret)",
        data=dfuse, family='gaussian')

    print(modelreg.fit(factors={"side": ["0", "1"], "stepval": ["0.0", "1.0",  "2.0"],"AM": ["0", "1"],
                                "pitchoftarg": ['1', '2', '3', '5', '13'], "talker": ["0.0", "1.0"], }, ordered=True,
                       REML=False,
                       old_optimizer=False))

    # looking at whether the response is correct or not
    dfcat = get_df_behav(ferrets=ferrets, includefaandmiss=True, startdate='04-01-2020', finishdate='01-10-2022')

    dfcat_use = dfcat[["pitchoftarg", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                       "timeToTarget", "DaysSinceStart", "AM",
                       "correctresp", "ferret", "stepval", "pitchofprecur"]]

    modelregcat = Lmer(
        "correctresp ~ talker*pitchoftarg +  side  + talker * stepval + stepval+timeToTarget + DaysSinceStart + AM + (1|ferret)",
        data=dfcat_use, family='binomial')

    print(modelregcat.fit(factors={"side": ["0", "1"], "stepval": ["0.0", "1.0",  "2.0"], "AM": ["0", "1"],
                                   "pitchoftarg": ['1', '2', '3', '5', '13'], "talker": ["0.0", "1.0"]}, REML=False,
                          old_optimizer=True))

    modelregcat_reduc = Lmer(
        "correctresp ~ talker*pitchoftarg  +  side + timeToTarget +talker*stepval+(1|ferret)",
        data=dfcat_use, family='binomial')

    print(modelregcat_reduc.fit(factors={"side": ["0", "1"],
                                         "pitchoftarg": ['1', '2', '3', '5', '13'], "talker": ["0.0", "1.0"], "stepval": ["0.0", "1.0",  "2.0"]},
                                REML=False,
                                old_optimizer=True))

    modelreg_reduc = Lmer(
        "realRelReleaseTimes ~ talker*(pitchoftarg)+side + talker*stepval+timeToTarget  + (1|ferret)",
        data=dfuse, family='gaussian')

    print(modelreg_reduc.fit(factors={"side": ["0", "1"],
                                      "pitchoftarg": ['1', '2', '3', '5', '13'], "talker": ["0.0", "1.0"],"stepval": ["0.0", "1.0",  "2.0"] },
                             ordered=True, REML=False,
                             old_optimizer=False))

    fig, ax = plt.subplots()

    ax = modelregcat.plot_summary()
    plt.title('Model Summary of Coefficients for P(Correct Responses)')
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = 'Intercept'
    labels[1] = 'targ - 124 Hz vs ref. 191 Hz'
    labels[2] = 'targ - 144 Hz vs ref. 191 Hz'
    labels[3] = 'targ - 251 Hz vs ref. 191 Hz'
    labels[4] = 'targ - 109 Hz vs ref. 191 Hz'
    labels[5] = 'side - left vs right'
    labels[6] = 'precur and targ same'
    labels[7] = 'time to target'
    labels[8] = 'days since start'
    labels[9] = 'AM'

    ax.set_yticklabels(labels)

    plt.show()
    fig, ax = plt.subplots()

    ax = modelreg.plot_summary()
    plt.title('Model Summary of Coefficients for Relative Release Times for Correct Responses')
    ax.set_yticklabels(labels)

    plt.show()
    # 1 is 191, 2 is 124, 3 is 144hz female, 5 is 251, 8 is 144hz male, 13 is109hz male
    # pitchof targ 1 is 124hz male, pitchoftarg4 is 109Hz Male

    fig, ax = plt.subplots()

    ax = modelregcat_reduc.plot_summary()
    plt.title('Model Summary of Coefficients for P(Correct Responses)')
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = 'Intercept'
    labels[1] = 'male talker vs ref. female talker'
    labels[2] = 'targ - 124 Hz vs ref. 191 Hz'
    labels[3] = 'targ - 144 Hz vs ref. 191 Hz'
    labels[4] = 'targ - 251 Hz vs ref. 191 Hz'
    labels[5] = 'side - right vs ref. left'
    labels[6] = 'time to target'
    labels[7] = 'pos. step from precursor to target'
    labels[8] = 'neg. step from precursor to target'
    labels[9] = 'targ pitch ~ step val'

   #ax.set_yticklabels(labels)
    plt.gca().get_yticklabels()[0].set_color("blue")
    plt.gca().get_yticklabels()[1].set_color("blue")
    plt.gca().get_yticklabels()[2].set_color("blue")
    plt.gca().get_yticklabels()[3].set_color("blue")
    plt.gca().get_yticklabels()[4].set_color("blue")
    plt.gca().get_yticklabels()[5].set_color("blue")
    plt.gca().get_yticklabels()[8].set_color("blue")
    plt.show()

    fig, ax = plt.subplots()

    ax = modelreg_reduc.plot_summary()

    plt.title('Model Summary of Coefficients for Correct Release Times')
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = 'Intercept'
    labels[1] = 'male talker vs ref. female talker'
    labels[2] = 'targ - 124 Hz vs ref. 191 Hz'
    labels[3] = 'targ - 144 Hz vs ref. 191 Hz'
    labels[4] = 'targ - 251 Hz vs ref. 191 Hz'
    labels[5] = 'side - right vs ref. left'
    labels[6] = 'pos. difference in F0 b/t precursor and talker'
    labels[7] = 'neg. difference in F0 b/t precursor and talker'
    labels[8] = 'time to target'
    labels[9] = 'talker corr. with stepval'

    #ax.set_yticklabels(labels, fontsize=10)
    plt.gca().get_yticklabels()[0].set_color("blue")
    plt.gca().get_yticklabels()[2].set_color("blue")
    plt.gca().get_yticklabels()[3].set_color("blue")
    plt.gca().get_yticklabels()[4].set_color("blue")
    plt.gca().get_yticklabels()[5].set_color("blue")
    plt.gca().get_yticklabels()[8].set_color("blue")

    plt.show()
    testmodel=pymer4.utils.pandas2R(modelreg_reduc.coefs)
    explainedvar = performance.r2_nakagawa(modelregcat_reduc.model_obj, by_group=False, tolerance=1e-05)
