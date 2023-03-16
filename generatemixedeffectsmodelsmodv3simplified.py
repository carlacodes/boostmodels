import sklearn.metrics
from rpy2.robjects import pandas2ri
import seaborn as sns
from instruments.io.BehaviourIO import BehaviourDataSet
from sklearn.inspection import permutation_importance
from instruments.behaviouralAnalysis import reactionTimeAnalysis  # outputbehaviordf
from sklearn.preprocessing import MinMaxScaler
from pymer4.models import Lmer
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import shap
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold

scaler = MinMaxScaler()
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import rpy2.robjects.numpy2ri
import sklearn
from sklearn.model_selection import train_test_split
from helpers.behaviouralhelpersformodels import *

rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr

# import R's "base" package
base = importr('base')
easystats = importr('easystats')
performance = importr('performance')
rstats = importr('stats')


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


def get_df_behav(path=None,
                 output=None,
                 ferrets=None,
                 includefaandmiss=False,
                 includemissonly=False,
                 startdate=None,
                 finishdate=None):
    if output is None:
        output = behaviourOutput

    if path is None:
        path = behaviouralDataPath

    allData, ferrets = extractAllFerretData(ferrets, path, startDate=startdate,
                                            finishDate=finishdate)
    fs = 24414.062500
    bigdata = pd.DataFrame()
    cosinesimfemale = np.load('D:/Stimuli/cosinesimvectorfemale.npy')
    cosinesimmale = np.load('D:/Stimuli/cosinesimvectormale.npy')
    numofferrets = allData['ferret'].unique()
    for ferret in numofferrets:
        newdata = allData[allData['ferret'] == ferret]
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
        precursorlist = newdata['distractors']
        catchtriallist = newdata['catchTrial']
        chosenresponse = newdata['response']
        realrelreleasetimelist = newdata['realRelReleaseTimes']
        pitchoftarg = np.empty(len(pitchshiftmat))
        pitchofprecur = np.empty(len(pitchshiftmat))
        stepval = np.empty(len(pitchshiftmat))

        precur_and_targ_same = np.empty(len(pitchshiftmat))
        talkerlist2 = np.empty(len(pitchshiftmat))

        correctresp = np.empty(shape=(0, 0))
        pastcorrectresp = np.empty(shape=(0, 0))
        pastcatchtrial = np.empty(shape=(0, 0))
        droplist = np.empty(shape=(0, 0))
        droplistnew = np.empty(shape=(0, 0))
        correspondcosinelist = np.empty(shape=(0, 0))

        for i in range(1, len(newdata['realRelReleaseTimes'].values)):
            chosenresponseindex = chosenresponse.values[i]
            pastcatchtrialindex = catchtriallist.values[i - 1]

            realrelreleasetime = realrelreleasetimelist.values[i]
            pastrealrelreleasetime = realrelreleasetimelist.values[i - 1]
            pastresponseindex = chosenresponse.values[(i - 1)]

            chosentrial = pitchshiftmat.values[i]
            is_all_zero = np.all((chosentrial == 0))
            if isinstance(chosentrial, float) or is_all_zero:
                chosentrial = talkermat.values[i].astype(int)

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

            targpos = np.where(chosendisttrial == 1)
            if ((
                        chosenresponseindex == 0 or chosenresponseindex == 1) and realrelreleasetime >= 0) or chosenresponseindex == 3:
                correctresp = np.append(correctresp, 1)
            else:
                correctresp = np.append(correctresp, 0)

            if ((
                        pastresponseindex == 0 or pastresponseindex == 1) and pastrealrelreleasetime >= 0) or pastresponseindex == 3:
                pastcorrectresp = np.append(pastcorrectresp, 1)
            else:
                pastcorrectresp = np.append(pastcorrectresp, 0)

            if pastcatchtrialindex == 1:
                pastcatchtrial = np.append(pastcatchtrial, 1)
            else:
                pastcatchtrial = np.append(pastcatchtrial, 0)

            try:
                if newdata['talker'].values[i] == 1:
                    correspondcosinelist = np.append(correspondcosinelist,
                                                     cosinesimfemale[int(chosendisttrial[targpos[0] - 1])])
                else:
                    correspondcosinelist = np.append(correspondcosinelist,
                                                     cosinesimmale[int(chosendisttrial[targpos[0] - 1])])
                if chosentrial[targpos[0]] == 8.0:
                    pitchoftarg[i] == 3.0
                else:
                    pitchoftarg[i] = chosentrial[targpos[0]]

                if chosentrial[targpos[0] - 1] == 8.0:
                    pitchofprecur[i] == 3
                else:
                    pitchofprecur[i] = chosentrial[targpos[0] - 1]
                    # 1 is 191, 2 is 124, 3 is 144hz female, 5 is 251, 8 is 144hz male, 13 is109hz male
                    # pitchof targ 1 is 124hz male, pitchoftarg4 is 109Hz Male

                if chosentrial[targpos[0] - 1] == 3.0:
                    stepval[i] = 1.0
                elif chosentrial[targpos[0] - 1] == 8.0:
                    stepval[i] = -1.0
                elif chosentrial[targpos[0] - 1] == 13.0:
                    stepval[i] = 1.0
                elif chosentrial[targpos[0] - 1] == 5.0:
                    stepval[i] = -1.0
                else:
                    stepval[i] = 0.0

                if pitchoftarg[i] == pitchofprecur[i]:
                    precur_and_targ_same[i] = 1
                else:
                    precur_and_targ_same[i] = 0
                if pitchofprecur[i] == 1.0:
                    pitchofprecur[i] = 4.0

                if pitchofprecur[i] == 13.0:
                    pitchofprecur[i] = 1.0

                if pitchoftarg[i] == 1.0:
                    pitchoftarg[i] = 4.0

                if pitchoftarg[i] == 13.0:
                    pitchoftarg[i] = 1.0


            except:
                indexdrop = newdata.iloc[i].name
                droplist = np.append(droplist, i - 1)
                # arrays START AT 0, but the index starts at 1, so the index is 1 less than the array
                droplistnew = np.append(droplistnew, indexdrop)
                continue

        newdata.drop(index=newdata.index[0],
                     axis=0,
                     inplace=True)
        newdata.drop(droplistnew, axis=0, inplace=True)
        droplist = [int(x) for x in droplist]  # drop corrupted metdata trials

        pitchoftarg = pitchoftarg.astype(int)
        pitchofprecur = pitchofprecur.astype(int)

        correctresp = correctresp[~np.isnan(correctresp)]
        correspondcosinelist = correspondcosinelist[~np.isnan(correspondcosinelist)]

        pitchoftarg = np.delete(pitchoftarg, 0)
        talkerlist2 = np.delete(talkerlist2, 0)
        stepval = np.delete(stepval, 0)
        pitchofprecur = np.delete(pitchofprecur, 0)
        precur_and_targ_same = np.delete(precur_and_targ_same, 0)

        pitchoftarg = np.delete(pitchoftarg, droplist)
        talkerlist2 = np.delete(talkerlist2, droplist)
        stepval = np.delete(stepval, droplist)

        newdata['pitchoftarg'] = pitchoftarg.tolist()
        pitchofprecur = np.delete(pitchofprecur, droplist)
        newdata['pitchofprecur'] = pitchofprecur.tolist()
        correctresp = np.delete(correctresp, droplist)
        pastcorrectresp = np.delete(pastcorrectresp, droplist)
        pastcatchtrial = np.delete(pastcatchtrial, droplist)
        precur_and_targ_same = np.delete(precur_and_targ_same, droplist)

        correctresp = correctresp.astype(int)
        pastcatchtrial = pastcatchtrial.astype(int)
        pastcorrectresp = pastcorrectresp.astype(int)

        newdata['correctresp'] = correctresp.tolist()
        newdata['pastcorrectresp'] = pastcorrectresp.tolist()
        newdata['talker'] = talkerlist2.tolist()
        newdata['pastcatchtrial'] = pastcatchtrial.tolist()
        newdata['stepval'] = stepval.tolist()
        newdata['realRelReleaseTimes'] = np.log(newdata['realRelReleaseTimes'])
        newdata['cosinesim'] = correspondcosinelist.tolist()
        precur_and_targ_same = precur_and_targ_same.astype(int)
        newdata['precur_and_targ_same'] = precur_and_targ_same.tolist()
        newdata['timeToTarget'] = newdata['timeToTarget'] / 24414.0625
        newdata['AM'] = newdata['AM'].astype(int)
        # make male talker lower values because it is lower in pitch
        #newdata['talker'] = newdata['talker'].replace({2: 0})

        # only look at v2 pitches from recent experiments
        newdata = newdata[(newdata.pitchoftarg == 1) | (newdata.pitchoftarg == 2) | (newdata.pitchoftarg == 3) | (
                newdata.pitchoftarg == 4) | (newdata.pitchoftarg == 5)]
        newdata = newdata[(newdata.pitchofprecur == 1) | (newdata.pitchofprecur == 2) | (newdata.pitchofprecur == 3) | (
                newdata.pitchofprecur == 4) | (newdata.pitchofprecur == 5)]

        newdata = newdata[(newdata.correctionTrial == 0)]  # | (allData.response == 7)
        newdata = newdata[(newdata.currAtten == 0)]  # | (allData.response == 7)

        if includefaandmiss is True:
            newdata = newdata[
                (newdata.response == 0) | (newdata.response == 1) | (newdata.response == 7) | (newdata.response == 5)]
        elif includemissonly is True:
            newdata = newdata[
                (newdata.response == 0) | (newdata.response == 1) | (newdata.response == 7) | (newdata.response == 3)]
        else:
            newdata = newdata[newdata.correctresp == 1]
            newdata = newdata[(newdata.catchTrial == 0)]
        bigdata = bigdata.append(newdata)
    return bigdata


# editing to extract different vars from df
def run_mixed_effects_analysis(ferrets):
    df = get_df_behav(ferrets=ferrets, includefaandmiss=False, startdate='04-01-2020', finishdate='09-03-2023')

    dfuse = df[["pitchoftarg", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                "timeToTarget", "DaysSinceStart", "AM",
                "realRelReleaseTimes", "ferret", "stepval", "pastcorrectresp", "pastcatchtrial", "trialNum"]]
    X = df[["pitchoftarg", "pitchofprecur", "talker", "side",
            "timeToTarget", "DaysSinceStart", "AM"]].to_numpy()

    modelreg = Lmer(
        "realRelReleaseTimes ~ talker*(pitchoftarg)+ talker*(stepval)+ side + timeToTarget + DaysSinceStart + AM  + (1|ferret)",
        data=dfuse)

    print(modelreg.fit(factors={"side": ["0", "1"], "stepval": ["0.0", "1.0", "-1.0"], "AM": ["0", "1"],
                                "pitchoftarg": ['1', '2', '3', '4', '5'], "talker": ["1.0", "2.0"], }, ordered=True,
                       REML=False,
                       old_optimizer=False))

    # looking at whether the response is correct or not
    dfcat = get_df_behav(ferrets=ferrets, includefaandmiss=True, startdate='04-01-2020', finishdate='01-10-2022')

    dfcat_use = dfcat[["pitchoftarg", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                       "targTimes", "DaysSinceStart", "AM", "timeToTarget",
                       "correctresp", "ferret", "stepval", "pastcorrectresp", "pastcatchtrial", "trialNum"]]

    modelregcat = Lmer(
        "correctresp ~ talker*pitchoftarg +  side  + talker * stepval + stepval+timeToTarget + DaysSinceStart + AM + (1|ferret)",
        data=dfcat_use, family='binomial')

    print(modelregcat.fit(factors={"side": ["0", "1"], "stepval": ["0.0", "-1.0", "1.0"], "AM": ["0", "1"],
                                   "pitchoftarg": ['1', '2', '3', '4', '5'], "talker": ["1.0", "2.0"]}, REML=False,
                          old_optimizer=True))

    modelregcat_reduc = Lmer(
        "correctresp ~ talker*pitchoftarg  +  side + timeToTarget +talker*stepval+(1|ferret)",
        data=dfcat_use, family='binomial')

    print(modelregcat_reduc.fit(factors={"side": ["0", "1"],
                                         "pitchoftarg": ['1', '2', '3', '4', '5'], "talker": ["1.0", "2.0"],
                                         "stepval": ["0.0", "-1.0", "1.0"]},
                                REML=False,
                                old_optimizer=True))

    modelreg_reduc = Lmer(
        "realRelReleaseTimes ~ talker*(pitchoftarg)+side + talker*stepval+timeToTarget  + (1|ferret)",
        data=dfuse)

    print(modelreg_reduc.fit(factors={"side": ["0", "1"],
                                      "pitchoftarg": ['1', '2', '3', '4', '5'], "talker": ["1.0", "2.0"],
                                      "stepval": ["0.0", "-1.0", "1.0"]},
                             ordered=True, REML=False,
                             old_optimizer=False))

    fig, ax = plt.subplots()

    ax = modelregcat.plot_summary()
    plt.title('Model Summary of Coefficients for P(Correct Responses)')
    labels = [item.get_text() for item in ax.get_yticklabels()]

    plt.show()

    ax = modelreg.plot_summary()
    plt.title('Model Summary of Coefficients for Relative Release Times for Correct Responses')
    ax.set_yticklabels(labels)
    plt.show()
    # 1 is 191, 2 is 124, 3 is 144hz female, 5 is 251, 8 is 144hz male, 13 is109hz male
    # pitchof targ 1 is 124hz male, pitchoftarg4 is 109Hz Male

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

    ax.set_yticklabels(labels)
    plt.gca().get_yticklabels()[0].set_color("blue")
    plt.gca().get_yticklabels()[1].set_color("blue")
    plt.gca().get_yticklabels()[2].set_color("blue")
    plt.gca().get_yticklabels()[3].set_color("blue")
    plt.gca().get_yticklabels()[4].set_color("blue")
    plt.gca().get_yticklabels()[5].set_color("blue")
    plt.gca().get_yticklabels()[8].set_color("blue")
    plt.show()

    ax = modelreg_reduc.plot_summary()

    plt.title('Model Summary of Coefficients for Correct Release Times')
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = 'Intercept'
    labels[1] = 'male talker vs ref. female talker'
    labels[2] = 'targ - 124 Hz vs ref. 191 Hz'
    labels[3] = 'targ - 144 Hz vs ref. 191 Hz'
    labels[4] = 'targ - 251 Hz vs ref. 191 Hz'
    labels[5] = 'side - right vs ref. left'
    labels[6] = 'pos. step in F0 \n b/t precursor and talker'
    labels[7] = 'neg. step in F0 \n b/t precursor and talker'
    labels[8] = 'time to target'
    labels[9] = 'talker corr. with stepval'
    ax.set_yticklabels(labels, fontsize=10)

    plt.gca().get_yticklabels()[0].set_color("blue")
    plt.gca().get_yticklabels()[2].set_color("blue")
    plt.gca().get_yticklabels()[3].set_color("blue")
    plt.gca().get_yticklabels()[4].set_color("blue")
    plt.gca().get_yticklabels()[5].set_color("blue")
    plt.gca().get_yticklabels()[8].set_color("blue")

    plt.show()
    explainedvar = performance.r2_nakagawa(modelregcat_reduc.model_obj, by_group=False, tolerance=1e-05)
    explainedvar_nagelkerke = performance.r2(modelregcat_reduc.model_obj)
    explainvarreleasetime = performance.r2_nakagawa(modelreg_reduc.model_obj, by_group=False, tolerance=1e-05)
    print(explainedvar)
    # side of audio likely adds behavioural noise to the data
    print(explainedvar_nagelkerke)
    ##the marginal R2 encompassing variance explained by only the fixed effects, and the conditional R2 comprising variance explained by both
    # fixed and random effects i.e. the variance explained by the whole model
    print(explainvarreleasetime)

    data_from_ferret = dfuse[(dfuse['pitchoftarg'] == 1) | (dfuse['pitchoftarg'] == 13)]
    data_from_ferret.isnull().values.any()
    predictedrelease = rstats.predict(modelreg_reduc.model_obj, type='response')
    # with localconverter(ro.default_converter + pandas2ri.converter):
    #     r_from_pd_df = ro.conversion.py2rpy(data_from_ferret)
    # predictedrelease_cruella=rstats.predict(modelreg_reduc.model_obj,  newdata=r_from_pd_df)
    predictedcorrectresp = rstats.predict(modelregcat_reduc.model_obj, type='response')
    # write all applicable dataframes to csv files
    p = 'D:/dfformixedmodels/'
    os.path.normpath(p)
    from pathlib import Path
    filepath = Path('D:/dfformixedmodels/dfuse.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    dfuse.to_csv(filepath)

    filepath = Path('D:/dfformixedmodels/dfcat.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    dfcat.to_csv(filepath)

    filepath = Path('D:/dfformixedmodels/df.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

    filepath = Path('D:/dfformixedmodels/dfcat_use.csv')
    dfcat_use.to_csv(filepath)
    return modelreg_reduc, modelregcat_reduc, modelregcat, modelreg, predictedrelease, dfuse, dfcat_use, predictedcorrectresp, explainedvar, explainvarreleasetime


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
                                                        random_state=123)
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

    kfold = KFold(n_splits=3, shuffle=True, random_state=123)
    results = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)

    mse = mean_squared_error(ypred, y_test)
    print("MSE: %.2f" % (mse))
    print("negative MSE: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(X_train)
    shap.summary_plot(shap_values, X_train)
    plt.show()
    return xg_reg, ypred, y_test, results

def extract_release_times_data(ferrets):
    df = get_df_behav(ferrets=ferrets, includefaandmiss=False, startdate='04-01-2020', finishdate='09-03-2023')

    dfuse = df[["pitchoftarg", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                "timeToTarget", "DaysSinceStart", "AM",
                "realRelReleaseTimes", "ferret", "stepval", "pastcorrectresp", "pastcatchtrial", "trialNum"]]
    return dfuse
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
    shap_values = shap.TreeExplainer(xg_reg).shap_values(dfx)
    fig, ax = plt.subplots(figsize=(15, 15))
    # title kwargs still does nothing so need this workaround for summary plots
    shap.summary_plot(shap_values, dfx, show=False)
    fig, ax = plt.gcf(), plt.gca()
    plt.title('Ranked list of features over their impact in predicting reaction time')
    plt.xlabel('SHAP value (impact on model output) on reaction time')

    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
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
    plt.savefig('figs/shapsummaryplot_allanimals.png')

    plt.show()

    shap.dependence_plot("timeToTarget", shap_values, dfx)  #

    explainer = shap.Explainer(xg_reg, dfx)
    shap_values2 = explainer(dfx)
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
                       title='Correct Responses - Reaction Time Model SHAP response \n vs. trial number')
    plt.show()

    return xg_reg, ypred, y_test, results


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

    xg_reg = lgb.LGBMRegressor(random_state=123, verbose=1)
    xg_reg.fit(X_train, y_train, verbose=1)
    ypred = xg_reg.predict(X_test)
    lgb.plot_importance(xg_reg)
    plt.title('feature importances for the LGBM Correct Release Times model for ferret ' + ferret_name)
    plt.show()

    kfold = KFold(n_splits=10)
    results = cross_val_score(xg_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
    mse_train = mean_squared_error(ypred, y_test)

    mse = mean_squared_error(ypred, y_test)
    print("MSE on test: %.4f" % (mse) + ferret_name)
    print("negative MSE training: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    shap_values = shap.TreeExplainer(xg_reg).shap_values(dfx)
    fig, ax = plt.subplots(figsize=(15, 15))
    # title kwargs still does nothing so need this workaround for summary plots
    shap.summary_plot(shap_values, dfx, show=False)
    fig, ax = plt.gcf(), plt.gca()
    plt.title('Ranked list of features over their impact in predicting reaction time for' + ferret_name)
    plt.xlabel('SHAP value (impact on model output) on reaction time' + ferret_name)

    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
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
    plt.show()

    return xg_reg, ypred, y_test, results, mse


def balanced_subsample(x, y, subsample_size=1.0):
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems * subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            this_xs = this_xs.reindex(np.random.permutation(this_xs.index))
        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = pd.concat(xs)
    ys = pd.Series(data=np.concatenate(ys), name='target')
    return xs, ys


def runlgbcorrectresponse(dfx, dfy, paramsinput=None):
    # params input from optuna study run on 06/03/2023
    paramsinput = {'colsample_bytree': 0.4253436090928764, 'alpha': 18.500902749816458, 'is_unbalanced': True,
                   'n_estimators': 8700, 'learning_rate': 0.45629236152754304, 'num_leaves': 670, 'max_depth': 3,
                   'min_data_in_leaf': 200, 'lambda_l1': 4, 'lambda_l2': 64, 'min_gain_to_split': 0.016768866636887758,
                   'bagging_fraction': 0.9, 'bagging_freq': 3, 'feature_fraction': 0.2}

    X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=123)
    print(X_train.shape)
    print(X_test.shape)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test)
    params2 = {"n_estimators": 9300,
               "scale_pos_weight": 0.3,
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
                                **paramsinput)  # colsample_bytree=0.4398528259745191, alpha=14.412788226345182,

    xg_reg.fit(X_train, y_train, eval_metric="cross_entropy_lambda", verbose=1000)
    ypred = xg_reg.predict_proba(X_test)
    # lgb.plot_importance(xg_reg)
    # plt.show()

    kfold = KFold(n_splits=10, shuffle=True, random_state=123)
    results = cross_val_score(xg_reg, X_test, y_test, scoring='accuracy', cv=kfold)
    bal_accuracy = cross_val_score(xg_reg, X_test, y_test, scoring='balanced_accuracy', cv=kfold)
    print("Accuracy: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    print('Balanced Accuracy: %.2f%%' % (np.mean(bal_accuracy) * 100.0))

    shap_values = shap.TreeExplainer(xg_reg).shap_values(dfx)
    shap.summary_plot(shap_values, dfx)
    plt.show()

    shap.dependence_plot("pitchoftarg", shap_values[0], dfx)  #
    plt.show()
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=10,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T)
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()

    explainer = shap.Explainer(xg_reg, X_train)

    shap_values2 = explainer(X_train)
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "precur_and_targ_same"])

    fig.tight_layout()
    plt.tight_layout()
    plt.subplots_adjust(left=-10, right=0.5)
    plt.show()

    shap.plots.scatter(shap_values2[:, "pitchoftarg"], color=shap_values2[:, "talker"])
    plt.show()

    shap.plots.scatter(shap_values2[:, "pitchoftarg"], color=shap_values2[:, "precur_and_targ_same"])
    plt.show()

    shap.plots.scatter(shap_values2[:, "precur_and_targ_same"], color=shap_values2[:, "talker"])
    plt.show()

    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "talker"])
    plt.show()

    return xg_reg, ypred, y_test, results, shap_values, X_train, y_train, bal_accuracy


def objective_releasetimes(trial, X, y):
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    #     colsample_bytree = 0.3, learning_rate = 0.1,
    # max_depth = 10, alpha = 10, n_estimators = 10, random_state = 42, verbose = 1
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.6),
        "alpha": trial.suggest_float("alpha", 5, 20),
        "n_estimators": trial.suggest_int("n_estimators", 2, 100, step=2),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
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


def run_optuna_study_releasetimes(X, y):
    study = optuna.create_study(direction="minimize", study_name="LGBM regressor")
    func = lambda trial: objective_releasetimes(trial, X, y)
    study.optimize(func, n_trials=1000)
    print("Number of finished trials: ", len(study.trials))
    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")
    return study


def run_optuna_study_falsealarm(dataframe, y):
    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    df_to_use = dataframe[
        ["cosinesim", "pitchofprecur", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
         "falsealarm", "pastcorrectresp", "temporalsim", "pastcatchtrial", "trialNum", "targTimes", ]]

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
        ["cosinesim", "pitchofprecur", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
         "falsealarm", "pastcorrectresp", "temporalsim", "pastcatchtrial", "trialNum", "targTimes", ]]

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


def runlgbcorrectrespornotwithoptuna(dataframe, paramsinput=None):
    if paramsinput is None:
        paramsinput = {'colsample_bytree': 0.4253436090928764, 'alpha': 18.500902749816458, 'is_unbalanced': True,
                       'n_estimators': 8700, 'learning_rate': 0.45629236152754304, 'num_leaves': 670, 'max_depth': 3,
                       'min_data_in_leaf': 200, 'lambda_l1': 4, 'lambda_l2': 64,
                       'min_gain_to_split': 0.016768866636887758, 'bagging_fraction': 0.9, 'bagging_freq': 3,
                       'feature_fraction': 0.2}

    df_to_use = dataframe[["pitchoftarg", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                           "targTimes", "DaysSinceStart", "AM", "cosinesim", "stepval", "pastcorrectresp",
                           "pastcatchtrial", "trialNum", "correctresp"]]

    col = 'correctresp'
    dfx = df_to_use.loc[:, df_to_use.columns != col]

    X_train, X_test, y_train, y_test = train_test_split(dfx, df_to_use['correctresp'], test_size=0.2, random_state=123)
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
    fig, ax = plt.subplots(figsize=(10, 65))
    shap.summary_plot(shap_values1, X_train, show=False)
    plt.title('Ranked list of features over their \n impact in predicting a correct response', fontsize=18)
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/correctrespmodel/correctresponsemodelrankedfeatures.png', dpi=500)

    plt.show()
    shap.dependence_plot("pitchofprecur", shap_values1[0], X_train)  #
    plt.show()
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=10,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T)
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/permutation_importance.png', dpi=500)
    plt.show()
    explainer = shap.Explainer(xg_reg, X_train)
    shap_values2 = explainer(X_train)
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "precur_and_targ_same"], ax=ax)
    plt.tight_layout()
    plt.subplots_adjust(left=-10, right=0.5)

    plt.show()
    shap.plots.scatter(shap_values2[:, "pitchofprecur"], color=shap_values2[:, "talker"])
    plt.show()

    shap.plots.scatter(shap_values2[:, "pitchofprecur"], color=shap_values2[:, "precur_and_targ_same"], show=False)
    plt.show()

    shap.plots.scatter(shap_values2[:, "precur_and_targ_same"], color=shap_values2[:, "talker"])
    plt.show()
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "talker"], show=False)
    plt.title('trial number \n vs. SHAP value impact')
    plt.ylabel('SHAP value', fontsize=18)
    plt.savefig('D:/behavmodelfigs/correctrespmodel/trialnumdepenencyplot.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "precur_and_targ_same"], show=False)
    plt.title('Cosine similarity \n vs. SHAP value impact')
    plt.ylabel('SHAP value', fontsize=18)
    plt.savefig('D:/behavmodelfigs/correctrespmodel/cosinesimdepenencyplot.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "precur_and_targ_same"], color=shap_values2[:, "cosinesim"], show=False)

    plt.title('Intra trial roving \n versus SHAP value impact', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.savefig('D:/behavmodelfigs/correctrespmodel/intratrialrovingcosinecolor.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "targTimes"], show=False)
    plt.title('CR model - Trial number versus SHAP value, \n colored by target presentation time', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xlabel('Trial number', fontsize=15)
    plt.savefig('D:/behavmodelfigs/correctrespmodel/trialnumtargtimecolor.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "targTimes"], color=shap_values2[:, "trialNum"], show=False)
    plt.title('CR model - Target times versus SHAP value, \n colored by trial number', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xlabel('Target presentation time', fontsize=15)
    plt.savefig('D:/behavmodelfigs/correctrespmodel/targtimestrialnumcolor.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "targTimes"], show=False)
    plt.title('Cosine Similarity as a function \n of SHAP values coloured by targTimes')
    plt.savefig('D:/behavmodelfigs/correctrespmodel/cosinesimtargtimes.png', dpi=500)
    plt.show()
    np.save('D:/behavmodelfigs/correctrespponseoptunaparams4_strat5kfold.npy', paramsinput)

    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "pitchoftarg"], show=False)
    plt.title('Cosine similarity as a function \n of SHAP values, coloured by the target pitch', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.savefig('D:/behavmodelfigs/correctrespmodel/cosinesimcolouredtalkers.png', dpi=500)
    plt.show()
    fig, ax = plt.subplots(figsize=(15, 35))
    shap.plots.scatter(shap_values2[:, "side"], color=shap_values2[:, "trialNum"], show=False)
    plt.title('SHAP values as a function of the side of the audio, \n coloured by the trial number', fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xticks([0, 1], ['Left', 'Right'], fontsize=18)
    plt.savefig('D:/behavmodelfigs/correctrespmodel/sidetrialnumbercolor.png', dpi=500)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 55))
    shap.plots.scatter(shap_values2[:, "pitchoftarg"], color=shap_values2[:, "targTimes"], show=False)
    plt.title('SHAP values as a function of the pitch of the target, \n coloured by the target presentation time',
              fontsize=18)
    plt.ylabel('SHAP value', fontsize=18)
    plt.xlabel('Pitch of target', fontsize=12)
    plt.xticks([1, 2, 3, 4, 5], ['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz'], fontsize=18)
    plt.savefig('D:/behavmodelfigs/correctrespmodel/pitchoftargcolouredbytargtimes.png', dpi=500)
    plt.show()

    return xg_reg, ypred, y_test, results, shap_values1, X_train, y_train, bal_accuracy, shap_values2


def runlgbfaornotwithoptuna(dataframe, paramsinput):
    df_to_use = dataframe[
        ["cosinesim", "pitchofprecur", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
         "falsealarm", "temporalsim", "pastcorrectresp", "pastcatchtrial", "trialNum", "targTimes", ]]

    col = 'falsealarm'
    dfx = df_to_use.loc[:, df_to_use.columns != col]
    # remove ferret as possible feature
    X_train, X_test, y_train, y_test = train_test_split(dfx, df_to_use['falsealarm'], test_size=0.2, random_state=123)
    print(X_train.shape)
    print(X_test.shape)
    # ran optuna study 06/03/2022 to find best params, balanced accuracy 57%, accuracy 63%
    paramsinput = {'colsample_bytree': 0.7024634011442671, 'alpha': 15.7349076305505, 'is_unbalanced': True,
                   'n_estimators': 6900, 'learning_rate': 0.3579458041084967, 'num_leaves': 1790, 'max_depth': 4,
                   'min_data_in_leaf': 200, 'lambda_l1': 0, 'lambda_l2': 24, 'min_gain_to_split': 2.34923345270416,
                   'bagging_fraction': 0.9, 'bagging_freq': 12, 'feature_fraction': 0.9}

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
    fig, ax = plt.subplots(figsize=(15, 65))
    shap.summary_plot(shap_values1, X_train, show=False)
    plt.title('Ranked list of features over their \n impact in predicting a false alarm', fontsize=18)
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/ranked_features_falsealarmmodel.png', dpi=500)
    plt.show()

    shap.dependence_plot("pitchofprecur", shap_values1[0], X_train)  #
    plt.show()
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=10,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T)
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/permutation_importance.png', dpi=500)
    plt.show()
    explainer = shap.Explainer(xg_reg, dfx)
    shap_values2 = explainer(X_train)
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "intra_trial_roving"])
    fig.tight_layout()
    plt.tight_layout()
    plt.subplots_adjust(left=-10, right=0.5)

    plt.show()
    shap.plots.scatter(shap_values2[:, "pitchofprecur"], color=shap_values2[:, "talker"])
    plt.show()

    shap.plots.scatter(shap_values2[:, "pitchofprecur"], color=shap_values2[:, "intra_trial_roving"], show=False)
    plt.show()

    shap.plots.scatter(shap_values2[:, "intra_trial_roving"], color=shap_values2[:, "talker"])
    plt.show()
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "talker"])
    plt.show()

    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "intra_trial_roving"], show=False)
    plt.title('False alarm model - Cosine similarity vs. SHAP value impact')
    plt.ylabel('SHAP value', fontsize=10)
    plt.savefig('D:/behavmodelfigs/cosinesimdepenencyplot.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "intra_trial_roving"], color=shap_values2[:, "cosinesim"], show=False)
    plt.title('False alarm model - Intra trial roving versus SHAP value impact')
    plt.ylabel('SHAP value', fontsize=10)
    plt.savefig('D:/behavmodelfigs/intratrialrovingcosinecolor.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "cosinesim"], show=False)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('False alarm model - Trial number versus SHAP value impact')
    plt.savefig('D:/behavmodelfigs/trialnumcosinecolor.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "pitchofprecur"], color=shap_values2[:, "targTimes"], show=False)
    plt.ylabel('SHAP value', fontsize=10)
    plt.title('False alarm model - pitch of the \n precursor word versus SHAP value impact', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Pitch of precursor word', fontsize=16)
    plt.savefig('D:/behavmodelfigs/pitchofprecurtargtimes.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "targTimes"], color=shap_values2[:, "cosinesim"], show=False)
    plt.title('False alarm model - Target Times coloured by Cosine Similarity vs Their Impact on the SHAP value')
    plt.ylabel('SHAP value', fontsize=10)
    plt.savefig('D:/behavmodelfigs/targtimescosinecolor.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "targTimes"], color=shap_values2[:, "trialNum"], show=False)
    plt.title('False alarm model - Target times \n coloured by trial number vs their SHAP value', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Target times', fontsize=16)
    plt.savefig('D:/behavmodelfigs/targtimestrialnum.png', dpi=500)
    plt.show()
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "targTimes"], show=False)
    plt.title('False alarm model - Trial number \n coloured by target times vs their SHAP value', fontsize=18)
    plt.ylabel('SHAP value', fontsize=16)
    plt.xlabel('Trial number', fontsize=16)
    plt.savefig('D:/behavmodelfigs/trialnumbercolortargtimes.png', dpi=500)
    plt.show()
    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "targTimes"], show=False)
    plt.title('Cosine Similarity as a function of SHAP values, coloured by targTimes')
    plt.ylabel('SHAP value', fontsize=10)
    plt.savefig('D:/behavmodelfigs/cosinesimtargtimes.png', dpi=500)
    plt.show()
    np.save('D:/behavmodelfigs/falsealarmoptunaparams_improveddf3_strat5kfold.npy', paramsinput)
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "talker"], show=False)
    plt.title('False alarm model - cosine similarity  \n as a function of SHAP values, coloured by talker')
    plt.ylabel('SHAP value corresponding to cosine sim.', fontsize=10)
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/cosinesimcolouredtalkers.png', dpi=500)
    plt.show()

    return xg_reg, ypred, y_test, results, shap_values1, X_train, y_train, bal_accuracy, shap_values2


def runlgbfaornot(dataframe):
    df_to_use = dataframe[
        ["cosinesim", "pitchofprecur", "talker", "side", "intra_trial_roving", "DaysSinceStart", "AM",
         "falsealarm", "pastcorrectresp", "pastcatchtrial", "trialNum", "targTimes", ]]

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

    kfold = KFold(n_splits=3, shuffle=True, random_state=123)
    results = cross_val_score(xg_reg, X_test, y_test, scoring='accuracy', cv=kfold)
    bal_accuracy = cross_val_score(xg_reg, X_test, y_test, scoring='balanced_accuracy', cv=kfold)
    print("Accuracy: %.2f%%" % (np.mean(results) * 100.0))
    print(results)
    print('Balanced Accuracy: %.2f%%' % (np.mean(bal_accuracy) * 100.0))

    shap_values1 = shap.TreeExplainer(xg_reg).shap_values(dfx)
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.summary_plot(shap_values1, dfx, show=False)
    plt.title('Ranked list of features over their impact in predicting a false alarm')
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/ranked_features.png', dpi=500)

    plt.show()
    shap.dependence_plot("pitchofprecur", shap_values1[0], dfx)  #
    plt.show()
    result = permutation_importance(xg_reg, X_test, y_test, n_repeats=10,
                                    random_state=123, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.barh(X_test.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T)
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/permutation_importance.png', dpi=500)
    plt.show()
    explainer = shap.Explainer(xg_reg, dfx)
    shap_values2 = explainer(X_train)
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "talker"], color=shap_values2[:, "intra_trial_roving"])
    fig.tight_layout()
    plt.tight_layout()
    plt.subplots_adjust(left=-10, right=0.5)

    plt.show()
    shap.plots.scatter(shap_values2[:, "pitchofprecur"], color=shap_values2[:, "talker"])
    plt.show()

    shap.plots.scatter(shap_values2[:, "pitchofprecur"], color=shap_values2[:, "intra_trial_roving"], show=False)
    plt.show()

    shap.plots.scatter(shap_values2[:, "intra_trial_roving"], color=shap_values2[:, "talker"])
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "talker"], show=False)
    plt.title('False alarm model - trial number as a function of SHAP values, coloured by talker')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "intra_trial_roving"], show=False)
    plt.title('False alarm model - SHAP values as a function of cosine similarity \n, coloured by intra trial roving')
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/cosinesimdepenencyplot.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "intra_trial_roving"], color=shap_values2[:, "cosinesim"], show=False)
    plt.savefig('D:/behavmodelfigs/intratrialrovingcosinecolor.png', dpi=500)

    plt.show()
    fig, ax = plt.subplots(figsize=(15, 15))
    shap.plots.scatter(shap_values2[:, "trialNum"], color=shap_values2[:, "pitchoftarg"], show=False)
    plt.title('False alarm model - trial number as a function of SHAP values, coloured by pitch of target')
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/trialnumcosinecolor.png', dpi=500)
    plt.show()

    fig, ax = plt.subplots(figsize=(18, 15))
    shap.plots.scatter(shap_values2[:, "targTimes"], color=shap_values2[:, "cosinesim"], show=False)
    plt.title('shap values for FA model as a function of target times, coloured by cosine similarity')
    fig.tight_layout()
    plt.savefig('D:/behavmodelfigs/targtimescosinecolor.png', dpi=500)
    plt.show()

    shap.plots.scatter(shap_values2[:, "cosinesim"], color=shap_values2[:, "targTimes"], show=False)
    plt.title('Cosine Similarity as a function of SHAP values, coloured by targTimes')
    plt.savefig('D:/behavmodelfigs/cosinesimtargtimes.png', dpi=500)
    plt.show()

    return xg_reg, ypred, y_test, results, shap_values1, X_train, y_train, bal_accuracy, shap_values2


def runfalsealarmpipeline(ferrets):
    resultingfa_df = behaviouralhelperscg.get_false_alarm_behavdata(ferrets=ferrets, startdate='04-01-2020',
                                                                    finishdate='01-03-2023')
    len_of_data_male = {}
    len_of_data_female = {}
    len_of_data_female_intra = {}
    len_of_data_female_inter = {}
    len_of_data_male_intra = {}
    len_of_data_male_inter = {}
    for i in range(0, len(ferrets)):
        noncorrectiondata = resultingfa_df[resultingfa_df['correctionTrial'] == 0]
        noncorrectiondata = noncorrectiondata[noncorrectiondata['currAtten'] == 0]


        len_of_data_male[ferrets[i]] = len(noncorrectiondata[(noncorrectiondata['ferret'] == i) & (noncorrectiondata['talker'] == 2.0) & (noncorrectiondata['control_trial'] == 1)])
        len_of_data_female[ferrets[i]] = len(noncorrectiondata[(noncorrectiondata['ferret'] == i) & (noncorrectiondata['talker'] == 1.0) & (noncorrectiondata['control_trial'] == 1)])

        interdata = noncorrectiondata[noncorrectiondata['inter_trial_roving'] == 1]
        intradata = noncorrectiondata[noncorrectiondata['intra_trial_roving'] == 1]
        len_of_data_female_intra[ferrets[i]] = len(intradata[(intradata['ferret'] == i) & (intradata['talker'] == 1.0)])
        len_of_data_female_inter[ferrets[i]] = len(interdata[(interdata['ferret'] == i) & (interdata['talker'] == 1.0)])

        len_of_data_male_inter[ferrets[i]] = len(interdata[(interdata['ferret'] == i) & (interdata['talker'] == 2.0)])
        len_of_data_male_intra[ferrets[i]] = len(intradata[(intradata['ferret'] == i ) & (intradata['talker'] == 2.0)])


    filepath = Path('D:/dfformixedmodels/falsealarmmodel_dfuse.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    resultingfa_df.to_csv(filepath)
    study = run_optuna_study_falsealarm(resultingfa_df, resultingfa_df['falsealarm'].to_numpy())
    print(study.best_params)
    # best_params = np.load('D:/behavmodelfigs/falsealarmoptunaparams3_strat5kfold.npy', allow_pickle=True).item()

    xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runlgbfaornotwithoptuna(
        resultingfa_df, study.best_params)
    # np.save('D:/behavmodelfigs/falsealarmoptunaparams2.npy', study.best_params)

    return xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2


def run_correct_responsepipeline(ferrets):
    resultingcr_df = get_df_behav(ferrets=ferrets, includefaandmiss=False, includemissonly=True, startdate='04-01-2020',
                                  finishdate='01-10-2022')
    filepath = Path('D:/dfformixedmodels/correctresponsemodel_dfuse.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    resultingcr_df.to_csv(filepath)
    # old BEST PARAMS without clove's data
    best_params = np.load('D:/behavmodelfigs/correctrespponseoptunaparams4_strat5kfold.npy', allow_pickle=True).item()
    xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runlgbcorrectrespornotwithoptuna(
        resultingcr_df)
    return xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2


def run_reaction_time_fa_pipleine_female(ferrets):
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    df_use = resultingdf.loc[:, resultingdf.columns != 'ferret']
    df_use = df_use.loc[df_use['intra_trial_roving'] == 0]
    df_use = df_use.loc[df_use['talker'] == 1]
    df_use = df_use.loc[:, df_use.columns != 'targTimes']
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
    plt.savefig('D:/behavmodelfigs/ranked_features_rxntimealarmhitmodel.png', dpi=500)
    plt.show()

    return resultingdf


def run_reaction_time_fa_pipleine_male(ferrets):
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    df_use = resultingdf.loc[:, resultingdf.columns != 'ferret']
    df_use = df_use.loc[df_use['intra_trial_roving'] == 0]
    df_use = df_use.loc[df_use['talker'] == 2]
    df_use = df_use.loc[:, df_use.columns != 'targTimes']
    df_use = df_use.loc[:, df_use.columns != 'stepval']
    df_use = df_use.loc[:, df_use.columns != 'side']
    df_use = df_use.loc[:, df_use.columns != 'AM']

    df_use = df_use.loc[:, df_use.columns != 'distractor_or_fa']
    df_use = df_use.loc[:, df_use.columns != 'realRelReleaseTimes']

    col = 'centreRelease'
    dfx = df_use.loc[:, df_use.columns != col]
    col = 'ferret'
    col2 = ['target', 'startResponseTime', 'distractors', 'recBlock', 'lickRelease2', 'lickReleaseCount',
            'PitchShiftMat', 'attenOrder', 'dDurs', 'tempAttens', 'currAttenList', 'attenList', 'fName', 'Level',
            'dates', 'ferretname', 'noiseType', 'noiseFile']
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
    plt.savefig('D:/behavmodelfigs/ranked_features_rxntimealarmhitmodel_male.png', dpi=500)
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
    plt.savefig('D:/behavmodelfigs/proportion_correct_responses_by_side.png', dpi=500)

    plt.show()
    df_left_by_ferret = {}
    df_right_by_ferret = {}

    # now plot by ferret ID
    ferrets = [0, 1, 2, 3]
    for ferret in ferrets:
        df_left_test = df_use.loc[df_use['side'] == 0]
        df_right_test = df_use.loc[df_use['side'] == 1]
        df_left_by_ferret[ferret] = df_left_test.loc[df_left_test['ferret'] == ferret]
        df_right_by_ferret[ferret] = df_right_test.loc[df_right_test['ferret'] == ferret]

    ax, fig = plt.subplots(figsize=(10, 12))
    plt.bar(['left - zola', 'right - zola', 'left - cru', 'right - cru', 'left - tina', 'right-tina', 'left - mac',
             'right-mac'], [df_left_by_ferret[0]['correct'].mean(), df_right_by_ferret[0]['correct'].mean(),
                            df_left_by_ferret[1]['correct'].mean(), df_right_by_ferret[1]['correct'].mean(),
                            df_left_by_ferret[2]['correct'].mean(), df_right_by_ferret[2]['correct'].mean(),
                            df_left_by_ferret[3]['correct'].mean(), df_right_by_ferret[3]['correct'].mean()])
    plt.title('Proportion of correct responses by side registered by sensors, \n  irrespective of talker, by ferret ID',
              fontsize=15)
    plt.xticks(rotation=45, fontsize=12)  # rotate the x axis labels
    plt.ylim(0, 1)

    plt.ylabel('proportion of correct responses', fontsize=13)
    plt.savefig('D:/behavmodelfigs/proportion_correct_responses_by_side_by_ferret.png', dpi=500)
    plt.show()
    return df_left, df_right


def plot_reaction_times(ferrets):
    # plot the reaction times by animal
    resultingdf = behaviouralhelperscg.get_reactiontime_data(ferrets=ferrets, startdate='04-01-2020',
                                                             finishdate='01-10-2022')
    df_use = resultingdf

    df_left_by_ferret = {}
    df_female = df_use.loc[df_use['talker'] == 1]
    df_female_control = df_female.loc[df_female['intra_trial_roving'] == 0]

    df_female = df_use.loc[df_use['talker'] == 1]
    df_female_rove = df_female.loc[df_female['intra_trial_roving'] == 1]

    df_male = df_use.loc[df_use['talker'] == 2]
    df_male_control = df_male.loc[df_male['intra_trial_roving'] == 0]
    df_male_rove = df_male.loc[df_male['intra_trial_roving'] == 1]

    # now plot generally by all ferrets
    ax, fig = plt.subplots(figsize=(10, 12))
    sns.distplot(df_use['realRelReleaseTimes'], hist=True, kde=False, color='blue',
                 hist_kws={'edgecolor': 'black'})
    plt.title('Distribution of reaction times, \n irrespective of talker and ferret', fontsize=15)
    plt.show()

    # now plot by talker, showing reaction times
    sns.distplot(df_female_control['realRelReleaseTimes'], color='blue', label='control F0')
    sns.distplot(df_female_rove['realRelReleaseTimes'], color='red', label='roved F0')
    plt.title('Reaction times for the female talker, \n irrespective of ferret', fontsize=15)
    plt.legend(fontsize=10)
    plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
    plt.show()

    sns.distplot(df_male_control['realRelReleaseTimes'], color='green', label='control F0')
    sns.distplot(df_male_rove['realRelReleaseTimes'], color='orange', label='roved F0')
    plt.title('Reaction times for the male talker, \n irrespective of ferret', fontsize=15)
    plt.legend(fontsize=10)
    plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
    plt.show()

    df_by_ferret = {}
    df_by_ferret_f_control = {}
    df_by_ferret_f_rove = {}
    df_by_ferret_m_control = {}
    df_by_ferret_m_rove = {}

    # now plot by ferret ID
    ferrets = [0, 1, 2, 3]
    for ferret in ferrets:
        df_by_ferret_f_control[ferret] = df_female_control.loc[df_female_control['ferret'] == ferret]
        df_by_ferret_f_rove[ferret] = df_female_rove.loc[df_female_rove['ferret'] == ferret]
        df_by_ferret_m_control[ferret] = df_male_rove.loc[df_male_rove['ferret'] == ferret]
        df_by_ferret_m_rove[ferret] = df_male_control.loc[df_male_control['ferret'] == ferret]
    ferret_labels = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni']
    for ferret in ferrets:
        sns.distplot(df_by_ferret_f_control[ferret]['realRelReleaseTimes'], color='blue', label='control F0, female')
        sns.distplot(df_by_ferret_f_rove[ferret]['realRelReleaseTimes'], color='red', label='roved F0, female')
        sns.distplot(df_by_ferret_m_control[ferret]['realRelReleaseTimes'], color='green', label='control F0, male')
        sns.distplot(df_by_ferret_m_rove[ferret]['realRelReleaseTimes'], color='orange', label='roved F0, male')
        plt.title('Reaction times for ferret ID ' + str(ferret_labels[ferret]), fontsize=15)
        plt.legend(fontsize=10)
        plt.xlabel('reaction time relative to target presentation (s)', fontsize=13)
        plt.show()

    return df_by_ferret

    #


if __name__ == '__main__':
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    # df_by_ferretdict = plot_reaction_times(ferrets)
    #
    # df_left, df_right = plot_correct_response_byside(ferrets)
    #
    # test_df = run_reaction_time_fa_pipleine(ferrets)
    #
    # test_df2 = run_reaction_time_fa_pipleine_male(ferrets)

    # xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = runfalsealarmpipeline(ferrets)

    # xg_reg2, ypred2, y_test2, results2, shap_values, X_train, y_train, bal_accuracy, shap_values2 = run_correct_responsepipeline(ferrets)

    # modelreg_reduc, modelregcat_reduc, modelregcat, modelreg, predictedrelease, df_use, dfcat_use, predictedcorrectresp, explainedvar, explainvarreleasetime = run_mixed_effects_analysis(
    #     ferrets)

    df_use = extract_release_times_data(ferrets)

    col = 'realRelReleaseTimes'
    dfx = df_use.loc[:, df_use.columns != col]


    # remove ferret as possible feature
    col2 = 'ferret'
    dfx = dfx.loc[:, dfx.columns != col2]

    # col3 = 'pitchofprecur'
    # dfx = dfx.loc[:, dfx.columns != col3]

    # col3 = 'stepval'
    # dfx = dfx.loc[:, dfx.columns != col3]

    study_release_times = run_optuna_study_releasetimes(dfx.to_numpy(), df_use[col].to_numpy())
    xg_reg, ypred, y_test, results = runlgbreleasetimes(dfx.to_numpy(), df_use[col].to_numpy(), paramsinput=study_release_times.best_params)
    # count = 0
    # for ferret_id in ferrets:
    #     xg_reg, ypred, y_test, results, mse = runlgbreleasetimes_for_a_ferret(df_use, ferret=count,
    #                                                                           ferret_name=ferret_id)
    #     count += 1
