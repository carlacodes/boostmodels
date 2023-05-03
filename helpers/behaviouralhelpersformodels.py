from instruments.config import behaviouralDataPath, behaviourOutput
from sklearn.preprocessing import MinMaxScaler
from instruments.helpers.extract_helpers import extractAllFerretData
import pandas as pd
import numpy as np
import pandas as pd


class behaviouralhelperscg():
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
            misslist = np.where((correctresp==0)|(correctresp==1), correctresp^1, correctresp)
            newdata['misslist'] = misslist.tolist()
            newdata['correctresp'] = correctresp.tolist()
            newdata['pastcorrectresp'] = pastcorrectresp.tolist()
            newdata['talker'] = talkerlist2.tolist()
            newdata['pastcatchtrial'] = pastcatchtrial.tolist()
            newdata['stepval'] = stepval.tolist()
            # newdata['realRelReleaseTimes'] = np.log(newdata['realRelReleaseTimes'])
            newdata['cosinesim'] = correspondcosinelist.tolist()
            precur_and_targ_same = precur_and_targ_same.astype(int)
            newdata['precur_and_targ_same'] = precur_and_targ_same.tolist()
            newdata['timeToTarget'] = newdata['timeToTarget'] / 24414.0625
            newdata['AM'] = newdata['AM'].astype(int)

            # only look at v2 pitches from recent experiments
            newdata = newdata[(newdata.pitchoftarg == 1) | (newdata.pitchoftarg == 2) | (newdata.pitchoftarg == 3) | (
                    newdata.pitchoftarg == 4) | (newdata.pitchoftarg == 5)]
            newdata = newdata[
                (newdata.pitchofprecur == 1) | (newdata.pitchofprecur == 2) | (newdata.pitchofprecur == 3) | (
                        newdata.pitchofprecur == 4) | (newdata.pitchofprecur == 5)]

            newdata = newdata[(newdata.correctionTrial == 0)]  # | (allData.response == 7)
            newdata = newdata[(newdata.currAtten == 0)]  # | (allData.response == 7)

            if includefaandmiss is True:
                newdata = newdata[
                    (newdata.response == 0) | (newdata.response == 1) | (newdata.response == 7) | (
                                newdata.response == 5)]
            elif includemissonly is True:
                newdata = newdata[
                    (newdata.response == 0) | (newdata.response == 1) | (newdata.response == 7) | (
                                newdata.response == 3)]
            else:
                newdata = newdata[newdata.correctresp == 1]
                newdata = newdata[(newdata.catchTrial == 0)]
            bigdata = bigdata.append(newdata)
        return bigdata

    def get_false_alarm_behavdata(path=None,
                                  output=None,
                                  ferrets=None,
                                  startdate=None,
                                  finishdate=None):
        if output is None:
            output = behaviourOutput

        if path is None:
            path = behaviouralDataPath

        allData, ferrets = extractAllFerretData(ferrets, path, startDate=startdate,
                                                finishDate=finishdate)
        len_of_data = {}
        fs = 24414.062500
        for i in range(0, len(ferrets)):
            noncorrectiondata = allData[allData['correctionTrial'] == 0]
            noncorrectiondata = noncorrectiondata[noncorrectiondata['currAtten'] == 0]
            len_of_data[ferrets[i]] = len(noncorrectiondata[noncorrectiondata['ferret'] == i])

        bigdata = pd.DataFrame()
        numofferrets = allData['ferret'].unique()
        for ferret in numofferrets:
            print(ferret)
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
            distractor_or_fa = np.empty(len(pitchshiftmat))
            gradinpitch = np.empty(len(pitchshiftmat))
            gradinpitchprecur = np.empty(len(pitchshiftmat))
            timetotarglist = np.empty(len(pitchshiftmat))

            precur_and_targ_same = np.empty(len(pitchshiftmat))
            intra_trial_roving = []
            inter_trial_roving = []
            control_trial = []
            talkerlist2 = np.empty(len(pitchshiftmat))

            falsealarm = np.empty(shape=(0, 0))
            correctresp = np.empty(shape=(0, 0))
            pastcorrectresp = np.empty(shape=(0, 0))
            pastcatchtrial = np.empty(shape=(0, 0))
            droplist = np.empty(shape=(0, 0))
            droplistnew = np.empty(shape=(0, 0))

            for i in range(1, len(newdata['realRelReleaseTimes'].values)):
                chosenresponseindex = chosenresponse.values[i]
                pastcatchtrialindex = catchtriallist.values[i - 1]

                realrelreleasetime = realrelreleasetimelist.values[i]
                pastrealrelreleasetime = realrelreleasetimelist.values[i - 1]
                pastresponseindex = chosenresponse.values[(i - 1)]

                chosentrial = pitchshiftmat.values[i]
                is_all_zero = np.all((chosentrial == 0))
                if is_all_zero:
                    control_trial.append(0)
                else:
                    control_trial.append(1)

                if isinstance(chosentrial, float) or is_all_zero:
                    chosentrial = talkermat.values[i].astype(int)
                    intra_trial_roving.append(0)
                else:
                    intra_trial_roving.append(1)
                chosentalker = talkerlist.values[i]

                if chosentalker == 3 or chosentalker == 5 or chosentalker == 8 or chosentalker == 13:
                    inter_trial_roving.append(1)
                else:
                    inter_trial_roving.append(0)

                chosendisttrial = precursorlist.values[i]
                if chosentalker == 3:
                    chosentalker = 1
                if chosentalker == 8:
                    chosentalker = 2
                if chosentalker == 13:
                    chosentalker = 2
                if chosentalker == 5:
                    chosentalker = 1
                talkerlist2[i] = chosentalker

                if ((
                            chosenresponseindex == 0 or chosenresponseindex == 1) and realrelreleasetime >= 0) or chosenresponseindex == 3 or chosenresponseindex == 7:
                    falsealarm = np.append(falsealarm, 0)
                else:
                    falsealarm = np.append(falsealarm, 1)
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
                    targpos = np.where(chosendisttrial == 1)
                    distractor_or_fa[i] = chosendisttrial[targpos[0] - 1]

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

                    pitchoftarg[i] = np.nan
                    if isinstance(chosentrial, int):
                        pitchofprecur[i] = chosentrial
                    else:
                        pitchofprecur[i] = chosentrial[-1]
                    if pitchofprecur[i] == 1.0:
                        pitchofprecur[i] = 4.0

                    if pitchofprecur[i] == 13.0:
                        pitchofprecur[i] = 1.0

                    if pitchoftarg[i] == 1.0:
                        pitchoftarg[i] = 4.0

                    if pitchoftarg[i] == 13.0:
                        pitchoftarg[i] = 1.0
                    distractor_or_fa[i] = chosendisttrial[-1]
                    continue
            newdata.drop(index=newdata.index[0],
                         axis=0,
                         inplace=True)
            droplist = [int(x) for x in droplist]  # drop corrupted metdata trials

            pitchoftarg = pitchoftarg.astype(int)
            pitchofprecur = pitchofprecur.astype(int)
            falsealarm = falsealarm[~np.isnan(falsealarm)]
            correctresp = correctresp[~np.isnan(correctresp)]

            pitchoftarg = np.delete(pitchoftarg, 0)
            talkerlist2 = np.delete(talkerlist2, 0)
            distractor_or_fa = np.delete(distractor_or_fa, 0)
            stepval = np.delete(stepval, 0)
            pitchofprecur = np.delete(pitchofprecur, 0)
            # intra_trial_roving = np.delete(intra_trial_roving, 0)

            newdata['pitchoftarg'] = pitchoftarg.tolist()
            newdata['pitchofprecur'] = pitchofprecur.tolist()

            falsealarm = falsealarm.astype(int)
            pastcatchtrial = pastcatchtrial.astype(int)
            pastcorrectresp = pastcorrectresp.astype(int)

            newdata['falsealarm'] = falsealarm.tolist()
            newdata['intra_trial_roving'] = intra_trial_roving
            newdata['inter_trial_roving'] = inter_trial_roving
            newdata['control_trial'] = control_trial
            newdata['correctresp'] = correctresp.tolist()
            newdata['distractor_or_fa'] = distractor_or_fa.tolist()
            newdata['pastcorrectresp'] = pastcorrectresp.tolist()
            newdata['talker'] = talkerlist2.tolist()
            newdata['pastcatchtrial'] = pastcatchtrial.tolist()
            newdata['stepval'] = stepval.tolist()
            newdata['timeToTarget'] = newdata['timeToTarget'] / 24414.0625
            newdata['AM'] = newdata['AM'].astype(int)
            newdata = newdata[newdata['distractor_or_fa'].values <= 57]

            cosinesimfemale = np.load('D:/Stimuli/cosinesimvectorfemale.npy')
            cosinesimmale = np.load('D:/Stimuli/cosinesimvectormale.npy')
            temporalsimfemale = np.load('D:/Stimuli/temporalcorrfemale.npy')
            temporalsimmale = np.load('D:/Stimuli/temporalcorrmale.npy')

            distinds = newdata['distractor_or_fa'].values
            distinds = distinds - 1;
            correspondcosinelist = []
            correspondtempsimlist = []
            for i in range(len(distinds)):
                if newdata['talker'].values[i] == 0:
                    correspondcosinelist.append(cosinesimfemale[int(distinds[i])])
                    correspondtempsimlist.append(temporalsimfemale[int(distinds[i])])
                else:
                    correspondcosinelist.append(cosinesimmale[int(distinds[i])])
                    correspondtempsimlist.append(temporalsimmale[int(distinds[i])])
            newdata['cosinesim'] = correspondcosinelist
            newdata['temporalsim'] = correspondtempsimlist

            newdata = newdata[(newdata.talker == 1) | (newdata.talker == 2) | (newdata.talker == 3) | (
                    newdata.talker == 4) | (newdata.talker == 5)]

            newdata = newdata[(newdata.pitchofprecur == 1) | (newdata.pitchofprecur == 2) | (newdata.pitchofprecur == 3) | (
                    newdata.pitchofprecur == 4) | (newdata.pitchofprecur == 5)]

            newdata = newdata[(newdata.pitchoftarg == 1) | (newdata.pitchoftarg == 2) | (newdata.pitchoftarg == 3) | (
                    newdata.pitchoftarg == 4) | (newdata.pitchoftarg == 5)]
            # newdata = newdata[
            #     (newdata.pitchofprecur == 1) | (newdata.pitchofprecur == 2) | (newdata.pitchofprecur == 3) | (
            #             newdata.pitchofprecur == 4) | (newdata.pitchofprecur == 5)]

            newdata = newdata[(newdata.correctionTrial == 0)]  # | (allData.response == 7)
            newdata = newdata[(newdata.currAtten == 0)]  # | (allData.response == 7)
            # newdata = newdata[(newdata.catchTrial == 0)]  # | (allData.response == 7)

            bigdata = bigdata.append(newdata)
        return bigdata

    def get_reactiontime_data(path=None,
                              output=None,
                              ferrets=None,
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
        numofferrets = allData['ferret'].unique()
        for ferret in numofferrets:
            print(ferret)

            newdata = allData[allData['ferret'] == ferret]
            newdata = newdata[
                (newdata.response == 1) | (newdata.response == 0) | (newdata.response == 5)]  # remove all misses
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
            distractor_or_fa = []
            intra_trial_roving = []
            inter_trial_roving = []
            control_trial = []

            talkerlist2 = np.empty(len(pitchshiftmat))
            falsealarm = np.empty(shape=(0, 0))
            correctresp = np.empty(shape=(0, 0))
            pastcorrectresp = np.empty(shape=(0, 0))
            pastcatchtrial = np.empty(shape=(0, 0))

            for i in range(1, len(newdata['realRelReleaseTimes'].values)):
                chosenresponseindex = chosenresponse.values[i]
                pastcatchtrialindex = catchtriallist.values[i - 1]

                realrelreleasetime = realrelreleasetimelist.values[i]
                pastrealrelreleasetime = realrelreleasetimelist.values[i - 1]
                pastresponseindex = chosenresponse.values[(i - 1)]
                current_distractors = distractors.values[i]
                current_dDurs = newdata['dDurs'].values[i] / 24414.062500
                current_releasetime = newdata['centreRelease'].values[i]
                curr_dur_list = []
                current_dist_list = []
                for i2 in range(0, len(current_distractors)):
                    current_dur = np.sum(current_dDurs[0:(i2 + 1)])

                    if current_dur <= current_releasetime:
                        # print('current dur', current_dur)
                        curr_dur_list.append(current_dur)
                        current_dist_list.append(current_distractors[i2])
                    else:
                        break

                try:
                    distractor_or_fa.append(current_dist_list[-1])
                except:
                    distractor_or_fa.append(current_distractors[0])

                chosentrial = pitchshiftmat.values[i]
                is_all_zero = np.all((chosentrial == 0))

                if is_all_zero:
                    control_trial.append(0)
                else:
                    control_trial.append(1)

                if isinstance(chosentrial, float) or is_all_zero:
                    chosentrial = talkermat.values[i].astype(int)
                    intra_trial_roving.append(0)
                else:
                    intra_trial_roving.append(1)

                chosentalker = talkerlist.values[i]
                print(i)
                # find where talkerlist.values == 3

                if chosentalker == 3 or chosentalker == 5 or chosentalker == 8 or chosentalker == 13:
                    print('inter detected')
                    inter_trial_roving.append(1)
                else:
                    print(chosentalker)
                    inter_trial_roving.append(0)

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

                if ((
                            chosenresponseindex == 0 or chosenresponseindex == 1) and realrelreleasetime >= 0) or chosenresponseindex == 3 or chosenresponseindex == 7:
                    falsealarm = np.append(falsealarm, 0)
                else:
                    falsealarm = np.append(falsealarm, 1)
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
                    targpos = np.where(chosendisttrial == 1)

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

                    if pitchofprecur[i] == 1.0:
                        pitchofprecur[i] = 4.0

                    if pitchofprecur[i] == 13.0:
                        pitchofprecur[i] = 1.0

                    if pitchoftarg[i] == 1.0:
                        pitchoftarg[i] = 4.0

                    if pitchoftarg[i] == 13.0:
                        pitchoftarg[i] = 1.0


                except:
                    # arrays START AT 0, but the index starts at 1, so the index is 1 less than the array

                    pitchoftarg[i] = np.nan
                    if isinstance(chosentrial, int):
                        pitchofprecur[i] = chosentrial
                    else:
                        pitchofprecur[i] = chosentrial[-1]
                    if pitchofprecur[i] == 1.0:
                        pitchofprecur[i] = 4.0

                    if pitchofprecur[i] == 13.0:
                        pitchofprecur[i] = 1.0

                    if pitchoftarg[i] == 1.0:
                        pitchoftarg[i] = 4.0

                    if pitchoftarg[i] == 13.0:
                        pitchoftarg[i] = 1.0
                    continue
            newdata.drop(index=newdata.index[0],
                         axis=0,
                         inplace=True)

            pitchoftarg = pitchoftarg.astype(int)
            pitchofprecur = pitchofprecur.astype(int)
            falsealarm = falsealarm[~np.isnan(falsealarm)]
            correctresp = correctresp[~np.isnan(correctresp)]

            pitchoftarg = np.delete(pitchoftarg, 0)
            talkerlist2 = np.delete(talkerlist2, 0)
            stepval = np.delete(stepval, 0)
            pitchofprecur = np.delete(pitchofprecur, 0)


            newdata['pitchoftarg'] = pitchoftarg.tolist()

            newdata['pitchofprecur'] = pitchofprecur.tolist()

            falsealarm = falsealarm.astype(int)
            pastcatchtrial = pastcatchtrial.astype(int)
            pastcorrectresp = pastcorrectresp.astype(int)

            newdata['falsealarm'] = falsealarm.tolist()
            newdata['control_trial'] = control_trial
            newdata['intra_trial_roving'] = intra_trial_roving
            newdata['inter_trial_roving'] = inter_trial_roving
            newdata['correctresp'] = correctresp.tolist()
            newdata['distractor_or_fa'] = distractor_or_fa
            newdata['pastcorrectresp'] = pastcorrectresp.tolist()
            newdata['pastcatchtrial'] = pastcatchtrial.tolist()
            newdata['stepval'] = stepval.tolist()
            newdata['timeToTarget'] = newdata['timeToTarget'] / 24414.0625
            newdata['AM'] = newdata['AM'].astype(int)
            newdata['talker'] = newdata['talker']
            newdata = newdata[newdata['distractor_or_fa'].values <= 57]

            cosinesimfemale = np.load('D:/Stimuli/cosinesimvectorfemale.npy')
            cosinesimmale = np.load('D:/Stimuli/cosinesimvectormale.npy')

            distinds = newdata['distractor_or_fa'].values
            distinds = distinds - 1;
            correspondcosinelist = []
            for i in range(len(distinds)):
                if newdata['talker'].values[i] == 0:
                    correspondcosinelist.append(cosinesimfemale[int(distinds[i])])
                else:
                    correspondcosinelist.append(cosinesimmale[int(distinds[i])])
            newdata['cosinesim'] = correspondcosinelist

            newdata = newdata[(newdata.correctionTrial == 0)]  # | (allData.response == 7)
            newdata = newdata[(newdata.currAtten == 0)]  # | (allData.response == 7)

            emptydistracotrindexdict_categorical = dict.fromkeys((range(1, 58)))
            emptydistracotrindexdict_categorical = {str(k): str(v) for k, v in
                                                    emptydistracotrindexdict_categorical.items()}

            dataframeversion = pd.DataFrame.from_dict(emptydistracotrindexdict_categorical, orient='index')
            for i in range(0, len(newdata)):
                # now declare function to get the distractor indices that the animal fa-ed to or the correct distractor
                if str(int(newdata['distractor_or_fa'].values[i])) in emptydistracotrindexdict_categorical:
                    exception_key = str(int(newdata['distractor_or_fa'].values[i]))
                    # if exception_key != 1:
                    #     print('nonone distracotr key =', i)
                    # combined = np.array([])
                    if i == 0:
                        # emptydistracotrindexdict_categorical[newdata['distractor_or_fa'].values[i]] = newdata['centreRelease'].values[i]
                        for key, value in emptydistracotrindexdict_categorical.items():
                            if key == exception_key:
                                emptydistracotrindexdict_categorical[key] = float(newdata['centreRelease'].values[i])
                            else:
                                emptydistracotrindexdict_categorical[key] = float('nan')
                    else:

                        for key, value in emptydistracotrindexdict_categorical.items():
                            if key == exception_key:
                                current = emptydistracotrindexdict_categorical[key]
                                combined = np.concatenate((current, [float(newdata['centreRelease'].values[i])]),
                                                          axis=None)
                                emptydistracotrindexdict_categorical[key] = combined
                            else:
                                current = emptydistracotrindexdict_categorical[key]
                                combined = np.concatenate((current, [float('nan')]), axis=None)
                                emptydistracotrindexdict_categorical[key] = combined


                else:
                    print("Does not exist")
                    print(str(int(newdata['distractor_or_fa'].values[i])))
            for keyh in emptydistracotrindexdict_categorical:
                selectedcol = emptydistracotrindexdict_categorical[keyh]
                newdata[keyh] = selectedcol

            # convert to dataframe
            #
            # dataframeversion = pd.DataFrame.from_dict(emptydistracotrindexdict_categorical, orient='index')
            # dataframeversion2 = pd.DataFrame(emptydistracotrindexdict_categorical, index=[0])
            # dataframeversion = dataframeversion.transpose()
            #
            # df = pd.DataFrame(emptydistracotrindexdict_categorical, columns=(range(1,56)))
            # newdata2 = newdata.append(dataframeversion)
            # newdata['distindex'] = emptydistracotrindexdict_categorical
            bigdata = bigdata.append(newdata)
        return bigdata

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
