from instruments.config import behaviouralDataPath, behaviourOutput
from sklearn.preprocessing import MinMaxScaler
from instruments.helpers.extract_helpers import extractAllFerretData
import pandas as pd
import numpy as np
import pandas as pd
import datetime as dt


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
            pitchoftarg =[]
            pitchofprecur = []
            stepval = np.empty(len(pitchshiftmat))
            precur_and_targ_same = []
            talkerlist2 = np.empty(len(pitchshiftmat))

            correctresp = np.empty(shape=(0, 0))
            pastcorrectresp = np.empty(shape=(0, 0))
            pastcatchtrial = np.empty(shape=(0, 0))
            droplist = np.empty(shape=(0, 0))
            droplistnew = np.empty(shape=(0, 0))
            intra_trial_roving = []
            inter_trial_roving = []
            control_trial = []
            for i in range(1, len(newdata['realRelReleaseTimes'].values)):
                chosenresponseindex = chosenresponse.values[i]
                pastcatchtrialindex = catchtriallist.values[i - 1]
                realrelreleasetime = realrelreleasetimelist.values[i]
                pastrealrelreleasetime = realrelreleasetimelist.values[i - 1]
                pastresponseindex = chosenresponse.values[(i - 1)]

                chosentrial = pitchshiftmat.values[i]
                is_all_zero = np.all((chosentrial == 0))
                chosentalker = talkerlist.values[i]

                import numbers
                if isinstance(chosentrial, float) and (chosentalker ==1 or chosentalker ==2):
                    control_trial.append(1)
                elif is_all_zero and (chosentalker ==1 or chosentalker ==2):
                    control_trial.append(1)
                else:
                    control_trial.append(0)

                if isinstance(pitchshiftmat.values[i], float):
                    print('intra not detected')
                    intra_trial_roving.append(0)
                    chosentrial = talkermat.values[i]
                elif is_all_zero:
                    print('intra not detected')
                    intra_trial_roving.append(0)
                    chosentrial = talkermat.values[i]

                else:
                    intra_trial_roving.append(1)



                chosendisttrial = precursorlist.values[i]
                if chosentalker == 3 or chosentalker == 5 or chosentalker == 8 or chosentalker == 13:
                    print('inter detected')
                    inter_trial_roving.append(1)
                else:
                    inter_trial_roving.append(0)

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

                targpos = np.where(chosendisttrial == 1)
                try:
                    targpos = int(targpos[0])
                    precur_pos = targpos - 1

                    if np.sum(newdata['dDurs'].values[i][:targpos]) / fs <= newdata['centreRelease'].values[i] - \
                            newdata['absentTime'].values[i] or newdata['response'].values[i] == 7:
                        if chosentrial[targpos] == chosentrial[targpos -1 ]:
                            precur_and_targ_same.append(1)
                        else:
                            precur_and_targ_same.append(0)

                        if chosentrial[targpos] == 8.0:
                            pitchoftarg.append(float(3))
                        elif chosentrial[targpos] == 13.0:
                            pitchoftarg.append(float(1))
                        elif chosentrial[targpos] == 1.0:
                            pitchoftarg.append(float(4))
                        else:
                            pitchoftarg.append(float(chosentrial[targpos]))


                    else:
                        pitchoftarg.append(np.nan)
                        precur_and_targ_same.append(np.nan)

                    if np.sum(newdata['dDurs'].values[i][:precur_pos]) / fs <= newdata['centreRelease'].values[i] - \
                            newdata['absentTime'].values[i] or newdata['response'].values[i] == 7:
                        if chosentrial[precur_pos] == 8.0:
                            pitchofprecur.append(float(3))
                        elif chosentrial[precur_pos] == 13.0:
                            pitchofprecur.append(float(1))
                        elif chosentrial[precur_pos] == 1.0:
                            pitchofprecur.append(float(4))
                        else:
                            pitchofprecur.append(float(chosentrial[precur_pos]))

                    else:
                        pitchofprecur.append(np.nan)
                except:
                    pitchoftarg.append(np.nan)
                    pitchofprecur.append(np.nan)
                    precur_and_targ_same.append(np.nan)

            newdata.drop(index=newdata.index[0],
                         axis=0,
                         inplace=True)
            newdata.drop(droplistnew, axis=0, inplace=True)
            droplist = [int(x) for x in droplist]  # drop corrupted metdata trials



            correctresp = correctresp[~np.isnan(correctresp)]

            talkerlist2 = np.delete(talkerlist2, 0)
            stepval = np.delete(stepval, 0)

            pitchoftarg = np.delete(pitchoftarg, droplist)
            talkerlist2 = np.delete(talkerlist2, droplist)
            stepval = np.delete(stepval, droplist)

            newdata['pitchoftarg'] = pitchoftarg
            newdata['pitchofprecur'] = pitchofprecur
            correctresp = np.delete(correctresp, droplist)
            pastcorrectresp = np.delete(pastcorrectresp, droplist)
            pastcatchtrial = np.delete(pastcatchtrial, droplist)
            precur_and_targ_same = np.delete(precur_and_targ_same, droplist)
            inter_trial_roving = np.delete(inter_trial_roving, droplist)
            intra_trial_roving = np.delete(intra_trial_roving, droplist)
            control_trial = np.delete(control_trial, droplist)

            correctresp = correctresp.astype(int)
            pastcatchtrial = pastcatchtrial.astype(int)
            pastcorrectresp = pastcorrectresp.astype(int)
            misslist = np.where((correctresp == 0) | (correctresp == 1), correctresp ^ 1, correctresp)
            newdata['misslist'] = misslist.tolist()
            newdata['correctresp'] = correctresp.tolist()
            newdata['pastcorrectresp'] = pastcorrectresp.tolist()
            newdata['inter_trial_roving'] = inter_trial_roving.tolist()
            newdata['intra_trial_roving'] = intra_trial_roving.tolist()
            newdata['talker'] = talkerlist2.tolist()
            newdata['pastcatchtrial'] = pastcatchtrial.tolist()
            newdata['stepval'] = stepval.tolist()
            newdata['control_trial'] = control_trial.tolist()

            # newdata['realRelReleaseTimes'] = np.log(newdata['realRelReleaseTimes'])
            # precur_and_targ_same = precur_and_targ_same.astype(int)
            newdata['precur_and_targ_same'] = precur_and_targ_same

            newdata['timeToTarget'] = newdata['timeToTarget'] / 24414.0625
            newdata['AM'] = newdata['AM'].astype(int)

            # only look at v2 pitches from recent experiments
            newdata = newdata[(newdata.pitchoftarg == 1) | (newdata.pitchoftarg == 2) | (newdata.pitchoftarg == 3) | (
                    newdata.pitchoftarg == 4) | (newdata.pitchoftarg == 5)| (newdata.pitchofprecur.isnull())]
            newdata = newdata[
                (newdata.pitchofprecur == 1) | (newdata.pitchofprecur == 2) | (newdata.pitchofprecur == 3) | (
                        newdata.pitchofprecur == 4) | (newdata.pitchofprecur == 5) | (newdata.pitchofprecur.isnull())]

            newdata = newdata[(newdata.correctionTrial == 0)]  # | (allData.response == 7)
            newdata = newdata[(newdata.currAtten == 0)]  # | (allData.response == 7)

            if includefaandmiss is True:
                newdata = newdata[
                    (newdata.response == 0) | (newdata.response == 1) | (newdata.response == 7) | (
                            newdata.response == 5)]
            elif includemissonly is True:
                newdata = newdata[
                    (newdata.response == 0) | (newdata.response == 1) | (newdata.response == 7) ]
            else:
                newdata = newdata[newdata.correctresp == 1]
                newdata = newdata[(newdata.catchTrial == 0)]
            bigdata = bigdata.append(newdata)
        return bigdata

    def matlab2datetime(matlab_datenum):
        day = dt.datetime.fromordinal(int(matlab_datenum))
        dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)
        return day + dayfrac

    def get_df_rxntimebydist(path=None,
                     output=None,
                     ferrets=None,
                     includefa = False,
                     includemissonly=False,
                     startdate=None,
                     finishdate=None, talker_param=1):
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
            newdata = allData[allData['ferret'] == ferret]

            newdata['targTimes'] = newdata['timeToTarget'] / fs

            newdata['centreRelease'] = newdata['lickRelease'] - newdata['startTrialLick']
            newdata['relReleaseTimes'] = newdata['centreRelease'] - newdata['targTimes']
            newdata['realRelReleaseTimes'] = newdata['relReleaseTimes'] - newdata['absentTime']
            distractors = newdata['distractors']
            #make new column for each distractor, and put the rxn time in of the absolute release time
            for i00 in range(0,57):
                #make an array of nans the length of the dataframe
                newdata['dist' + str(i00+1)] = np.full((len(distractors)),np.nan)

            EasyListF = [1, 5, 20, 2, 42, 56, 32, 57, 33, 11];
            EasyList = [1, 6, 22, 2, 49, 56, 38, 57, 39, 13];

            for i0 in range(0, len(distractors)):
                dist_trial = distractors.values[i0]
                for dist in dist_trial:
                    if dist <=57:
                        #calculate position of distractor in trial
                        distpos = np.where(dist_trial == dist)[0][0]
                        #calculate rxn time of distractor
                        #check if string of fname contains 39 or 43, and check the day of the experiment
                        dateofstart = newdata['startTime'].values[i0]
                        day = dt.datetime.fromordinal(int(dateofstart))
                        dayfrac = dt.timedelta(days=dateofstart % 1) - dt.timedelta(days=366)
                        exdatestart = day + dayfrac

                        if 'level_39' in newdata['fName'].values[i0] or 'level_43' in newdata['fName'].values[i0] and talker_param == 1:
                            distlabel = EasyListF[dist-1]
                        elif 'level_39' in newdata['fName'].values[i0] or 'level_43' in newdata['fName'].values[i0]  and talker_param == 2:
                            distlabel = EasyList[dist-1]
                        elif dist == 56 and newdata['ferretname'].values[i0] == 'F1702_Zola' and exdatestart >= dt.datetime(2021, 6, 21, 0, 0, 0):
                            print('pink noise detected')
                            distlabel = 57
                        else:
                            distlabel = dist


                        if np.sum(newdata['dDurs'].values[i0][:distpos-1])/fs <= newdata['centreRelease'].values[i0]:
                            newdata['dist' + str(distlabel)].values[i0] = np.sum(newdata['dDurs'].values[i0][:distpos-1])/fs


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
            correspondcosinelist = np.empty(shape=(0, 0))
            intra_trial_roving = []
            inter_trial_roving = []
            control_trial = []
            for i in range(1, len(newdata['realRelReleaseTimes'].values)):
                chosenresponseindex = chosenresponse.values[i]
                pastcatchtrialindex = catchtriallist.values[i - 1]
                realrelreleasetime = realrelreleasetimelist.values[i]
                pastrealrelreleasetime = realrelreleasetimelist.values[i - 1]
                pastresponseindex = chosenresponse.values[(i - 1)]

                chosentrial = pitchshiftmat.values[i]
                is_all_zero = np.all((chosentrial == 0))
                chosentalker = talkerlist.values[i]

                import numbers
                if isinstance(chosentrial, float) and (chosentalker == 1 or chosentalker == 2):
                    control_trial.append(1)
                elif is_all_zero and (chosentalker == 1 or chosentalker == 2):
                    control_trial.append(1)
                else:
                    control_trial.append(0)

                if isinstance(chosentrial, float):
                    print('intra not detected')
                    intra_trial_roving.append(0)
                    chosentrial = talkermat.values[i]
                elif is_all_zero:
                    print('intra not detected')
                    intra_trial_roving.append(0)
                else:
                    intra_trial_roving.append(1)

                chosendisttrial = precursorlist.values[i]
                if chosentalker == 3 or chosentalker == 5 or chosentalker == 8 or chosentalker == 13:
                    print('inter detected')
                    inter_trial_roving.append(1)
                else:
                    print(chosentalker)
                    inter_trial_roving.append(0)
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


            newdata.drop(index=newdata.index[0],
                         axis=0,
                         inplace=True)
            # newdata.drop(droplistnew, axis=0, inplace=True)
            # droplist = [int(x) for x in droplist]  # drop corrupted metdata trials

            pitchoftarg = pitchoftarg.astype(int)
            pitchofprecur = pitchofprecur.astype(int)

            correctresp = correctresp[~np.isnan(correctresp)]

            pitchoftarg = np.delete(pitchoftarg, 0)
            talkerlist2 = np.delete(talkerlist2, 0)
            stepval = np.delete(stepval, 0)
            pitchofprecur = np.delete(pitchofprecur, 0)
            precur_and_targ_same = np.delete(precur_and_targ_same, 0)



            newdata['pitchoftarg'] = pitchoftarg.tolist()
            newdata['pitchofprecur'] = pitchofprecur.tolist()


            correctresp = correctresp.astype(int)
            pastcatchtrial = pastcatchtrial.astype(int)
            pastcorrectresp = pastcorrectresp.astype(int)
            misslist = np.where((correctresp == 0) | (correctresp == 1), correctresp ^ 1, correctresp)
            newdata['misslist'] = misslist.tolist()
            newdata['correctresp'] = correctresp.tolist()
            newdata['pastcorrectresp'] = pastcorrectresp.tolist()
            newdata['inter_trial_roving'] = inter_trial_roving
            newdata['intra_trial_roving'] = intra_trial_roving
            newdata['talker'] = talkerlist2.tolist()
            newdata['pastcatchtrial'] = pastcatchtrial.tolist()
            newdata['stepval'] = stepval.tolist()
            newdata['control_trial'] = control_trial
            # newdata['realRelReleaseTimes'] = np.log(newdata['realRelReleaseTimes'])
            precur_and_targ_same = precur_and_targ_same.astype(int)
            newdata['precur_and_targ_same'] = precur_and_targ_same.tolist()
            newdata['timeToTarget'] = newdata['timeToTarget'] / 24414.0625
            newdata['AM'] = newdata['AM'].astype(int)

            newdata = newdata[(newdata.correctionTrial == 0)]  # | (allData.response == 7)
            newdata = newdata[(newdata.currAtten == 0)]
            newdata = newdata[(newdata.talker == talker_param)]

            if includefa is True:
                newdata = newdata[
                    (newdata.response == 0) | (newdata.response == 1) |  (
                            newdata.response == 5) ]
            elif includemissonly is True:
                newdata = newdata[
                    (newdata.response == 0) | (newdata.response == 1) | (newdata.response == 7) | (
                            newdata.response == 3)]
            else:
                newdata = newdata[newdata.correctresp == 1]
                newdata = newdata[(newdata.catchTrial == 0)]
            bigdata = bigdata.append(newdata)
        return bigdata
    def matlab2datetime(matlab_datenum):
        day = dt.datetime.fromordinal(int(matlab_datenum))
        dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)
        return day + dayfrac

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
            newdata = newdata[newdata['catchTrial'] == 0]
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

            stepval = np.empty(len(pitchshiftmat))
            distractor_or_fa = np.empty(len(pitchshiftmat))

            intra_trial_roving = []
            inter_trial_roving = []
            control_trial = []
            pitchoftarg = []
            pitchofprecur = []
            pitchof0oflastword = []
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


                targpos = np.where(chosendisttrial == 1)
                #figure out the F0 of when the ferret releases by iteratively summing up the duration of each word
                #and comparing it to the time of the release
                count = -1
                distractordurationoftrial = newdata['dDurs'].values[i]
                if newdata['response'].values[i] != 3 and newdata['response'].values[i] != 7: #if it's a correct catch trial then the release time is inf
                    while np.sum(distractordurationoftrial[0:count+1]) / fs < newdata['centreRelease'].values[i] - newdata['absentTime'].values[i]:
                        count = count + 1
                    if chosentrial[count] == 8.0:
                        pitchof0oflastword.append(float(3))
                    elif chosentrial[count] == 13.0:
                        pitchof0oflastword.append(float(1))
                    elif chosentrial[count] == 1.0:
                        pitchof0oflastword.append(float(4))
                    else:
                        pitchof0oflastword.append(float(chosentrial[count]))


                    #
                    # for k2 in range(0, len(distractordurationoftrial)+1):
                    #     if np.sum(distractordurationoftrial[0:k2]) / fs < newdata['centreRelease'].values[i] - newdata['absentTime'].values[i]:
                    #         count = count + 1
                    #     else:
                    #         if chosentrial[count] == 8.0:
                    #             pitchof0oflastword.append(float(3))
                    #             break
                    #         elif chosentrial[count] == 13.0:
                    #             pitchof0oflastword.append(float(1))
                    #             break
                    #         elif chosentrial[count] == 1.0:
                    #             pitchof0oflastword.append(float(4))
                    #             break
                    #         else:
                    #             pitchof0oflastword.append(float(chosentrial[count]))
                    #             break


                else:
                    pitchof0oflastword.append(chosentrial[-1])


                try:
                    targpos = int(targpos[0])
                    precur_pos = targpos - 1

                    if np.sum(newdata['dDurs'].values[i][:targpos]) / fs <= newdata['centreRelease'].values[i] - newdata['absentTime'].values[i]:
                        if chosentrial[targpos] == 8.0:
                            pitchoftarg.append(float(3))
                        elif chosentrial[targpos] == 13.0:
                            pitchoftarg.append(float(1))
                        elif chosentrial[targpos] == 1.0:
                            pitchoftarg.append(float(4))
                        else:
                            pitchoftarg.append( float(chosentrial[targpos]))
                    else:
                        pitchoftarg.append(np.nan)

                    if np.sum(newdata['dDurs'].values[i][:precur_pos]) / fs <= newdata['centreRelease'].values[i]- newdata['absentTime'].values[i]:
                        if chosentrial[precur_pos] == 8.0:
                            pitchofprecur.append(float(3))
                        elif chosentrial[precur_pos] == 13.0:
                            pitchofprecur.append(float(1))
                        elif chosentrial[precur_pos] == 1.0:
                            pitchofprecur.append(float(4))
                        else:
                            pitchofprecur.append(float(chosentrial[precur_pos]))
                    else:
                        pitchofprecur.append(np.nan)
                except:
                    pitchoftarg.append(np.nan)
                    pitchofprecur.append(np.nan)

                print('at trial'+str(i))

            newdata.drop(index=newdata.index[0],
                         axis=0,
                         inplace=True)

            # pitchoftarg = pitchoftarg.astype(int)
            # pitchofprecur = pitchofprecur.astype(int)
            falsealarm = falsealarm[~np.isnan(falsealarm)]
            correctresp = correctresp[~np.isnan(correctresp)]

            talkerlist2 = np.delete(talkerlist2, 0)
            distractor_or_fa = np.delete(distractor_or_fa, 0)
            stepval = np.delete(stepval, 0)

            pitchof0oflastword = [float(d) for d in pitchof0oflastword]
            newdata['pitchoftarg'] = pitchoftarg
            newdata['pitchofprecur'] = pitchofprecur
            newdata['pitchof0oflastword'] = pitchof0oflastword

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

            # newdata = newdata[
            #     (newdata.pitchofprecur == 1) | (newdata.pitchofprecur == 2) | (newdata.pitchofprecur == 3) | (
            #             newdata.pitchofprecur == 4) | (newdata.pitchofprecur == 5)]
            #
            # newdata = newdata[(newdata.pitchoftarg == 1) | (newdata.pitchoftarg == 2) | (newdata.pitchoftarg == 3) | (
            #         newdata.pitchoftarg == 4) | (newdata.pitchoftarg == 5)]
            newdata = newdata[
                (newdata.pitchofprecur == 1) | (newdata.pitchofprecur == 2) | (newdata.pitchofprecur == 3) | (
                        newdata.pitchofprecur == 4) | (newdata.pitchofprecur == 5) | (newdata.pitchofprecur.isnull())]

            newdata = newdata[
                (newdata.pitchof0oflastword == 1) | (newdata.pitchof0oflastword == 2) | (newdata.pitchof0oflastword == 3) | (
                        newdata.pitchof0oflastword == 4) | (newdata.pitchof0oflastword == 5) ]

            newdata = newdata[(newdata.correctionTrial == 0)]  # | (allData.response == 7)
            newdata = newdata[(newdata.currAtten == 0)]  # | (allData.response == 7)
            # newdata = newdata[(newdata.catchTrial == 0)]  # | (allData.response == 7)

            bigdata = bigdata.append(newdata)
        return bigdata

    def get_stats_df(path=None,
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

            stepval = np.empty(len(pitchshiftmat))
            distractor_or_fa = np.empty(len(pitchshiftmat))

            intra_trial_roving = []
            inter_trial_roving = []
            control_trial = []
            pitchoftarg = []
            f0_list = []
            pitchofprecur = []
            talkerlist2 = np.empty(len(pitchshiftmat))
            falsealarm = np.empty(shape=(0, 0))
            correctresp = np.empty(shape=(0, 0))

            #classify hits as realRelRelease times between 0 and 2s
            newdata['hit'] = (newdata['realRelReleaseTimes'] >= 0) & (newdata['realRelReleaseTimes'] <= 2)



            for i in range(0, len(newdata['realRelReleaseTimes'].values)):
                chosenresponseindex = chosenresponse.values[i]

                realrelreleasetime = realrelreleasetimelist.values[i]

                chosentrial = pitchshiftmat.values[i]
                is_all_zero = np.all((chosentrial == 0))
                chosentalker = talkerlist.values[i]

                if is_all_zero or (isinstance(chosentrial, float) and( chosentalker ==2 or chosentalker==1)):
                    control_trial.append(1)
                else:
                    control_trial.append(0)

                if isinstance(chosentrial, float) or is_all_zero:
                    chosentrial = talkermat.values[i].astype(int)
                    intra_trial_roving.append(0)
                else:
                    intra_trial_roving.append(1)

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

                # if ((
                #             chosenresponseindex == 0 or chosenresponseindex == 1) and realrelreleasetime >= 0) or chosenresponseindex == 3 or chosenresponseindex == 7:
                if chosenresponseindex == 5:
                    falsealarm = np.append(falsealarm, 1)
                else:
                    falsealarm = np.append(falsealarm, 0)
                if ((
                            chosenresponseindex == 0 or chosenresponseindex == 1) and realrelreleasetime >= 0) or chosenresponseindex == 3:
                    correctresp = np.append(correctresp, 1)
                else:
                    correctresp = np.append(correctresp, 0)

                targpos = np.where(chosendisttrial == 1)
                try:
                    targpos = int(targpos[0])
                    precur_pos = targpos - 1

                    if chosentrial[targpos] == 8.0:
                        f0_list.append(float(3))
                    elif chosentrial[targpos] == 13.0:
                        f0_list.append(float(1))
                    elif chosentrial[targpos] == 1.0:
                        f0_list.append(float(4))
                    else:
                        f0_list.append(float(chosentrial[targpos]))

                    if np.sum(newdata['dDurs'].values[i][:targpos]) / fs <= newdata['centreRelease'].values[i] - newdata['absentTime'].values[i] or newdata['response'].values[i] == 7:
                        if chosentrial[targpos] == 8.0:
                            pitchoftarg.append(float(3))
                        elif chosentrial[targpos] == 13.0:
                            pitchoftarg.append(float(1))
                        elif chosentrial[targpos] == 1.0:
                            pitchoftarg.append(float(4))
                        else:
                            pitchoftarg.append( float(chosentrial[targpos]))
                    else:
                        pitchoftarg.append(np.nan)

                    if np.sum(newdata['dDurs'].values[i][:precur_pos]) / fs <= newdata['centreRelease'].values[i]- newdata['absentTime'].values[i] or newdata['response'].values[i] == 7:
                        if chosentrial[precur_pos] == 8.0:
                            pitchofprecur.append(float(3))
                        elif chosentrial[precur_pos] == 13.0:
                            pitchofprecur.append(float(1))
                        elif chosentrial[precur_pos] == 1.0:
                            pitchofprecur.append(float(4))
                        else:
                            pitchofprecur.append(float(chosentrial[precur_pos]))
                    else:
                        pitchofprecur.append(np.nan)
                except:
                    pitchoftarg.append(np.nan)
                    pitchofprecur.append(np.nan)
                    f0_list.append(np.nan)

            falsealarm = falsealarm[~np.isnan(falsealarm)]
            correctresp = correctresp[~np.isnan(correctresp)]

            newdata['pitchoftarg'] = pitchoftarg
            newdata['pitchofprecur'] = pitchofprecur

            falsealarm = falsealarm.astype(int)


            newdata['falsealarm'] = falsealarm.tolist()
            newdata['intra_trial_roving'] = intra_trial_roving
            newdata['inter_trial_roving'] = inter_trial_roving
            newdata['control_trial'] = control_trial
            newdata['correctresp'] = correctresp.tolist()
            newdata['distractor_or_fa'] = distractor_or_fa.tolist()
            newdata['talker'] = talkerlist2.tolist()
            newdata['stepval'] = stepval.tolist()
            newdata['timeToTarget'] = newdata['timeToTarget'] / 24414.0625
            newdata['AM'] = newdata['AM'].astype(int)
            newdata['f0'] = f0_list
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

            # newdata = newdata[
            #     (newdata.pitchofprecur == 1) | (newdata.pitchofprecur == 2) | (newdata.pitchofprecur == 3) | (
            #             newdata.pitchofprecur == 4) | (newdata.pitchofprecur == 5) | (newdata.pitchofprecur.isnull())]

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

                if is_all_zero or (isinstance(chosentrial, float) and( (newdata['talker'].values == 2).any() or (newdata['talker'].values == 1).any())):
                    control_trial.append(1)
                else:
                    control_trial.append(0)

                if isinstance(chosentrial, float) or is_all_zero:
                    chosentrial = talkermat.values[i].astype(int)
                    intra_trial_roving.append(0)
                else:
                    intra_trial_roving.append(1)

                chosentalker = talkerlist.values[i]
                print(i)
                # find where talkerlist.values == 3

                if chosentalker == 3 or chosentalker == 5 or chosentalker == 8 or chosentalker == 13:
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
