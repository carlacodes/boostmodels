
from instruments.config import behaviouralDataPath, behaviourOutput
from sklearn.preprocessing import MinMaxScaler
from instruments.helpers.extract_helpers import extractAllFerretData
import pandas as pd
import numpy as np
import rpy2.robjects.numpy2ri
import pandas as pd


class behaviouralhelperscg:
    def get_false_alarm_behavdata(path=None,
                     output=None,
                     ferrets=None,
                     includefaandmiss=False,
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
            # newdata = allData.iloc(allData['ferret'] == ferret)
            newdata = allData[allData['ferret'] == ferret]
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

            precursorlist = newdata['distractors']
            catchtriallist = newdata['catchTrial']
            chosenresponse = newdata['response']
            realrelreleasetimelist = newdata['realRelReleaseTimes']
            pitchoftarg = np.empty(len(pitchshiftmat))
            pitchofprecur = np.empty(len(pitchshiftmat))
            stepval = np.empty(len(pitchshiftmat))
            gradinpitch = np.empty(len(pitchshiftmat))
            gradinpitchprecur = np.empty(len(pitchshiftmat))
            timetotarglist = np.empty(len(pitchshiftmat))

            precur_and_targ_same = np.empty(len(pitchshiftmat))
            talkerlist2 = np.empty(len(pitchshiftmat))

            correctresp = np.empty(shape=(0, 0))
            pastcorrectresp = np.empty(shape=(0, 0))
            pastcatchtrial = np.empty(shape=(0, 0))
            droplist = np.empty(shape=(0, 0))
            droplistnew = np.empty(shape=(0, 0))
            print(len(newdata['realRelReleaseTimes'].values))

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
                    ##arrays START AT 0, but the index starts at 1, so the index is 1 less than the array
                    droplistnew = np.append(droplistnew, indexdrop)
                    continue
            # newdata.drop(0, axis=0, inplace=True)  # drop first trial for each animal
            # accidentally dropping all catch trials?
            ##TODO: CHECK THIS
            newdata.drop(index=newdata.index[0],
                         axis=0,
                         inplace=True)
            newdata.drop(droplistnew, axis=0, inplace=True)
            droplist = [int(x) for x in droplist]  # drop corrupted metdata trials

            pitchoftarg = pitchoftarg.astype(int)
            pitchofprecur = pitchofprecur.astype(int)
            correctresp = correctresp[~np.isnan(correctresp)]

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
            precur_and_targ_same = precur_and_targ_same.astype(int)
            newdata['precur_and_targ_same'] = precur_and_targ_same.tolist()
            newdata['timeToTarget'] = newdata['timeToTarget'] / 24414.0625
            newdata['AM'] = newdata['AM'].astype(int)
            newdata['talker'] = newdata['talker'] - 1

            newdata = newdata[(newdata.pitchoftarg == 1) | (newdata.pitchoftarg == 2) | (newdata.pitchoftarg == 3) | (
                    newdata.pitchoftarg == 4) | (newdata.pitchoftarg == 5)]
            newdata = newdata[
                (newdata.pitchofprecur == 1) | (newdata.pitchofprecur == 2) | (newdata.pitchofprecur == 3) | (
                        newdata.pitchofprecur == 4) | (newdata.pitchofprecur == 5)]

            newdata = newdata[(newdata.correctionTrial == 0)]  # | (allData.response == 7)
            newdata = newdata[(newdata.currAtten == 0)]  # | (allData.response == 7)
            # if includefaandmiss is True:
            #     newdata = newdata[(newdata.response == 0) | (newdata.response == 1) | (newdata.response == 7)]
            # else:
            #     newdata = newdata[(newdata.response == 0) | (newdata.response == 1)]
            #     newdata = newdata[(newdata.catchTrial == 0)]
            bigdata = bigdata.append(newdata)
        return bigdata