

class CalculateStats:
    def get_stats(currData,
                 respTimeWind=[],
                 TargTimeWind=[],
                 returnhitTrials=False,
                 fs=24414.062500):
        currData.reset_index(drop=True, inplace=True)
        stats = {}
        dataLeftTrials = currData[currData['side'] == 0]
        dataRightTrials = currData[currData['side'] == 1]

        dataNonCorrectionTrial = currData[currData['correctionTrial'] == 0]
        dataLeftTrialsNonCorrection = dataNonCorrectionTrial[dataNonCorrectionTrial['side'] == 0]
        dataRightTrialsNonCorrection = dataNonCorrectionTrial[dataNonCorrectionTrial['side'] == 1]

        dataCorrectionTrial = currData[currData['correctionTrial'] == 1]
        correctionTrialFracR = (len(dataRightTrials) - len(dataRightTrialsNonCorrection)) / len(dataRightTrials)
        correctionTrialFracL = (len(dataLeftTrials) - len(dataLeftTrialsNonCorrection)) / len(dataLeftTrials)

        correctionTrialFrac = len(dataCorrectionTrial) / len(currData)

        absentTime = currData['absentTime'][0]
        currData['targTimes'] = currData['timeToTarget'] / fs

        currData['centreRelease'] = currData['lickRelease'] - currData['startTrialLick']
        currData['relReleaseTimes'] = currData['centreRelease'] - currData['targTimes']
        currData['realRelReleaseTimes'] = currData['relReleaseTimes'] - currData['absentTime']
        # currData['relReleaseTimes'] = currData['realRelReleaseTimes']

        dataNonCorrectionNonCatch = currData[(currData['correctionTrial'] != 1)
                                             & (currData['catchTrial'] != 1)]

        dataLeftNonCorrNonCatch = dataNonCorrectionNonCatch[dataNonCorrectionNonCatch['side'] == 0]
        dataRightNonCorrNonCatch = dataNonCorrectionNonCatch[dataNonCorrectionNonCatch['side'] == 1]

        dataCatchNonCorr = currData[(currData['correctionTrial'] != 1)
                                    & (currData['catchTrial'] == 1)]

        dataTimesClean = currData[(currData['centreRelease'] != np.inf) & (currData['centreRelease'].notna())]
        dataTimesClean.reset_index(drop=True, inplace=True)

        stats['timing'] = {}
        if not respTimeWind:
            stats['timing']['edges1'] = np.linspace(0, np.ceil(max(dataTimesClean['centreRelease'])),
                                                    int((np.ceil(max(dataTimesClean['centreRelease'])) + 0.2) / 0.2),
                                                    endpoint=True)
        else:
            stats['timing']['edges1'] = np.linspace(respTimeWind[0], respTimeWind[1], int((respTimeWind[1] + 0.2) / 0.2))

        stats['timing']['responseTimeDistribution'] = \
        np.histogram(dataTimesClean['centreRelease'] - dataTimesClean['absentTime'], bins=stats['timing']['edges1'])[0]
        stats['timing']['targetTimeDistribution'] = \
        np.histogram(dataTimesClean['targTimes'], bins=stats['timing']['edges1'])[0]

        if not TargTimeWind:
            stats['timing']['edges2'] = np.arange(-np.ceil(max(abs(dataTimesClean['relReleaseTimes']))),
                                                  np.ceil(max(abs(dataTimesClean['relReleaseTimes']))) + 0.1, 0.1).round(1)
        else:
            stats['timing']['edges2'] = np.linspace(TargTimeWind[0], TargTimeWind[1], int((TargTimeWind[1] + 0.2) / 0.2))

        stats['timing']['releaseTimeDistribution'] = \
        np.histogram(dataTimesClean['relReleaseTimes'] - dataTimesClean['absentTime'], bins=stats['timing']['edges2'])[0]
        timeInds = np.digitize(dataTimesClean['targTimes'], stats['timing']['edges1'])
        stats['timing']['binProb'] = {}
        for i in range(len(stats['timing']['edges1'])):
            tempInds = np.where(timeInds == i)
            stats['timing']['binProb'][i] = np.mean(
                [(k > 0) & (k < 2) for k in dataTimesClean['relReleaseTimes'].iloc[tempInds]])

        # Calculate hits, false alarms and correct rejections
        stats['correcTrialFracR'] = correctionTrialFracR
        stats['correcTrialFracL'] = correctionTrialFracL
        stats['correctionTrialFrac'] = correctionTrialFrac

        stats['hits'] = np.mean([(t >= 0 + absentTime) & (t <= 2) for t in dataNonCorrectionNonCatch['relReleaseTimes']])
        stats['nTrialsForHits'] = len(dataNonCorrectionNonCatch)
        stats['leftHits'] = np.mean([(t >= 0 + absentTime) & (t <= 2) for t in dataLeftNonCorrNonCatch['relReleaseTimes']])
        stats['rightHits'] = np.mean(
            [(t >= 0 + absentTime) & (t <= 2) for t in dataRightNonCorrNonCatch['relReleaseTimes']])
        stats['firstTalkerHits'] = np.mean([(t >= 0 + absentTime) & (t <= 2) for t in
                                            dataNonCorrectionNonCatch[dataNonCorrectionNonCatch['talker'] == 1][
                                                'relReleaseTimes']])
        stats['secondTalkerHits'] = np.mean([(t >= 0 + absentTime) & (t <= 2) for t in
                                             dataNonCorrectionNonCatch[dataNonCorrectionNonCatch['talker'] == 2][
                                                 'relReleaseTimes']])

        dataHits = dataNonCorrectionNonCatch[
            (dataNonCorrectionNonCatch['relReleaseTimes'] >= 0) & (dataNonCorrectionNonCatch['relReleaseTimes'] <= 2)]
        stats['correctLateralisation'] = np.mean(
            [response == side for (response, side) in zip(dataHits['response'], dataHits['side'])])
        stats['lateralhits'] = np.mean([response == side for (response, side) in
                                        zip(dataNonCorrectionNonCatch['response'], dataNonCorrectionNonCatch['side'])])
        stats['bufferhits'] = np.mean([t == np.inf for t in dataCatchNonCorr['relReleaseTimes']])
        stats['catchtrialFA'] = np.mean([t != np.inf for t in dataCatchNonCorr['lickRelease']])
        stats['nTrialsForCatchTrialFA'] = len(dataCatchNonCorr)

        stats['allHits'] = np.mean([not (math.isnan(t)) for t in dataNonCorrectionNonCatch['relReleaseTimes']])
        stats['allFA'] = np.mean(currData.loc[currData['response'] == 5, 'relReleaseTimes'])
        stats['midFA'] = np.mean(
            [(t - absentTime >= -2) & (t - absentTime < 0) for t in dataNonCorrectionNonCatch['relReleaseTimes']])
        stats['badhits'] = math.nan
        # stats['firstTalkerCatchFA']=np.mean([(t>=-2)&(t<0) for t in dataCatchNonCorr[dataCatchNonCorr['talker']==1]['relReleaseTimes']])
        # stats['secondTalkerCatchFA']=np.mean([(t>=-2)&(t<0) for t in dataCatchNonCorr[dataCatchNonCorr['talker']==2]['relReleaseTimes']])
        stats['firstTalkerCatchFA'] = np.mean(
            [t != np.inf for t in dataCatchNonCorr[dataCatchNonCorr['talker'] == 1]['lickRelease']])
        stats['secondTalkerCatchFA'] = np.mean(
            [t != np.inf for t in dataCatchNonCorr[dataCatchNonCorr['talker'] == 2]['lickRelease']])

        # if task with noise, separate trials with and without noise. And separate
        # trials with and without attenuation on distractor words. Then, calculate
        # hit and FA rate for these trials.
        if 'currNoiseAtten' in dataNonCorrectionNonCatch.keys():

            silenceThresh = 60  # Above which attenuation a trial is considered in silence (as opposed to noise)

            noisebreakdown = {}
            for wordAtten in [0, 1]:  # 0: currAtten = 0, 1: currAtten>0
                if wordAtten == 0:
                    atten = 'nonAtten'
                    tempData = dataNonCorrectionNonCatch[dataNonCorrectionNonCatch['currAtten'] == 0]
                    tempDataCatch = dataCatchNonCorr[dataCatchNonCorr['currAtten'] == 0]
                else:
                    atten = 'atten'
                    tempData = dataNonCorrectionNonCatch[dataNonCorrectionNonCatch['currAtten'] > 0]
                    tempDataCatch = dataCatchNonCorr[dataCatchNonCorr['currAtten'] > 0]

                noisebreakdown[atten] = {}
                for noise in np.unique(dataNonCorrectionTrial['noiseType']):
                    noisebreakdown[atten][noise] = {}
                    for noiseAtten in np.unique(dataNonCorrectionTrial['currNoiseAtten']):
                        if noiseAtten < silenceThresh:
                            noisebreakdown[atten][noise][noiseAtten] = {}
                            tempDataNoise = tempData[
                                (tempData['noiseType'] == noise) & (tempData['currNoiseAtten'] == noiseAtten)]
                            tempDataNoiseCatch = tempDataCatch[
                                (tempDataCatch['noiseType'] == noise) & (tempDataCatch['currNoiseAtten'] == noiseAtten)]

                            noisebreakdown[atten][noise][noiseAtten]['hits'] = np.mean(
                                [(t - absentTime >= 0) & (t <= 2) for t in tempDataNoise['relReleaseTimes']])
                            noisebreakdown[atten][noise][noiseAtten]['nbforhits'] = len(tempDataNoise)
                            noisebreakdown[atten][noise][noiseAtten]['catchtrialFA'] = np.mean(
                                [(t - absentTime >= -2) & (t - absentTime < 0) for t in
                                 tempDataNoiseCatch['relReleaseTimes']])
                            noisebreakdown[atten][noise][noiseAtten]['nbforcatchtrialFA'] = len(tempDataNoiseCatch)

                tempdataSilence = tempData[tempData['currNoiseAtten'] >= silenceThresh]
                tempdataSilenceCatch = tempDataCatch[tempDataCatch['currNoiseAtten'] >= silenceThresh]

                noisebreakdown[atten]['Silence'] = {}
                noisebreakdown[atten]['Silence']['hits'] = np.mean(
                    [(t - absentTime >= 0) & (t <= 2) for t in tempdataSilence['relReleaseTimes']])
                noisebreakdown[atten]['Silence']['nbforhits'] = len(tempdataSilence)
                noisebreakdown[atten]['Silence']['catchtrialFA'] = np.mean(
                    [(t - absentTime >= -2) & (t - absentTime < 0) for t in tempdataSilenceCatch['relReleaseTimes']])
                noisebreakdown[atten]['Silence']['nbforcatchtrialFA'] = len(tempdataSilenceCatch)

            stats['noisebreakdown'] = noisebreakdown

            # Should be deleted and replaced by something more elegant
            # But other functions (ExtractByParams for tracking behaviour weekly)
            # are using this messy structure so it will do for now:
            dataNoiseNonAtten = dataNonCorrectionNonCatch[
                (dataNonCorrectionNonCatch['currNoiseAtten'] <= 60) & (dataNonCorrectionNonCatch['currAtten'] == 0)]
            dataNoiseAtten = dataNonCorrectionNonCatch[
                (dataNonCorrectionNonCatch['currNoiseAtten'] <= 60) & (dataNonCorrectionNonCatch['currAtten'] != 0)]
            dataNoiseNonAttenCatch = dataCatchNonCorr[
                (dataCatchNonCorr['currNoiseAtten'] <= 60) & (dataCatchNonCorr['currAtten'] == 0)]
            dataNoiseAttenCatch = dataCatchNonCorr[
                (dataCatchNonCorr['currNoiseAtten'] <= 60) & (dataCatchNonCorr['currAtten'] != 0)]

            dataNonNoiseNonAtten = dataNonCorrectionNonCatch[
                (dataNonCorrectionNonCatch['currNoiseAtten'] > 60) & (dataNonCorrectionNonCatch['currAtten'] == 0)]
            dataNonNoiseAtten = dataNonCorrectionNonCatch[
                (dataNonCorrectionNonCatch['currNoiseAtten'] > 60) & (dataNonCorrectionNonCatch['currAtten'] != 0)]
            dataNonNoiseNonAttenCatch = dataCatchNonCorr[
                (dataCatchNonCorr['currNoiseAtten'] > 60) & (dataCatchNonCorr['currAtten'] == 0)]
            dataNonNoiseAttenCatch = dataCatchNonCorr[
                (dataCatchNonCorr['currNoiseAtten'] > 60) & (dataCatchNonCorr['currAtten'] != 0)]

            stats['hitNoiseNonAtten'] = np.mean([(t >= 0) & (t <= 2) for t in dataNoiseNonAtten['relReleaseTimes']])
            stats['nbNoiseNonAtten'] = len(dataNoiseNonAtten)

            stats['hitNoiseAtten'] = np.mean([(t >= 0) & (t <= 2) for t in dataNoiseAtten['relReleaseTimes']])
            stats['nbNoiseAtten'] = len(dataNoiseAtten)

            stats['hitNonNoiseNonAtten'] = np.mean([(t >= 0) & (t <= 2) for t in dataNonNoiseNonAtten['relReleaseTimes']])
            stats['nbNonNoiseNonAtten'] = len(dataNonNoiseNonAtten)
            stats['FANonNoiseNonAtten'] = np.mean([t != np.inf for t in dataNonNoiseNonAttenCatch['lickRelease']])
            stats['nbCatchNonNoiseNonAtten'] = len(dataNonNoiseNonAttenCatch)

            stats['hitNonNoiseAtten'] = np.mean([(t >= 0) & (t <= 2) for t in dataNonNoiseAtten['relReleaseTimes']])
            stats['nbNonNoiseAtten'] = len(dataNonNoiseAtten)
            stats['FANonNoiseAtten'] = np.mean([t != np.inf for t in dataNonNoiseAttenCatch['lickRelease']])
            stats['nbCatchNonNoiseAtten'] = len(dataNonNoiseAttenCatch)

            # dataWhiteNoiseNonAtten = dataNoiseNonAtten[dataNoiseNonAtten['noiseType']=='WhiteNoise']
            # dataWhiteNoiseAtten = dataNoiseAtten[dataNoiseAtten['noiseType']=='WhiteNoise']
            # dataWhiteNoiseNonAttenCatch = dataNoiseNonAttenCatch[dataNoiseNonAttenCatch['noiseType']=='WhiteNoise']
            # dataWhiteNoiseAttenCatch = dataNoiseAttenCatch[dataNoiseAttenCatch['noiseType']=='WhiteNoise']

            # stats['hitWhiteNoiseNonAtten']=np.mean([(t>=0)&(t<=2) for t in dataWhiteNoiseNonAtten['relReleaseTimes']])
            # stats['nbWhiteNoiseNonAtten']=len(dataWhiteNoiseNonAtten)
            # stats['FAWhiteNoiseNonAtten']=np.mean([t!=np.inf for t in dataWhiteNoiseNonAttenCatch['lickRelease']])
            # stats['nbCatchWhiteNoiseNonAtten']=len(dataWhiteNoiseNonAttenCatch)

            # stats['hitWhiteNoiseAtten']=np.mean([(t>=0)&(t<=2) for t in dataWhiteNoiseAtten['relReleaseTimes']])
            # stats['nbWhiteNoiseAtten']=len(dataWhiteNoiseAtten)
            # stats['FAWhiteNoiseAtten']=np.mean([t!=np.inf for t in dataWhiteNoiseAttenCatch['lickRelease']])
            # stats['nbCatchWhiteNoiseAtten']=len(dataWhiteNoiseAttenCatch)

            # dataPinkNoiseNonAtten = dataNoiseNonAtten[dataNoiseNonAtten['noiseType']=='PinkNoise']
            # dataPinkNoiseAtten = dataNoiseAtten[dataNoiseAtten['noiseType']=='PinkNoise']
            # dataPinkNoiseNonAttenCatch = dataNoiseNonAttenCatch[dataNoiseNonAttenCatch['noiseType']=='PinkNoise']
            # dataPinkNoiseAttenCatch = dataNoiseAttenCatch[dataNoiseAttenCatch['noiseType']=='PinkNoise']

            # stats['hitPinkNoiseNonAtten']=np.mean([(t>=0)&(t<=2) for t in dataPinkNoiseNonAtten['relReleaseTimes']])
            # stats['nbPinkNoiseNonAtten']=len(dataPinkNoiseNonAtten)
            # stats['FAPinkNoiseNonAtten']=np.mean([t!=np.inf for t in dataPinkNoiseNonAttenCatch['lickRelease']])
            # stats['nbCatchPinkNoiseNonAtten']=len(dataPinkNoiseNonAttenCatch)

            # stats['hitPinkNoiseAtten']=np.mean([(t>=0)&(t<=2) for t in dataPinkNoiseAtten['relReleaseTimes']])
            # stats['nbPinkNoiseAtten']=len(dataPinkNoiseAtten)
            # stats['FAPinkNoiseAtten']=np.mean([t!=np.inf for t in dataPinkNoiseAttenCatch['lickRelease']])
            # stats['nbCatchPinkNoiseAtten']=len(dataPinkNoiseAttenCatch)

            # dataSSNNonAtten = dataNoiseNonAtten[dataNoiseNonAtten['noiseType']=='SSN']
            # dataSSNAtten = dataNoiseAtten[dataNoiseAtten['noiseType']=='SSN']
            # dataSSNNonAttenCatch = dataNoiseNonAttenCatch[dataNoiseNonAttenCatch['noiseType']=='SSN']
            # dataSSNAttenCatch = dataNoiseAttenCatch[dataNoiseAttenCatch['noiseType']=='SSN']

            # stats['hitSSNNonAtten']=np.mean([(t>=0)&(t<=2) for t in dataSSNNonAtten['relReleaseTimes']])
            # stats['nbSSNNonAtten']=len(dataSSNNonAtten)
            # stats['FASSNNonAtten']=np.mean([t!=np.inf for t in dataSSNNonAttenCatch['lickRelease']])
            # stats['nbCatchSSNNonAtten']=len(dataSSNNonAttenCatch)

            # stats['hitSSNAtten']=np.mean([(t>=0)&(t<=2) for t in dataSSNAtten['relReleaseTimes']])
            # stats['nbSSNAtten']=len(dataSSNAtten)
            # stats['FASSNAtten']=np.mean([t!=np.inf for t in dataSSNAttenCatch['lickRelease']])
            # stats['nbCatchSSNAtten']=len(dataSSNAttenCatch)

        # std error for hits
        stats['stderrorhits'] = np.sqrt((stats['hits'] * (1 - stats['hits'])) / len(dataNonCorrectionNonCatch))
        stats['stderrorlateralhits'] = np.sqrt((stats['lateralhits'] * (1 - stats['lateralhits'])) / len(currData))

        stats['stderrorFA'] = np.sqrt((stats['midFA'] * (1 - stats['midFA'])) / len(dataNonCorrectionNonCatch))
        stats['stderrorcatchtrialFA'] = np.sqrt(
            (stats['catchtrialFA'] * (1 - stats['catchtrialFA'])) / len(dataCatchNonCorr))

        stats['corrRej'] = np.mean([t == np.inf for t in dataCatchNonCorr['relReleaseTimes']])

        # stats['d'] = dprime(stats['hits'],stats['allFA'])
        stats['MidD'] = dprime(stats['hits'], stats['midFA'])
        stats['d'] = dprime(stats['hits'], stats['catchtrialFA'])
        # stats['alld'] = dprime(stats['allHits'],stats['allFA'])

        out = [stats, currData['relReleaseTimes']]

        if returnhitTrials:
            hitTrials = dataNonCorrectionNonCatch[
                (dataNonCorrectionNonCatch['relReleaseTimes'] >= 0) & (dataNonCorrectionNonCatch['relReleaseTimes'] <= 2)]
            out.append(hitTrials)

        return (out)

