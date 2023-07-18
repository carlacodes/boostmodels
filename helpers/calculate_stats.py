import pandas as pd
import numpy as np
import math

from instruments.helpers.analysis_helpers import dprime
from scipy.stats import norm

class CalculateStats:
    def dprime(pHit, pFA):

        # Convert to Z scores
        zHit = norm.ppf(pHit)
        zFA = norm.ppf(pFA)

        # Calculate d-prime
        d = zHit - zFA

        return d

    def bias(pHit, pFA):
        zHit = norm.ppf(pHit)
        zFA = norm.ppf(pFA)
        c = -0.5 * (zHit+ zFA)
        return c

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
        currData['realRelReleaseTimes'] = currData['centreRelease'] - currData['targTimes']
        currData['realRelReleaseTimes'] = currData['realRelReleaseTimes'] - currData['absentTime']
        # currData['realRelReleaseTimes'] = currData['realRelReleaseTimes']

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
            stats['timing']['edges2'] = np.arange(-np.ceil(max(abs(dataTimesClean['realRelReleaseTimes']))),
                                                  np.ceil(max(abs(dataTimesClean['realRelReleaseTimes']))) + 0.1, 0.1).round(1)
        else:
            stats['timing']['edges2'] = np.linspace(TargTimeWind[0], TargTimeWind[1], int((TargTimeWind[1] + 0.2) / 0.2))

        stats['timing']['releaseTimeDistribution'] = \
        np.histogram(dataTimesClean['realRelReleaseTimes'] - dataTimesClean['absentTime'], bins=stats['timing']['edges2'])[0]
        timeInds = np.digitize(dataTimesClean['targTimes'], stats['timing']['edges1'])
        stats['timing']['binProb'] = {}
        for i in range(len(stats['timing']['edges1'])):
            tempInds = np.where(timeInds == i)
            stats['timing']['binProb'][i] = np.mean(
                [(k > 0) & (k < 2) for k in dataTimesClean['realRelReleaseTimes'].iloc[tempInds]])

        # Calculate hits, false alarms and correct rejections
        stats['correcTrialFracR'] = correctionTrialFracR
        stats['correcTrialFracL'] = correctionTrialFracL
        stats['correctionTrialFrac'] = correctionTrialFrac

        stats['hits'] = np.mean([(t >= 0 ) & (t <= 2) for t in dataNonCorrectionNonCatch['realRelReleaseTimes']])
        stats['nTrialsForHits'] = len(dataNonCorrectionNonCatch)
        stats['leftHits'] = np.mean([(t >= 0 + absentTime) & (t <= 2) for t in dataLeftNonCorrNonCatch['realRelReleaseTimes']])
        stats['rightHits'] = np.mean(
            [(t >= 0 + absentTime) & (t <= 2) for t in dataRightNonCorrNonCatch['realRelReleaseTimes']])
        stats['firstTalkerHits'] = np.mean([(t >= 0 + absentTime) & (t <= 2) for t in
                                            dataNonCorrectionNonCatch[dataNonCorrectionNonCatch['talker'] == 1][
                                                'realRelReleaseTimes']])
        stats['secondTalkerHits'] = np.mean([(t >= 0 + absentTime) & (t <= 2) for t in
                                             dataNonCorrectionNonCatch[dataNonCorrectionNonCatch['talker'] == 2][
                                                 'realRelReleaseTimes']])

        dataHits = dataNonCorrectionNonCatch[
            (dataNonCorrectionNonCatch['realRelReleaseTimes'] >= 0) & (dataNonCorrectionNonCatch['realRelReleaseTimes'] <= 2)]
        stats['correctLateralisation'] = np.mean(
            [response == side for (response, side) in zip(dataHits['response'], dataHits['side'])])
        stats['lateralhits'] = np.mean([response == side for (response, side) in
                                        zip(dataNonCorrectionNonCatch['response'], dataNonCorrectionNonCatch['side'])])
        stats['bufferhits'] = np.mean([t == np.inf for t in dataCatchNonCorr['realRelReleaseTimes']])
        stats['catchtrialFA'] = np.mean([t != np.inf for t in dataCatchNonCorr['lickRelease']])
        stats['nTrialsForCatchTrialFA'] = len(dataCatchNonCorr)

        stats['allHits'] = np.mean([not (math.isnan(t)) for t in dataNonCorrectionNonCatch['realRelReleaseTimes']])
        stats['allFA'] = np.mean(currData.loc[currData['response'] == 5, 'realRelReleaseTimes'])
        stats['midFA'] = np.mean(
            [(t - absentTime >= -2) & (t - absentTime < 0) for t in dataNonCorrectionNonCatch['realRelReleaseTimes']])
        stats['badhits'] = math.nan
        # stats['firstTalkerCatchFA']=np.mean([(t>=-2)&(t<0) for t in dataCatchNonCorr[dataCatchNonCorr['talker']==1]['realRelReleaseTimes']])
        # stats['secondTalkerCatchFA']=np.mean([(t>=-2)&(t<0) for t in dataCatchNonCorr[dataCatchNonCorr['talker']==2]['realRelReleaseTimes']])
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



        # std error for hits
        stats['stderrorhits'] = np.sqrt((stats['hits'] * (1 - stats['hits'])) / len(dataNonCorrectionNonCatch))
        stats['stderrorlateralhits'] = np.sqrt((stats['lateralhits'] * (1 - stats['lateralhits'])) / len(currData))

        stats['stderrorFA'] = np.sqrt((stats['midFA'] * (1 - stats['midFA'])) / len(dataNonCorrectionNonCatch))
        stats['stderrorcatchtrialFA'] = np.sqrt(
            (stats['catchtrialFA'] * (1 - stats['catchtrialFA'])) / len(dataCatchNonCorr))

        stats['corrRej'] = np.mean([t == np.inf for t in dataCatchNonCorr['realRelReleaseTimes']])

        stats['MidD'] = dprime(stats['hits'], stats['midFA'])
        stats['d'] = dprime(stats['hits'], stats['catchtrialFA'])

        out = [stats, currData['realRelReleaseTimes']]

        if returnhitTrials:
            hitTrials = dataNonCorrectionNonCatch[
                (dataNonCorrectionNonCatch['realRelReleaseTimes'] >= 0) & (dataNonCorrectionNonCatch['realRelReleaseTimes'] <= 2)]
            out.append(hitTrials)

        return (out)


