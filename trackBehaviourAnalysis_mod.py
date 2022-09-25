import click
import instruments
from instruments.io.BehaviourIO import BehaviourDataSet, WeekBehaviourDataSet
from instruments.config import behaviouralDataPath, behaviourOutput
from instruments.behaviouralAnalysis import createWeekBehaviourFigs, reactionTimeAnalysis, outputbehaviordf
import math
import numpy as np


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
    if ferret == 'F1702_Zola':
        ferrData = ferrData.loc[(ferrData.dates != '2021-10-04 10:25:00')]

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

    allData = dataSet._load()

    # for ferret in ferrets:
    ferret = ferrets
    ferrData = allData.loc[allData.ferretname == ferret]
    if ferret == 'F1702_Zola':
        ferrData = ferrData.loc[(ferrData.dates != '2021-10-04 10:25:00')]
    pitchshiftmat = allData['PitchShiftMat']
    precursorlist = allData['distractors']
    pitchoftarg = np.empty(len(pitchshiftmat))
    for i in range(0, len(pitchshiftmat)):
        chosentrial = pitchshiftmat[i]
        chosendisttrial = precursorlist[i]

        targpos = np.where(chosendisttrial == 1)

        if not isinstance(chosentrial, (np.ndarray, np.generic)):
            if math.isnan(chosentrial):
                chosentrial = np.zeros(5)
        try:
            chosentrial = chosentrial[chosentrial != 0]
        except:
            continue
        if chosentrial.size == 0:
            pitchoftarg[i] = 0

        else:
            try:
                pitchoftarg[i] = chosentrial[targpos]
            except:
                pitchoftarg[i] = 0

    allData['pitchoftarg'] = pitchoftarg.tolist()

    return allData

    # ferretFigs = reactionTimeAnalysis(ferrData)
    # dataSet._save(figs=ferretFigs, file_name='reaction_times_{}_{}_{}.pdf'.format(ferret, startdate, finishdate))


# editing to extract different vars from df
# cli.add_command(cli_reaction_time)

if __name__ == '__main__':
    df = get_df_behav(ferrets='F1702_Zola', startdate='04-01-2020', finishdate='04-01-2022')
    cli_reaction_time(ferrets='F1702_Zola', startdate='04-01-2020', finishdate='04-01-2022')
