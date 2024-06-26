import scikit_posthocs as sp
from statsmodels.stats.multicomp import MultiComparison
from matplotlib.colors import ListedColormap
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
import scipy.stats as stats
import matplotlib.pyplot as plt
from helpers.behaviouralhelpersformodels import *
from helpers.calculate_stats import *


def kw_test(df):
    '''run a kruskal wallis test on the data
    :param df: dataframe
    :return: kw_dict'''

    conditions = ['inter_trial_roving', 'intra_trial_roving', 'control_trial']
    columns_to_compare = ['realRelReleaseTimes']
    # Create an empty dictionary to store results
    kw_dict = {}
    # Iterate over each unique talker
    for talker in df['talker'].unique():
        kw_dict[talker] = {}

        # Get data for the current talker
        data_talker_whole = df[df['talker'] == talker]

        # Perform Kruskal-Wallis test for each column and conditions
        for column in columns_to_compare:
            kw_dict[talker][column] = {}
            for ferret in df['ferret'].unique():
                kw_dict[talker][column][ferret] = {}
                groups = []

                data_talker = data_talker_whole[data_talker_whole['ferret'] == ferret]

                for condition in conditions:
                    if column == 'hit' or column == 'realRelReleaseTimes':
                        data_condition = data_talker[(data_talker[condition] == 1) & (data_talker['catchTrial'] != 1)][column]
                    elif column == 'falsealarm':
                        data_condition = data_talker[(data_talker[condition] == 1) & (data_talker['catchTrial'] != 0)][column]
                    else:
                        data_condition = data_talker[data_talker[condition] == 1][column]
                    if column == 'hit' or column == 'realRelReleaseTimes':
                        data_condition = data_condition.dropna()
                    groups.append(data_condition)

                # Perform Kruskal-Wallis test for the current column and conditions
                kw_stat, kw_p_value = stats.kruskal(*groups)
                dunn_results = None
                if kw_p_value < 0.05:  # Assuming a significance level of 0.05
                    # Perform post hoc test (Dunn's test)
                    dunn_results = sp.posthoc_dunn(groups,
                                                   p_adjust='bonferroni')  # You can choose a different p-adjust method

                    # Print Dunn's test results
                    print("Dunn's test results:")
                    print(dunn_results)


                data_total = np.concatenate(groups)

                k = len(groups)
                n = len(data_total)

                # Calculate Eta-squared effect size
                eta_squared = (kw_stat - k + 1) / (n - k)
                kw_dict[talker][column][ferret]['kw_stat'] = kw_stat
                kw_dict[talker][column][ferret]['p_value'] = kw_p_value
                kw_dict[talker][column][ferret]['effect_size'] = eta_squared
                kw_dict[talker][column][ferret]['dunn_result'] = dunn_results

    kw_dict_all_df = pd.DataFrame.from_dict(kw_dict)
    kw_dict_all_df.to_csv(f'D:\mixedeffectmodelsbehavioural\metrics/kw_dict_byrovetype.csv')
    return kw_dict


def run_stats_calc(df, ferrets, pitch_param = 'control_trial'):
    '''run stats calculations on the dataframe
    :param df: dataframe
    :param ferrets: list of ferrets
    :param pitch_param: pitch parameter
    :return: stats_dict_all, stats_dict'''

    df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    df_noncorrection = df[(df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    talkers = [1,2]
    stats_dict = {}
    stats_dict[1] = {}
    stats_dict[2] = {}
    stats_dict[1][pitch_param] = {}
    stats_dict[2][pitch_param] = {}

    stats_dict[1][pitch_param]['hits'] = {}
    stats_dict[1][pitch_param]['false_alarms']= {}
    stats_dict[1][pitch_param]['false_alarms_catch']= {}

    stats_dict[1][pitch_param]['correct_response']= {}
    stats_dict[1][pitch_param]['dprime']= {}
    stats_dict[1][pitch_param]['bias']= {}


    stats_dict[2][pitch_param]['hits'] ={}
    stats_dict[2][pitch_param]['false_alarms'] = {}
    stats_dict[2][pitch_param]['false_alarms_catch']= {}

    stats_dict[2][pitch_param]['correct_response'] = {}
    stats_dict[2][pitch_param]['dprime']= {}
    stats_dict[2][pitch_param]['bias']= {}

    count = 0
    for ferret in ferrets:

        selected_ferret = df_noncatchnoncorrection[df_noncatchnoncorrection['ferret'] == count]
        selected_ferret_catch = df_catchnoncorrection[df_catchnoncorrection['ferret'] == count]
        selected_ferret_all = df_noncorrection[df_noncorrection['ferret'] == count]

        for talker in talkers:
            selected_ferret_talker = selected_ferret[selected_ferret['talker'] == talker]
            selected_ferret_all_talker = selected_ferret_all[selected_ferret_all['talker'] == talker]

            selected_ferret_talker_hitrate = selected_ferret_talker[selected_ferret_talker['response'] != 5]

            selected_ferret_catch_talker = selected_ferret_catch[selected_ferret_catch['talker'] == talker]

            stats_dict[talker][pitch_param]['hits'][ferret] = np.mean(selected_ferret_talker_hitrate['hit'])
            stats_dict[talker][pitch_param]['false_alarms'][ferret] = np.mean(selected_ferret_all_talker['falsealarm'])
            stats_dict[talker][pitch_param]['false_alarms_catch'][ferret] = np.mean(selected_ferret_catch_talker['falsealarm'])

            stats_dict[talker][pitch_param]['dprime'][ferret] = CalculateStats.dprime(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
            stats_dict[talker][pitch_param]['bias'][ferret] = CalculateStats.bias(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
            #%Correct(hit + CR / hits + misses + CR + FA)
            stats_dict[talker][pitch_param]['correct_response'][ferret] = (len(selected_ferret_talker[selected_ferret_talker['hit']==True]) + len(selected_ferret_catch_talker[selected_ferret_catch_talker['response'] == 3]))/ (len(selected_ferret_talker) + len(selected_ferret_catch_talker))
        count += 1
    stats_dict_all = {}
    stats_dict_all[1] ={}
    stats_dict_all[2] ={}

    stats_dict_all[1][pitch_param]= {}
    stats_dict_all[2][pitch_param]= {}

    for talker in talkers:
        df_noncatchnoncorrection_talker = df_noncatchnoncorrection[df_noncatchnoncorrection['talker'] == talker]
        df_noncorrection_talker = df_noncorrection[df_noncorrection['talker'] == talker]
        df_noncatchnoncorrection_talker_hitrate = df_noncatchnoncorrection_talker[df_noncatchnoncorrection_talker['response'] != 5]
        df_catchnoncorrection_talker = df_catchnoncorrection[df_catchnoncorrection['talker'] == talker]

        correct_response = np.mean(list(stats_dict[talker][pitch_param]['correct_response'].values()))
        hits = np.mean(list(stats_dict[talker][pitch_param]['hits'].values()))
        false_alarms = np.mean(list(stats_dict[talker][pitch_param]['false_alarms'].values()))
        false_alarms_catch = np.mean(list(stats_dict[talker][pitch_param]['false_alarms_catch'].values()))

        stats_dict_all[talker][pitch_param]['hits'] = hits
        stats_dict_all[talker][pitch_param]['false_alarms'] = false_alarms
        stats_dict_all[talker][pitch_param]['false_alarms_catch'] = false_alarms_catch

        stats_dict_all[talker][pitch_param]['correct_response'] = correct_response
        stats_dict_all[talker][pitch_param]['dprime'] = CalculateStats.dprime(hits, false_alarms)
        stats_dict_all[talker][pitch_param]['bias'] = CalculateStats.bias(hits, false_alarms)

    return stats_dict_all, stats_dict
# def run_stats_calc_by_pitch(df, ferrets, stats_dict, pitch_param = 'control_trial'):
#     '''run stats calculations on the dataframe separated by pitch / F0
#     :param df: dataframe
#     :param ferrets: list of ferrets
#     :param pitch_param: pitch parameter
#     :return: stats_dict_all, stats_dict'''
#
#     df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
#     df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
#     df_noncorrection = df[(df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
#     stats_dict[1] = {}
#     stats_dict[2] = {}
#     stats_dict[3] = {}
#     stats_dict[4] = {}
#     stats_dict[5] = {}
#
#     stats_dict[1]['hits'] = {}
#     stats_dict[1]['false_alarms']= {}
#     stats_dict[1]['correct_response']= {}
#     stats_dict[1]['dprime']= {}
#     stats_dict[1]['bias']= {}
#
#
#     stats_dict[2]['hits'] ={}
#     stats_dict[2]['false_alarms'] = {}
#     stats_dict[2]['correct_response'] = {}
#     stats_dict[2]['dprime']= {}
#     stats_dict[2]['bias']= {}
#
#     stats_dict[3]['hits'] ={}
#     stats_dict[3]['false_alarms'] = {}
#     stats_dict[3]['correct_response'] = {}
#     stats_dict[3]['dprime']= {}
#     stats_dict[3]['bias']= {}
#
#     stats_dict[4]['hits'] ={}
#     stats_dict[4]['false_alarms'] = {}
#     stats_dict[4]['correct_response'] = {}
#     stats_dict[4]['dprime']= {}
#     stats_dict[4]['bias']= {}
#
#     stats_dict[5]['hits'] ={}
#     stats_dict[5]['false_alarms'] = {}
#     stats_dict[5]['correct_response'] = {}
#     stats_dict[5]['dprime']= {}
#     stats_dict[5]['bias']= {}
#
#
#     count = 0
#     pitch_list = [1,2,3,4,5]
#     for ferret in ferrets:
#
#         selected_ferret = df_noncatchnoncorrection[df_noncatchnoncorrection['ferret'] == count]
#         selected_ferret_catch = df_catchnoncorrection[df_catchnoncorrection['ferret'] == count]
#         selected_ferret_all = df_noncorrection[df_noncorrection['ferret'] == count]
#
#         for pitch in pitch_list:
#             selected_ferret_talker = selected_ferret[selected_ferret['pitchoftarg'] == pitch]
#             selected_ferret_all_talker = selected_ferret_all[selected_ferret_all['pitchoftarg'] == pitch]
#
#             selected_ferret_talker_hitrate = selected_ferret_talker[selected_ferret_talker['response'] != 5]
#
#             selected_ferret_catch_talker = selected_ferret_catch[selected_ferret_catch['pitchoftarg'] == pitch]
#
#             stats_dict[pitch]['hits'][ferret] = np.mean(selected_ferret_talker_hitrate['hit'])
#             stats_dict[pitch]['false_alarms'][ferret] = np.mean(selected_ferret_all_talker['falsealarm'])
#             stats_dict[pitch]['dprime'][ferret] = CalculateStats.dprime(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
#             stats_dict[pitch]['bias'][ferret] = CalculateStats.bias(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
#             #%Correct(hit + CR / hits + misses + CR + FA)
#             stats_dict[pitch]['correct_response'][ferret] = (len(selected_ferret_talker[selected_ferret_talker['hit']==True]) + len(selected_ferret_catch_talker[selected_ferret_catch_talker['response'] == 3]))/ (len(selected_ferret_talker) + len(selected_ferret_catch_talker))
#         count += 1
#     stats_dict_all = {}
#
#     stats_dict_all[1]= {}
#     stats_dict_all[2]= {}
#     stats_dict_all[3]= {}
#     stats_dict_all[4]= {}
#     stats_dict_all[5]= {}
#
#
#     for pitch in pitch_list:
#         df_noncatchnoncorrection_talker = df_noncatchnoncorrection[df_noncatchnoncorrection['pitchoftarg'] == pitch]
#         df_noncorrection_talker = df_noncorrection[df_noncorrection['pitchoftarg'] == pitch]
#         df_noncatchnoncorrection_talker_hitrate = df_noncatchnoncorrection_talker[df_noncatchnoncorrection_talker['response'] != 5]
#
#         df_catchnoncorrection_talker = df_catchnoncorrection[df_catchnoncorrection['pitchoftarg'] == pitch]
#
#         correct_response = np.mean(list(stats_dict[pitch]['correct_response'].values()))
#         hits = np.mean(list(stats_dict[pitch]['hits'].values()))
#         false_alarms = np.mean(list(stats_dict[pitch]['false_alarms'].values()))
#
#         stats_dict_all[pitch]['hits'] = hits
#         stats_dict_all[pitch]['false_alarms'] = false_alarms
#         stats_dict_all[pitch]['correct_response'] = correct_response
#         stats_dict_all[pitch]['dprime'] = CalculateStats.dprime(hits, false_alarms)
#         stats_dict_all[pitch]['bias'] = CalculateStats.bias(hits, false_alarms)
#
#     return stats_dict_all, stats_dict

def run_stats_calc_by_pitch_mf(df, ferrets, stats_dict, pitch_param = 'inter_trial_roving'):
    ''' run stats calculations on the dataframe separated by pitch type and talker
    :param df: dataframe
    :param ferrets: list of ferrets
    :param pitch_param: pitch parameter
    :return: stats_dict_all, stats_dict'''
    if pitch_param == None:
        df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0)]
        df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0)]
        df_noncorrection = df[(df['correctionTrial'] == 0)]

    else:

        df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0) & ((df[pitch_param] == 1) |(df['control_trial'] == 1))]
        df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0) & ((df[pitch_param] == 1) |(df['control_trial'] == 1))]
        df_noncorrection = df[(df['correctionTrial'] == 0) & ((df[pitch_param] == 1) | (df['control_trial'] == 1))]
    stats_dict[1] = {}
    stats_dict[2] = {}
    stats_dict[3] = {}
    stats_dict[4] = {}
    stats_dict[5] = {}
    stats_dict[6] = {}



    stats_dict[1]['hits'] = {}
    stats_dict[1]['false_alarms']= {}
    stats_dict[1]['correct_response']= {}
    stats_dict[1]['dprime']= {}
    stats_dict[1]['bias']= {}


    stats_dict[2]['hits'] ={}
    stats_dict[2]['false_alarms'] = {}
    stats_dict[2]['correct_response'] = {}
    stats_dict[2]['dprime']= {}
    stats_dict[2]['bias'] = {}

    stats_dict[3]['hits'] ={}

    stats_dict[3]['false_alarms'] = {}
    stats_dict[3]['correct_response'] = {}
    stats_dict[3]['dprime']= {}
    stats_dict[3]['bias'] = {}

    stats_dict[4]['hits'] ={}
    stats_dict[4]['false_alarms'] = {}
    stats_dict[4]['correct_response'] = {}
    stats_dict[4]['dprime']= {}
    stats_dict[4]['bias'] = {}

    stats_dict[5]['hits'] ={}
    stats_dict[5]['false_alarms'] = {}
    stats_dict[5]['correct_response'] = {}
    stats_dict[5]['dprime']= {}
    stats_dict[5]['bias'] = {}
    stats_dict[6]['hits'] ={}
    stats_dict[6]['false_alarms'] = {}
    stats_dict[6]['correct_response'] = {}
    stats_dict[6]['dprime']= {}
    stats_dict[6]['bias'] = {}

    count = 0
    pitch_list = [1,2,3,3,4,5]
    for ferret in ferrets:

        selected_ferret = df_noncatchnoncorrection[df_noncatchnoncorrection['ferret'] == count]
        selected_ferret_catch = df_catchnoncorrection[df_catchnoncorrection['ferret'] == count]
        selected_ferret_all = df_noncorrection[df_noncorrection['ferret'] == count]

        for i, pitch in enumerate(pitch_list):
            if i ==2:
                selected_ferret_talker = selected_ferret[selected_ferret['pitchoftarg'] == pitch]
                selected_ferret_talker = selected_ferret_talker[selected_ferret_talker['talker'] == 2]


                selected_ferret_all_talker = selected_ferret_all[selected_ferret_all['f0'] == 3]
                selected_ferret_all_talker = selected_ferret_all_talker[selected_ferret_all_talker['talker'] == 2]
            elif i == 3:
                selected_ferret_talker = selected_ferret[selected_ferret['pitchoftarg'] == 3]
                selected_ferret_talker = selected_ferret_talker[selected_ferret_talker['talker'] == 1]


                selected_ferret_all_talker = selected_ferret_all[selected_ferret_all['f0'] == 3]
                selected_ferret_all_talker = selected_ferret_all_talker[selected_ferret_all_talker['talker'] == 1]
            else:
                selected_ferret_talker = selected_ferret[selected_ferret['f0'] == pitch]

                selected_ferret_all_talker = selected_ferret_all[selected_ferret_all['f0'] == pitch]

            selected_ferret_talker_hitrate = selected_ferret_talker[selected_ferret_talker['response'] != 5]

            selected_ferret_catch_talker = selected_ferret_catch[selected_ferret_catch['pitchoftarg'] == pitch]

            stats_dict[i+1]['hits'][ferret] = np.mean(selected_ferret_talker_hitrate['hit'])
            stats_dict[i+1]['false_alarms'][ferret] = np.mean(selected_ferret_all_talker['falsealarm'])
            stats_dict[i+1]['dprime'][ferret] = CalculateStats.dprime(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
            stats_dict[i+1]['bias'][ferret] = CalculateStats.bias(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
            #%Correct(hit + CR / hits + misses + CR + FA)
            stats_dict[i+1]['correct_response'][ferret] = (len(selected_ferret_talker[selected_ferret_talker['hit']==True]) + len(selected_ferret_catch_talker[selected_ferret_catch_talker['response'] == 3]))/ (len(selected_ferret_talker) + len(selected_ferret_catch_talker))
        count += 1
    stats_dict_all = {}
    stats_dict_all[1]= {}
    stats_dict_all[2]= {}
    stats_dict_all[3]= {}
    stats_dict_all[4]= {}
    stats_dict_all[5]= {}
    stats_dict_all[6] = {}

    for i, pitch in enumerate(pitch_list):
        correct_response = np.mean(list(stats_dict[i+1]['correct_response'].values()))
        hits = np.mean(list(stats_dict[i+1]['hits'].values()))
        false_alarms = np.mean(list(stats_dict[i+1]['false_alarms'].values()))
        stats_dict_all[i+1]['hits'] = hits
        stats_dict_all[i+1]['false_alarms'] = false_alarms
        stats_dict_all[i+1]['correct_response'] = correct_response
        stats_dict_all[i+1]['dprime'] = CalculateStats.dprime(hits, false_alarms)
        stats_dict_all[i+1]['bias'] = CalculateStats.bias(hits, false_alarms)

    return stats_dict_all, stats_dict
def run_stats_calc_by_pitch(df, ferrets, stats_dict, pitch_param = 'inter_trial_roving', kw_test = True):
    ''' run stats calculations on the dataframe separated by pitch type and talker
    :param df: dataframe
    :param ferrets: list of ferrets
    :param pitch_param: pitch parameter
    :return: stats_dict_all, stats_dict'''

    if pitch_param == None:
        df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0)]
        df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0)]
        df_noncorrection = df[(df['correctionTrial'] == 0)]
    else:
        df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0) & ((df[pitch_param] == 1) |(df['control_trial'] == 1))]
        df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0) & ((df[pitch_param] == 1) |(df['control_trial'] == 1))]
        df_noncorrection = df[(df['correctionTrial'] == 0) & ((df[pitch_param] == 1) | (df['control_trial'] == 1))]

    stats_dict[1] = {}
    stats_dict[2] = {}
    stats_dict[3] = {}
    stats_dict[4] = {}
    stats_dict[5] = {}

    stats_dict[1]['hits'] = {}
    stats_dict[1]['false_alarms']= {}
    stats_dict[1]['false_alarms_catch']= {}

    stats_dict[1]['correct_response']= {}

    stats_dict[1]['dprime']= {}
    stats_dict[1]['bias']= {}


    stats_dict[2]['hits'] ={}
    stats_dict[2]['false_alarms'] = {}
    stats_dict[2]['false_alarms_catch']= {}

    stats_dict[2]['correct_response'] = {}
    stats_dict[2]['dprime']= {}
    stats_dict[2]['bias'] = {}

    stats_dict[3]['hits'] ={}

    stats_dict[3]['false_alarms'] = {}
    stats_dict[3]['false_alarms_catch']= {}

    stats_dict[3]['correct_response'] = {}
    stats_dict[3]['dprime']= {}
    stats_dict[3]['bias'] = {}

    stats_dict[4]['hits'] ={}
    stats_dict[4]['false_alarms'] = {}
    stats_dict[4]['false_alarms_catch']= {}

    stats_dict[4]['correct_response'] = {}
    stats_dict[4]['dprime']= {}
    stats_dict[4]['bias'] = {}

    stats_dict[5]['hits'] ={}
    stats_dict[5]['false_alarms'] = {}
    stats_dict[5]['false_alarms_catch']= {}
    stats_dict[5]['correct_response'] = {}
    stats_dict[5]['dprime']= {}
    stats_dict[5]['bias'] = {}

    count = 0
    pitch_list = [1,2,3,4,5]
    data_dict = {}
    for i in pitch_list:
        data_dict[i] = {}
        data_dict[i] = df_noncorrection[df_noncorrection['f0'] == i]
    kw_dict_all = {}
    kw_dict_all[1] = {}
    kw_dict_all[2] = {}
    kw_dict_all[3] = {}
    kw_dict_all[4] = {}
    kw_dict_all[5] = {}

    rm_anova_dataframe = pd.DataFrame(columns=['pitch', 'ferret', 'hits', 'false_alarms', 'correct_response'])
    for ferret in ferrets:
        selected_ferret = df_noncatchnoncorrection[df_noncatchnoncorrection['ferret'] == count]
        selected_ferret_catch = df_catchnoncorrection[df_catchnoncorrection['ferret'] == count]
        selected_ferret_all = df_noncorrection[df_noncorrection['ferret'] == count]
        for pitch in pitch_list:
            selected_ferret_talker = selected_ferret[selected_ferret['pitchoftarg'] == pitch]
            selected_ferret_all_talker = selected_ferret_all[selected_ferret_all['f0'] == pitch]

            selected_ferret_talker_hitrate = selected_ferret_talker[selected_ferret_talker['response'] != 5]

            selected_ferret_catch_talker = selected_ferret_catch[selected_ferret_catch['pitchoftarg'] == pitch]
            stats_dict[pitch]['hits'][ferret] = np.mean(selected_ferret_talker_hitrate['hit'])
            # kw_dict_all[pitch]['hits'] = list(selected_ferret_talker_hitrate['hit'])
            stats_dict[pitch]['false_alarms'][ferret] = np.mean(selected_ferret_all_talker['falsealarm'])
            stats_dict[pitch]['false_alarms_catch'][ferret] = np.mean(selected_ferret_catch_talker['falsealarm'])


            stats_dict[pitch]['dprime'][ferret] = CalculateStats.dprime(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
            stats_dict[pitch]['bias'][ferret] = CalculateStats.bias(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
            #%Correct(hit + CR / hits + misses + CR + FA)
            stats_dict[pitch]['correct_response'][ferret] = (len(selected_ferret_talker[selected_ferret_talker['hit']==True]) + len(selected_ferret_catch_talker[selected_ferret_catch_talker['response'] == 3]))/ (len(selected_ferret_talker) + len(selected_ferret_catch_talker))

            rm_anova_dataframe = rm_anova_dataframe.append({'pitch': pitch, 'ferret': ferret, 'hits':  np.mean(selected_ferret_talker_hitrate['hit']), 'false_alarms': np.mean(selected_ferret_all_talker['falsealarm']), 'correct_response': stats_dict[pitch]['correct_response'][ferret]}, ignore_index=True
                                      )
        count += 1
    anovaresults = {}
    for value in ['hits', 'false_alarms', 'correct_response']:
        rm = AnovaRM(rm_anova_dataframe, value, 'ferret', within=['pitch'])
        res = rm.fit()
        #store res in a dictionary
        anovaresults[value] = res
        #run tukey post hoc test
        print(res)
        #export to csv

        numerator = res.anova_table['F Value'][0] * res.anova_table['Num DF'][0]
        denominator = numerator + res.anova_table['Den DF'][0]
        partial_eta_squared = numerator / denominator

        res.anova_table.to_csv(f'D:\mixedeffectmodelsbehavioural\metrics/anova_results_bypitch_{value}.csv')
        #export eta_squared to csv
        eta_squared_df = pd.DataFrame({'partial_eta_squared': [partial_eta_squared]})
        eta_squared_df.to_csv(f'D:\mixedeffectmodelsbehavioural\metrics/eta_squared_bypitch_{value}.csv')
        mc = MultiComparison(rm_anova_dataframe[value], rm_anova_dataframe['pitch'])
        result = mc.tukeyhsd()
        #get the eta squared value


        # Print post hoc test results
        print(result)

        # Export post hoc test results to CSV
        posthoc_df = pd.DataFrame(result._results_table.data[1:], columns=result._results_table.data[0])
        posthoc_df.to_csv(f'D:/mixedeffectmodelsbehavioural/metrics/posthoc_results_{value}.csv')

    stats_dict_all = {}
    stats_dict_all[1]= {}
    stats_dict_all[2]= {}
    stats_dict_all[3]= {}
    stats_dict_all[4]= {}
    stats_dict_all[5]= {}

    for pitch in pitch_list:
        df_noncatchnoncorrection_talker = df_noncatchnoncorrection[df_noncatchnoncorrection['pitchoftarg'] == pitch]
        df_noncorrection_talker = df_noncorrection[df_noncorrection['pitchoftarg'] == pitch]
        df_noncatchnoncorrection_talker_hitrate = df_noncatchnoncorrection_talker[df_noncatchnoncorrection_talker['response'] != 5]
        df_catchnoncorrection_talker = df_catchnoncorrection[df_catchnoncorrection['pitchoftarg'] == pitch]

        correct_response = np.mean(list(stats_dict[pitch]['correct_response'].values()))
        hits = np.mean(list(stats_dict[pitch]['hits'].values()))
        false_alarms = np.mean(list(stats_dict[pitch]['false_alarms'].values()))

        if kw_test == True:

            kw_dict_all[pitch]['hits'] = list(data_dict[pitch]['hit'])
            kw_dict_all[pitch]['false_alarms'] = list(data_dict[pitch]['falsealarm'])
            #remove all na values from the data

            kw_dict_all[pitch]['reaction_time'] = list(data_dict[pitch]['realRelReleaseTimes'].dropna())

        stats_dict_all[pitch]['hits'] = hits
        stats_dict_all[pitch]['false_alarms'] = false_alarms
        stats_dict_all[pitch]['correct_response'] = correct_response
        stats_dict_all[pitch]['dprime'] = CalculateStats.dprime(hits, false_alarms)
        stats_dict_all[pitch]['bias'] = CalculateStats.bias(hits, false_alarms)

    if kw_test == True:
        kw_dict_rxntime = {}
        columns_to_compare = ['realRelReleaseTimes']
        for column in columns_to_compare:
            kw_dict_rxntime[column] = {}
            for ferret in ferrets:
                kw_dict_rxntime[column][ferret] = {}
                group_values = []
                for condition in pitch_list:
                    kw_dict_rxntime[column][ferret][condition] = {}
                    data = data_dict[condition]
                    data = data[data['ferretname'] == ferret]

                    # if column == 'reaction_time':
                    values_to_append = data[column].dropna()
                    group_values.append(values_to_append)

                    # Perform Kruskal-Wallis test for the current column, condition, and talker
                kw_stat, kw_p_value = stats.kruskal(*group_values)
                data_total = np.concatenate(group_values)
                dunn_results = None
                if kw_p_value < 0.05:  # Assuming a significance level of 0.05
                    # Perform post hoc test (Dunn's test)
                    dunn_results = sp.posthoc_dunn(group_values,
                                                   p_adjust='bonferroni')  # You can choose a different p-adjust method

                    # Print Dunn's test results
                    print("Dunn's test results:")
                    print(dunn_results)
                # Print Kruskal-Wallis test results

                k = len(group_values)
                n = len(data_total)

                # Calculate Eta-squared effect size
                eta_squared = (kw_stat - k + 1) / (n - k)

                # compute the effect size
                kw_dict_rxntime[column][ferret] = {'kw_stat': kw_stat, 'p_value': kw_p_value,
                                   'effect_size': eta_squared, 'dunn_result': dunn_results}
        kw_dict_all_df = pd.DataFrame.from_dict(kw_dict_rxntime)
        kw_dict_all_df.to_csv(f'D:\mixedeffectmodelsbehavioural\metrics/kw_dict_bypitch.csv')
        return stats_dict_all, stats_dict, kw_dict

    return stats_dict_all, stats_dict

def plot_stats(stats_dict_all_combined, stats_dict_combined):
    ''' plot general stats across all ferrets
    :param stats_dict_all_combined: dictionary of stats across all ferrets
    :param stats_dict_combined: dictionary of stats by ferret
    :return: None'''

    #generate bar plots
    stats = pd.DataFrame.from_dict(stats_dict_all_combined)
    x = np.arange(len(stats_dict_all_combined))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    gap_width = 0.2  # Width of the gap between series
    text_width_pt = 419.67816  # Replace with your value

    # Convert the text width from points to inches
    text_width_inches = text_width_pt / 72.27

    fig, (ax3, ax1, ax2, ax4) = plt.subplots(1,4, layout='constrained',figsize=(1.6*text_width_inches,0.4*text_width_inches))
    #make a panel for the subplots to go into

    color_map = plt.cm.get_cmap('tab10')  # Choose a colormap

    for attribute, measurement in stats_dict_all_combined.items():
        for talker, measurement_data in measurement.items():
            print(measurement_data)
            if multiplier < 3:
                offset = width * multiplier
            else:
                offset = (gap_width) + (width * multiplier)  # Add gap offset for the second series

            color = color_map(
                np.where(np.array(list(measurement.keys())) == talker)[0][0])  # Assign color based on label
            rects = ax1.bar(offset, measurement_data['hits'], width, label='_nolegend_', color=color)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute][talker]['hits'].items():
                #add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax1.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],label='_nolegend_', edgecolors='black')
                count += 1

            multiplier += 1

    ax1.set_ylim(0, 1)
    ax1.set_ylabel('P(hits)')
    ax1.set_title('Hits')
    width = 0.25  # the width of the bars
    multiplier = 0
    gap_width = 0.2

    ax1.set_xticks([0.25, 1.25], ['Female', 'Male'])
    color_map = plt.cm.get_cmap('tab10')  # Choose a colormap

    for attribute, measurement in stats_dict_all_combined.items():
        for talker, measurement_data in measurement.items():
            print(measurement_data)
            if multiplier < 3:
                offset = width * multiplier
            else:
                offset = (gap_width) + (width * multiplier)  # Add gap offset for the second series

            color = color_map(
                np.where(np.array(list(measurement.keys())) == talker)[0][0])
            if multiplier> 3:
                label = talker
            else:
                label = '_nolegend_' # Assign color based on label
            rects = ax2.bar(offset, measurement_data['false_alarms'], width, label=talker, color=color)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 's', '<', 'd', "*"]

            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute][talker]['false_alarms'].items():
                #add jitter to offset
                print('ferret data', ferret_data)
                if multiplier < 1:
                    label_text = ferret
                else:
                    label_text = '_nolegend_'
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax2.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],label=label_text, edgecolors='black')
                count += 1
            # if multiplier !=2:
            multiplier += 1

    ax2.set_ylim(0, 1)
    # ax2.legend()
    ax2.legend(['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove', 'control F0', 'inter F0', 'intra F0' ], fontsize=6, loc='upper right')
    ax2.set_xticks([0.25, 1.25], ['Female', 'Male'])
    ax2.set_ylabel('p(FA)')
    ax2.set_title('False alarms')
    multiplier = 0
    gap_width = 0.2

    for attribute, measurement in stats_dict_all_combined.items():
        for talker, measurement_data in measurement.items():
            print(measurement_data)
            if multiplier < 3:
                offset = width * multiplier
            else:
                offset = (gap_width) + (width * multiplier)  # Add gap offset for the second series

            color = color_map(
                np.where(np.array(list(measurement.keys())) == talker)[0][0])  # Assign color based on label
            rects = ax3.bar(offset, measurement_data['correct_response'], width, label=talker, color=color)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute][talker]['correct_response'].items():
                #add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax3.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],  label='_nolegend_', edgecolors='black')
                count += 1

            multiplier += 1

    ax3.set_ylim(0, 1)
    ax3.set_xticks([0.25, 1.25], ['Female', 'Male'])
    ax3.set_ylabel('p(correct)')
    ax3.set_title('Correct responses')
    multiplier = 0
    for attribute, measurement in stats_dict_all_combined.items():
        for talker, measurement_data in measurement.items():
            print(measurement_data)
            if multiplier < 3:
                offset = width * multiplier
            else:
                offset = (gap_width) + (width * multiplier)  # Add gap offset for the second series

            color = color_map(
                np.where(np.array(list(measurement.keys())) == talker)[0][0])  # Assign color based on label
            rects = ax4.bar(offset, measurement_data['dprime'], width, label=talker, color=color)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute][talker]['dprime'].items():
                #add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)

                ax4.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],  label='_nolegend_', edgecolors='black')
                count += 1

            multiplier += 1

    ax4.set_xticks([0.25, 1.25], ['Female', 'Male'])
    ax4.set_ylabel('d\'')
    ax4.set_title('d\'')


    plt.savefig('figs/proportion_hits_falsealarms_correctresp_dprime_bytalker_01052024.png', dpi = 500, bbox_inches='tight')
    plt.show()

    #add an additional plot for catch trials during false alarms
    fig, ax = plt.subplots()
    multiplier = 0
    for attribute, measurement in stats_dict_all_combined.items():
        for talker, measurement_data in measurement.items():
            print(measurement_data)
            if multiplier < 3:
                offset = width * multiplier
            else:
                offset = (gap_width) + (width * multiplier)
            color = color_map(
                np.where(np.array(list(measurement.keys())) == talker)[0][0])
            if multiplier > 3:
                label = talker
            else:
                label = '_nolegend_'
            rects = ax.bar(offset, measurement_data['false_alarms_catch'], width, label=label, color=color)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute][talker]['false_alarms_catch'].items():
                #add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                if multiplier < 1:
                    label_text = ferret
                else:
                    label_text = '_nolegend_'
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count], label=label_text, edgecolors='black')
                count += 1

            multiplier += 1
    ax.set_xticks([0.25, 1.25], ['Female', 'Male'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('false alarms during catch trials')
    ax.set_title('false alarms during catch trials')
    plt.legend(fontsize=8, loc = 'upper right')

    plt.savefig('figs/falsealarms_catch_bytalker_01052024.png', dpi=500, bbox_inches='tight')

    fig, ax = plt.subplots()
    multiplier = 0
    for attribute, measurement in stats_dict_all_combined.items():
        for talker, measurement_data in measurement.items():
            print(measurement_data)
            if multiplier < 3:
                offset = width * multiplier
            else:
                offset = (gap_width) + (width * multiplier)  # Add gap offset for the second series

            color = color_map(
                np.where(np.array(list(measurement.keys())) == talker)[0][0])  # Assign color based on label
            if multiplier > 3:
                label = talker
            else:
                label = '_nolegend_'
            rects = ax.bar(offset, measurement_data['bias'], width, label=label, color=color)
            # scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute][talker]['bias'].items():
                # add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                if multiplier < 1:
                    label_text = ferret
                else:
                    label_text = '_nolegend_'
                ax.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count], label=label_text,
                            edgecolors='black')
                count += 1

            multiplier += 1

    ax.set_xticks([0.25, 1.25], ['Female', 'Male'], fontsize = 15)
    ax.set_ylabel('bias', fontsize = 15)
    ax.set_title('Bias across talkers', fontsize = 18)
    plt.legend(fontsize=8, loc = 'lower right')
    plt.savefig('figs/bias_bytalker_01052024.png', dpi=500, bbox_inches='tight')
    plt.show()
    return

def plot_stats_by_pitch(stats_dict_all_combined, stats_dict_combined, stats_dict_all_inter, stats_dict_inter, stats_dict_all_intra, stats_dict_intra):
    ''' plot general stats across all ferrets
    :param stats_dict_all_combined: dictionary of stats across all ferrets
    :param stats_dict_combined: dictionary of stats by ferret
    :return: None'''
    width = 0.25  # the width of the bars
    multiplier = 0
    gap_width = 0.2  # Width of the gap between series
    ferret_ids = ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']
    text_width_pt = 419.67816  # Replace with your value
    # Convert the text width from points to inches
    text_width_inches = text_width_pt / 72.27
    fig, ((ax1, ax2)) = plt.subplots(2,1, layout='constrained',figsize=(0.8*text_width_inches,0.9*text_width_inches))
    #make a panel for the subplots to go into
    color_map = plt.cm.get_cmap('tab20')
    colors_to_remove = [0, 1, 2, 3 ,4,5]

    # Create a colormap without the specified colors
    new_colors = [color_map(i) for i in range(color_map.N) if i not in colors_to_remove]
    color_map = ListedColormap(new_colors)

    #remove orange and green from the color map, make a custom colour map
    for attribute, measurement in stats_dict_all_combined.items():
            offset = width * multiplier
            color = color_map(attribute)  # Assign color based on label
            rects = ax1.bar(offset, measurement['hits'], width, label='_nolegend_', color=color)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute]['hits'].items():
                #add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax1.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],label='_nolegend_', edgecolors='black')
                count += 1

            multiplier += 1

    ax1.set_ylim(0, 1)
    ax1.set_ylabel('P(hit) by target word F0', fontsize = 12)
    ax1.set_title('Hits')

    width = 0.25  # the width of the bars
    multiplier = 0
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['109', '124', '144', '191', '251 '])
    ax1.set_xlabel('F0 of target (Hz)')

    for attribute, measurement in stats_dict_all_inter.items():
            offset = width * multiplier
            # Add gap offset for the second series
            color = color_map(attribute)  # Assign color based on label
            rects = ax2.bar(offset, measurement['false_alarms'], width, label='_nolegend_', color=color)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_inter[attribute]['false_alarms'].items():
                #add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax2.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],label='_nolegend_', edgecolors='black')
                count += 1
            multiplier += 1
    if multiplier >= 5:
        multiplier = len(stats_dict_all_inter)+1
        # Add gap offset for the second series
        for talker, measurement in stats_dict_all_intra.items():
            offset = width * multiplier
            color = color_map(multiplier)
            rects = ax2.bar(offset, stats_dict_all_intra[talker]['intra_trial_roving']['false_alarms'], width, label='_nolegend_', color=color)
            count = 0
            for ferret, ferret_data in stats_dict_intra[talker]['intra_trial_roving']['false_alarms'].items():
                # add jitter to offset
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                if multiplier == 7:
                    ax2.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],
                                label=ferret_ids[count], edgecolors='black')
                else:
                    ax2.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count], label='_nolegend_',
                            edgecolors='black')
                count += 1
            multiplier += 1

    ax2.set_ylim(0, 1)
    ax2.legend( loc='upper left')
    ax2.set_ylabel('P(FA) by target word F0', fontsize = 12)
    ax2.set_title('False alarms')
    ax2.set_xlabel('F0 (Hz)')
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.5, 1.75], ['109 Hz', '124 Hz', '144 Hz', '191 Hz', '251 Hz', 'intra - female', 'intra - male '], rotation=45)
    plt.subplots_adjust(wspace=0.0, hspace=0.38)

    plt.savefig('figs/proportionofhitsbyF0_noaxisannotation.pdf', dpi = 500, bbox_inches='tight')
    plt.show()
    return


def plot_stats_by_pitch_lineplot(stats_dict_all_combined, stats_dict_combined, stats_dict_all_inter, stats_dict_inter, stats_dict_all_intra, stats_dict_intra):
    ''' plot general stats across all ferrets, lineplot style
    :param stats_dict_all_combined: dictionary of stats across all ferrets
    :param stats_dict_combined: dictionary of stats by ferret
    :param stats_dict_all_inter: dictionary of stats across all ferrets for inter trial roving
    :param stats_dict_inter: dictionary of stats by ferret for inter trial roving
    :param stats_dict_all_intra: dictionary of stats across all ferrets for intra trial roving
    :param stats_dict_intra: dictionary of stats by ferret for intra trial roving

    :return: None'''

    width = 0.25  # the width of the bars
    multiplier = 0
    ferret_ids = ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']
    text_width_pt = 419.67816  # Replace with your value

    # Convert the text width from points to inches
    text_width_inches = text_width_pt / 72.27
    fig, ((ax1, ax2)) = plt.subplots(2,1, layout='constrained',figsize=(0.8*text_width_inches,0.9*text_width_inches))
    #make a panel for the subplots to go into

    color_map = plt.cm.get_cmap('tab10')  # Choose a colormap
    offsets1 = []
    hits = []
    offsets2 = []
    false_alarms = []
    for attribute, measurement in stats_dict_all_combined.items():

            offset = width * multiplier
            if multiplier >= 3:
                offset = width * (multiplier-1)
            color = color_map(attribute)  # Assign color based on label
            hits.append(measurement['hits'])
            offsets1.append(offset)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute]['hits'].items():
                #add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax1.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],label='_nolegend_', edgecolors='black')
                count += 1
            multiplier += 1
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('P(hit) by target word F0', fontsize = 12)
    ax1.set_title('Hits')

    width = 0.25  # the width of the bars
    multiplier = 0
    for attribute, measurement in stats_dict_all_inter.items():
            offset = width * multiplier
            if multiplier >= 3:
                offset = width * (multiplier-1)
            # Add gap offset for the second series
            offsets2.append(offset)
            false_alarms.append(measurement['false_alarms'])
            color = color_map(attribute)  # Assign color based on label
            # rects = ax2.bar(offset, measurement['false_alarms'], width, label='_nolegend_', color=color)
            #do a line plot
            ax2.plot(offset, measurement['false_alarms'], color=color, linestyle = '--', marker = 'o', label='_nolegend_', markersize = 12, alpha = 0.5)

            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['p', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_inter[attribute]['false_alarms'].items():
                #add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                if multiplier < 1:
                    label_text = ferret_ids[count]
                else:
                    label_text = '_nolegend_'
                ax2.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],label=label_text, edgecolors='black')
                count += 1

            multiplier += 1

    offsets1_f = offsets1[len(offsets1)//2:]
    hits_f = hits[len(hits)//2:]
    offsets1_m = offsets1[:len(offsets1)//2]
    hits_m = hits[:len(hits)//2]

    ax1.plot(offsets1_f, hits_f, color='grey', linestyle='--', marker='o', label='Female Talker', markersize=12, alpha=0.5)
    ax1.plot(offsets1_m, hits_m, color='blue', linestyle='--', marker='o', label='Male Talker', markersize=12, alpha=0.5)

    offsets2_f = offsets2[len(offsets2)//2:]
    false_alarms_f = false_alarms[len(false_alarms)//2:]
    offsets2_m = offsets2[:len(offsets2)//2]
    false_alarms_m = false_alarms[:len(false_alarms)//2]

    ax2.plot(offsets2_f, false_alarms_f,  color='grey', linestyle='--', marker='o', label='Female Talker', markersize=12, alpha=0.5)
    ax2.plot(offsets2_m, false_alarms_m,  color='blue', linestyle='--', marker='o', label='Male Talker', markersize=12, alpha=0.5)

    ax2.set_ylim(0, 1)
    ax2.legend( loc='upper left', fontsize = 8)
    ax2.set_ylabel('P(FA) by target word F0', fontsize = 12)
    ax2.set_title('False alarms')
    ax2.set_xlabel('F0 of target (Hz)')
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['109', '124', '144', '191', '251 '])
    ax1.set_xlabel('F0 (Hz)')
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['109', '124', '144', '191', '251 '])

    # plt.suptitle('Proportion of hits, false alarms,\n correct responses and d\' by talker')
    plt.subplots_adjust(wspace=0.0, hspace=0.38)
    plt.savefig('figs/proportionofhits_FA_byF0_noaxisannotation.pdf', dpi = 500, bbox_inches='tight')
    plt.savefig('figs/proportionofhits_FA_byF0_noaxisannotation.png', dpi = 500, bbox_inches='tight')
    plt.show()

    ##no do dprime and correct response
    multiplier = 0
    text_width_pt = 419.67816  # Replace with your value

    # Convert the text width from points to inches
    text_width_inches = text_width_pt / 72.27
    fig, ((ax1, ax2)) = plt.subplots(2,1, layout='constrained',figsize=(0.8*text_width_inches,0.9*text_width_inches))
    #make a panel for the subplots to go into
    color_map = plt.cm.get_cmap('tab10')  # Choose a colormap
    offsets1 = []
    hits = []
    offsets2 = []
    false_alarms = []
    for attribute, measurement in stats_dict_all_combined.items():

            offset = width * multiplier
            if multiplier >= 3:
                offset = width * (multiplier-1)

            color = color_map(attribute)  # Assign color based on label
            hits.append(measurement['dprime'])
            offsets1.append(offset)
            #scatter plot the corresponding individual ferret data, each ferret is a different marker shape
            marker_list = ['o', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_combined[attribute]['dprime'].items():
                #add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax1.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],label='_nolegend_', edgecolors='black')
                count += 1

            multiplier += 1
    ax1.set_ylabel("d'", fontsize = 12)
    ax1.set_title("d'")
    width = 0.25  # the width of the bars
    multiplier = 0
    for attribute, measurement in stats_dict_all_inter.items():
            offset = width * multiplier
            if multiplier >= 3:
                offset = width * (multiplier-1)
            offsets2.append(offset)
            false_alarms.append(measurement['correct_response'])
            color = color_map(attribute)  # Assign color based on label

            marker_list = ['p', 's', '<', 'd', "*"]
            count = 0
            for ferret, ferret_data in stats_dict_inter[attribute]['correct_response'].items():
                #add jitter to offset
                print('ferret', ferret)
                print('ferret data', ferret_data)
                offset_jitter = offset + np.random.uniform(-0.05, 0.05)
                ax2.scatter(offset_jitter, ferret_data, 25, color=color, marker=marker_list[count],label='_nolegend_', edgecolors='black')
                count += 1

            multiplier += 1
    offsets1_f = offsets1[len(offsets1)//2:]
    hits_f = hits[len(hits)//2:]
    offsets1_m = offsets1[:len(offsets1)//2]
    hits_m = hits[:len(hits)//2]
    ax1.plot(offsets1_f, hits_f, color='grey', linestyle='--', marker='o', label='Female Talker', markersize=12, alpha=0.5)
    ax1.plot(offsets1_m, hits_m, color='blue', linestyle='--', marker='o', label='Male Talker', markersize=12, alpha=0.5)

    offsets2_f = offsets2[len(offsets2)//2:]
    false_alarms_f = false_alarms[len(false_alarms)//2:]
    offsets2_m = offsets2[:len(offsets2)//2]
    false_alarms_m = false_alarms[:len(false_alarms)//2]

    ax2.plot(offsets2_f, false_alarms_f,  color='grey', linestyle='--', marker='o', label='Female Talker', markersize=12, alpha=0.5)
    ax2.plot(offsets2_m, false_alarms_m,  color='blue', linestyle='--', marker='o', label='Male Talker', markersize=12, alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('P(Correct Response) by F0', fontsize = 12)
    ax2.set_title('P(Correct Response)')
    ax2.set_xlabel('F0 of target (Hz)')
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0,], ['109', '124', '144', '191', '251 '])
    ax1.set_xlabel('F0 (Hz)')
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['109', '124', '144', '191', '251 '])
    plt.subplots_adjust(wspace=0.0, hspace=0.38)
    plt.savefig('figs/proportionofhits_dprimecorrect_response_byF0_noaxisannotation.pdf', dpi = 500, bbox_inches='tight')
    plt.savefig('figs/proportionofhits_dprimecorrect_response_byF0_noaxisannotation.png', dpi = 500, bbox_inches='tight')
    plt.show()
    return


def run_barplot_pipeline():
    ''' run the pipeline to generate the bar plots
    :return: None'''

    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    df = behaviouralhelperscg.get_stats_df(ferrets=ferrets, startdate='04-01-2016', finishdate='01-03-2023', path = 'D:/Data/L27andL28/')
    pitch_type_list = ['control_trial', 'inter_trial_roving', 'intra_trial_roving']
    stats_dict_all_combined = {}
    stats_dict_combined = {}
    stats_dict_all_combined[1] = {}
    stats_dict_all_combined[2] = {}
    stats_dict_combined[1] = {}
    stats_dict_combined[2] = {}

    for pitch in pitch_type_list:
        stats_dict_all, stats_dict = run_stats_calc(df, ferrets, pitch_param=pitch)
        stats_dict_all_combined[1][pitch] = stats_dict_all[1][pitch]
        stats_dict_all_combined[2][pitch] = stats_dict_all[2][pitch]
        stats_dict_combined[1][pitch] = stats_dict[1][pitch]
        stats_dict_combined[2][pitch] = stats_dict[2][pitch]
    plot_stats(stats_dict_all_combined, stats_dict_combined)

def run_simulated_releasetimes():
    ''' run a simulation of release times'''
    min_time = 0
    max_time = 6
    # Number of response times to simulate
    num_samples = 1000
    # Simulate response times using a uniform distribution
    simulated_response_times = np.random.uniform(min_time, max_time, num_samples)
    # Calculate the mean of the simulated response times
    mean_response_time = np.mean(simulated_response_times)
    #plot the distribution
    fig, ax = plt.subplots()
    ax.hist(simulated_response_times, bins=100)
    ax.set_xlabel('Response time (s)')
    ax.set_ylabel('Count')
    ax.set_title('Simulated Response Times')
    plt.show()

    # Calculate the actual mean squared error (MSE)
    actual_mean = (max_time + min_time) / 2  # Actual mean of the uniform distribution
    mse = np.mean((simulated_response_times - actual_mean) ** 2)

    print(f"Simulated Mean Response Time: {mean_response_time:.4f} seconds")
    print(f"Actual Mean Squared Error (MSE): {mse:.4f}")
    return

def run_repeated_anova(stats_dict_inter, stats_dict_intra, stats_dict_control):
    ''' run a repeated measures anova on the data
    :param stats_dict_inter: dictionary of stats for inter trial roving
    :param stats_dict_intra: dictionary of stats for intra trial roving
    :param stats_dict_control: dictionary of stats for control trials
    :return: None'''

    stat_dict_inter2 = {}
    stat_dict_intra2 = {}
    stat_dict_control2 = {}

    for key in stats_dict_control.keys():
        stat_dict_control2[key] = {}
        for trialkey in stats_dict_control[key].keys():
            for statkey in stats_dict_control[key][trialkey].keys():
                list = []

                for ferretkey in stats_dict_control[key][trialkey][statkey].keys():
                    list.append(stats_dict_control[key][trialkey][statkey][ferretkey])
                stat_dict_control2[key][statkey] = list

    for key in stats_dict_intra.keys():
        stat_dict_intra2[key] = {}
        for trialkey in stats_dict_intra[key].keys():
            for statkey in stats_dict_intra[key][trialkey].keys():
                list = []

                for ferretkey in stats_dict_intra[key][trialkey][statkey].keys():
                    list.append(stats_dict_intra[key][trialkey][statkey][ferretkey])
                stat_dict_intra2[key][statkey] = list
    for key in stats_dict_inter.keys():
        stat_dict_inter2[key] = {}
        for trialkey in stats_dict_inter[key].keys():
            for statkey in stats_dict_inter[key][trialkey].keys():
                list = []

                for ferretkey in stats_dict_inter[key][trialkey][statkey].keys():
                    list.append(stats_dict_inter[key][trialkey][statkey][ferretkey])
                stat_dict_inter2[key][statkey] = list

    for talker in [1,2]:
        stats_dict_inter2 = pd.DataFrame.from_dict(stat_dict_inter2[talker])
        stats_dict_intra2 = pd.DataFrame.from_dict(stat_dict_intra2[talker])
        stats_dict_control2 = pd.DataFrame.from_dict(stat_dict_control2[talker])
        #add a roving type column
        stats_dict_inter2['roving_type'] = 'inter'
        stats_dict_intra2['roving_type'] = 'intra'
        stats_dict_control2['roving_type'] = 'control'
        #concatenate the dataframes
        stats_dict_all = pd.concat([stats_dict_inter2, stats_dict_intra2, stats_dict_control2])
        #make a dataframe with the data
        stats_dict_all = stats_dict_all.reset_index()
        #rename the index column
        stats_dict_all = stats_dict_all.rename(columns = {'index': 'ferret'})

        for value in ['hits', 'false_alarms', 'correct_response', 'dprime', 'bias']:
            anovaresults = AnovaRM(stats_dict_all, value, 'ferret', within=['roving_type']).fit()

            posthoc = MultiComparison(stats_dict_all[value], stats_dict_all['roving_type'])
            posthocresults = posthoc.tukeyhsd()
            print(posthocresults)
            numerator = anovaresults.anova_table['F Value'][0] * anovaresults.anova_table['Num DF'][0]
            denominator = numerator + anovaresults.anova_table['Den DF'][0]
            partial_eta_squared = numerator / denominator

            eta_squared_df = pd.DataFrame({'partial_eta_squared': [partial_eta_squared]})
            eta_squared_df.to_csv(f'D:\mixedeffectmodelsbehavioural\metrics/eta_squared_rovingtype_{value}_talker_{talker}.csv')
            #export to csv
            tukey_df = pd.DataFrame(data=posthocresults._results_table.data[1:],
                                    columns=posthocresults._results_table.data[0])

            tukey_df.to_csv(f'D:\mixedeffectmodelsbehavioural\metrics/posthocresults_by_rovingtype_{value}_talker_{talker}.csv')
            anovaresults.anova_table.to_csv(f'D:\mixedeffectmodelsbehavioural\metrics/anova_results_by_rovingtype_{value}_talker_{talker}.csv')


    stats_dict_all = pd.DataFrame()
    for talker in [1,2]:
        stat_dict_inter3 = pd.DataFrame.from_dict( stat_dict_inter2[talker])
        stat_dict_intra3 = pd.DataFrame.from_dict(stat_dict_intra2[talker])
        stat_dict_control3 = pd.DataFrame.from_dict(stat_dict_control2[talker])
        stat_dict_inter3['talker'] = talker
        stat_dict_intra3['talker'] = talker
        stat_dict_control3['talker'] = talker
        stat_dict_inter3['roving_type'] = 'inter'
        stat_dict_intra3['roving_type'] = 'intra'
        stat_dict_control3['roving_type'] = 'control'
        #append to the dataframe
        stats_dict_all = stats_dict_all.append(stat_dict_intra3)
        stats_dict_all = stats_dict_all.append(stat_dict_inter3)
        stats_dict_all = stats_dict_all.append(stat_dict_control3)
    #make a dataframe with the data
    stats_dict_all = stats_dict_all.reset_index()
    #rename the index column
    stats_dict_all = stats_dict_all.rename(columns = {'index': 'ferret'})


    for value in ['hits', 'false_alarms', 'correct_response', 'dprime', 'bias']:
        anovaresults = pg.rm_anova(data=stats_dict_all, dv=value, within=['roving_type', 'talker'], subject='ferret')
        if 'roving_type * talker' in anovaresults.values:
            # index_of_interaction = anovaresults.columns.get_loc('roving_type * talker')
            #get the valuse in the Source column
            #get the index of the interaction
            index_of_interaction = anovaresults.index[anovaresults['Source'] == 'roving_type * talker'].tolist()[0]
            if anovaresults['p-GG-corr'][index_of_interaction] < 0.05:
                # Interaction is significant, perform posthoc comparisons
                posthoc = pg.pairwise_ttests(data=stats_dict_all, dv=value, within=['roving_type', 'talker'],
                                             subject='ferret', padjust='bonf')
                posthocresults = posthoc
                female_df = stats_dict_all[stats_dict_all['talker'] == 1]
                male_df = stats_dict_all[stats_dict_all['talker'] == 2]
                female_tukey = pg.pairwise_tukey(data=female_df, dv=value, between='roving_type')
                male_tukey = pg.pairwise_tukey(data=male_df, dv=value, between='roving_type')
                #add a talker column
                female_tukey['talker'] = np.ones(len(female_tukey))
                male_tukey['talker'] = np.ones(len(male_tukey)) * 2
                posthoc = pd.concat([female_tukey, male_tukey])
            else:
                # No interaction, perform comparisons for roving type only
                posthoc = pg.pairwise_tukey(data=stats_dict_all, dv=value, between='roving_type')
        numerator = anovaresults['F'][0] * anovaresults['ddof1'][0]
        denominator = numerator + anovaresults['ddof2'][0]
        partial_eta_squared = numerator / denominator

        eta_squared_df = pd.DataFrame({'partial_eta_squared': [partial_eta_squared]})
        anovaresults.to_csv(
            f'D:\mixedeffectmodelsbehavioural\metrics/rmanova_twolayer_{value}_acrosstalkers.csv')

        posthoc.to_csv(
            f'D:\mixedeffectmodelsbehavioural\metrics/posthocresults_twolayer_{value}_acrosstalkers.csv')
        print(anovaresults)
    return



if __name__ == '__main__':
    run_simulated_releasetimes()
    stats_dict_empty = {}
    # run_barplot_pipeline()
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    df = behaviouralhelperscg.get_stats_df(ferrets=ferrets, startdate='04-01-2016', finishdate='01-03-2023', path='D:/Data/L27andL28/')
    df_null =behaviouralhelperscg.get_stats_df_null_distribution(ferrets=ferrets, startdate='04-01-2016', finishdate='01-03-2023', path='D:/Data/L27andL28/')
    kw_dict =  kw_test(df)
    stats_dict_all_inter, stats_dict_inter = run_stats_calc_by_pitch_mf(df, ferrets, stats_dict_empty, pitch_param='inter_trial_roving')
    stats_dict_all_intermf, stats_dict_intermf = run_stats_calc(df, ferrets, pitch_param='inter_trial_roving')
    stats_dict_all_intra, stats_dict_intra = run_stats_calc(df, ferrets, pitch_param='intra_trial_roving')
    stats_dict_all_control, stats_dict_control = run_stats_calc(df, ferrets, pitch_param='control_trial')

    run_repeated_anova(stats_dict_intermf, stats_dict_intra, stats_dict_control)

    stats_dict_all_bypitch, stats_dict_bypitch = run_stats_calc_by_pitch_mf(df, ferrets, stats_dict_empty, pitch_param=None)


    plot_stats_by_pitch(stats_dict_all_bypitch, stats_dict_bypitch, stats_dict_all_inter, stats_dict_inter, stats_dict_all_intra, stats_dict_intra)
    plot_stats_by_pitch_lineplot(stats_dict_all_bypitch, stats_dict_bypitch, stats_dict_all_inter, stats_dict_inter, stats_dict_all_intra, stats_dict_intra)

    stats_dict_all_bypitch, stats_dict_bypitch, kw_dict_bypitch = run_stats_calc_by_pitch(df, ferrets, stats_dict_empty, pitch_param=None)
    stats_dict_all_intra, stats_dict_intra = run_stats_calc(df, ferrets, pitch_param='intra_trial_roving')




    # plot_stats_by_pitch(stats_dict_all_bypitch, stats_dict_bypitch, stats_dict_all_inter, stats_dict_inter, stats_dict_all_intra, stats_dict_intra)


