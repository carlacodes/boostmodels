import sklearn.metrics
# from rpy2.robjects import pandas2ri
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

import shap
import matplotlib
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold
# scaler = MinMaxScaler()
import os
import xgboost as xgb
import matplotlib.pyplot as plt
# import rpy2.robjects.numpy2ri
import matplotlib.colors as mcolors
import sklearn
from sklearn.model_selection import train_test_split
from helpers.behaviouralhelpersformodels import *
from helpers.calculate_stats import *




def run_stats_calc(df, ferrets, pitch_param = 'control_trial'):

    df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    df_noncorrection = df[(df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    count = int(0)
    # stats_dict[pitch_param] = {}
    # stats_dict[pitch_param]['hits'] = {}
    # stats_dict[pitch_param]['false_alarms'] = {}
    # stats_dict[pitch_param]['correct_response'] = {}

    talkers = [1,2]
    stats_dict = {}
    stats_dict[1] = {}
    stats_dict[2] = {}
    stats_dict[1][pitch_param] = {}
    stats_dict[2][pitch_param] = {}

    stats_dict[1][pitch_param]['hits'] = {}
    stats_dict[1][pitch_param]['false_alarms']= {}
    stats_dict[1][pitch_param]['correct_response']= {}
    stats_dict[1][pitch_param]['dprime']= {}
    stats_dict[1][pitch_param]['bias']= {}


    stats_dict[2][pitch_param]['hits'] ={}
    stats_dict[2][pitch_param]['false_alarms'] = {}
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
        # hits = np.mean(df_noncatchnoncorrection_talker_hitrate['hit'])
        # false_alarms = np.mean(df_noncorrection_talker['falsealarm'])
        # correct_rejections = np.mean(df_catchnoncorrection_talker['response'] == 3)
        # correct_response =  (len(df_noncatchnoncorrection_talker[df_noncatchnoncorrection_talker['hit']==True]) + len(df_catchnoncorrection_talker[df_catchnoncorrection_talker['response'] == 3]))/ (len(df_noncorrection_talker))
        #take mean of all the values in the dictionary


        correct_response = np.mean(list(stats_dict[talker][pitch_param]['correct_response'].values()))
        hits = np.mean(list(stats_dict[talker][pitch_param]['hits'].values()))
        false_alarms = np.mean(list(stats_dict[talker][pitch_param]['false_alarms'].values()))

        stats_dict_all[talker][pitch_param]['hits'] = hits
        stats_dict_all[talker][pitch_param]['false_alarms'] = false_alarms
        stats_dict_all[talker][pitch_param]['correct_response'] = correct_response
        stats_dict_all[talker][pitch_param]['dprime'] = CalculateStats.dprime(hits, false_alarms)
        stats_dict_all[talker][pitch_param]['bias'] = CalculateStats.bias(hits, false_alarms)

    return stats_dict_all, stats_dict
def run_stats_calc_by_pitch(df, ferrets, stats_dict, pitch_param = 'control_trial'):

    df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    df_noncorrection = df[(df['correctionTrial'] == 0) & (df[pitch_param] == 1)]
    count = int(0)


    talkers = [1,2]
    stats_dict[1] = {}
    stats_dict[2] = {}
    stats_dict[3] = {}
    stats_dict[4] = {}
    stats_dict[5] = {}



    stats_dict[1]['hits'] = {}
    stats_dict[1]['false_alarms']= {}
    stats_dict[1]['correct_response']= {}
    stats_dict[1]['dprime']= {}
    stats_dict[1]['bias']= {}


    stats_dict[2]['hits'] ={}
    stats_dict[2]['false_alarms'] = {}
    stats_dict[2]['correct_response'] = {}
    stats_dict[2]['dprime']= {}
    stats_dict[2]['bias']= {}

    stats_dict[3]['hits'] ={}
    stats_dict[3]['false_alarms'] = {}
    stats_dict[3]['correct_response'] = {}
    stats_dict[3]['dprime']= {}
    stats_dict[3]['bias']= {}

    stats_dict[4]['hits'] ={}
    stats_dict[4]['false_alarms'] = {}
    stats_dict[4]['correct_response'] = {}
    stats_dict[4]['dprime']= {}
    stats_dict[4]['bias']= {}

    stats_dict[5]['hits'] ={}
    stats_dict[5]['false_alarms'] = {}
    stats_dict[5]['correct_response'] = {}
    stats_dict[5]['dprime']= {}
    stats_dict[5]['bias']= {}


    count = 0
    pitch_list = [1,2,3,4,5]
    for ferret in ferrets:

        selected_ferret = df_noncatchnoncorrection[df_noncatchnoncorrection['ferret'] == count]
        selected_ferret_catch = df_catchnoncorrection[df_catchnoncorrection['ferret'] == count]
        selected_ferret_all = df_noncorrection[df_noncorrection['ferret'] == count]

        for pitch in pitch_list:
            selected_ferret_talker = selected_ferret[selected_ferret['pitchoftarg'] == pitch]
            selected_ferret_all_talker = selected_ferret_all[selected_ferret_all['pitchoftarg'] == pitch]

            selected_ferret_talker_hitrate = selected_ferret_talker[selected_ferret_talker['response'] != 5]

            selected_ferret_catch_talker = selected_ferret_catch[selected_ferret_catch['pitchoftarg'] == pitch]

            stats_dict[pitch]['hits'][ferret] = np.mean(selected_ferret_talker_hitrate['hit'])
            stats_dict[pitch]['false_alarms'][ferret] = np.mean(selected_ferret_all_talker['falsealarm'])
            stats_dict[pitch]['dprime'][ferret] = CalculateStats.dprime(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
            stats_dict[pitch]['bias'][ferret] = CalculateStats.bias(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
            #%Correct(hit + CR / hits + misses + CR + FA)
            stats_dict[pitch]['correct_response'][ferret] = (len(selected_ferret_talker[selected_ferret_talker['hit']==True]) + len(selected_ferret_catch_talker[selected_ferret_catch_talker['response'] == 3]))/ (len(selected_ferret_talker) + len(selected_ferret_catch_talker))
        count += 1
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
        # hits = np.mean(df_noncatchnoncorrection_talker_hitrate['hit'])
        # false_alarms = np.mean(df_noncorrection_talker['falsealarm'])
        # correct_rejections = np.mean(df_catchnoncorrection_talker['response'] == 3)
        # correct_response =  (len(df_noncatchnoncorrection_talker[df_noncatchnoncorrection_talker['hit']==True]) + len(df_catchnoncorrection_talker[df_catchnoncorrection_talker['response'] == 3]))/ (len(df_noncorrection_talker))
        #take mean of all the values in the dictionary


        correct_response = np.mean(list(stats_dict[pitch]['correct_response'].values()))
        hits = np.mean(list(stats_dict[pitch]['hits'].values()))
        false_alarms = np.mean(list(stats_dict[pitch]['false_alarms'].values()))

        stats_dict_all[pitch]['hits'] = hits
        stats_dict_all[pitch]['false_alarms'] = false_alarms
        stats_dict_all[pitch]['correct_response'] = correct_response
        stats_dict_all[pitch]['dprime'] = CalculateStats.dprime(hits, false_alarms)
        stats_dict_all[pitch]['bias'] = CalculateStats.bias(hits, false_alarms)

    return stats_dict_all, stats_dict

def run_stats_calc_by_pitch(df, ferrets, stats_dict, pitch_param = 'inter_trial_roving'):
    if pitch_param == None:
        df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0)]
        df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0)]
        df_noncorrection = df[(df['correctionTrial'] == 0)]

    else:

        df_noncatchnoncorrection = df[(df['catchTrial'] == 0) & (df['correctionTrial'] == 0) & ((df[pitch_param] == 1) |(df['control_trial'] == 1))]
        df_catchnoncorrection = df[(df['catchTrial'] == 1) & (df['correctionTrial'] == 0) & ((df[pitch_param] == 1) |(df['control_trial'] == 1))]
        df_noncorrection = df[(df['correctionTrial'] == 0) & ((df[pitch_param] == 1) | (df['control_trial'] == 1))]
    count = int(0)


    talkers = [1,2]
    stats_dict[1] = {}
    stats_dict[2] = {}
    stats_dict[3] = {}
    stats_dict[4] = {}
    stats_dict[5] = {}



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


    count = 0
    pitch_list = [1,2,3,4,5]
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
            stats_dict[pitch]['false_alarms'][ferret] = np.mean(selected_ferret_all_talker['falsealarm'])
            stats_dict[pitch]['dprime'][ferret] = CalculateStats.dprime(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
            stats_dict[pitch]['bias'][ferret] = CalculateStats.bias(np.mean(selected_ferret_talker_hitrate['hit']), np.mean(selected_ferret_all_talker['falsealarm']))
            #%Correct(hit + CR / hits + misses + CR + FA)
            stats_dict[pitch]['correct_response'][ferret] = (len(selected_ferret_talker[selected_ferret_talker['hit']==True]) + len(selected_ferret_catch_talker[selected_ferret_catch_talker['response'] == 3]))/ (len(selected_ferret_talker) + len(selected_ferret_catch_talker))
        count += 1
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
        # hits = np.mean(df_noncatchnoncorrection_talker_hitrate['hit'])
        # false_alarms = np.mean(df_noncorrection_talker['falsealarm'])
        # correct_rejections = np.mean(df_catchnoncorrection_talker['response'] == 3)
        # correct_response =  (len(df_noncatchnoncorrection_talker[df_noncatchnoncorrection_talker['hit']==True]) + len(df_catchnoncorrection_talker[df_catchnoncorrection_talker['response'] == 3]))/ (len(df_noncorrection_talker))
        #take mean of all the values in the dictionary


        correct_response = np.mean(list(stats_dict[pitch]['correct_response'].values()))
        hits = np.mean(list(stats_dict[pitch]['hits'].values()))
        false_alarms = np.mean(list(stats_dict[pitch]['false_alarms'].values()))

        stats_dict_all[pitch]['hits'] = hits
        stats_dict_all[pitch]['false_alarms'] = false_alarms
        stats_dict_all[pitch]['correct_response'] = correct_response
        stats_dict_all[pitch]['dprime'] = CalculateStats.dprime(hits, false_alarms)
        stats_dict_all[pitch]['bias'] = CalculateStats.bias(hits, false_alarms)

    return stats_dict_all, stats_dict

def plot_stats(stats_dict_all_combined, stats_dict_combined):

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

    def get_axis_limits(ax, scale=1):
        return ax.get_xlim()[0] * scale, (ax.get_ylim()[1] * scale)

    import matplotlib.font_manager as fm

    # ax1.annotate('a)', xy=get_axis_limits(ax1))
    # ax2.annotate('b)', xy=get_axis_limits(ax2))
    # ax3.annotate('c)', xy=get_axis_limits(ax3))
    # ax4.annotate('d)', xy=get_axis_limits(ax4))
    title_y = ax1.title.get_position()[1]  # Get the y-coordinate of the title
    font_props = fm.FontProperties(weight='bold', size = 9)

    # ax1.annotate('A', xy=get_axis_limits(ax1), xytext=(-0.1, ax1.title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props, zorder=10)
    # ax2.annotate('B', xy=get_axis_limits(ax2), xytext=(-0.1, ax2.title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    # ax3.annotate('C', xy=get_axis_limits(ax3), xytext=(-0.1, ax3.title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    # ax4.annotate('D', xy=get_axis_limits(ax4), xytext=(-0.1, ax4.title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)

    # plt.suptitle('Proportion of hits, false alarms, correct responses and d\' by talker')
    # plt.suptitle('Proportion of hits, false alarms,\n correct responses and d\' by talker')

    plt.savefig('figs/proportion_hits_falsealarms_correctresp_dprime_bytalker_2706.png', dpi = 500, bbox_inches='tight')
    plt.show()

    #plot the bias in a separate figure
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
    plt.savefig('figs/bias_bytalker_2706.png', dpi=500, bbox_inches='tight')
    plt.show()








    #get proportion of hits and false alarms for the dataframe
    fig, ax = plt.subplots()


def plot_stats_by_pitch(stats_dict_all_combined, stats_dict_combined, stats_dict_all_inter, stats_dict_inter, stats_dict_all_intra, stats_dict_intra):

    #generate bar plots
    stats = pd.DataFrame.from_dict(stats_dict_all_combined)
    x = np.arange(len(stats_dict_all_combined))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    gap_width = 0.2  # Width of the gap between series
    ferret_ids = ['F1702', 'F1815', 'F1803', 'F2002', 'F2105']
    text_width_pt = 419.67816  # Replace with your value

    # Convert the text width from points to inches
    text_width_inches = text_width_pt / 72.27
    fig, ((ax1, ax2)) = plt.subplots(2,1, layout='constrained',figsize=(0.8*text_width_inches,0.9*text_width_inches))
    #make a panel for the subplots to go into

    color_map = plt.cm.get_cmap('tab10')  # Choose a colormap

    for attribute, measurement in stats_dict_all_combined.items():

            offset = width * multiplier
    # Add gap offset for the second series

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
    gap_width = 0.2


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


    def get_axis_limits(ax, scale=1):
        return ax.get_xlim()[0] * scale, (ax.get_ylim()[1] * scale)

    import matplotlib.font_manager as fm
    #
    # ax1.annotate('a)', xy=get_axis_limits(ax1))
    # ax2.annotate('b)', xy=get_axis_limits(ax2))
    # # ax3.annotate('c)', xy=get_axis_limits(ax3))
    # # ax4.annotate('d)', xy=get_axis_limits(ax4))
    # title_y = ax1.title.get_position()[1]  # Get the y-coordinate of the title
    # font_props = fm.FontProperties(weight='bold')
    font_props = fm.FontProperties(weight='bold', size=9)

    # ax1.annotate('A', xy=get_axis_limits(ax1), xytext=(-0.1, ax1.title.get_position()[1]+0.05), textcoords='axes fraction', fontproperties = font_props, zorder=10)
    # ax2.annotate('B', xy=get_axis_limits(ax2), xytext=(-0.1, ax2.title.get_position()[1]+0.05), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    # ax3.annotate('c)', xy=get_axis_limits(ax3), xytext=(-0.1, ax3.title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    # ax4.annotate('d)', xy=get_axis_limits(ax4), xytext=(-0.1, ax4.title.get_position()[1]+0.1), textcoords='axes fraction', fontproperties = font_props,zorder=10)
    #
    # plt.suptitle('Proportion of hits, false alarms,\n correct responses and d\' by talker')
    plt.subplots_adjust(wspace=0.0, hspace=0.38)

    plt.savefig('figs/proportionofhitsbyF0_noaxisannotation.pdf', dpi = 500, bbox_inches='tight')
    plt.show()


def run_barplot_pipeline():
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    df = behaviouralhelperscg.get_stats_df(ferrets=ferrets, startdate='04-01-2016', finishdate='01-03-2023')
    stats_dict = {}
    pitch_type_list = ['control_trial', 'inter_trial_roving', 'intra_trial_roving']
    stats_dict_all_combined = {}
    stats_dict_combined = {}
    stats_dict_all_combined[1] = {}
    stats_dict_all_combined[2] = {}
    stats_dict_combined[1] = {}
    stats_dict_combined[2] = {}

    for pitch in pitch_type_list:
        stats_dict_all, stats_dict = run_stats_calc(df, ferrets, pitch_param=pitch)
        # append to dataframe

        stats_dict_all_combined[1][pitch] = stats_dict_all[1][pitch]
        stats_dict_all_combined[2][pitch] = stats_dict_all[2][pitch]

        stats_dict_combined[1][pitch] = stats_dict[1][pitch]
        stats_dict_combined[2][pitch] = stats_dict[2][pitch]

    plot_stats(stats_dict_all_combined, stats_dict_combined)

def run_simulated_releasetimes():
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

if __name__ == '__main__':
    run_simulated_releasetimes()
    stats_dict_empty = {}
    # run_barplot_pipeline()
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']
    df = behaviouralhelperscg.get_stats_df(ferrets=ferrets, startdate='04-01-2016', finishdate='01-03-2023')
    stats_dict_all_inter, stats_dict_inter = run_stats_calc_by_pitch(df, ferrets, stats_dict_empty, pitch_param='inter_trial_roving')

    stats_dict_all_bypitch, stats_dict_bypitch = run_stats_calc_by_pitch(df, ferrets, stats_dict_empty, pitch_param=None)
    stats_dict_all_intra, stats_dict_intra = run_stats_calc(df, ferrets, pitch_param='intra_trial_roving')
    plot_stats_by_pitch(stats_dict_all_bypitch, stats_dict_bypitch, stats_dict_all_inter, stats_dict_inter, stats_dict_all_intra, stats_dict_intra)



