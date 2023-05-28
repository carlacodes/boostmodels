import mat73
import numpy as np
import os
import sys
import scipy
import scipy.io
import numpy as np
from scipy.signal import resample
import argparse
from scipy import spatial

import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics

def run_word_durations(dirfemale):

    # load mat file
    mat = scipy.io.loadmat(dirfemale)
    word_dictionary = mat['s']
    frequency = 24414.0625
    word_dictionary2 = word_dictionary[0]
    word_dictionary2 = word_dictionary2[0]
    word_dictionary_np = np.array(word_dictionary2)
    word_dictionary_np = word_dictionary_np[0]
    word_times = embed_word_durations(word_dictionary_np, frequency)

    # word_dictionary2=word_dictionary[0,0]
    # word_dictionary = np.array(word_durations, dtype=np.float64)
    #
    # # embed word durations in wav file
    # data = embed_word_durations(data, word_durations, fs)
    #
    # # save wav file
    # wav.write(args.wavfile, fs, data)
    return word_times, word_dictionary_np


def run_word_durations_male(dirfemale):

    # load mat file
    mat = scipy.io.loadmat(dirfemale)
    word_dictionary = mat['maleSounds']
    print('run_word_durations male')

    frequency = 24414.0625
    word_dictionary2 = word_dictionary[0]
    word_dictionary2 = word_dictionary2[3] #taking the fourth stimuli set in matlab, 3rd index in python
    word_dictionary_np = np.array(word_dictionary2)
    word_dictionary_np=word_dictionary_np[0]
    word_times = embed_word_durations(word_dictionary_np, frequency)

    # word_dictionary2=word_dictionary[0,0]
    # word_dictionary = np.array(word_durations, dtype=np.float64)
    #
    # # embed word durations in wav file
    # data = embed_word_durations(data, word_durations, fs)
    #
    # # save wav file
    # wav.write(args.wavfile, fs, data)
    return word_times, word_dictionary_np


def calc_cosine_similarity(x, y):
    # x and y are vectors
    # returns the cosine similarity between x and y
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
def scaledata(datain, minval, maxval):
    dataout = datain - min(datain.flatten())
    dataout = (dataout / (max(datain.flatten()) - min(datain.flatten()))) * (maxval - minval)
    dataout = dataout + minval
    return dataout




def calc_temporal_sim(data, pos):
    """ calculate temporal similarity between word and all other words in the audio file"""
    # data is a vector of audio samples
    # pos is the position of the word in the audio file
    # returns the cosine similarity between the word and the audio file
    # get word
    word = data[pos]
    word = np.reshape(word, (1, len(word)))
    word = np.squeeze(word)
    zeros_array = np.zeros((round(24414.0625 * 0.08)))
    word = np.concatenate((word, zeros_array), axis=0)
    word = scaledata(word, -7797, 7797)

    # get cosine similarity
    for i in range(0, len(data)):
        otherword = data[i]
        otherword = np.squeeze(otherword)
        otherword = np.concatenate((otherword, zeros_array), axis=0)
        otherword = scaledata(otherword, -7797, 7797)
        #take the hanning window to downsample
        window_word = np.hanning(len(word))
        window_word_other = np.hanning(len(otherword))
        x_win = word * window_word
        x_win_other = otherword * window_word_other
        word_downsampled = resample(x_win, num=int(len(word)*1000/24414.0625))
        otherword_downsampled = resample(x_win_other, num=int(len(otherword)*1000/24414.0625))

        # Calculate the cross-correlation between the two signals
        corr = np.correlate(word_downsampled, otherword_downsampled, mode='full')

        # Normalize the cross-correlation
        corr = corr / np.sqrt(np.sum(word_downsampled ** 2) * np.sum(otherword_downsampled ** 2))
        # Compute the temporal similarity as the maximum value of the cross-correlation
        similarity = np.max(corr)
        if i == 0:
            corrvector = similarity
        else:
            corrvector = np.append(corrvector, similarity)
    return corrvector

def calc_cosine_acrossdata(data, pos):
    """ calculate cosine similarity between word and all other words in the audio file"""
    # data is a vector of audio samples
    # pos is the position of the word in the audio file
    # returns the cosine similarity between the word and the audio file

    # get word
    word = data[pos]
    word = np.reshape(word, (1, len(word)))
    word = np.squeeze(word)
    zeros_array = np.zeros((round(24414.0625 * 0.08)))

    word = np.concatenate((word, zeros_array), axis=0)
    word = scaledata(word, -7797, 7797)

    # get cosine similarity
    for i in range(0, len(data)):
        otherword = data[i]
        otherword = np.squeeze(otherword)
        otherword = np.concatenate((otherword, zeros_array), axis=0)
        otherword = scaledata(otherword, -7797, 7797)

        if len(otherword) > np.size(word,0):
            #add zero padding to shorter word
            #print('yes')
            word = np.concatenate((word, np.zeros(abs(np.size(word,0) - len(otherword)))))
            word = np.fft.fft(word)
        elif len(otherword) < np.size(word,0):
            #print('less than')
            #otherword = np.pad(otherword, (1, abs(np.size(word,1)- len(otherword))), 'constant')
            otherword = np.concatenate((otherword, np.zeros(abs(np.size(word,0) - len(otherword)))))
            #take the fft
            otherword = np.fft.fft(otherword)
        cosinesim = calc_cosine_similarity(word, otherword)
        cosinesim = np.real(cosinesim)
        print(cosinesim)
        if i == 0:
            cosinesimvector = cosinesim
        else:
            cosinesimvector = np.append(cosinesimvector, cosinesim)
    return cosinesimvector
    # data is a vector of audio samples
    # fs is the sampling frequency in Hz
    # returns a vector of audio samples with word durations embedded in it



def embed_word_durations(data, fs):
    # word_durations is a vector of word durations in seconds
    # data is a vector of audio samples
    # fs is the sampling frequency in Hz
    # returns a vector of audio samples with word durations embedded in it

    # convert word durations to samples
    word_dictionary_length_list = []
    for i in range(len(data)):
        word_dictionary_length = len(data[i])/fs
        word_dictionary_length_list.append(word_dictionary_length)

    # embed word durations in data


    return word_dictionary_length_list


def main():
    dirfemale = 'D:/Stimuli/19122022/FemaleSounds24k_addedPinkNoiseRevTargetdB.mat'
    dirmale = 'D:/Stimuli/19122022/MaleSounds24k_addedPinkNoiseRevTargetdB.mat'
    word_times, worddictionary_female= run_word_durations(dirfemale)
    #print(word_times)
    cosinesimvectorfemale = calc_cosine_acrossdata(worddictionary_female, 0)
    word_times_male, worddictionary_male = run_word_durations_male(dirmale)
    # word_times_male_normalised = word_times_male/word_times_male[0]
    # word_times_female_normalised = word_times/word_times[0]

    cosinesimvectormale = calc_cosine_acrossdata(worddictionary_male,0)

    temporalcorrmale = calc_temporal_sim(worddictionary_male, 0)
    temporalcorrfemale = calc_temporal_sim(worddictionary_female, 0)

    np.save('D:/Stimuli/temporalcorrmale.npy', temporalcorrmale)
    np.save('D:/Stimuli/temporalcorrfemale.npy', temporalcorrfemale)

    np.save('D:/Stimuli/cosinesimvectorfemale.npy', cosinesimvectorfemale)
    np.save('D:/Stimuli/cosinesimvectormale.npy', cosinesimvectormale)


if __name__ == "__main__":
    main()
