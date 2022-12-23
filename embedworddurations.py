import mat73
import numpy as np
import os
import sys
import scipy
import scipy.io
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


def calc_cosine_acrossdata(data, pos):
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

    #word = word / np.linalg.norm(word)

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
        elif len(otherword) <  np.size(word,0):
            #print('less than')
            #otherword = np.pad(otherword, (1, abs(np.size(word,1)- len(otherword))), 'constant')
            otherword = np.concatenate((otherword, np.zeros(abs(np.size(word,0) - len(otherword)))))
        cosinesim = calc_cosine_similarity(word, otherword)
        print(cosinesim)
        if i == 0:
            cosinesimvector = cosinesim
        else:
            cosinesimvector = np.append(cosinesimvector, cosinesim)

    #cosine_similarity = calc_cosine_similarity(data, word)

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
    np.save('D:/Stimuli/cosinesimvectorfemale.npy', cosinesimvectorfemale)
    np.save('D:/Stimuli/cosinesimvectormale.npy', cosinesimvectormale)


if __name__ == "__main__":
    main()
