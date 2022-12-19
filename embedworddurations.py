import mat73
import numpy as np
import os
import sys
import scipy
import scipy.io
import argparse
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

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
    return word_times


def run_word_durations_male(dirfemale):

    # load mat file
    mat = scipy.io.loadmat(dirfemale)
    word_dictionary = mat['maleSounds']

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
    return word_times
def calc_cosine_similarity(x, y):
    # x and y are vectors
    # returns the cosine similarity between x and y
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def calc_cosine_acrossdata(data, pos):
    # data is a vector of audio samples
    # pos is the position of the word in the audio file
    # returns the cosine similarity between the word and the audio file

    # get word
    word = data[pos]

    # get cosine similarity
    cosine_similarity = calc_cosine_similarity(data, word)

    return cosine_similarity
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
    word_times = run_word_durations(dirfemale)
    print(word_times)
    cosinesimvectorfemale = calc_cosine_acrossdata(word_times, 0)
    word_times_male = run_word_durations_male(dirmale)
    cosinesimvectormale = calc_cosine_acrossdata(word_times_male,0)
    np.save('cosinesimvectorfemale', cosinesimvectorfemale)
    np.save('cosinesimvectormale', cosinesimvectormale)


if __name__ == "__main__":
    main()
