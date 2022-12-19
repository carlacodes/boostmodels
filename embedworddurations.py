import mat73
import numpy as np
import os
import sys
import argparse
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

def run_word_durations():
    parser = argparse.ArgumentParser(description='Embed word durations in wav file')
    parser.add_argument('wavfile', help='wav file')
    parser.add_argument('matfile', help='mat file')
    args = parser.parse_args()

    # load wav file
    fs, data = wav.read(args.wavfile)
    if data.ndim > 1:
        data = data[:,0]

    # load mat file
    mat = mat73.loadmat(args.matfile)
    word_durations = mat['word_durations']
    word_durations = np.array(word_durations, dtype=np.float64)

    # embed word durations in wav file
    data = embed_word_durations(data, word_durations, fs)

    # save wav file
    wav.write(args.wavfile, fs, data)

def embed_word_durations(data, word_durations, fs):
    # word_durations is a vector of word durations in seconds
    # data is a vector of audio samples
    # fs is the sampling frequency in Hz
    # returns a vector of audio samples with word durations embedded in it

    # convert word durations to samples
    word_durations = np.round(word_durations * fs).astype(np.int32)

    # embed word durations in data
    data = np.repeat(data, word_durations)

    return data


def main():
    run_word_durations()


if __name__ == "__main__":
    main()
