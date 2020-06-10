# in this code we will preprocess audios with MFCC and spectrogram and then save them as numpy file.
# TODO 1 : Change PATH
# TODO 2 : install pydub ffmpeg and other libs
# TODO 3 : prepare other csv files too.
# TODO 4 : create numpy file for other csv files too.
# TODO 5 : Run this script on kaggle for turkey. With All Todos.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
import pydub
import IPython.display as ipd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile # reading the wavfile
import os

# =========== < new import for approach 2 > ======================
from numpy.lib.stride_tricks import as_strided

from pydub import AudioSegment
from python_speech_features import mfcc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Lambda)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint   
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Activation
# =========== </new import for approach 2> ======================
import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle


SIGNAL_RATE = 16000 # we convert audios to this SIGNAL_RATE
RNG_SEED = 123
NumberOfSamples = 1823 # write number of samples here

# say address to your Files. NEED TO CHANGE
PATH  = '/kaggle/input/trsst/tr/'
PATH2 = '/kaggle/input/tr-attention-features/'

def drop_col(df, cols):
    for col in cols:
        df.drop([col], axis=1, inplace=True)


def prepare_dataset(df_address):
    df =pd.read_csv(df_address, delimiter='\t', encoding='utf-8')
    drop_col(df, ['age', 'up_votes', 'client_id', 'accent', 'down_votes', 'gender'])
    return df

#prepare validation and train data just for now
partition='train'
train_attention = prepare_dataset(PATH+ 'train' + '.tsv' )

partition = 'dev'
dev_attention = prepare_dataset(PATH+ 'dev' + '.tsv')

# this is the way to pad all the y of audios to same size.
def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram
    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).
    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x
    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)  

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs


def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
#     asssume we don't need next line ==>  PATH + 'clips/' + filename
    audio, sample_rate = librosa.load(PATH + 'clips/' + filename, sr=16000, duration = 10)
    audio = padarray(audio, 160000)
    if audio.ndim >= 2:
        audio = np.mean(audio, 1)
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of "
                            " sample rate")
    if step > window:
        raise ValueError("step size must not be greater than window size")
    hop_length = int(0.001 * step * sample_rate)
    fft_length = int(0.001 * window * sample_rate)
    pxx, freqs = spectrogram(
        audio, fft_length=fft_length, sample_rate=sample_rate,
        hop_length=hop_length)
#     print(pxx)
#     print(freqs)
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))


def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    int_sequence = []
#     text = text.lower()
#     print(text)
    for c in text:
#         print(c)
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence

def int_sequence_to_text(int_sequence):
    """ Convert an integer sequence to text """
    text = []
    for c in int_sequence:
        ch = index_map[c]
        text.append(ch)
    return text

def calc_mfcc(filename):
    dim=26
    audio, sample_rate = librosa.load(PATH + 'clips/' + filename, sr=16000, duration = 10)
    audio = padarray(audio, 160000)
    return mfcc(audio, sample_rate, numcep=dim)

def load_audio(filename):
    audio, sample_rate = librosa.load(PATH + 'clips/' + filename, sr=16000, duration = 10)
    audio = padarray(audio, 160000)
    return audio, filename 
#     return np.zeros(160000), filename


# create numpy file
features_extracted = []
train_features = [] # (No. audios, time_seq, feature)
for path in train_attention['path']:
    features_extracted.append(calc_mfcc(path))

with open('features_extracted_train_attention.npy', 'wb') as f:
    np.save(f, np.array(features_extracted))

# create numpy file
dev_features = [] # (No. audios, time_seq, feature)
for path in dev_attention['path']:
    features_extracted.append(calc_mfcc(path))

with open('features_extracted_dev_attention.npy', 'wb') as f:
    np.save(f, np.array(features_extracted))