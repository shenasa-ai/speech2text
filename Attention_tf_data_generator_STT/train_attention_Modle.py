# TODO 1 : Remove unnecessary imports
# TODO 2 : Change PATH
# TODO 3 : funciton all these Parts
# TODO 4 : Some vars are not defined here .
# TODO 5 : select callbacks
# TODO 6 : Run this Script in KAGGLE

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

# ============================================ PART ==================================
# first of all load preprocessed data.
# loaad numpy file
with open(PATH2 + 'features_extracted_train_attention.npy', 'rb') as f:
    train_features = np.load(f)
    
with open(PATH2 + 'features_extracted_dev_attention.npy', 'rb') as f:
    dev_features = np.load(f)

# Tokenizing
# Store captions and image names in vectors
all_captions = []
tokenized_captions = []
# for annot in train_attention['sentence']:
#     caption = '<start> ' + annot + ' <end>'
#     all_captions.append(caption)


# ============================================ PART ==================================

tokenizer = tf.keras.preprocessing.text.Tokenizer( oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ', char_level=True, lower=True )
tokenizer.fit_on_texts( list(dev_attention['sentence']) + list(train_attention['sentence']) )
# tokenizer.fit_on_texts(train_attention['sentence'])
train_seqs = tokenizer.texts_to_sequences(train_attention['sentence'])
dev_seqs = tokenizer.texts_to_sequences(dev_attention['sentence'])

print(f'sentence : {train_attention["sentence"][0]} , tokenized length : {len(train_seqs[0])}, original sentence length : {len(train_attention["sentence"][0])} ')

# ============================================ PART ==================================

# add start and end to each sentence. add start and end to dict . add padding to dict and sentences. 
start_index_in_dict = len(tokenizer.word_index) + 1
end_index_in_dict = len(tokenizer.word_index) + 2

tokenizer.word_index['<start>'] = start_index_in_dict
tokenizer.index_word[start_index_in_dict] = '<start>'
tokenizer.word_index['<end>'] = end_index_in_dict
tokenizer.index_word[end_index_in_dict] = '<end>'
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
def myfunc(x):
    x.insert(0,start_index_in_dict)
    x.insert(len(x),end_index_in_dict)
    return x

train_seqs = list(map(myfunc, train_seqs))
dev_seqs = list(map(myfunc, dev_seqs))

cap_vector_train = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
cap_vector_dev = tf.keras.preprocessing.sequence.pad_sequences(dev_seqs, padding='post')


# ============================================ PART ==================================

# Store captions and image names in vectors
all_captions = []
tokenized_captions = []
# for annot in train_attention['sentence']:
#     caption = '<start> ' + annot + ' <end>'
#     all_captions.append(caption)

tokenizer = tf.keras.preprocessing.text.Tokenizer( oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ', char_level=True, lower=True )
tokenizer.fit_on_texts( list(dev_attention['sentence']) + list(train_attention['sentence']) )
# tokenizer.fit_on_texts(train_attention['sentence'])
train_seqs = tokenizer.texts_to_sequences(train_attention['sentence'])
dev_seqs = tokenizer.texts_to_sequences(dev_attention['sentence'])

# ============================================ PART ==================================


# add start and end to each sentence. add start and end to dict . add padding to dict and sentences. 
start_index_in_dict = len(tokenizer.word_index) + 1
end_index_in_dict = len(tokenizer.word_index) + 2

tokenizer.word_index['<start>'] = start_index_in_dict
tokenizer.index_word[start_index_in_dict] = '<start>'
tokenizer.word_index['<end>'] = end_index_in_dict
tokenizer.index_word[end_index_in_dict] = '<end>'
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
def myfunc(x):
    x.insert(0,start_index_in_dict)
    x.insert(len(x),end_index_in_dict)
    return x

train_seqs = list(map(myfunc, train_seqs))
dev_seqs = list(map(myfunc, dev_seqs))

cap_vector_train = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
cap_vector_dev = tf.keras.preprocessing.sequence.pad_sequences(dev_seqs, padding='post')



# ============================================ PART ==================================



# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 32
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(train_features) // BATCH_SIZE
# Shape of the vector extracted from mfcc is (999, 26), for spectrogram is (999, 166)
# These two variables represent that vector shape
features_shape = 26 # 166 for spectrogram
attention_features_shape = 999

dataset = tf.data.Dataset.from_tensor_slices((train_features, cap_vector_train))
# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# ============================================ PART ==================================



class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 999, embedding_dim => mand dadam (256) )

    # hidden shape == (batch_size, hidden_size).  hidden hidden state ghabli too balas.
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 999, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 999, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights



class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 999, embedding_dim (256))
        self.fc = tf.keras.layers.Dense(256)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x




# ================== part  ===================
class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units ,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    
    context_vector, attention_weights = self.attention(features, hidden)
    
    x = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(x, 1)], axis=-1)
#     print(x.shape)
    
    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


#  start to use Our classes.
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

# ============== part ==========
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

# ============== part ============

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
#  == part ==

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])



# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []

@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from audio to audio
  hidden = decoder.reset_state(batch_size=target.shape[0])
  current_batch_size = img_tensor.shape[0]
  dec_input = tf.expand_dims([float(tokenizer.word_index['<start>'])] * current_batch_size, 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(float(target[:, i]), 1)
#           print(target[:,i])
#           print(predictions[:, i])
#           dec_input = predictions[:, i]
        

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss




EPOCHS = 20

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
      ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))