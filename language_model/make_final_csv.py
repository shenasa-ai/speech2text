import pandas as pd
import os
import string
import re
from numpy import array, argmax, random, take
import pandas as pd
import os
import fileinput
from fa import convert
from sklearn.utils import shuffle
import numpy as np

# Unicodes
persian_alpha_codepoints = '\u0621-\u0628\u062A-\u063A\u0641-\u0642\u0644-\u0648\u064E-\u0651\u0655\u067E\u0686\u0698\u06A9\u06AF\u06BE\u06CC'
persian_num_codepoints = '\u06F0-\u06F9'
arabic_numbers_codepoints = '\u0660-\u0669'
space_codepoints ='\u0020\u2000-\u200F\u2028-\u202F'
additional_arabic_characters_codepoints = '\u0629\u0643\u0649-\u064B\u064D\u06D5'






# =====================================   Select 120 of each speaker =============================
# pure_validated = pd.read_csv('~/Masoud/mozilla data/fa/corpus/fa/validated.tsv', sep='\t')
# pure_validated.groupby('client_id').apply(
#                      lambda x: x.iloc[:120]).reset_index(drop=True).to_csv('./test.tsv', sep='\t', index=False)
# =====================================   /Select 120 of each speaker =============================


def clean_row(line):
    line = re.sub(re.compile(r'\([^)]*\)'), '', line)

    # split with tab and remove nim fasele
    line = line.replace('\u200c', ' ')

    # Just remain persian alphabete and numbers
    line = re.sub(r"[^" + persian_alpha_codepoints +
     persian_num_codepoints +
     additional_arabic_characters_codepoints+
     arabic_numbers_codepoints +
     space_codepoints +
     '1234567890\n' + "]", "", line)

    line = re.sub(r"[" +
                  space_codepoints+ "]", " ", line)


    # Remove or Substitude some characters.   ـ
    # این نتوین   ً و ء حفظ میشه
    line = re.sub(r"[" + 'ّ'  + 'ٌ' + 'ـ' + 'َ' + 'ِ' + 'ٕ'  + 'ٍ' + 'ُ' + 'ْ' + "]", '', line)
    # Be careful this editor VSC shows the character in wrong order
    line = re.sub('ؤ', 'و', line)
    line = re.sub('ة', 'ه', line)
    line = re.sub('ك', 'ک', line)
    line = re.sub('ى', 'ی', line)
    line = re.sub('ي', 'ی', line)
    line = re.sub('ە', 'ه', line)
    line = re.sub('ئ', 'ی', line)
    line = re.sub('أ', 'ا', line)
    line = re.sub('إ', 'ا', line)



    # remove multiple spaces with just one space 
    line = re.sub(' +', ' ', line)

    # remove multiple strings from first and last of lines
    line = line.strip()
    line = re.sub(r"(\d+)", lambda x : convert(int(x.group(0))), line)
    return line

# ========================== Clean Selected Data ===================================
# myNewTrain = pd.read_csv('~/Masoud/mozilla data/myNewTrain.tsv', sep='\t')
# myNewTrain['sentence'] = myNewTrain.sentence.apply(lambda x : clean_row(x))

# myNewTrain.to_csv('./myNewTrainCleaned.tsv', sep='\t', index=False)


# ========================== /Clean Selected Data ===================================


# ============================   SPLIT IT TO TRAIN AND TESTDEV ===> THEN SPLIT TESTDEV TO TEST AND DEV  ===================================
# myNewTrainCleaned = pd.read_csv('~/Masoud/mozilla data/myNewTrainCleaned.tsv', sep='\t')

# train=myNewTrainCleaned.sample(frac=0.75,random_state=200) #random state is a seed value
# test_dev=myNewTrainCleaned.drop(train.index)

# dev=test_dev.sample(frac=0.8,random_state=200) #random state is a seed value
# test=test_dev.drop(dev.index)

# train.to_csv('./mozilla_wav_train.tsv', sep='\t', index=False)
# dev.to_csv('./mozilla_wav_dev.tsv', sep='\t', index=False)
# test.to_csv('./mozilla_wav_test.tsv', sep='\t', index=False)

# ============================   / SPLIT IT TO TRAIN AND TESTDEV ===> THEN SPLIT TESTDEV TO TEST AND DEV  ===================================


# ======================================= Concat my csv file with mozilla files  =====================================

# train_mozilla = pd.read_csv('~/Masoud/mozilla data/fa/corpus/fa/clips/train.csv' )
# test_mozilla = pd.read_csv('~/Masoud/mozilla data/fa/corpus/fa/clips/test.csv' )
# dev_mozilla = pd.read_csv('~/Masoud/mozilla data/fa/corpus/fa/clips/dev.csv' )

# train_radio = pd.read_csv('~/Masoud/mozilla data/my_wav_train.csv' )
# test_radio = pd.read_csv('~/Masoud/mozilla data/my_wav_test.csv' )
# dev_radio = pd.read_csv('~/Masoud/mozilla data/my_wav_dev.csv' )

# # print(train_mozilla.shape)
# # print(test_mozilla.shape)
# # print(dev_mozilla.shape)

# # print(train_radio.shape)
# # print(test_radio.shape)
# # print(dev_radio.shape)

# train_mr = shuffle(pd.concat([train_radio, train_mozilla]))
# test_mr = shuffle(pd.concat([test_mozilla, test_radio]))
# dev_mr = shuffle(pd.concat([dev_mozilla, dev_radio]))

# train_mr.to_csv('./train_mr.csv', index=False)
# test_mr.to_csv('./test_mr.csv', index=False)
# dev_mr.to_csv('./dev_mr.csv', index=False)


# =====================================  / Concat my csv file with mozilla files  ================================



# =============================================  remove rows with no transcript ======================================
# train_mr = pd.read_csv('~/Masoud/mozilla_data/fa/corpus/fa/clips/train_mr.csv' )
# test_mr = pd.read_csv('~/Masoud/mozilla_data/fa/corpus/fa/clips/test_mr.csv' )
# dev_mr = pd.read_csv('~/Masoud/mozilla_data/fa/corpus/fa/clips/dev_mr.csv' )

# print(train_mr.isnull().sum())
# print(test_mr.isnull().sum())
# print(dev_mr.isnull().sum())

# train_mr.dropna(inplace=True).to_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/train_mr.csv', index=False)
# test_mr.dropna(inplace=True).to_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/test_mr.csv', index=False)
# dev_mr.dropna(inplace=True).to_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/dev_mr.csv', index=False)
# Gode_Khatereh_radio-goftego-96_02_14-16_30_199.wav


# print(train_mr.isnull().sum())
# print(test_mr.isnull().sum())
# print(dev_mr.isnull().sum())


# =============================================  / remove rows with no transcript ======================================


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   TENSORSPEECH APPROACH !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# =============================================  / add duration and create tensroSpeech tsv ======================================
# import librosa
# train_mr = pd.read_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/train_mr.csv' )
# test_mr = pd.read_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/test_mr.csv' )
# dev_mr = pd.read_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/dev_mr.csv' )

# train_mr_tensor = pd.read_csv('./train_mr.csv' )
# test_mr_tensor = pd.read_csv('./test_mr.csv' )
# dev_mr_tensor = pd.read_csv('./dev_mr.csv' )

# # test.to_csv('./mozilla_wav_test.tsv', sep='\t', index=False)

# wav_files_folder = '/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/'

# print(train_mr.isnull().sum())

# def calc_duration(row):
#     y, sr = librosa.load(wav_files_folder + row, sr=None)
#     return librosa.get_duration(y, sr)

# train_mr['wav_filesize'] = train_mr['wav_filename'].apply(lambda x : calc_duration(x))
# train_mr.rename({"wav_filename": "PATH", "wav_filesize": "DURATION", "transcript": "TRANSCRIPT"}).to_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/train_mr_tensor.tsv', sep='\t', index=False)
# train_mr.to_csv('./train_mr_tensor.tsv', sep='\t', index=False)


# test_mr['wav_filesize'] = test_mr['wav_filename'].apply(lambda x : calc_duration(x))
# test_mr.rename({"wav_filename": "PATH", "wav_filesize": "DURATION", "transcript": "TRANSCRIPT"}).to_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/test_mr_tensor.tsv', sep='\t', index=False)
# test_mr.rename({"wav_filename": "PATH", "wav_filesize": "DURATION", "transcript": "TRANSCRIPT"}).to_csv('./test_mr_tensor.tsv', sep='\t', index=False)


# dev_mr['wav_filesize'] = dev_mr['wav_filename'].apply(lambda x : calc_duration(x))
# dev_mr.rename({"wav_filename": "PATH", "wav_filesize": "DURATION", "transcript": "TRANSCRIPT"}).to_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/dev_mr_tensor.tsv', sep='\t', index=False)
# dev_mr.rename({"wav_filename": "PATH", "wav_filesize": "DURATION", "transcript": "TRANSCRIPT"}).to_csv('./dev_mr_tensor.tsv', sep='\t', index=False)





# =============================================  / add duration and create tensroSpeech tsv ======================================

# =============================================  / create small  csv ======================================


# small_test = test_mr.sample(frac=0.1,random_state=200)
# small_train = train_mr.sample(frac=0.01,random_state=200)
# small_dev = dev_mr.sample(frac=0.03,random_state=200)

# small_test.to_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/test_mr_small.csv', index=False)
# small_train.to_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/train_mr_small.csv', index=False)
# small_dev.to_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/dev_mr_small.csv', index=False)


# test_mr = pd.read_csv('/home/robodoc/Masoud/mozilla_data/fa/corpus/fa/clips/test_mr_small.csv' )
# print(test_mr.shape)



# =============================================  / create small  csv ======================================
