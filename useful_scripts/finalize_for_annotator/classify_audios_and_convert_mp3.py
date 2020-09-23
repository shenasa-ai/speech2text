# ========================= Imports =====================
# be careful name of script is so Important. pydub.py or chunk.py will cause error.
from pydub import AudioSegment
from pydub.utils import make_chunks
import pandas as pd
import os
import speech_recognition as sr
import subprocess
import requests
import shutil
from pydub.silence import split_on_silence
import glob
from functools import partial

# ============================================== <Variables> ==========================================

# ========================= <NOTE CHANGE NAME> =============================
ANNOTATIONS_CSV = os.getcwd() + '/Goftegoo_Jamiat_Hal_Ayande.csv'
ANNOTATIONS_CSV_UPDATED = os.getcwd() + '/Goftegoo_Jamiat_Hal_Ayande_D79.csv'
ANNOTATOR_CSV = os.getcwd() + '/Goftegoo_Jamiat_Hal_Ayande_U79.csv'

PATH_CHUNKED_AUDIOS = os.getcwd() + '/Goftegoo_audios_chunked_Jamiat_Hal_Ayande/audios_chunked/'
PATH_Classify_Folder = os.getcwd() + '/Goftegoo_Jamiat_Hal_Ayande_U79_classify_folder/'

PRPGRAM_NAME = 'JAMIAT_HAL_AYANDE'
# ========================= </NOTE CHANGE NAME> =============================

PATH_WAV_TO_MP3 = os.getcwd() + '/wav_to_mp3/'
Final_CSV = pd.DataFrame(columns=['wav_filename', 'wav_filesize', 'transcript', 'confidence_level'])
FIRST_CSV = pd.read_csv(ANNOTATIONS_CSV)
chunked = os.listdir(PATH_CHUNKED_AUDIOS)
classify = os.listdir(PATH_Classify_Folder)
MY_THRESHOLD = 0.79
# print(os.listdir(os.getcwd() + '/useful_scripts/clips/'))

# ============================================== </Variables> ==========================================

def add_specific_name_to_CSV_and_Audios(row):
    os.rename(PATH_CHUNKED_AUDIOS + row.wav_filename, PATH_CHUNKED_AUDIOS + PRPGRAM_NAME + '_' + row.wav_filename)
    return PRPGRAM_NAME + '_' + row.wav_filename

def change_file_name_suffix(row):
    array_of_wave_file_no_suffix = row.wav_filename.split('.')[:-1]
    array_of_wave_file_no_suffix.insert(len(array_of_wave_file_no_suffix), 'mp3')
    return '.'.join(array_of_wave_file_no_suffix)

def save_csv_file(df, csv_file_name):
    # df['mostafa_koon'] = df.apply(change_file_name_suffix ,axis=1)
    df['wav_filename'] = df.apply(change_file_name_suffix ,axis=1)
    df.to_csv(os.getcwd() + f'/useful_scripts/{csv_file_name}.csv', index=False, encoding='utf-8-sig')

def extract_mp3_from_wav_command(wav_file, output) :
    command = "ffmpeg -i {wav_file} -ac 1 -ab 64000 -ar 16000 {output}".format(wav_file=wav_file, output=output)
    subprocess.call(command,shell=True)

def extract_mp3_from_wav(wav_files, source_path, dest_path) :
    for wav_file in wav_files:
        extract_mp3_from_wav_command(source_path + wav_file, dest_path + wav_file.split('.')[0] + '.mp3')

def classify_voices(threshold, row):
    global Final_CSV
    global FIRST_CSV
    if row.confidence_level > threshold:
        # recreate _FINAL
        Final_CSV = Final_CSV.append(row)

        # cut audio to another folder
        shutil.move(PATH_CHUNKED_AUDIOS + row.wav_filename, PATH_Classify_Folder)

        # remove that row from prev dataFrame
        FIRST_CSV.drop(FIRST_CSV[FIRST_CSV.wav_filename == row.wav_filename].index, axis=0, inplace=True)





# ========================= Rename ======================

df = pd.read_csv('./Goftegoo_Jamiat_Hal_Ayande.csv')
df['wav_filename'] = df.apply(add_specific_name_to_CSV_and_Audios, axis=1)
df.to_csv("Goftegoo_Jamiat_Hal_Ayande.csv", index=False, encoding='utf-8-sig')
# ========================= /Rename ======================

# ========================= classify ========================
# FIRST_CSV.apply(partial(classify_voices, MY_THRESHOLD),axis=1)
# if not os.path.isfile(ANNOTATOR_CSV):
#     print("s")
#     Final_CSV.to_csv(ANNOTATOR_CSV,
#      index=False, encoding='utf-8-sig')
#     FIRST_CSV.to_csv(ANNOTATIONS_CSV_UPDATED,
#      index=False, encoding='utf-8-sig')
# ========================= /classify ========================


# ========================= convert to mp3 ========================
# extract_mp3_from_wav(classify, PATH_Classify_Folder, PATH_WAV_TO_MP3)
# ========================= /convert to mp3 ========================




