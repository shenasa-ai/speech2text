# ========================= How To Use this Script =====================

# TODO : Remove clips and audios files . each time you use
# TODO : You Have to re fill clips and audios directory.
# TODO : name properly each csvfile you will create as output.

"""
directory guid to use this script

-parent dir
    |
    |
    ------ clips
    |
    |
    ------ audios
    |
    |
    ------ audios_chunked
    |
    |
    myScript.py ( we are here right now )
    data.csv ( this will be your output )
"""
"""
Steps to use this script each time.

1- paste your videos in the clips directory

2- use related function to extracat audio from videos

3- use related functions to chunk audios

4- use related function to transcribe audio files

5- output your csv file
"""

# ========================= Imports =====================
# be careful name of script is so Important. pydub.py or chunk.py will cause error.
from pydub import AudioSegment
from pydub.utils import make_chunks
import pandas as pd
import os
import speech_recognition as sr
import subprocess


# ========================= Variables =====================
PATH_VIDEOS_SRC = os.getcwd() + '/useful_scripts/clips/'
PATH_EXTRACTED_AUDIO = os.getcwd() + '/useful_scripts/audios/'
PATH_CHUNKED_AUDIOS = os.getcwd() + '/useful_scripts/audios_chunked/'

videos = os.listdir(PATH_VIDEOS_SRC)


# ======================================= Functions ================================
def extract_audio(video,output):
    # to add flag that overrride if file exist or not : -y or -n
    # -y means yes override
    # -n means neer override
    command = "ffmpeg -i {video} -acodec pcm_s16le -ac 1 -ar 16000 {output}".format(video=video, output=output)
    subprocess.call(command,shell=True)

def extract_videos(videos_array, source_path, dest_path):
    for video in videos:
        if video.split('.')[-1] == "mp4" or video.split('.')[-1] == 'mkv':
            extract_audio(source_path + video ,dest_path + video.split('.')[0] + '.wav')

def slice_one_audio_files(audio_name, extracted_audios_path, destination_path, chunk_length_ms= 10000):
    # Imagin you have a long audio file.
    myaudio = AudioSegment.from_file(extracted_audios_path + audio_name , "wav") 
    #  Here I said 10 second chunks. you can change them        
    chunk_length_ms = chunk_length_ms # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of ten sec

    #Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
        chunk_name = audio_name + "chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        # You should have a directory with this name to avoid error
        chunk.export(destination_path + chunk_name, format="wav")

def prepare_audios_to_be_chunked(extracted_audios_path, destination_path):
    audios = os.listdir(extracted_audios_path)
    for audio in audios:
        slice_one_audio_files(audio, extracted_audios_path, destination_path, 10000)

def transcribe_audios(audio_chunkec_path):
    dataset_csv = pd.DataFrame(columns=['wav_filename', 'wav_filesize', 'transcript'])
    #  now let's loop over this audio files
    for audio_name in os.listdir(audio_chunkec_path):
        r = sr.Recognizer()
        chunked_audio = sr.AudioFile(audio_chunkec_path + audio_name)
        audio_file_size = os.stat(audio_chunkec_path + audio_name).st_size
        with chunked_audio as source:
            audio = r.record(source)
        audio_transcribe = r.recognize_google(audio, language='fa-IR')
        print('========================================')
        print(audio_transcribe)
        print(audio_name)
        print('========================================')
        new_row = {'wav_filename' : audio_name,'wav_filesize' : audio_file_size,
         'transcript' : audio_transcribe}
        #append row to the dataframe
        dataset_csv = dataset_csv.append(new_row, ignore_index=True)
        # print(audio_name)
    return dataset_csv

# ===================================   use functions in your needs order ==========================
# extract_videos(videos, PATH_VIDEOS_SRC, PATH_EXTRACTED_AUDIO)

# prepare_audios_to_be_chunked(PATH_EXTRACTED_AUDIO, PATH_CHUNKED_AUDIOS)

mycsv = transcribe_audios(PATH_CHUNKED_AUDIOS)
# NOTE : CHANGE NAME CHANGE NAME CHANGE NAME CHANGE NAME
mycsv.to_csv( os.getcwd() + "/useful_scripts/dorehami.csv", index=False)
