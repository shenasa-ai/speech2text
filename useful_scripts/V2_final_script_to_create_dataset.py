# ========================= How To Use this Script =====================

# TODO : Remove clips and audios files . each time you use
# TODO : You Have to re fill clips and audios directory.
# TODO : name properly each csvfile you will create as output.

"""
directory guid to use this script

-parent dir( here is useful_scripts )
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
import requests

# ========================= Variables =====================
PATH_VIDEOS_SRC = os.getcwd() + '/useful_scripts/clips/'
PATH_EXTRACTED_AUDIO = os.getcwd() + '/useful_scripts/audios/'
PATH_CHUNKED_AUDIOS = os.getcwd() + '/useful_scripts/audios_chunked/'

API_TOKEN = 'mgJdvEhrOJrbWIhM'

videos = os.listdir(PATH_VIDEOS_SRC)
audios = os.listdir(PATH_EXTRACTED_AUDIO)
# print(os.listdir(os.getcwd() + '/useful_scripts/audios/'))
# ======================================= Functions ================================
def extract_audio(video,output):
    # to add flag that overrride if file exist or not : -y or -n
    # -y means yes override
    # -n means neer override
    command = "ffmpeg -i {video} -acodec pcm_s16le -ac 1 -ar 16000 {output}".format(video=video, output=output)
    subprocess.call(command,shell=True)

def extract_videos(videos_array, source_path, dest_path):
    for video in videos:
        if video.split('.')[-1] == "mp4" or video.split('.')[-1] == 'mkv' or video.split('.')[-1] == 'webm':
            extract_audio(source_path + video ,dest_path + video.split('.')[0] + '.wav')

def extract_audio_mp3_command(video,output):
    # to add flag that overrride if file exist or not : -y or -n
    # -y means yes override
    # -n means neer override    ffmpeg -i video.mp4 -f mp3 -ab 192000 -vn music.mp3
    command = "ffmpeg -i {video} -f mp3 -ab 192000 -vn {output}".format(video=video, output=output)
    subprocess.call(command,shell=True)

def extract_videos_mp3_type(videos_array, source_path, dest_path):
    for video in videos:
        if video.split('.')[-1] == "mp4" or video.split('.')[-1] == 'mkv' or video.split('.')[-1] == 'webm':
            extract_audio_mp3_command(source_path + video ,dest_path + video.split('.')[0] + '.mp3')


def extract_mp3_from_wav_command(wav_file, output) :
    command = "ffmpeg -i {wav_file} -ac 1 -ab 64000 -ar 16000 {output}".format(wav_file=wav_file, output=output)
    subprocess.call(command,shell=True)

def extract_mp3_from_wav(wav_files, source_path, dest_path) :
    for wav_file in wav_files:
        extract_mp3_from_wav_command(source_path + wav_file, dest_path + wav_file.split('.')[0] + '.mp3')


def slice_one_audio_files(audio_name, extracted_audios_path, destination_path, format_flag, chunk_length_ms= 10000):
    # Imagin you have a long audio file.
    myaudio = AudioSegment.from_file(extracted_audios_path + audio_name , format_flag) 
    #  Here I said 10 second chunks. you can change them        
    chunk_length_ms = chunk_length_ms # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of ten sec

    #Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
        chunk_name = audio_name + "chunk{0}.".format(i) + format_flag
        print("exporting", chunk_name)
        # You should have a directory with this name to avoid error
        chunk.export(destination_path + chunk_name, format="wav")

def prepare_audios_to_be_chunked(extracted_audios_path, destination_path, format_flag):
    if format_flag == 'mp3':
        audios = [i for i in os.listdir(extracted_audios_path) if i.split('.')[-1] == 'mp3']
    elif format_flag == 'wav':
        audios = [i for i in os.listdir(extracted_audios_path) if i.split('.')[-1] == 'wav']
    
    for audio in audios:
        print(audio)
        slice_one_audio_files(audio, extracted_audios_path, destination_path, format_flag, 10000)

def transcribe_audios(audio_chunkec_path):
    dataset_csv = pd.DataFrame(columns=['wav_filename', 'wav_filesize', 'transcript'])
    #  now let's loop over this audio files
    for audio_name in os.listdir(audio_chunkec_path):
        r = sr.Recognizer()
        chunked_audio = sr.AudioFile(audio_chunkec_path + audio_name)
        audio_file_size = os.stat(audio_chunkec_path + audio_name).st_size
        with chunked_audio as source:
            audio = r.record(source)
        audio_transcribe = ""
        print(audio_name)
        print(audio_transcribe)
        try: 
            audio_transcribe = r.recognize_google(audio, language='fa-IR')
        except:
            print("error is happening")
            audio_transcribe = "ERROR"
        finally:
            new_row = {'wav_filename' : audio_name,'wav_filesize' : audio_file_size,
            'transcript' : audio_transcribe}
            # append row to the dataframe
            dataset_csv = dataset_csv.append(new_row, ignore_index=True)
            # NOTE : ******************************************* CHANGE NAME CHANGE NAME CHANGE NAME  *******************************************
            dataset_csv.to_csv( os.getcwd() + "/useful_scripts/dorehamiPartThree.csv", index=False)
    return dataset_csv
  

def transcribe_irani(audio_chunked_path, api_token):
    for audio_name in os.listdir(audio_chunked_path)[0:1]:
        URL = "https://www.iotype.com/api/recognize/file"
        files = {'file': open(audio_chunked_path + audio_name, 'rb')}
        PARAMS = {'token':api_token}  
        r = requests.post(url = URL, params = PARAMS, files=files) 
        data = r.json() 
        print(data)



# ===================================   use functions in your needs order ================================
# extract_videos(videos, PATH_VIDEOS_SRC, PATH_EXTRACTED_AUDIO)

# extract_mp3_from_wav(audios, PATH_EXTRACTED_AUDIO, PATH_EXTRACTED_AUDIO)

# prepare_audios_to_be_chunked(PATH_EXTRACTED_AUDIO, PATH_CHUNKED_AUDIOS, 'wav')

# NOTE : ******************************************* CHANGE NAME CHANGE NAME CHANGE NAME  *******************************************
mycsv = transcribe_audios(PATH_CHUNKED_AUDIOS)
# NOTE : ******************************************* CHANGE NAME CHANGE NAME CHANGE NAME  *******************************************
mycsv.to_csv( os.getcwd() + "/useful_scripts/dorehamiPartThree.csv", index=False)


# transcribe_irani(PATH_CHUNKED_AUDIOS, API_TOKEN)


#         # command = "CURL -v -F token={api_token} -F audio=@{audio_name} https://www.iotype.com/api/recognize/file".format(api_token=api_token, audio_name = audio_chunked_path + audio_name)
        # subprocess.call(command,shell=True)