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
import shutil
from pydub.silence import split_on_silence
import glob

# ========================= Variables =====================
PATH_VIDEOS_SRC = os.getcwd() + '/useful_scripts/clips/'
PATH_EXTRACTED_AUDIO = os.getcwd() + '/useful_scripts/audios/'
PATH_CHUNKED_AUDIOS = os.getcwd() + '/useful_scripts/audios_chunked/'
PATH_NON_STOP_TALKER = os.getcwd() + '/useful_scripts/NON_STOP_TALKER/'

API_TOKEN = 'mgJdvEhrOJrbWIhM'

videos = os.listdir(PATH_VIDEOS_SRC)
audios = os.listdir(PATH_EXTRACTED_AUDIO)
# print(os.listdir(os.getcwd() + '/useful_scripts/clips/'))
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

def extract_mp3_from_wav_command(wav_file, output) :
    command = "ffmpeg -i {wav_file} -ac 1 -ab 64000 -ar 16000 {output}".format(wav_file=wav_file, output=output)
    subprocess.call(command,shell=True)

def extract_mp3_from_wav(wav_files, source_path, dest_path) :
    for wav_file in wav_files:
        extract_mp3_from_wav_command(source_path + wav_file, dest_path + wav_file.split('.')[0] + '.mp3')

def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

def slice_on_sokoot(audio , source_path ,destination_path):
    num = source_path + audio.split('.')[0]
    # Load your audio.
    song = AudioSegment.from_wav(num + '.wav')
    # Split track where the silence is 2 seconds or more and get chunks using 
    # the imported function.
    # print(split_on_silence (song))
    chunks = split_on_silence (
        # Use the loaded audio.
        song, 
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len = 450,
        # Consider a chunk silent if it's quieter than -16 dBFS.
        # (You may want to adjust this parameter.)
        silence_thresh = -22
    )

    # Process each chunk with your parameters
    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=500)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(audio_chunk, -22.0)

        # Export the audio chunk with new bitrate.
        print("Exporting chunk{0}.wav.".format(i))
        normalized_chunk.export(
            "/"+destination_path + audio.split('.')[0] +"_{0}.wav".format(i),
            # bitrate = "192k",
            format = "wav"
        )

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
        # slice_one_audio_files(audio, extracted_audios_path, destination_path, format_flag, 10000)
        slice_on_sokoot(audio, PATH_EXTRACTED_AUDIO, PATH_CHUNKED_AUDIOS)

def update_audios_array(audio_chunkec_path, path_to_current_csv):
    audio_files = os.listdir(audio_chunkec_path)
    if os.path.exists(path_to_current_csv):
        annotated_files = pd.read_csv(path_to_current_csv, error_bad_lines=False)
        annotated_files = annotated_files['wav_filename'].values.tolist()
        for i in list(set(annotated_files)):
            if i in audio_files:
                audio_files.remove(i)
    else:
        pass
    return audio_files


def transcribe_audios(audio_chunkec_path, csv_name):
    PATH_TO_CURRENT_CSV = os.getcwd() + f'/useful_scripts/{csv_name}.csv'
    audios_sanitized = update_audios_array(PATH_CHUNKED_AUDIOS, PATH_TO_CURRENT_CSV)
    if os.path.exists(PATH_TO_CURRENT_CSV):
        dataset_csv = pd.read_csv(PATH_TO_CURRENT_CSV, error_bad_lines=False)
    else:     
        dataset_csv = pd.DataFrame(columns=['wav_filename', 'wav_filesize', 'transcript', 'confidence_level'])
    #  now let's loop over this audio files
    for audio_name in audios_sanitized:
        r = sr.Recognizer()
        chunked_audio = sr.AudioFile(audio_chunkec_path + audio_name)
        audio_file_size = os.stat(audio_chunkec_path + audio_name).st_size
        if audio_file_size <= 500000:
            if audio_file_size <= 50000:
                os.remove(audio_chunkec_path + audio_name)
            else:
                with chunked_audio as source:
                    audio = r.record(source)
                audio_transcribe = ""
                # for confidence : 
                #  -1 means google couldn't detect any speech.
                #  -2 means detected some but all are weak.
                #  -3 Error happend. 
                #  [0, 1] means real probability.
                confidence = -1
                print(audio_name)
                try: 
                    audio_transcribe = r.recognize_google(audio, language='fa-IR', show_all=True)
                except:
                    print("error is happening")
                    audio_transcribe = "ERROR"
                    confidence = -3
                finally:
                    if len(audio_transcribe) > 0 :
                        if 'confidence' in audio_transcribe['alternative'][0]: confidence = audio_transcribe['alternative'][0]['confidence']
                        else : confidence = -2
                        new_row = {'wav_filename' : audio_name,'wav_filesize' : audio_file_size,
                        'transcript' : audio_transcribe['alternative'][0]['transcript'],
                        'confidence_level' : confidence }
                    else :
                        new_row = {'wav_filename' : audio_name,'wav_filesize' : audio_file_size,
                        'transcript' : 'Google_Detected_No_Speech',
                        'confidence_level' : confidence }
                    
                    # # append row to the dataframe
                    dataset_csv = dataset_csv.append(new_row, ignore_index=True)
                    dataset_csv.to_csv( os.getcwd() + "/useful_scripts/" + csv_name + ".csv", index=False)
        else:
            shutil.move(audio_chunkec_path + audio_name, PATH_NON_STOP_TALKER)
    return dataset_csv
  
def transcribe_irani(audio_chunked_path, api_token):
    for audio_name in os.listdir(audio_chunked_path)[0:1]:
        URL = "https://www.iotype.com/api/recognize/file"
        files = {'file': open(audio_chunked_path + audio_name, 'rb')}
        PARAMS = {'token':api_token}  
        r = requests.post(url = URL, params = PARAMS, files=files) 
        data = r.json() 
        print(data)


# ========================================================
                # CHANGE CSV SUFFIX AND FORMAT #
# ========================================================
def change_file_name_suffix(row):
    array_of_wave_file_no_suffix = row.wav_filename.split('.')[:-1]
    array_of_wave_file_no_suffix.insert(len(array_of_wave_file_no_suffix), 'mp3')
    return '.'.join(array_of_wave_file_no_suffix)

def save_csv_file(df, csv_file_name):
    # df['mostafa_koon'] = df.apply(change_file_name_suffix ,axis=1)
    df['wav_filename'] = df.apply(change_file_name_suffix ,axis=1)
    df.to_csv(os.getcwd() + f'/useful_scripts/{csv_file_name}.csv', index=False, encoding='utf-8-sig')


# ===================================  My Own Testing ================================
# extract_videos(videos, PATH_VIDEOS_SRC, PATH_EXTRACTED_AUDIO)

# NOTE : YOU SHOULD FIND THRESHHOLD AND LENGTH FOR EACH PROGRAMM. that is unique for them.
# prepare_audios_to_be_chunked(PATH_EXTRACTED_AUDIO, PATH_CHUNKED_AUDIOS, 'wav')

# extract_mp3_from_wav(audios, PATH_EXTRACTED_AUDIO, PATH_EXTRACTED_AUDIO)


# NOTE : ******************************************* CHANGE NAME CHANGE NAME CHANGE NAME  *******************************************
mycsv = transcribe_audios(PATH_CHUNKED_AUDIOS, 'Taghato_E_Farhang_Part4')
# NOTE : ******************************************* CHANGE NAME CHANGE NAME CHANGE NAME  *******************************************
mycsv.to_csv( os.getcwd() + "/useful_scripts/Taghato_E_Farhang_Part4.csv", index=False)


# dataFrame = pd.read_csv( os.getcwd() + '/useful_scripts/test.csv')
# save_csv_file(dataFrame, 'khandevanehPartElevenMP3')

# ================================================  /My own testing ================================================================


# NOTE : در فراخوانی تابع نام فایل اکسل را درست وارد کنید. نام تکراری با فایل های قبلی ندهید
# NOTE : حتما بررسی کنید فولدر های موجود در ابتدا خالی باشند. به جز فولدر مربوط به فیلم ها یا همان کلیپز
# def extract_chunk_transcribe(csv_name):
#     extract_videos(videos, PATH_VIDEOS_SRC, PATH_EXTRACTED_AUDIO)
#     prepare_audios_to_be_chunked(PATH_EXTRACTED_AUDIO, PATH_CHUNKED_AUDIOS, 'wav')
#     mycsv = transcribe_audios(PATH_CHUNKED_AUDIOS, csv_name)
#     mycsv.to_csv( os.getcwd() + "/useful_scripts/ " + csv_name + ".csv", index=False)


# extract_chunk_transcribe('FILE_NAME')