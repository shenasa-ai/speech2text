"""
ALL scripts in theis directory are some part of final_script_to_create_dataset.py

If you want to see final result of our work in creating dataset. check final_script_to_create_dataset.py file
"""



import pandas as pd
import os
import speech_recognition as sr
import subprocess

# ffmpeg -i 111.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav


PATH_SRC = os.getcwd() + '/useful_scripts/clips/'
PATH_DEST = os.getcwd() + '/useful_scripts/audios/'

videos = os.listdir(PATH_SRC)

def extract_audio(video,output):
    command = "ffmpeg -i {video} -acodec pcm_s16le -ac 1 -ar 16000 {output}".format(video=video, output=output)
    subprocess.call(command,shell=True)

# extract_audio(PATH + 't1.mp4',PATH+ 't1.wav')
def extract_videos(videos_array, source_path, dest_path):
    for video in videos:
        if video.split('.')[-1] == "mp4" or video.split('.')[-1] == 'mkv':
            extract_audio(source_path + video ,dest_path + video.split('.')[0] + '.wav')

extract_videos(videos, PATH_SRC, PATH_DEST)
