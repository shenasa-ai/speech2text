import pandas as pd
import os
import speech_recognition as sr
import subprocess

def extract_audio(video,output):
    command = "ffmpeg -i {video} -ac 1  -f flac -vn {output}".format(video=video, output=output)
    subprocess.call(command,shell=True)

extract_audio('dm.MOV','dm-new.flac')