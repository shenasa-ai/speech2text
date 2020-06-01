"""
    In this code I'll show you how to use speech_recognition lib for persian and english files.
    this library is used to transcribe your audio files. and can use many WebSerivces like :
    Google,IBM,Microsoft,..

    Here I used google speech recognition

    TIP : Google cloud speech recognition is different from google speech recognition
    NOTE: You need INTERNET access to use google speech recognition.
    NOTE : YOUR AUDIO TYPE MUST BE in WAV Format
"""
# import lib
import speech_recognition as sr
import os
# Just say where is your directory
PATH = os.getcwd() + '/useful_scripts/Speech_Recognition_Lib_usage/'
# create object
r = sr.Recognizer()
# read those 2 files
persian = sr.AudioFile(PATH + 'common_voice_fa_20239522.wav')
harvard = sr.AudioFile(PATH + 'harvard.wav')

# get info of that audio with r.record
with harvard as source:
    audio = r.record(source)

with persian as source2:
    audio2 = r.record(source2)

# print type of objects
print(type(audio))
# Print text of my audio
print(r.recognize_google(audio, language='en_US'))

print(type(audio2))
print(r.recognize_google(audio2, language='fa-IR'))
