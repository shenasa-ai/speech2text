"""
ALL scripts in theis directory are some part of final_script_to_create_dataset.py

If you want to see final result of our work in creating dataset. check final_script_to_create_dataset.py file
"""



"""
Okay, let's see how we can use libraries to downalod a video. then convert it to audio. then pass it to google 
and get text of what they are saying.

"""

# first I assume you have installed : 
# 1- ffmpeg ( for convert video to audio). install guid :  http://ubuntuhandbook.org/index.php/2019/08/install-ffmpeg-4-2-ubuntu-18-04/#comment-3671234
# 2- you have an application to cut audio file to smaller ones. I use Audacity
# 3- you have installed youtube-dl lib. install guid https://github.com/ytdl-org/youtube-dl
# 4- you need speech recognition audio file. so let's try 

# step One : download sooriland video use next line commadn in terminal
# youtube-dl https://www.youtube.com/watch?v=59NrpQmwY0s


# Step Two : Convert video to audio :
#  more info at (http://www.savvyadmin.com/extract-audio-from-video-files-to-wav-using-ffmpeg/)
# more info at : ( https://gist.github.com/whizkydee/804d7e290f46c73f55a84db8a8936d74 )
# ffmpeg -i 1.webm -acodec pcm_s16le -ac 2 audio.wav

# cut audio file to shorter audios. use any app. I use Audacity

# export this shorter audio and name it a_10s.wav
# use next line codes to get text of this audio.
import speech_recognition as sr
import os
# Just say where is your directory
PATH = os.getcwd() + '/useful_scripts/'

r = sr.Recognizer()
harvard = sr.AudioFile(PATH + 'a_10s.wav')

with harvard as source:
    audio = r.record(source)
    
for i in  range(1000):
    print(r.recognize_google(audio, language='fa-IR'))
    print(i)
    print("====   ====")

# print(type(audio))

# print(r.recognize_google(audio, language='fa-IR'))
