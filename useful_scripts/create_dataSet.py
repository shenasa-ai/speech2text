
"""
ALL scripts in theis directory are some part of final_script_to_create_dataset.py

If you want to see final result of our work in creating dataset. check final_script_to_create_dataset.py file
"""




import pandas as pd
import os
import speech_recognition as sr
# path to directory of clips. which contain all audio files.
PATH = os.getcwd() + '/useful_scripts/clips/'
# cretea csv file to use in custom models and mozilla deep speech
dataset_csv = pd.DataFrame(columns=['age', 'up_votes', 'client_id', 'accent', 'down_votes', 'gender', 'path', 'sentence'])

#  now let's loop over this audio files
for audio_name in os.listdir(PATH):
    r = sr.Recognizer()
    harvard = sr.AudioFile(PATH + audio_name)

    with harvard as source:
        audio = r.record(source)
    new_row = {'age':22, 'up_votes':0, 'client_id':111, 'accent':'',
     'down_votes' : 0, 'gender': 'male', 'path' : audio_name, 'sentence' : r.recognize_google(audio, language='fa-IR')}
    #append row to the dataframe
    dataset_csv = dataset_csv.append(new_row, ignore_index=True)
    print(audio_name)



dataset_csv.to_csv( os.getcwd() + "/useful_scripts/train.csv", index=False)