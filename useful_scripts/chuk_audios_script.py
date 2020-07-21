"""
ALL scripts in theis directory are some part of final_script_to_create_dataset.py

If you want to see final result of our work in creating dataset. check final_script_to_create_dataset.py file
"""




# be careful name of script is so Important. pydub.py or chunk.py will cause error.
from pydub import AudioSegment
from pydub.utils import make_chunks
# Imagin you have a long audio file.
myaudio = AudioSegment.from_file("audio.wav" , "wav") 
#  Here I said 10 second chunks. you can change them
chunk_length_ms = 10000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of ten sec

#Export all of the individual chunks as wav files

for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    print("exporting", chunk_name)
    # You should have a directory with this name to avoid error
    chunk.export('./audios_chunked/' + chunk_name, format="wav")