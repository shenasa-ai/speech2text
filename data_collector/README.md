
# Data Collector 📁

this folder contain my crawler to crawl radio data.


TIP : If your OS is Linux change SEPRATOR variable to '/' if you are in windows set to \\\
 
<br>
 
```
you should have this directories to be able to run this code.

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
    ------ audios_chunked_79
    |
    |
    ------ NONE_STOP_TALKER
    |
    |
    parallel.py ( the  )
    data.csv ( this will be your output )
    data_79.csv ( this will be your output )
```
TIP : when you start to use this code, all folders must be empty except clips.

TIP2 : script will automatically remove videos and useless files . if you want to wave them comment the part of code you need.

what are these folders?
*   **clips** : put your mp4 or mkv files here
*   **audios** : script will convert your mp4(mkv) files and put them here
*   **audios_chunked** : script will chunk audios by silence . and put them here
*   **audios_chunked_79** : script will move audio files with having more than a confidence level number( how to change confidence level? check script you'll find variable.
*   **NONE_STOP_TALKER** : strategy changed. now this folder is doing nothing( to be honest this is doing something. large files will move here or removed. I choose removing you can keep them. check script)


## Run the code ⚙️

put your clips in clips directory. make sure other folders are empty.change the csv file name in 2 last line of the code. then run it. :)
