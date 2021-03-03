# Speech to Text 🚀
This repo create ASR system by using toolkits PLUS own implementations.


Toolkits used: 
- [Mozilla Deep Speech](#mozilla-deep-speech)
- [DeepSpeech2](#deepspeech2) 
- [Wave2vec](#wav2vec-20) 
- [Jasper](#jasper) 
- [Our Implementations](#my-own-implementations)

<br>

# Prerequisites 📋

You need to know about RNNs, Attention mechanism, CTC, a little pandas/numpy, Tensorflow, KERAS and NLP Stuff(e.g. transformers, text cleaning etc. ). Also knowing about Spectrograms, MFCC, Filter Bank will help you to understand preprocess of audios. <br><br>
if you don't know any of these stuff you can check Wiki page there are many links

<br>

# Models Output 🎯
<table width="100%">
 <tr>
   <th width="30%"> model Name  </th> 
   <th width="40%"> DataSet </th> 
   <th width="70%"> Loss  </th> 
 </tr>

<tr>
   <td width="25%"> our own implementations : (first try) </td>
   <td width="25%"> common_voice_en (400h) </td>
 <td width="25%"> 74 ( So bad results / many models tested. ) </td>
 </tr>

 <tr>
   <td width="25%">Deep Speech 1 : Mozilla (first try) </td>
   <td width="25%"> common_voice + tv programms + radio programs (totally 300h) </td>
 <td width="25%"> 28 </td>
 </tr>

  <tr>
   <td width="25%">Deep Speech 1 : Mozilla (second try) </td>
   <td width="25%"> common_voice + tv programms + radio programs (totally 300h) </td>
 <td width="25%"> 25</td>
 </tr>

  <tr>
   <td width="25%">Deep Speech 1 : Mozilla + Transfer Learning (third try) </td>
   <td width="25%"> common_voice + tv programms + radio programs (totally 300h) </td>
 <td width="25%"> 24</td>
 </tr>

  <tr>
   <td width="25%">Deep Speech 1 : Mozilla + Transfer Learning (third try) </td>
   <td width="25%"> common_voice + tv programms + radio programs (totally 1000h) </td>
 <td width="25%"> 22 </td>
 </tr>

  <tr>
   <td width="25%">Deep Speech 2 : Tnesor_Speech (first try) </td>
   <td width="25%"> common_voice + tv programms + radio programs (totally 1000h) </td>
 <td width="25%"> Soon ... </td>
 </tr>

  <tr>
   <td width="25%">Wave2vec 2.0 </td>
   <td width="25%"> common_voice + tv programms + radio programs (totally 1000h) </td>
 <td width="25%"> Soon ... </td>
 </tr>

   <tr>
   <td width="25%"> Jasper </td>
   <td width="25%"> common_voice + tv programms + radio programs (totally 1000h) </td>
 <td width="25%"> Soon ... </td>
 </tr>

</table>


# Dataset we used 📁
there are many public datasets for English. But for persian there is not enough and free STT dataset.So we created our own data crawler for collecting data.

[common voice dataset](https://voice.mozilla.org/en/datasets) is a rich free dataset.<br>



# Mozilla Deep Speech
### clone and download common voice dataset 
To use this toolkit you must first do what  [this link ](https://deepspeech.readthedocs.io/en/latest/TRAINING.html) says or follow short Installation blow.

Currently we are using DeepSpeech V0.9.3

#### Short Installation
- Clone and create virtual enviroments
```
git clone --branch v0.9.3 https://github.com/mozilla/DeepSpeech
cd DeepSpeech
pip3 install --upgrade pip==20.2.2 wheel==0.34.2 setuptools==49.6.0
pip3 install --upgrade -e .
pip install tensorflow-gpu==1.15.4
```

After cloning and install dependecies you need [common voice dataset](https://voice.mozilla.org/en/datasets).
Download the proper language and then preprocessing. all the steps for preprocess common voice dataset is documented in [here](https://deepspeech.readthedocs.io/en/latest/TRAINING.html) too.

- Start training
```
‫‪python3‬‬ ‫‪DeepSpeech.py‬‬ ‫‪--train_files‬‬ ‫‪../data/CV/en/clips/train.csv‬‬ ‫‪--dev_files‬‬ \
‫‪../data/CV/en/clips/dev.csv‬‬ ‫‪--test_files‬‬ ‫‪../data/CV/en/clips/test.csv‬‬
```
- Test your model with this `TEST_STT_MODEL.ipynb` file.

Pretrained checkpoints of model is [here] (https://drive.google.com/drive/folders/1MEzzao1ksPcQzuMOU2KNGPfyh49Xs1Ll?usp=sharing).

### your own dataset
if you want to create your own dataset you need these TIPS : 
*   all your audio files must be in wav format
*   your audios must be in :  16khz / 16bitpersample / mono / no noise ( optional ) 
*   don't use more than 30 minutes of audio files recorded by one person
*   at least you must have 300h of data.
*   you need a csv file for all the audio files you have. you can see the structure of csv file after you preprocessed the common voice files by import_cv2.py( this python file is in Deepspeech repo. you'll have it after cloning)
*   and also you need to clean your transcripts (Some more NLP task :) ).
*   (optional) if you want to use my crawler to create dataset you can check crawler.py (in crawler folder ) and parallel.py(in data_collector folder) file.
*   (optional) if you want to use my transcription desktop application you can check [ciacada repository](https://github.com/shenasa-ai/cicada-audio-annotation-tool)
*  ( optional ) you can use my scripts to clean transcripts and make final csv file. they are available in github
* don't worry about creating spectrogram/FastFourier/MFCC/Filterbank , Mozilla will do it all for you.
* TODO :  need to remember more tips.

### Language model
you need language model for testing the model. the language model is trained on [Kenlm](https://github.com/kpu/kenlm)

the steps to train language model is [here](https://deepspeech.readthedocs.io/en/v0.9.3/Scorer.html)

text file size : 2GB

test file to optimize : 20M


TIP : You can use just this language model :| not any other.
<br>

after all these steps it means you have your dataset ready and you want to train . if you want to train on English there is no more steps **BUT** if you are in another language ( like persian ) you need to check transfer learning part of [this link](https://deepspeech.readthedocs.io/en/latest/TRAINING.html#transfer-learning-new-alphabet)
TIP : don't forget to change alphabete.txt

Question : can I use Other languages checkpoints to start transfer learning? Sure, do it. But remember to drop weights of last N layers.

you may need the meaning of flags to use all the abilities of mozilla deep speech. [check their Documentation](https://deepspeech.readthedocs.io/en/v0.9.3/Flags.html)

Tip : if you faced with CUDA/CudNN errors. try to use conda and install proper versions.

<br>
<br>
<hr>

## DeepSpeech2
using TensorSpeech
[Link to repository](https://github.com/TensorSpeech/TensorFlowASR)
their repo is really complete and you can pass their steps to train a model but I will say some tips : 

*   to change any option you need to change config.yml file
*   Remember to change alphabetes. you need to change the vocabulary in config.yml file
* the dataset in this repo is a little different. you must have tsv file. and columns hav different names and values
*   (optional) to prepare your own dataset for this approach you can use my script. it is available in github

### Language model
in this repo you can use any language model. even kenlm.


<br>
<hr>

## Wav2vec 2.0
using facebook fairseq toolkit
SOON 


<br>
<hr>

## Jasper
using Nvidia NeMo Toolkit
Soon

## My Own Implementations

### Installation 🔧

Some libraries you need to install. I'll list them here ( These are the most important ) : <br>
* ffmpeg
* pydub
* python_speech_features
* numpy V1.18.1
* pandas V1.0.1
* tensorflow  V2.1.0
* sklearn V0.22.2.post1
* librosa V0.7.2
<br> <br>
In next line I'll show how to install three of them ( I used these commands to install in Kaggle Notebooks too.)<br>

```
pip install pydub
```

``` 
pip install python_speech_features
```

``` 
!apt-get install -y ffmpeg
``` 

the codes are developed to use commonvoice data. make sure your data are in that format.


# Hyperparameters

all the steps for training models are documented in a txt file. they are uploaded in google drive. you can check there if needed.

<br>

# WIKI page 📖
Visit our wiki page for more info about Tutorials, useful Links, Hardware Info, Result and other things. 


<br>

# Contributing 🖇️

If you want to help us for better models and new approaches, please contact us, we will be happy
<br>
Email : Soon...