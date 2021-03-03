import pandas as pd
import os
import string
import re
from numpy import array, argmax, random, take
import pandas as pd
import os
import fileinput
from fa import convert
from sklearn.utils import shuffle
import numpy as np

# Unicodes
persian_alpha_codepoints = '\u0621-\u0628\u062A-\u063A\u0641-\u0642\u0644-\u0648\u064E-\u0651\u0655\u067E\u0686\u0698\u06A9\u06AF\u06BE\u06CC'
persian_num_codepoints = '\u06F0-\u06F9'
arabic_numbers_codepoints = '\u0660-\u0669'
space_codepoints ='\u0020\u2000-\u200F\u2028-\u202F'
additional_arabic_characters_codepoints = '\u0629\u0643\u0649-\u064B\u064D\u06D5'


def clean_row(line):
    line = re.sub(re.compile(r'\([^)]*\)'), '', line)

    # split with tab and remove nim fasele
    line = line.replace('\u200c', ' ')

    # Just remain persian alphabete and numbers
    line = re.sub(r"[^" + persian_alpha_codepoints +
     persian_num_codepoints +
     additional_arabic_characters_codepoints+
     arabic_numbers_codepoints +
     space_codepoints +
     '1234567890\n' + "]", "", line)

    line = re.sub(r"[" +
                  space_codepoints+ "]", " ", line)


    # Remove or Substitude some characters.   ـ
    # این نتوین   ً و ء حفظ میشه
    line = re.sub(r"[" + 'ّ'  + 'ٌ' + 'ـ' + 'َ' + 'ِ' + 'ٕ'  + 'ٍ' + 'ُ' + 'ْ' + "]", '', line)
    # Be careful this editor VSC shows the character in wrong order
    line = re.sub('ؤ', 'و', line)
    line = re.sub('ة', 'ه', line)
    line = re.sub('ك', 'ک', line)
    line = re.sub('ى', 'ی', line)
    line = re.sub('ي', 'ی', line)
    line = re.sub('ە', 'ه', line)
    line = re.sub('ئ', 'ی', line)
    line = re.sub('أ', 'ا', line)
    line = re.sub('إ', 'ا', line)



    # remove multiple spaces with just one space 
    line = re.sub(' +', ' ', line)

    # remove multiple strings from first and last of lines
    line = line.strip()
    line = re.sub(r"(\d+)", lambda x : convert(int(x.group(0))), line)
    return line

# ========================== Clean Selected Data ===================================
myNewTrain = pd.read_csv('~/Masoud/mozilla data/myNewTrain.tsv', sep='\t')
myNewTrain['sentence'] = myNewTrain.sentence.apply(lambda x : clean_row(x))

myNewTrain.to_csv('./myNewTrainCleaned.tsv', sep='\t', index=False)


# ========================== /Clean Selected Data ===================================