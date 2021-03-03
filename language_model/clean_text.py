import string
import re
from numpy import array, argmax, random, take
import pandas as pd
import os
import fileinput
from fa import convert
import tqdm

'''
NOTE : This Python file is for Leipzig Data.

HOW TO DO THIS? Put them all in another array and then write a txt file.
OR edit this one?

**************************************** TODO : RUN THIS IN JUPYTER NOTEBOOK  *****************************************************

TODO : seprate each sentence.  =======================  DONE =========================
TODO : seprate sentence and line number.  =======================  DONE =========================
TODO : remove words inside paranthesis  =======================  DONE =========================
TODO : replace nim fasele \u200c with space   =======================  DONE =========================
TODO : Remove Strings with Hour Format from each line.  =======================  DONE =========================
TODO : Remove Strings with Date yy/mm/dd yyyy/mm/dd yy/m/dd Format from each line.  =======================  DONE =========================
TODO : remove every thing except Persian Alphabete and Numbers =======================  DONE =========================
TODO : substitute all kinds of sapce with normal space  =======================  DONE =========================
TODO : remove rows which are just numbers.  =======================  DONE =========================
TODO : Num to word  check =======================  DONE =========================


TODO : FIND SOME CHARACTERS WHICH SHOULD CHANGE TO A PERSIAN CHAR  مثلا انواع ک
    NOTE : ؤ ==> و  =========  DONE ===========
    NOTE : إ حذف شود    =========  DONE ===========
    NOTE :  U+064E	َ	d9 8e	ARABIC FATHA    HAZF    =========  DONE ===========
            U+064F	ُ	d9 8f	ARABIC DAMMA    HAZF   =========  DONE ===========
            U+0650	ِ	d9 90	ARABIC KASRA    HAZF   =========  DONE ===========
            U+0651	ّ	d9 91	ARABIC SHADDA    HAZF =========  DONE ===========
            U+0655	ٕ	d9 95	ARABIC HAMZA BELOW    HAZF   =========  DONE ===========
            ـ حذف بشه  =========  DONE ===========
    NOTE :  ة ===> ه  =========  DONE ===========
    NOTE : ك ===> ک  =========  DONE ===========
    NOTE : ى ===> ی  =========  DONE ===========
    NOTE : ي ===> ی  =========  DONE ===========
    NOTE : ە ===> ه  =========  DONE ===========

TODO NOTE : Find Some Lines in Text wich has weird format Remov them. next lines are for : Leipzig2 => news_2011
        کلید واژه
        نظرات خوانندگان
        
TODO : Remove Multiple Spaces.  =======================  DONE ==============


'''

my_text_url = os.getcwd() + "/voa_text_corpus/"
my_text_file = 'sample.txt'

# Unicodes
persian_alpha_codepoints = '\u0621-\u0628\u062A-\u063A\u0641-\u0642\u0644-\u0648\u064E-\u0651\u0655\u067E\u0686\u0698\u06A9\u06AF\u06BE\u06CC'
persian_num_codepoints = '\u06F0-\u06F9'
arabic_numbers_codepoints = '\u0660-\u0669'
space_codepoints ='\u0020\u2000-\u200F\u2028-\u202F'
additional_arabic_characters_codepoints = '\u0629\u0643\u0649-\u064B\u064D\u06D5'


with open(my_text_url + my_text_file,'r',encoding='utf-8') as r:
    with open(my_text_url + 'bb.txt','w',encoding='utf-8') as w:
        for line in r:
            # remove words between paranthesis and thos paranthesis too.
            line = re.sub(re.compile(r'\([^)]*\)'), '', line)


            # split with tab and remove nim fasele
            line = line.split('\t')[1].replace('\u200c', ' ')

            #Remove Hour formats : hh:mm:ss 
            line = re.sub('\d{2}\:\d{2}:\d{2}', '',line)
            line = re.sub('\d{2}\:\d{2}', '',line)
            line = re.sub('\d{1}\:\d{2}', '',line)

            # Remove Date formats : yy(yy):mm(mm):ss(ss)
            line = re.sub(r'[0-9]{2}[\/,:][0-9]{2}[\/,:][0-9]{2,4}', "", line)
            line = re.sub(r'[۰-۹]{2,4}[\/,:][۰-۹]{2,4}[\/,:][۰-۹]{2,4}', "", line)

            # Just remain persian alphabete and numbers
            line = re.sub(r"[^" + persian_alpha_codepoints +
             persian_num_codepoints +
             additional_arabic_characters_codepoints+
             arabic_numbers_codepoints +
             space_codepoints +
             '1234567890\n' + "]", "", line)


             # change all kinds of sapce with normal space
            line = re.sub(r"[" +
                  space_codepoints+ "]", " ", line)

            # change nim fasele with space
            line = re.sub(r"[\u200c]", " ", line)


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

            # if line is just numbers. ignore it.
            if re.sub(r"[" + space_codepoints + "]", "", line).isnumeric():
                continue

            # # NOTE : THIS PART IS SO Dependent On TEXT TYPE. next Line is for : Leipzig2 => news_2011
            # if 'کلید واژه'  in line:
            #     continue
            # if 'نظرات خوانندگان'  in line:
            #     continue
            # num2word persian
            line = re.sub(r"(\d+)", lambda x : convert(int(x.group(0))), line)


            # TEST
            line = re.sub(r"[\n]", " ", line)
            w.write(line + '\n')



