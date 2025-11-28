#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:38:54 2023

@author: m1dcs04
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:12:44 2023
"""
import os
from bs4 import BeautifulSoup, NavigableString, Tag
import requests
import pandas as pd
from tqdm import tqdm
import re
import string
import numpy as np
import glob
import nltk
import nameparser
from nltk.corpus import stopwords
import requests
stopwords = set(stopwords.words('english'))
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#---------------------


# Parent path
parent_path = 'fomc_transcript/'

# Transcript directory
transcript_road = 'fomc_transcript/data/processed/Transcripts/'
transcript_ls = r'fomc_transcript/data/processed/Transcripts/*.csv'
transcript_files = glob.glob(transcript_ls)

#------------------------------------------------------------------------------
laughs_all = []
datee = []
transcript_bodies = []

laughs_eyes_total = []
laughs_yous_total = []
laughs_nonpersons_total = []

laughter_word = ["[Laughter]", "[laughter]"]
myself_pronouns = [" i "," i'll ", " me ", " mine ", " my ", " myself "]
you_pronouns = [" you ", " you're ", " you are "]
nonperson_pronouns = [" they ", " they'll ", " they're "]

laughs_body = []
#------------------------------------------------------------------------------


for i in np.arange(0, len(transcript_files)):
    eye = i
    csv = pd.read_csv(transcript_files[0])
    date_pick = str(csv['date'][1])
    transcript_string = ' '.join(csv['clean_transcript_text'].astype(str))
    laughter_count = transcript_string.count("[Laughter]") + transcript_string.count("[laughter]")
    laughs_all.append(laughter_count)
    datee.append(date_pick)
    transcript_bodies.append(transcript_string)
    laughter_statements = []
    speaker_of_laughter_statements = []
    for k in np.arange(0, len(csv['clean_transcript_text'])):
        string_to_check = csv['clean_transcript_text'][k]
        for z in np.arange(0, len(laughter_word)):
            if str(laughter_word[z]) in str(string_to_check) :
                #print("One of the words in laughter words is in clean_transcript_text")
                laughter_statements.append(string_to_check)
        #laughter_string = ' '.join(laughter_statements)
        #laughs_body.append(laughter_string)
            else:
                print("One of the words in laughter words is NOT in clean_transcript_text")  
        laughs_eyes = []
        laugh_yous = []
        laughs_np = []
        for b in np.arange(0,len(laughter_statements)):
            for p in np.arange(0, len(myself_pronouns)):
                    if str(myself_pronouns[p]) in laughter_statements[b]:
                        laughs_eyes.append(1)
                    else:
                        laughs_eyes.append(0)
            for y in np.arange(0, len(you_pronouns)):
                    if str(you_pronouns[y]) in laughter_statements[b]:
                        laugh_yous.append(1)
                    else:
                        laugh_yous.append(0)
            for t in np.arange(0, len(nonperson_pronouns)):
                    if str(nonperson_pronouns[t]) in laughter_statements[b]:
                        laughs_np.append(1)
                    else:
                        laughs_np.append(0)
    laughs_eyes_total.append(sum(laughs_eyes))
    laughs_yous_total.append(sum(laugh_yous))
    laughs_nonpersons_total.append(sum(laughs_np))
    laugh_merged_for_speaker = pd.DataFrame()
    laugh_merged_for_speaker['clean_transcript_text'] = laughter_statements
    total_laughs_speaker = pd.merge(csv, laugh_merged_for_speaker, on = 'clean_transcript_text', how = 'inner')
    for speak in np.arange(0, len(total_laughs_speaker)):
        mr_laugher = total_laughs_speaker['Speaker']

        
    
    
    
    
    
            
            
            
            
#---
df = pd.DataFrame()
#---

df['date'] = datee
df['All laughs'] = laughs_all
#df['body'] = transcript_bodies
df['pre_laugh_eye'] = laughs_eyes_total
df['pre_laugh_yew'] = laughs_yous_total
df['pre_laugh_nonperson'] = laughs_nonpersons_total


#---Create our chair dummy
df['chair_dummy'] = "0"

for i in np.arange(0, len(df)):
    #Yellen's first FOMC was March18-19, 2014
    if int(df['date'][i]) >= 20140319:
        df['chair_dummy'][i] = "1"
    else:
        df['chair_dummy'][i] == "0"



#-----
df['perc_joke_i'] = (df['pre_laugh_eye'] / (df['pre_laugh_eye'] + df['pre_laugh_yew'] + df['pre_laugh_nonperson'])) * 100
df['perc_joke_u'] = (df['pre_laugh_yew'] / (df['pre_laugh_eye'] + df['pre_laugh_yew'] + df['pre_laugh_nonperson'])) * 100
df['perc_joke_np'] = (df['pre_laugh_nonperson'] / (df['pre_laugh_eye'] + df['pre_laugh_yew'] + df['pre_laugh_nonperson'])) * 100



df.to_csv("fomc_transcript/output/jokes.csv")


#---
laugh_analysis = pd.read_csv("fomc_transcript/output/laughs.csv")
laugh_analysis['date'].astype(str)
#----
total = laugh_analysis.merge(df, on = "date", how = "outer")
#-----------------------------------------------------------------------------------
#------------------------------------------------------------------------------


for i in np.arange(0, len(transcript_files)):
    eye = i
    csv = pd.read_csv(transcript_files[i])
    date_pick = str(csv['date'][1])
    transcript_string = ' '.join(csv['clean_transcript_text'].astype(str))
    laughter_count = transcript_string.count("[Laughter]") + transcript_string.count("[laughter]")
    laughs_all.append(laughter_count)
    datee.append(date_pick)
    transcript_bodies.append(transcript_string)
    laughter_statements = []
    for k in np.arange(0, len(csv['clean_transcript_text'])):
        string_to_check = csv['clean_transcript_text'][k]
        for z in np.arange(0, len(laughter_word)):
            if str(laughter_word[z]) in str(string_to_check) :
                #print("One of the words in laughter words is in clean_transcript_text")
                laughter_statements.append(string_to_check)
        #laughter_string = ' '.join(laughter_statements)
        #laughs_body.append(laughter_string)
            else:
                print("One of the words in laughter words is NOT in clean_transcript_text")
        laughs_eyes = []
        laugh_yous = []
        laughs_np = []
        for b in np.arange(0,len(laughter_statements)):
            for p in np.arange(0, len(myself_pronouns)):
                    if str(myself_pronouns[p]) in laughter_statements[b]:
                        laughs_eyes.append(1)
                    else:
                        laughs_eyes.append(0)
            for y in np.arange(0, len(you_pronouns)):
                    if str(you_pronouns[y]) in laughter_statements[b]:
                        laugh_yous.append(1)
                    else:
                        laugh_yous.append(0)
            for t in np.arange(0, len(nonperson_pronouns)):
                    if str(nonperson_pronouns[t]) in laughter_statements[b]:
                        laughs_np.append(1)
                    else:
                        laughs_np.append(0)
    laughs_eyes_total.append(sum(laughs_eyes))
    laughs_yous_total.append(sum(laugh_yous))
    laughs_nonpersons_total.append(sum(laughs_np))


