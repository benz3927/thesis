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

#-------------------------
#1 is female
#0 is male
df = pd.DataFrame()
#-------------------------
laughs_all = []
datee = []
talking = []

for i in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[i])
    date_pick = str(csv['date'][1])
    talks = len(csv['Speaker'].unique())
    talking.append(talks)
    transcript_string = ' '.join(csv['clean_transcript_text'].astype(str))
    laughter_count = transcript_string.count("[Laughter]") + transcript_string.count("[laughter]")
    laughs_all.append(laughter_count)
    datee.append(date_pick)
    
#----

df['date'] = datee
df['All laughs'] = laughs_all


#---Create our chair dummy
df['chair_dummy'] = "0"

for i in np.arange(0, len(df)):
    #Yellen's first FOMC was March18-19, 2014
    if int(df['date'][i]) >= 20140319:
        df['chair_dummy'][i] = "1"
    else:
        df['chair_dummy'][i] == "0"



#----------
#datee_2 = []
csv_full_laughs = []

for y in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[y])
    csv['clean_transcript_text'] = csv['clean_transcript_text'].fillna("none")
    #date_pick = str(csv['date'][1])
    #datee_2.append(date_pick)
    laughs_per_row = []
    for i in np.arange(0, len(csv['clean_transcript_text'])):
        transcript_text = csv['clean_transcript_text'][i]
        laughter_count_per = transcript_text.count("[Laughter]") + transcript_text.count("[laughter]")
        laughs_per_row.append(laughter_count_per)
    csv['Laughs'] = laughs_per_row
    #Drop rows that males are speaker
    csv = csv[csv["Speaker"].str.contains("MS ") == True]
    csv_sum_fem_laughs = sum(csv['Laughs'])
    csv_full_laughs.append(csv_sum_fem_laughs)


df['Laughs After MS SPEAKER'] = csv_full_laughs


#----------
#datee_2 = []
csv_full_laughs = []

for y in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[y])
    csv['clean_transcript_text'] = csv['clean_transcript_text'].fillna("none")
    #date_pick = str(csv['date'][1])
    #datee_2.append(date_pick)
    laughs_per_row = []
    for i in np.arange(0, len(csv['clean_transcript_text'])):
        transcript_text = csv['clean_transcript_text'][i]
        laughter_count_per = transcript_text.count("[Laughter]") + transcript_text.count("[laughter]")
        laughs_per_row.append(laughter_count_per)
    csv['Laughs'] = laughs_per_row
    #Drop rows that males are speaker
    csv = csv[csv["Speaker"].str.contains("MR ") == True]
    csv_sum_fem_laughs = sum(csv['Laughs'])
    csv_full_laughs.append(csv_sum_fem_laughs)


df['Laughs After MR SPEAKER'] = csv_full_laughs


#----CHAIR
#----------
#datee_2 = []

csv_full_laughs = []

for y in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[y])
    csv['clean_transcript_text'] = csv['clean_transcript_text'].fillna("none")
    #date_pick = str(csv['date'][1])
    #datee_2.append(date_pick)
    laughs_per_row = []
    for i in np.arange(0, len(csv['clean_transcript_text'])):
        transcript_text = csv['clean_transcript_text'][i]
        laughter_count_per = transcript_text.count("[Laughter]") + transcript_text.count("[laughter]")
        laughs_per_row.append(laughter_count_per)
    csv['Laughs'] = laughs_per_row
    #Drop rows that males are speaker
    csv = csv[csv["Speaker"].str.contains("CHAIR ") == True]
    csv_sum_fem_laughs = sum(csv['Laughs'])
    csv_full_laughs.append(csv_sum_fem_laughs)


df['Laughs After CHAIR'] = csv_full_laughs



csv_full_laughs = []

for y in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[y])
    csv['clean_transcript_text'] = csv['clean_transcript_text'].fillna("none")
    #date_pick = str(csv['date'][1])
    #datee_2.append(date_pick)
    laughs_per_row = []
    for i in np.arange(0, len(csv['clean_transcript_text'])):
        transcript_text = csv['clean_transcript_text'][i]
        laughter_count_per = transcript_text.count("[Laughter]") + transcript_text.count("[laughter]")
        laughs_per_row.append(laughter_count_per)
    csv['Laughs'] = laughs_per_row
    #Drop rows that males are speaker
    csv = csv[csv["Speaker"].str.contains("CHAIRMAN ") == True]
    csv_sum_fem_laughs = sum(csv['Laughs'])
    csv_full_laughs.append(csv_sum_fem_laughs)


df['Laughs After CHAIRMAN'] = csv_full_laughs

csv_full_laughs = []


for y in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[y])
    csv['clean_transcript_text'] = csv['clean_transcript_text'].fillna("none")
    #date_pick = str(csv['date'][1])
    #datee_2.append(date_pick)
    laughs_per_row = []
    for i in np.arange(0, len(csv['clean_transcript_text'])):
        transcript_text = csv['clean_transcript_text'][i]
        laughter_count_per = transcript_text.count("[Laughter]") + transcript_text.count("[laughter]")
        laughs_per_row.append(laughter_count_per)
    csv['Laughs'] = laughs_per_row
    #Drop rows that males are speaker
    csv = csv[csv["Speaker"].str.contains("GOVERNOR ") == True]
    csv_sum_fem_laughs = sum(csv['Laughs'])
    csv_full_laughs.append(csv_sum_fem_laughs)


df['Laughs After GOVERNOR'] = csv_full_laughs

csv_full_laughs = []


for y in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[y])
    csv['clean_transcript_text'] = csv['clean_transcript_text'].fillna("none")
    #date_pick = str(csv['date'][1])
    #datee_2.append(date_pick)
    laughs_per_row = []
    for i in np.arange(0, len(csv['clean_transcript_text'])):
        transcript_text = csv['clean_transcript_text'][i]
        laughter_count_per = transcript_text.count("[Laughter]") + transcript_text.count("[laughter]")
        laughs_per_row.append(laughter_count_per)
    csv['Laughs'] = laughs_per_row
    #Drop rows that males are speaker
    csv = csv[csv["Speaker"].str.contains("PRESIDENT ") == True]
    csv_sum_fem_laughs = sum(csv['Laughs'])
    csv_full_laughs.append(csv_sum_fem_laughs)


df['Laughs After PRESIDENT'] = csv_full_laughs

#--------------------------------AFTER YELLEN (FIRST FEMALE CHAIR)
df['Laughs after FEMALE CHAIR'] = "0"

for i in np.arange(0, len(df)):
    #Yellen's first FOMC was March18-19, 2014
    if int(df['date'][i]) >= 20140319:
        df['Laughs after FEMALE CHAIR'][i] = df['Laughs After CHAIRMAN'][i] + df['Laughs After CHAIR'][i]
    else:
        df['Laughs after FEMALE CHAIR'][i] == "0"


#--------------------------------
df.to_csv("fomc_transcript/output/laughs.csv")
