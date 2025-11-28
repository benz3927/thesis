#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:51:30 2023

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

#-------------------------
#1 is female
#0 is male
df = pd.DataFrame()
#-------------------------
protests_all = []
crisis_all = []
datee = []
talking = []

for i in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[i])
    date_pick = str(csv['date'][1])
    talks = len(csv['Speaker'].unique())
    talking.append(talks)
    transcript_string = ' '.join(csv['clean_transcript_text'].astype(str))
    crisis_count = transcript_string.count("unrest") 
    protest_count = transcript_string.count("protest")
    protests_all.append(protest_count)
    crisis_all.append(crisis_count)
    datee.append(date_pick)
    
#----

df['date'] = datee
df['Protest'] = protests_all
df['Unrest'] = crisis_all




#--------------------------------
df.to_csv("fomc_transcript/output/unrest.csv")
df.to_csv("GPPR/data/fomc_transcript_unrest.csv")