
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
from nltk.corpus import stopwords
import requests
stopwords = set(stopwords.words('english'))
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#---------------------


#====
data = pd.read_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/06_collapsed_with_vote.csv")
data = data.drop(columns= ['Unnamed: 0.1','Unnamed: 0'])
data['date'] = data['date'].astype(str)

meeting_stats = pd.read_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/output/attendance_voter_breakdown.csv")
meeting_stats = meeting_stats.drop(columns= ['Unnamed: 0'])
len(list(set(meeting_stats['date'])))
meeting_stats['Total Voters'] = meeting_stats['Female Voters'] + meeting_stats['Male Voters']
meeting_stats['date'] = meeting_stats['date'].astype(str)


total = pd.merge(data, meeting_stats, on = 'date', how = 'outer')
total = total.drop(['Voter_transcript', 'Attendance_transcript'], axis = 1)
#Bring in Buckets!---

buckets = pd.read_csv('/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/buckets.csv')
#buckets = buckets.drop(columns= ['Unnamed: 0'])
buckets['date'] = buckets['date'].astype(str)
buckets = buckets[buckets['date'].notna()]

buckets_date = []
for d in np.arange(0, len(buckets['date'])):
    date_int = buckets['date'][d][:]
    buckets_date.append(date_int)
    
buckets['date'] = buckets_date
buckets['date'] = buckets['date'].astype(str)

total_total = pd.merge(buckets, total, on = ['date'], how = 'outer')

total_total = total_total[total_total['speaker'].notna()]

#Drop some outliers
len(list(set(total_total['date'])))
#

total_total.to_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/final.csv")