#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:30:00 2023

@author: m1dcs04
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:31:15 2023

@author: m1dcs04
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
pd.set_option('display.float_format', lambda x: '%.5f' % x)
#Parent_Path
parent_path = 'fomc_transcript/'

# Attendees
att_road = 'fomc_transcript/data/processed/Attendees/'
att_ls = r'fomc_transcript/data/processed/Attendees/*.csv'
att_files = glob.glob(att_ls)

# Voters
vote_road = 'fomc_transcript/data/processed/Voters/'
vote_ls = r'fomc_transcript/data/processed/Voters/*.csv'
vote_files = glob.glob(vote_ls)
#--------Get the percentage of each bucket for attendance!

meeting_stats = pd.read_csv("fomc_transcript/output/attendance_voter_breakdown.csv")

#------
df = pd.DataFrame()
#===================

att_unique = []
att_csv = []

for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        #date_pick = str(csv['date'][1])
        att_csv.append(csv)
        att_unique.append(csv['Position'].unique())
        
att_unique= np.concatenate(att_unique).ravel().tolist()
att_unique = list(set(att_unique))

att_unique

bucket_1 = ['senior vice president','senior vice preisdent','preisdent','assistant vice president','president-elect','chair','president elect','first vice preisndet','Chairman','chairman', 'vice president','executive vice president','governor','group vice president','Vice Chairman','vice chairman','executvie vice president','executice vice president','president','first vice president',]
bucket_2 = ['special adviser to the board','special adviser to the chair','special assistant', 'special policy adviser to the president','deputy general counsel','senior special adviser to the board', 'senior special adviser','senior sepcial adviser to the board','special policy advisor to the president','deputy generl counsel','visiting senior adviser','special advisor to the board','deputy secretary counsel','advisor to the president','special adviser','senior special advisor to the chair','senior advisor', 'senior special adviser to the chair','special policy advisor','deputy congressional liason','deputy general cousnel','senior adviser', 'adviser to the board']
bucket_3 = ['records management analyst','sectio chief','secretary','research officer','open market secretariat assistant','open market secretariant assistant','deputy director','economic policy advisor','secretariat assistant','open market secretariat','open market operations manager','visiting associate director','information management analyst','senior research officer','system open market account manger','dpeuty director','officer','deputy staff director','open market secretariat specialist','financial economist','deputy associate director','system open market manager','visiting reserve bank officer','special assistant to the director','visitng reserve bank officer','senior attorney','financial analyst','information manager','senior counsel','senior economic advisor','associate director','dpeuty associate director','monetary advisor','adivser','senior financial analyst ','seciton chief','senior financial analyst','secretary of the board','senior techincal editor','special assitant to the board','senior economist','section chief','open market secretary specialist','general counsel','senior associate','assistant congressional liasion','research assistant','research adviser','open makret secretariat specialist','economic adviser','senior economic adviser','manager','system open market account manager','assistant directors','seniro associate director','assistant economist','senior information manager','associate economists','associate general counsel','principal economist','adviser','assistant general counsel','assitant to the board','deputy secretary','managing senior counsel','markets officer','research economist','associate economist','assistant secretary','visiting research bank officer','associate secretary','manager for foreign operations','senior economic project manager','special counsel','senior research advisor', 'director','senior associate directgor','research','assistant director','senior research adviser','assistant to the director','temporary manager','acting director', 'staff assistant','senior associate director','special assistant to the board','system open market account mananger','senior project manager','manager for domestic operations','assistant to the board','policy adviser','consultant','economic advisor','records project manager','deputy manager','project manager','assistant to the secretary','secretary and economist','senior research economist','group manager','senior professional economist','senior research advisor senior economist','economist']


bucket1_total = []
bucket2_total = []
bucket3_total = []

for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        bucket1_total_int = []
        for a in np.arange(0, len(csv['Position'])):
            if csv['Position'][a] in bucket_1:
                bucket1_total_int.append(1)
            else:
                bucket1_total_int.append(0)
        bucket1_total.append(sum(bucket1_total_int))
                
for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        bucket2_total_int = []
        for a in np.arange(0, len(csv['Position'])):
            if csv['Position'][a] in bucket_2:
                bucket2_total_int.append(1)
            else:
                bucket2_total_int.append(0)
        bucket2_total.append(sum(bucket2_total_int))



for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        bucket3_total_int = []
        for a in np.arange(0, len(csv['Position'])):
            if csv['Position'][a] in bucket_3:
                bucket3_total_int.append(1)
            else:
                bucket3_total_int.append(0)
        bucket3_total.append(sum(bucket3_total_int))


date = []                
for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        date_pluck_int = []
        for d in np.arange(0, len(csv['date'])):
            date_pluck = csv['date'][1]
            date_pluck_int.append(date_pluck)
        date.append(str(date_pluck_int[1]))
            
            
df = pd.DataFrame()

df['date'] = date
df['date'] = df['date'].astype(str)
df['bucket_1'] = bucket1_total
df['bucket_2'] = bucket2_total
df['bucket_3'] = bucket3_total




df['total'] = df['bucket_1'] + df['bucket_2'] + df['bucket_3']

df['bucket1_perc'] = df['bucket_1'] / df['total']

df['bucket2_perc'] = df['bucket_2'] / df['total']

df['bucket3_perc'] = df['bucket_3'] / df['total']



#Append to final dataset


#total = pd.read_csv('fomc_transcript/data/processed/sets/raw_pre_collapse_after_variable_made.csv')

#total['date'] = total['date'].astype(str)

#total = pd.merge(total, df, on = ['date'])


df.to_csv('fomc_transcript/data/processed/sets/buckets.csv')
