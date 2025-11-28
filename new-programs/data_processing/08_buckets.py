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
from nltk.corpus import stopwords
import requests
stopwords = set(stopwords.words('english'))
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
#Parent_Path
parent_path = 'fomc_transcript/'

# Attendees
att_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Attendees/'
att_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Attendees/*.csv'
att_files = glob.glob(att_ls)

# Voters
vote_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Voters/'
vote_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Voters/*.csv'
vote_files = glob.glob(vote_ls)
#--------Get the percentage of each bucket for attendance!

meeting_stats = pd.read_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/output/attendance_voter_breakdown.csv")


#----Append all attendee buckets
attendees_list = ['']
greeting_list = ['']
position_list = ['']

for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        #date_pick = str(csv['date'][1])
        attendees_list.extend(csv['Clean Names'])
        greeting_list.extend(csv['Greeting'])
        position_list.extend(csv['Position'])

del attendees_list[0]
del greeting_list[0]
del position_list[0]
##
#------
df = pd.DataFrame()
df_long = pd.DataFrame()
df_long['Clean Names'] = attendees_list
df_long['Greeting'] = greeting_list
df_long['Position'] = position_list

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

bucket_1 = ['chair','chairman','preisdent','president elect','Chairman','Vice Chairman','president','vice chairman','governor','president-elect',]
bucket_2 = ['executice vice president','senior vice preisdent','vice president','executvie vice president','senior vice president','assistant vice president','first vice preisndet','first vice president','executive vice president','group vice president',]
bucket_3 = ['deputy general counsel','secretary and economist', 'senior special adviser to the board','deputy secretary','senior special adviser to the chair','senior sepcial adviser to the board','secretary of the board','deputy general cousnel','special policy advisor to the president','assistant to the board','special adviser to the chair','deputy congressional liason','deputy generl counsel','special assitant to the board','special advisor to the board','assistant to the secretary','assitant to the board','special counsel','adviser to the board','acting director','special assistant to the board','special policy adviser to the president','advisor to the president','special adviser to the board', 'director','deputy secretary counsel','senior special advisor to the chair','assistant secretary', 'secretary', 'associate secretary', 'secretary and economist', 'deputy secretary', 'secretary', 'assistant secretary', 'deputy secretary', 'secretary and economist', 'assistant secretary', 'secretary']
bucket_4 = ['associate economist', 'manager','senior associate','senior advisor','economic policy advisor','research','secretariat assistant','special assistant','assistant to the director','general counsel','deputy staff director','senior information manager','assistant directors','group manager','senior counsel','deputy director','financial economist','senior research economist','associate director','visiting associate director','research assistant','visiting senior adviser','assistant general counsel','senior research advisor senior economist','policy adviser','open market secretariant assistant','open market operations manager','system open market account mananger','senior economic advisor','assistant director','visiting research bank officer','senior professional economist','system open market manager','senior associate director','senior economist','senior special adviser','information manager','senior research officer','senior economic project manager','information management analyst','system open market account manger','senior financial analyst ','adivser','visiting reserve bank officer','special adviser','system open market account manager','monetary advisor','economic adviser','section chief','research economist','sectio chief','dpeuty associate director','markets officer','senior research adviser','economist','manager for domestic operations','open market secretariat assistant','open market secretary specialist','consultant','senior economic adviser','open market secretariat specialist','open market secretariat','open makret secretariat specialist','dpeuty director','deputy manager','project manager','records management analyst','special assistant to the director','deputy associate director','associate general counsel', 'senior associate directgor','assistant economist','assistant congressional liasion','seniro associate director','economic advisor','senior attorney','associate economists','staff assistant','senior financial analyst','manager for foreign operations','seciton chief','research officer','records project manager','senior research advisor','principal economist','temporary manager','senior project manager','managing senior counsel','financial analyst','research adviser','visitng reserve bank officer','senior adviser','officer','senior techincal editor','special policy advisor','adviser']


bucket1_total = []
bucket2_total = []
bucket3_total = []
bucket4_total = []
bucket1_total_per_person = []
bucket2_total_per_person = []
bucket3_total_per_person = []
bucket4_total_per_person = []
dates = []

for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        bucket1_total_int = []
        
        for a in np.arange(0, len(csv['Position'])):
            dates.append(csv['date'][a])
            if csv['Position'][a] in bucket_1:
                bucket1_total_int.append(1)
                bucket1_total_per_person.append(1)
                
            else:
                bucket1_total_int.append(0)
                bucket1_total_per_person.append(0)
        bucket1_total.append(sum(bucket1_total_int))


for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        bucket2_total_int = []
        for a in np.arange(0, len(csv['Position'])):
            if csv['Position'][a] in bucket_2:
                bucket2_total_int.append(1)
                bucket2_total_per_person.append(1)
            else:
                bucket2_total_int.append(0)
                bucket2_total_per_person.append(0)
        bucket2_total.append(sum(bucket2_total_int))



for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        bucket3_total_int = []
        for a in np.arange(0, len(csv['Position'])):
            if csv['Position'][a] in bucket_3:
                bucket3_total_int.append(1)
                bucket3_total_per_person.append(1)
            else:
                bucket3_total_int.append(0)
                bucket3_total_per_person.append(0)
        bucket3_total.append(sum(bucket3_total_int))
        
for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        bucket4_total_int = []
        for a in np.arange(0, len(csv['Position'])):
            if csv['Position'][a] in bucket_4:
                bucket4_total_int.append(1)
                bucket4_total_per_person.append(1)
            else:
                bucket4_total_int.append(0)
                bucket4_total_per_person.append(0)
        bucket4_total.append(sum(bucket4_total_int))


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
df['bucket_4'] = bucket4_total

df_long['date'] = dates
df_long['bucket_1'] = bucket1_total_per_person
df_long['bucket_2'] = bucket2_total_per_person
df_long['bucket_3'] = bucket3_total_per_person
df_long['bucket_4'] = bucket4_total_per_person




df['total'] = df['bucket_1'] + df['bucket_2'] + df['bucket_3'] + df['bucket_4']

df['bucket1_perc'] = df['bucket_1'] / df['total']

df['bucket2_perc'] = df['bucket_2'] / df['total']

df['bucket3_perc'] = df['bucket_3'] / df['total']

df['bucket4_perc'] = df['bucket_4'] / df['total']


#Append to final dataset


#total = pd.read_csv('fomc_transcript/data/processed/sets/raw_pre_collapse_after_variable_made.csv')

#total['date'] = total['date'].astype(str)

#total = pd.merge(total, df, on = ['date'])

len(list(set(df['date'])))

df.to_csv('/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/buckets.csv')
df_long.to_csv('/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/buckets_dummy.csv')
