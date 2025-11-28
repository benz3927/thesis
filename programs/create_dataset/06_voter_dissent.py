#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:48:23 2023

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

#--------Get the percentage of vote dissent and the voter who dissented per meeting!
voter_names = []
voter_dates = []
voter_position = []

for i in np.arange(0,len(vote_files)):
    with open(vote_files[i]) as fhandler:
        csv = pd.read_csv(vote_files[i])
        if 'Voter' in csv.columns:
            for v in np.arange(0, len(csv['Voter'])):
                voter_names.append(csv['Voter'][v])
                voter_dates.append(csv['date'][v])
                voter_position.append(csv['Greeting'][v])
        else:
            for g in np.arange(0, len(csv['Voters'])):
                voter_names.append(csv['Voters'][g])
                voter_dates.append(csv['date'][g])
                voter_position.append(csv['Greeting'][g])

#------------------------------------------------------------------------
df_voter_for_df = pd.DataFrame()
df_voter_for_df['short_name'] = voter_names
df_voter_for_df['short_name'] = df_voter_for_df['short_name'].str.lower()
df_voter_for_df['Position'] = voter_position
df_voter_for_df['date'] = voter_dates
df_voter_for_df['voter_dum'] = 1

df_voter_for_df.to_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/voter_dummy.csv")
#------------------------------------------------------------------------

total_yes = []
total_no = []

for i in np.arange(0,len(vote_files)):
    with open(vote_files[i]) as fhandler:
        csv = pd.read_csv(vote_files[i])
        yes_es = []
        no_s = []
        for v in np.arange(0, len(csv['Vote'])):
            if csv['Vote'][v] == 'Yes':
                yes_es.append(1)
            else:
                no_s.append(1)
        total_yes.append(sum(yes_es))
        total_no.append(sum(no_s))        
                
                
date = []                
for i in np.arange(0,len(vote_files)):
    with open(vote_files[i]) as fhandler:
        csv = pd.read_csv(vote_files[i])
        date_pluck_int = []
        for d in np.arange(0, len(csv['date'])):
            date_pluck = csv['date'][1]
            date_pluck_int.append(date_pluck)
        date.append(str(date_pluck_int[1]))
                            
                
df = pd.DataFrame()

df['date'] = date
df['date'] = df['date'].astype(str)
df['total_yes_vote'] = total_yes
df['total_no_vote'] = total_no
df['total_vote'] = df['total_yes_vote'] + df['total_no_vote']
df['perc_disset_vote'] = df['total_no_vote'] / df['total_vote']

 #----------Now we need to get a dummy for the dissenter per meeting!


dissenter_date = []
dissenter = []
dissenter_vote = []


for i in np.arange(0,len(vote_files)):
    with open(vote_files[i]) as fhandler:
        csv = pd.read_csv(vote_files[i])
        csv['Voters'] = csv.iloc[:,0].str.lower()

        for v in np.arange(0, len(csv['Vote'])):
            if csv['Vote'][v] == 'No':
                dissenter.append(csv['Voters'][v])
                dissenter_date.append(csv['date'][v])
                dissenter_vote.append(1)
            else:
                dissenter.append(csv['Voters'][v])
                dissenter_date.append(csv['date'][v])
                dissenter_vote.append(0)     
                              
                
df_dissenter = pd.DataFrame()
df_dissenter['date'] = dissenter_date
df_dissenter['short_name'] = dissenter
df_dissenter['dissenter_dummy'] = dissenter_vote



#-------------Bring in entire dataset



total = pd.read_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/04_collapsed.csv")

#total = total.drop(columns = ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'X'])

total['speaker'] = total['speaker'].str.lower()


short_name = []

for n in np.arange(0, len(total['speaker'])):
    sentence = list(total['speaker'][n].split(" "))
    if len(sentence) > 1:
        short_name_int = sentence[-1]
        short_name.append(short_name_int)
    elif len(sentence) == 1:
        short_name.append(sentence[0])
    else:
        short_name.append(0)



total['short_name'] = short_name
#-------------------

total_2 = pd.merge(total, df_dissenter, on = ['short_name', 'date'], how = 'outer')
total_2['date'] = total_2['date'].astype(str)

total_2 = pd.merge(total_2, df, on = ['date'])

len(list(set(total_2['date'])))
#----------MERGE WITH MEETING STATISTICS--------------------#

total_2.to_csv('/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/06_collapsed_with_vote.csv')

