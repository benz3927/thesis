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

pd.set_option('display.float_format', lambda x: '%,g' % x)
pd.set_option('display.float_format', str)

#Parent_Path
parent_path = 'fomc_transcript/'

# Attendees
att_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Attendees/'
att_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Attendees/*.csv'
att_files = glob.glob(att_ls)

#==================
#===================
att_length = []
fem = []
male = []
datee = []

for i in np.arange(0,len(att_files)):
    with open(att_files[i]) as fhandler:
        csv = pd.read_csv(att_files[i])
        att_length.append(len(csv))
        date_pick = str(csv['date'][1])
        male_count = csv['Greeting'].str.contains("Mr").sum()
        fem_count = csv['Greeting'].str.contains("Ms").sum()
        datee.append(date_pick)
        fem.append(fem_count)
        male.append(male_count)
#==================
df = pd.DataFrame()
df['Transcript'] = att_files

df['Attendees'] = att_length
df['Male'] = male
df['Female'] = fem
df['date'] = datee


df['Percent Male Att'] = df['Male'] / df['Attendees']
df['Percent Female Att'] = df['Female'] / df['Attendees']

#===================
len(list(set(df['date'])))


df.to_csv('/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/output/attendance.csv')

        