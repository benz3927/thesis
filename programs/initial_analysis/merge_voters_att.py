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
#----Get a date!

#Bring in the attendance CSV

att_csv = pd.read_csv("fomc_transcript/output/attendance.csv")
att_csv = att_csv.drop(columns=['Unnamed: 0'], axis = 1)

voter_csv = pd.read_csv("fomc_transcript/output/voters.csv")
voter_csv = voter_csv.drop(columns=['Unnamed: 0'], axis = 1)
#-----------------
#join the dataframes on the transcript name

total = pd.merge(voter_csv, att_csv, on = 'date', how = 'outer')

#------Attendees NON Voters

total['ATT NOT VOTERS'] = total['Attendees'] - total['Number of Voters']

#----LAUGHS
laughs = pd.read_csv("fomc_transcript/output/laughs.csv")

total = pd.merge(total, laughs, on = 'date', how = 'outer')


total.to_csv('fomc_transcript/output/total.csv')