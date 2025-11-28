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


# Parent path

#Find the current working directory
print(os.getcwd())
os.chdir("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/")

# HTML directory
htmls_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/htmls/'
htmls_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/htmls/*.html'
html_files = glob.glob(htmls_ls)


# Attendees
att_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Attendees/'
att_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Attendees/*.csv'
att_files = glob.glob(att_ls)

# Voters
vote_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Voters/'
vote_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Voters/*.csv'
vote_files = glob.glob(vote_ls)

#===================
df = pd.DataFrame()
#===================
vote_length = []
datee = []
fem_vote = []
male_vote = []

for i in np.arange(0,len(vote_files)):
    with open(vote_files[i]) as fhandler:
        csv = pd.read_csv(vote_files[i])
        date_pick = str(csv['date'][1])
        vote_length.append(len(csv))
        datee.append(date_pick)
#==================
df['Transcript'] = vote_files
df['Number of Voters'] = vote_length
df['date'] = datee
df.to_csv('/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/output/voters.csv')
df['date'] = df['date'].apply(str)
#===Merge Gender to Voters
att = pd.read_csv("fomc_transcript/output/attendance.csv")
att = att.drop(columns=['Unnamed: 0'], axis = 1)
att['date'] = att['date'].apply(str)
#total_outer = pd.merge(df, att, on = 'date', how = 'outer')

total = pd.merge(df, att, on = 'date', how = 'inner')

#------Attendees NON Voters

total['ATT NOT VOTERS'] = total['Attendees'] - total['Number of Voters']

#Rename Columns----------------------------------------------------
total = total.rename(columns={"Transcript_x" : "Voter_transcript", "Transcript_y" : "Attendance_transcript"})
#------------------------------------------------------------------
fem_voters = []
male_voters = []
date_pick = []
voter_length = []
nonvoter_length = []
non_male_voters = []
non_female_voters = []



for i in np.arange(0, len(total)):
    trs_csv = pd.read_csv(total['Attendance_transcript'][i])
    vote_csv = pd.read_csv(total['Voter_transcript'][i])
    vote_csv = vote_csv.rename(columns = {"Voters" : "Clean Names"})
    vote_csv = vote_csv.rename(columns = {"Voter" : "Clean Names"})
    vote_csv = vote_csv.rename(columns = {"Greeting": "Position"})
    vote_csv['Position'] = vote_csv['Position'].str.lower()
    trs_csv['Position'] = trs_csv['Position'].str.lower()
    inner_total_gender = pd.merge(trs_csv, vote_csv, on = ["Clean Names", "Position", "date"], how = "inner")
    inner_total_gender = inner_total_gender[['date', 'Clean Names', 'Greeting']]
    voter_number = len(vote_csv)
    inner_total_gender['Greeting'] = inner_total_gender['Greeting'].str.lower()
    male_count = inner_total_gender['Greeting'].str.count("mr").sum()
    fem_count = inner_total_gender['Greeting'].str.count("ms").sum()
    fem_count_2 = inner_total_gender['Greeting'].str.count("mrs").sum()
    fem_count = fem_count + fem_count_2
    date_picker = str(inner_total_gender['date'][1])
    fem_voters.append(fem_count)
    male_voters.append(male_count)
    date_pick.append(date_picker)
    voter_length.append(voter_number)
    #Outer
    outer_total_gender = pd.merge(trs_csv, vote_csv, on = ["Clean Names", "Position","date"], how = "outer")
    #outer_total_gender = outer_total_gender.rename(columns = {"Greeting_x": "Salutation", "Greeting_y": "Position", "date_x": "date"})
    outer_total_gender = outer_total_gender[['date', 'Clean Names', 'Greeting', 'Vote']]
    outer_total_gender = outer_total_gender[outer_total_gender.Vote != "Yes"]
    outer_total_gender = outer_total_gender[outer_total_gender.Vote != "No"]
    nonvoter_number = len(outer_total_gender)
    nonvoter_length.append(nonvoter_number)
    outer_total_gender['Salutation'] = outer_total_gender['Greeting'].str.lower()
    non_male_count = outer_total_gender['Greeting'].str.count("mr").sum()
    non_fem_count = outer_total_gender['Greeting'].str.count("ms").sum()
    non_fem_count_2 = outer_total_gender['Greeting'].str.count("mrs").sum()
    non_fem_count = non_fem_count + non_fem_count_2
    non_male_voters.append(non_male_count)
    non_female_voters.append(non_fem_count)
    #print(inner_total_gender)
    


#==================
df_per_vote = pd.DataFrame()
df_per_vote['date'] = date_pick
df_per_vote['Female Voters'] = fem_voters
df_per_vote['Male Voters'] = male_voters
df_per_vote['Female NON Voters'] = non_female_voters
df_per_vote['Male NON Voters'] = non_male_voters
#df_per_vote['Number of Voters'] = voter_length

#====Statistics
df_per_vote['Number of Voters'] = df_per_vote['Female Voters'] + df_per_vote['Male Voters']
df_per_vote['percent_fem_voter'] = df_per_vote['Female Voters'] / df_per_vote['Number of Voters']
df_per_vote['percent_male_voter'] = df_per_vote['Male Voters'] / df_per_vote['Number of Voters']

df_per_vote['Number of NonVoters'] = df_per_vote['Female NON Voters'] + df_per_vote['Male NON Voters']
df_per_vote['percent_female_NONvoters'] = df_per_vote['Female NON Voters'] / df_per_vote['Number of NonVoters']
df_per_vote['percent_male_NONvoters'] = df_per_vote['Male NON Voters'] / df_per_vote['Number of NonVoters']


total = total.drop(['Number of Voters'], axis = 1)
#===================

tt = pd.merge(total, df_per_vote, on = "date", how = "inner")
len(list(set(tt['date'])))





tt.to_csv('/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/output/attendance_voter_breakdown.csv')










