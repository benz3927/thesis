#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:22:35 2023

@author: m1dcs04
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 23:15:27 2023

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
from nltk.corpus import stopwords
import requests
stopwords = set(stopwords.words('english'))
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#---------------------


# Parent path
parent_path = 'fomc_transcript/'

# Transcript directory
transcript_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Transcripts/'
transcript_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Transcripts/*.csv'
transcript_files = glob.glob(transcript_ls)

# Attendee directory
att_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Attendees/'
att_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Attendees/*.csv'
att_files = glob.glob(att_ls)

#====
#data = pd.read_csv("fomc_transcript/data/processed/sets/owen_initial.csv")
#data = pd.read_csv("fomc_transcript/data/processed/sets/owen_final_08312023.csv")
#data = pd.read_csv("fomc_transcript/data/processed/sets/raw_pre_collapse_after_variable_made.csv")
data = pd.read_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/raw_pre_collapse.csv")
data = data.drop(['Unnamed: 0'], axis = 1)
#Collapse Variables
#The number of times a person speaks in a meeting
#Word count for that meeting for that person
total = data[['date', 'speaker', 'speak_count', 'speak_length']]
total = total.drop_duplicates()

#Take the average of how long people speak after that person
data_avg_next_speaker = data[['date', 'speaker', 'next_speaker_text_length']]
data_avg_next_speaker = data_avg_next_speaker.groupby(['date', 'speaker']).mean()
data_avg_next_speaker.columns = ['avg_next_speaker_length']

total = pd.merge(total, data_avg_next_speaker, on = ['date', 'speaker'])


#Change Mr and Ms to 1 and 0 respectively to calculate average of gender speaking after that person

gender_tally = []
for i in np.arange(0, len(data)):
    if data['Next.Speaker.Gender'][i] == "Mr" or data['Next.Speaker.Gender'][i] == "mr" :
        gender_tally.append(1)
    else:
        gender_tally.append(0)
        
data['gender_dummy'] = gender_tally

data_gender_perc = data[['date', 'speaker', 'gender_dummy']]
data_gender_perc = data_gender_perc.groupby(['date', 'speaker']).mean()
data_gender_perc.columns = ['avg_next_speaker_gender']
total = pd.merge(total, data_gender_perc, on = ['date', 'speaker'])


#Number of times others laugh after/during the time this person is talking
data_laughter = data[['date', 'speaker', 'laughter_first_speaker']]
data_laughter = data_laughter.groupby(['date', 'speaker']).sum()
data_laughter.columns = ['laughter_sum']
total = pd.merge(total, data_laughter, on = ['date', 'speaker'])


#If the individual introduces new topics - indicate with a a1
data_topics_new = data[['date','speaker', 'first_to_introduce_topic']]
data_topics_new = data_topics_new.groupby(['date', 'speaker']).sum()

total = pd.merge(total, data_topics_new, on = ['date', 'speaker'])

#Number of times a person is interrupted
interrupted = data[['date', 'speaker','disrupted_1_yes_0_no']]
#changes mean to sum
data_interrupted = interrupted.groupby(['date', 'speaker']).sum()
data_interrupted.columns = ['sum_interrupted']
total = pd.merge(total, data_interrupted, on = ['date', 'speaker'])

#number of times a person interrupts somebody else
#changes mean to sum
#? If this person gets interrupted.. conditional on being interrupted.. etc.
interrupter = data[['date', 'speaker','number_of_times_all_disrupted_per_date']]
interrupter = interrupter.drop_duplicates()
data_interrupter = interrupter.groupby(['date', 'speaker']).sum()
data_interrupter.columns = ['sum_interrupter']
total = pd.merge(total, data_interrupter, on = ['date', 'speaker'])


#Number of times a person uses hedging language
hedges = data[['date', 'speaker', 'hedges']]
hedges = hedges.groupby(['date', 'speaker']).sum()
hedges.columns = ['sum_hedge']
total = pd.merge(total, hedges, on = ['date', 'speaker'])




#data = pd.read_csv("fomc_transcript/data/processed/sets/owen_final_08312023.csv")
sentiment = pd.read_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/after_drive.csv")

#Per day
sentiment_per_day = sentiment[['date','sentiment']]
sentiment_per_day = sentiment_per_day.groupby(['date']).mean()
sentiment_per_day['avg_day_sentiment'] = sentiment_per_day['sentiment']
sentiment_per_day.drop(['sentiment'], axis = 1, inplace = True)

#Per speaker
sentiment_per_speaker = sentiment[['date','speaker','sentiment']]
sentiment_per_speaker = sentiment_per_speaker.groupby(['date', 'speaker']).mean()
sentiment_per_speaker['avg_sentiment_per_speaker'] = sentiment_per_speaker['sentiment']
sentiment_per_speaker.drop(['sentiment'], axis = 1, inplace = True)

#First, merge in sentiment_per_day
total = pd.merge(total, sentiment_per_day, on = ['date'])
total = pd.merge(total, sentiment_per_speaker, on = ['date', 'speaker'])

#Remove any speakers that should not be there! For example: HAIR YELLEN
total = total[total['speak_count'].notna()]

len(list(set(total['date'])))

total.to_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/04_collapsed.csv")





#--------------------------------------------------------------


#total.to_csv("fomc_transcript/data/processed/sets/owen_collapsed.csv")   
