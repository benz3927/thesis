#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:26:26 2023

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
#---------------------




#Run this after running the sentiment drive script
#-----------------------------------ANYTHING TO GO BACK TO?---------------
#Yes, go back to topics to create a column that indicates 1 for topic introduction
total = pd.read_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/after_drive.csv")
#total = total.drop(['transcript_text'], axis = 1)
total = total.drop(['Unnamed: 0'], axis = 1)
total = total.drop(['X'], axis = 1)
total = total.drop(['text_between_speakers'], axis = 1)
#total = total.drop(['speaker_location'], axis = 1)
#total = total.drop(['next_speaker_location'], axis = 1)
df_var_sentiment = total[['date', 'speaker', 'sentiment']]
df_var_sentiment = df_var_sentiment.groupby(['date', 'speaker']).mean()

df_var_sentiment.columns = ['average_sentiment']

total = pd.merge(total, df_var_sentiment, on = ['date', 'speaker'])

len(list(set(total['date'])))

total.to_csv('/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/raw_pre_collapse.csv')