#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 00:40:27 2023

@author: m1dcs04
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 27, 2022

@author: m1dcs04

Description: This script will parse each PDF transcript into an initial remarks
and Q&A CSV that we can then perform frequency counts and textual analysis on
for the FOMC transcripts.

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




# Parent path
parent_path = 'fomc_transcript/'

# HTML directory
htmls_road = 'fomc_transcript/htmls/'
htmls_ls = r'fomc_transcript/htmls/*.html'
html_files = glob.glob(htmls_ls)


#===================
for i in np.arange(0,len(html_files)):
    with open(html_files[i]) as fhandler:
        number_files = i
        indices_of_con_cap_words = []
        soup = BeautifulSoup(fhandler, 'html.parser')
        pure_text = soup.text
        pure_text = pure_text.replace("\n", ' ')
        pure_text = pure_text.replace(",", '')
        pure_text = pure_text.replace(".", '')
        #transcript = re.findall(r"Transcript of the(.+?)END OF MEETING", pure_text)
        transcript = pure_text
        indices_of_con_cap_words = re.findall('[A-Z][A-Z]+(?=\s[A-Z])(?:\s[A-Z][A-Z]+)+', pure_text)
        indices_positions = [(m.start(0), m.end(0)) for m in re.finditer('[A-Z][A-Z]+(?=\s[A-Z])(?:\s[A-Z][A-Z]+)+', pure_text)]
        text_from_indices = []
        for i in np.arange(0,len(indices_positions)):
            if  i+1 in np.arange(0,len(indices_positions)):
                start = indices_positions[i][0]
                end = indices_positions[i+1][0]
                t_text = pure_text[start:end]
                text_from_indices.append(t_text)
            else:
                start = indices_positions[i][0]
                end = -1
                t_text = pure_text[start:end]
                text_from_indices.append(t_text)
        
        df_transcript = pd.DataFrame(indices_of_con_cap_words, columns=['Speaker'])
        df_transcript['transcript_text'] = text_from_indices
        clean_transcript_text = []
        for i in np.arange(0,len(df_transcript)):
            if len(df_transcript['Speaker'][i].split()) == 2:
                clean_transcript_text.append(' '.join(df_transcript['transcript_text'][i].split()[2:]))
            elif len(df_transcript['Speaker'][i].split()) == 3:
                clean_transcript_text.append(' '.join(df_transcript['transcript_text'][i].split()[3:]))
            elif len(df_transcript['Speaker'][i].split()) == 1:
                clean_transcript_text.append(' '.join(df_transcript['transcript_text'][i].split()[1:]))
            else:
                clean_transcript_text.append(' '.join(df_transcript['transcript_text'][i].split()[1:]))
        df_transcript['clean_transcript_text'] = clean_transcript_text

        df_transcript.to_csv("fomc_transcript/data/processed/Transcripts/" + html_files[number_files][40:48] + "_t.csv" )

        
