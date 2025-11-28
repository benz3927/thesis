#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:41:47 2023

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
parent_path = '/if/research_ifs/fomc_transcript/'

# ATT
htmls_road = '/if/research_ifs/fomc_transcript/data/processed/Attendees/'
htmls_ls = r'/if/research_ifs/fomc_transcript/data/processed/Attendees/*.csv'
html_files = glob.glob(htmls_ls)
#Remove 199601 199701 199801 199901
html_files.remove('/if/research_ifs/fomc_transcript/data/processed/Attendees/19960131_att.csv')
html_files.remove('/if/research_ifs/fomc_transcript/data/processed/Attendees/19970205_att.csv')
html_files.remove('/if/research_ifs/fomc_transcript/data/processed/Attendees/19980204_att.csv')
html_files.remove('/if/research_ifs/fomc_transcript/data/processed/Attendees/19990203_att.csv')
#
#Mother files
mother_att_list = []
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20170201_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20160127_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20150128_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20140129_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20130130_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20120125_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20110126_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20100127_att.csv')))

mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20090128_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20080130_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20070131_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20060131_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20050202_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20040128_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20030129_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20020130_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20010131_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/20000202_att.csv')))

mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/Copy of 19990203_att_2.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/Copy of 19980204_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/Copy of 19970205_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/Copy of 19960131_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/19950201_att.csv')))
mother_att_list.append(html_files.pop(html_files.index('/if/research_ifs/fomc_transcript/data/processed/Attendees/19940204_att.csv')))

#
#
bucket_1 = ['chair', 'chairman', 'president', 'governor','senior vice president', 'first vice president','vice president', 'deputy secretary', 'deputy general counsel', 'deputy general counsl', 'vice president','executive vice president']
bucket_1 = [x.strip(' ') for x in bucket_1]

bucket_2 = ['senior special adviser to the board',  'secretary of the board', 'senior special adviser to the board', 'senior adviser', 'assistant secretary', 'deputy director',
            'senior special adviser', 'deputy manager', 'director', 'adviser', 'assistant to the board', 'assistant secretary',
            'assistant to the secretary', 'assistant vice president', 'deputy manager', 'deputy secretary', 'deputy staff director']
bucket_2 = [x.strip(' ') for x in bucket_2]

bucket_3 = ['section chief', 'secretary', 'manager', 'economist', 'senior economist', 'group manager', 'project manager', 'senior financial analyst',
            ' financial analyst', 'information', 'associate economist', 'principal economist','economist', 'general counsel', 'assistant general counsel', 'open market secretariat', 'open market secretariat assistant',
            'open market secretariat specialist', 'secretary and economist', 'senior research officer']
bucket_3 = [x.strip(' ') for x in bucket_3]

#Mother list
mother_roles = pd.read_csv(mother_att_list[0])
df = pd.concat([pd.read_csv(mother_att_list[1]),mother_roles])
df = pd.concat([pd.read_csv(mother_att_list[2]),df])
df = pd.concat([pd.read_csv(mother_att_list[3]),df])
df = pd.concat([pd.read_csv(mother_att_list[4]),df])
df = pd.concat([pd.read_csv(mother_att_list[5]),df])
df = pd.concat([pd.read_csv(mother_att_list[6]),df])
df = pd.concat([pd.read_csv(mother_att_list[7]),df])
df = pd.concat([pd.read_csv(mother_att_list[8]),df])
df = pd.concat([pd.read_csv(mother_att_list[9]),df])
df = pd.concat([pd.read_csv(mother_att_list[10]),df])
df = pd.concat([pd.read_csv(mother_att_list[11]),df])
df = pd.concat([pd.read_csv(mother_att_list[12]),df])
df = pd.concat([pd.read_csv(mother_att_list[13]),df])
df = pd.concat([pd.read_csv(mother_att_list[14]),df])
df = pd.concat([pd.read_csv(mother_att_list[15]),df])
df = pd.concat([pd.read_csv(mother_att_list[16]),df])
df = pd.concat([pd.read_csv(mother_att_list[17]),df])
df = pd.concat([pd.read_csv(mother_att_list[18]),df])


df['year'] = (df['date'].astype(str)).str[0:4]



everyone = []

for i in np.arange(0, len(html_files)):
    eye = i
    child = pd.read_csv(html_files[i])
    child['year'] = (child['date'].astype(str)).str[0:4]
    total = pd.merge(df, child, on = ["Clean Names", "Greeting","year"])
    total = total[['year', 'date_x', 'Clean Names', 'Greeting', 'Position_x']]
    total = total.rename(columns={'date_x': 'date', 'Position_x': 'Position'}) 
    everyone.append(total)
    
everyone = pd.concat(everyone)

everyone = pd.DataFrame(everyone)
everyone = everyone.reset_index()
everyone['Position'] = everyone['Position'].str.strip()

everyone['bucket_1'] = 0
everyone['bucket_2'] = 0
everyone["bucket_3"] = 0

#everyone.to_csv("/if/research_ifs/fomc_transcript/data/processed/everyone.csv")
for i in np.arange(0, len(everyone['Position'])):
    for z in np.arange(0, len(bucket_1)):
        if bucket_1[z] in everyone['Position'][i] :
            everyone['bucket_1'][i] = 1
        else:
            everyone['bucket_1'][i] = 0

    
for i in np.arange(0, len(everyone['Position'])):
    for b in np.arange(0, len(bucket_2)):
        if str(bucket_2[b]) in str(everyone['Position'][i]) :
            everyone['bucket_2'][i] = 1
        else:
            everyone['bucket_2'][i] = 0
            
for i in np.arange(0, len(everyone['Position'])):
    for g in np.arange(0, len(bucket_3)):
        if str(bucket_3[g]) in str(everyone['Position'][i]) :
            everyone['bucket_3'][i] = 1
        else:
            everyone['bucket_3'][i] = 0
    
    transcript_string = ' '.join(csv['clean_transcript_text'].astype(str))
    laughter_count = transcript_string.count("[Laughter]") + transcript_string.count("[laughter]")
    laughs_all.append(laughter_count)
    datee.append(date_pick
    transcript_bodies.append(transcript_string)
    laughter_statements = []
    speaker_of_laughter_statements = []
    for k in np.arange(0, len(csv['clean_transcript_text'])):
        string_to_check = csv['clean_transcript_text'][k]
        for z in np.arange(0, len(laughter_word)):
            if str(laughter_word[z]) in str(string_to_check) :
                #print("One of the words in laughter words is in clean_transcript_text")
                laughter_statements.append(string_to_check)
        #laughter_string = ' '.join(laughter_statements)
        #laughs_body.append(laughter_string)
            else:
                print("One of the words in laughter words is NOT in clean_transcript_text")  
        laughs_eyes = []
        laugh_yous = []
        laughs_np = []
        for b in np.arange(0,len(laughter_statements)):
            for p in np.arange(0, len(myself_pronouns)):
                    if str(myself_pronouns[p]) in laughter_statements[b]:
                        laughs_eyes.append(1)
                    else:
                        laughs_eyes.append(0)
            for y in np.arange(0, len(you_pronouns)):
                    if str(you_pronouns[y]) in laughter_statements[b]:
                        laugh_yous.append(1)
                    else:
                        laugh_yous.append(0)
            for t in np.arange(0, len(nonperson_pronouns)):
                    if str(nonperson_pronouns[t]) in laughter_statements[b]:
                        laughs_np.append(1)
                    else:
                        laughs_np.append(0)
    laughs_eyes_total.append(sum(laughs_eyes))
    laughs_yous_total.append(sum(laugh_yous))
    laughs_nonpersons_total.append(sum(laughs_np))
    laugh_merged_for_speaker = pd.DataFrame()
    laugh_merged_for_speaker['clean_transcript_text'] = laughter_statements
    total_laughs_speaker = pd.merge(csv, laugh_merged_for_speaker, on = 'clean_transcript_text', how = 'inner')
    for speak in np.arange(0, len(total_laughs_speaker)):
        mr_laugher = total_laughs_speaker['Speaker']

        




