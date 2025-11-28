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
parent_path = '/fomc_transcript/'

# HTML directory
htmls_road = '/fomc_transcript/htmls/'
htmls_ls = r'/fomc_transcript/htmls/*.html'
html_files = glob.glob(htmls_ls)
#=========================================================
non_names_please_delete = ['Group Manager', 'Special', 'International', 'Finance','Statistics', 'Principal', 'II', 'I', 'III', 'Payment', 'Supervision', 'Members', 'Office', 'Payments', 'Systems', 'Regulation', 'Alternate','Reserve Bank Operations', 'St Louis','Advisers', 'Account', 'Accounts', 'Louis', 'Federal Reserve System', 'Chair', 'Counsel', 'General Counsel', 'Economist', 'Alternate Members', 'Executive', 'Atlanta', 'Richmond', 'Senior', 'Federal Reserve Board',
                           'San Francisco', 'New York', 'Kansas City', 'Board', 'Senior Economist', 'Federal Reserve Banks',
                           'Chicago', 'Cleveland', 'Boston', 'St. Louis', 'Federal Reserve Bank', 'Philadelphia',
                          'Market Secretariat', 'Dallas','Market Committee', 'Associate', 'Director', 'Division', 'Research', 'Assistant Directors', 'Adviser',
                           'System Open Market Account', 'Project Manager', 'Monetary Affairs', 'International Finance',
                           'Legal', 'Research and Statistics', 'Assistant', 'Governors', 'Governor',
                           'Statistics','Reserve Bank','Financial', 'Financial Stability', 'Senior Advisers', 'Senior Adviser', 'Washington', 'D.C.', 'Federal Open Market Committee']

pattern_to_remove = r'\b(?:{})\b',format('|'.join(non_names_please_delete))

non_votes_please_delete = ['T   hank ','briefing', 'thank', 'you', 'Thank', 'Thank you', 'MADIGAN', 'MR']
voting_pattern_to_remove = '|'.join(non_votes_please_delete)

####Option 2
for i in np.arange(0,len(html_files)):
    with open(html_files[0]) as fhandler:
        number_files = 0
        soup = BeautifulSoup(fhandler, 'html.parser')
        pure_text = soup.text
        pure_text = pure_text.replace("\n", ' ')
        pure_text = pure_text.replace(",", '')
        pure_text = pure_text.replace(".", '')
        pure_text = re.sub(' +', ' ', pure_text)
        
        #indices_of_con_cap_words = re.findall('[A-Z][A-Z]+(?=\s[A-Z])(?:\s[A-Z][A-Z]+)+', pure_text)
        indices_positions = [(m.start(0), m.end(0)) for m in re.finditer('[A-Z][A-Z]+(?=\s[A-Z])(?:\s[A-Z][A-Z]+)+', pure_text)]
        
        first_page = re.findall("Meeting of the(.+?) Transcript", pure_text)
# =============================================================================
#         if len(first_page) ==0:
#             first_page = re.findall("Meeting of the(.+?) Transcript", pure_text)
# # =============================================================================
# #             first_page = re.findall("Meeting of the(.+?)Transcript of", pure_text)
# #             if len(first_page) ==0:
# #                 first_page = re.findall("Meeting of(.+?)Transcript of", pure_text)
# #             elif len(first_page) ==0:
# #                 first_page = re.findall("Meeting(.+?)Transcript", pure_text)
# #             elif len(first_page) ==0:
# #                 first_page = re.findall("Minute(.+?)Transcript", pure_text)
# #             else:
# #                 pass
# # =============================================================================
#         else:
#             first_page = re.findall("Minutes(.+?)Transcript ", pure_text)
#                         
# =============================================================================
        
            
        
        
        first_page_names = []  
        def extract_entities(text):
            for sent in nltk.sent_tokenize(text):
                for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                    if hasattr(chunk, 'node'):
                        a ="Name: ", ' '.join(c[0] for c in chunk.leaves())
                        first_page_names.append(a)
   
        extract_entities(first_page[0])
        df_names = pd.DataFrame(first_page_names)

        df_names = df_names.rename(columns = {0:'Title', 1:'Name'})
        df_names = df_names[['Title', 'Name']]

       # pattern_to_remove = r'\b(?:{})\b',format('|'.join(non_names_please_delete))
        df_names['Clean Names'] = df_names['Name'].str.replace(pattern_to_remove[1], '').str.replace('\d+','')

        df_names = df_names[df_names["Clean Names"] != ""]
        df_names = df_names[df_names["Clean Names"] != ' ']
        df_names = df_names[df_names["Clean Names"] != '  ']
        df_names = df_names[df_names["Clean Names"] != '   ']
        df_names = df_names.dropna()
        df_names = df_names[['Clean Names']]
        df_names["date"] = str(html_files[number_files][40:48])

        df_names.to_csv("/fomc_transcript/data/processed/Attendees/" + html_files[number_files][40:48] + "_att.csv" )


#==========================================================END ATTENDEES=============================================#

#=======================================================================BEGIN VOTERS=============================================#
voter_rows = pd.DataFrame({'Votes':pd.Series(dtype = 'str')})


for i in np.arange(1,len(html_files)):
    with open(html_files[i]) as fhandler:
        number_files = i
        soup = BeautifulSoup(fhandler, 'html.parser')
        pure_text = soup.text
        pure_text = pure_text.replace("\n", ' ')
        pure_text = pure_text.replace(",", '')
        pure_text = pure_text.replace(".", '')
        pure_text = re.sub(' +', ' ', pure_text)
        
        if len(re.findall(r"Call the roll(.+?)CHAIRMAN ", pure_text)) == 1:
            vote_page = re.findall(r"Call the roll(.+?)CHAIRMAN ", pure_text)
        elif len(re.findall(r"call the roll(.+?)CHAIRMAN ", pure_text)) == 1:
            vote_page = re.findall(r"call the roll(.+?)CHAIRMAN ", pure_text)
        elif len(re.findall(r"call the roll(.+?)CHAIR ", pure_text)) == 1:
            vote_page = re.findall(r"call the roll(.+?)CHAIR ", pure_text)
        elif len(re.findall(r"Call the roll(.+?)CHAIR ", pure_text)) == 1:
            vote_page = re.findall(r"Call the roll(.+?)CHAIR ", pure_text)
        elif len(re.findall(r"the roll(.+?)CHAIR ", pure_text)) == 1:
            vote_page = re.findall(r"the roll(.+?)CHAIR ", pure_text)
        elif len(re.findall(r"directive(.+?)CHAIR ", pure_text)) == 1:
            vote_page = re.findall(r"directive(.+?)CHAIR ", pure_text)
        elif len(re.findall(r"directive(.+?)CHAIRMAN ", pure_text)) == 1:
            vote_page = re.findall(r"directive(.+?)CHAIRMAN ", pure_text)
        elif len(re.findall(r"This vote(.+?)CHAIRMAN ", pure_text)) == 1:
            vote_page = re.findall(r"This vote(.+?)CHAIRMAN ", pure_text)
        elif len(re.findall(r"This vote(.+?)CHAIR", pure_text)) == 1:
            vote_page = re.findall(r"This vote(.+?)CHAIR ", pure_text)
        elif len(re.findall(r"associated directive(.+?)CHAIR", pure_text)) == 1:
            vote_page = re.findall(r"associated directive(.+?)CHAIR ", pure_text)
        elif len(re.findall(r"associated directive(.+?)CHAIRMAN", pure_text)) == 1:
            vote_page = re.findall(r"associated directive(.+?)CHAIRMAN ", pure_text)
        elif len(re.findall(r"This is a vote(.+?)Thank", pure_text)) == 1:
            vote_page = re.findall(r"associated directive(.+?)Thank ", pure_text)
        else:
            vote_page = ['Please revisit']
# =============================================================================
#         vote_page = re.findall("Call the roll(.+?)CHAIRMAN ", pure_text)
#         if len(vote_page) == 0:
#             vote_page = re.findall("call the roll(.+?)CHAIRMAN ", pure_text)
#             if len(vote_page) == 0:
#                 vote_page = re.findall("call the roll(.+?)CHAIR ", pure_text)
#                 if len(vote_page) == 0:
#                     vote_page = re.findall("Call the roll(.+?)CHAIR ", pure_text)
#                 else:
#                     pass
#             else:
#                 pass
#         else:
#             continue
# =============================================================================
                
        voter_names = []  
        def extract_entities(text):
            #for sent in nltk.sent_tokenize(text):
            voter = re.split("Chair|Vice Chairman|Governor|President|     ", vote_page[0])
            voter_names.append(voter)
   
        extract_entities(vote_page[0])
        df_voters = pd.DataFrame(voter_names[0])

        df_voters = df_voters.rename(columns = {0:'Voter Name'})
        df_voters['Voter Name'] = df_voters['Voter Name'].replace(r"^ +| +$", r"", regex = True)
        entity_tags = nltk.pos_tag(df_voters['Voter Name'])
        words =nltk.pos_tag(df_voters['Voter Name'])
        voters_to_keep = []
        for i in np.arange(0,len(words)):
            if words[i][1] == "NNP" or words[i][0] == "Yellen":
                voters_to_keep.append(words[i][0])
        df_voters = pd.DataFrame(voters_to_keep, columns = {"Voters":0})
        df_voters['Voters'] = df_voters['Voters'].str.replace(voting_pattern_to_remove, '').str.replace('\d+','')
        df_voters[df_voters["Voters"].str.contains("C   HAIR YELLEN") == False]
        df_voters[df_voters["Voters"].str.contains(" QUARLES So moved") == False]


    

        df_voters = df_voters[df_voters["Voters"] != ""]
        df_voters = df_voters[df_voters["Voters"] != ' ']
        df_voters = df_voters[df_voters["Voters"] != '  ']
        df_voters = df_voters[df_voters["Voters"] != '   ']
        df_voters = df_voters[df_voters["Voters"] != 'MS DANKER']
        df_voters = df_voters.dropna()
        df_voters = df_voters[['Voters']]
        df_voters
        yes_no_rows = df_voters[df_voters['Voters'].str.contains("Yes|yes|no|No")].index.values.tolist()
        df_voters.to_csv("/fomc_transcript/data/processed/Voters/" + html_files[number_files][40:48] + "_voters.csv" )
    
        
        
        
        
        
        #voter_rows = pd.DataFrame({'Votes':pd.Series(dtype = 'str')})
        voter_rows = ''
        for i in np.arange(0,len(yes_no_rows)):
            

            eye = yes_no_rows[i]
            voter_rows = df_voters.loc[eye].tolist()
            #voter_rows.loc[len(voter_rows.index)] = df_voters['']
            #print(voter_rows['Votes'])
            
            voter_rowss = ''
            voter_rows_words = []
            for i in np.arange(0, len(voter_rows)):
                if len(voter_rows[i].split()) >1:
                    voters_split = voter_rows[i].split()
                    voter_rowss = voters_split
                else:
                        voter_rows_words.append(voter_rows[i])
                        
                        
                        #print(voter_rowss)
        
        for i in np.arange(0,len(yes_no_rows)):
            df_voters = df_voters.drop([yes_no_rows[i]])
        
        df_voters[df_voters["Voters"].str.contains("Yes") == False]
        df_voters[df_voters["Voters"].str.contains("No") == False]
        
        if voter_rows != '' and len(voter_rows) == len(df_voters):
            df_voters['Votes'] = voter_rows
            df_voters["date"] = str(html_files[number_files][40:48])
            df_voters.to_csv("fomc_transcript/data/processed/Voters/" + html_files[number_files][40:48] + "_voters.csv" )
        elif voter_rows != '' and len(voter_rows) != len(df_voters):
            df_voters.to_csv("fomc_transcript/data/processed/Voters/" + html_files[number_files][40:48] + "_voters.csv" )
            print("The voter_rows are populated, but need to be cleaned")
        elif voter_rows == '' and len(voter_rows_words) == len(df_voters):
            df_voters['Votes'] = voter_rows_words
            df_voters["date"] = str(html_files[number_files][40:48])
            df_voters.to_csv("fomc_transcript/data/processed/Voters/" + html_files[number_files][40:48] + "_voters.csv" )
        elif voter_rows == '' and len(voter_rows_words) != len(df_voters):
            df_voters.to_csv("fomc_transcript/data/processed/Voters/" + html_files[number_files][40:48] + "_voters.csv" )
            print("The voter_rows_words are populated, but need to be cleaned")
        else:
            continue
            
                            #df_voters['Votes'] = voter_rowss
        #print(df_voters)

#=======================================================================END VOTERS=============================================#



    
 
