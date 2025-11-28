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
import re
import string
import nltk
import numpy as np
import glob
import requests
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Find the current working directory
print(os.getcwd())
os.chdir("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/")

# HTML directory
htmls_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/htmls/'
htmls_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/htmls/*.html'
html_files = glob.glob(htmls_ls)

#========READ PDF
# =============================================================================
# def get_human_names(text):
#     tokens = nltk.tokenize.word_tokenize(text)
#     pos = nltk.pos_tag(tokens)
#     sentt = nltk.ne_chunk(pos, binary = False)
#     
#     person = []
#     name = ""
#     for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
#         for leaf in subtree.leaves():
#             person.append(leaf[0])
#         if len(person) > 1: #avoid grabbing lone surnames
#             for part in person:
#                 name += part + ' '
#             if name[:-1] not in person_list:
#                 person_list.append(name[:-1])
#             name = ''
#         person = []  
# 
# get_human_names(first_page[0])
#     with open(html_files[1]) as fhandler:
#         soup = BeautifulSoup(fhandler, 'html.parser')
#         pure_text = soup.text
#         pure_text = pure_text.replace("\n", ' ')
#         pure_text = pure_text.replace(",", '')
#         pure_text = re.sub(' +', ' ', pure_text)
# 
# =============================================================================
non_names_please_delete = ['Group Manager', 'Special', 'International', 'Finance','Statistics', 'Principal', 'II', 'I', 'III', 'Payment', 'Supervision', 'Members', 'Office', 'Payments', 'Systems', 'Regulation', 'Alternate','Reserve Bank Operations', 'St Louis','Advisers', 'Account', 'Accounts', 'Louis', 'Federal Reserve System', 'Chair', 'Counsel', 'General Counsel', 'Economist', 'Alternate Members', 'Executive', 'Atlanta', 'Richmond', 'Senior', 'Federal Reserve Board',
                           'San Francisco', 'New York', 'Kansas City', 'Board', 'Senior Economist', 'Federal Reserve Banks',
                           'Chicago', 'Cleveland', 'Boston', 'St. Louis', 'Federal Reserve Bank', 'Philadelphia',
                           'Associate', 'Director', 'Division', 'Research', 'Assistant Directors', 'Adviser',
                           'System Open Market Account', 'Project Manager', 'Monetary Affairs', 'International Finance',
                           'Legal', 'Research and Statistics', 'Assistant', 'Governors', 'Governor',
                           'Statistics','Reserve Bank','Financial', 'Financial Stability', 'Senior Advisers', 'Senior Adviser', 'Washington', 'D.C.', 'Federal Open Market Committee']

pattern_to_remove = r'\b(?:{})\b',format('|'.join(non_names_please_delete))

non_votes_please_delete = ['T   hank ','briefing', 'thank', 'you', 'Thank', 'Thank you', 'MADIGAN', 'MR']
voting_pattern_to_remove = '|'.join(non_votes_please_delete)

####Option 2
for i in np.arange(0,len(html_files)):
    with open(html_files[i]) as fhandler:
        number_files = i
        soup = BeautifulSoup(fhandler, 'html.parser')
        pure_text = soup.text
        pure_text = pure_text.replace("\n", ' ')
        pure_text = pure_text.replace(",", '')
        pure_text = pure_text.replace(".", '')
        first_page = re.findall(r"Meeting of the Federal Open Market Committee(.+?)Transcript of the Federal Open Market Committee Meeting", pure_text)

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


        df_names.to_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/Attendees/" + html_files[number_files][79:87] + "_att.csv" )


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
        vote_page = re.findall(r"The vote will(.+?)CHAIR", pure_text)
        if len(vote_page) == 0:
            vote_page = re.findall(r"call roll(.+?)CHAIR", pure_text)
            if len(vote_page) == 0:
                vote_page = re.findall(r"call the roll(.+?)CHAIR", pure_text)
                if len(vote_page) < 5:
                    vote_page = re.findall(r"directive to the Desk(.+?)CHAIR", pure_text)
                
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
        df_voters = df_voters.dropna()
        df_voters = df_voters[['Voters']]
        yes_no_rows = df_voters[df_voters['Voters'].str.contains("Yes|yes|no|No")].index.values.tolist()
       
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
            df_voters.to_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/Voters/" + html_files[number_files][79:87] + "_voters.csv" )
        elif voter_rows != '' and len(voter_rows) != len(df_voters):
            print("The voter_rows are populated, but need to be cleaned")
        elif voter_rows == '' and len(voter_rows_words) == len(df_voters):
            df_voters['Votes'] = voter_rows_words
            df_voters.to_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/Voters/" + html_files[number_files][79:87] + "_voters.csv" )
        elif voter_rows == '' and len(voter_rows_words) != len(df_voters):
            print("The voter_rows_words are populated, but need to be cleaned")
        else:
            continue
            
                            #df_voters['Votes'] = voter_rowss
        print(df_voters)

#=======================================================================END VOTERS=============================================#



    
 
