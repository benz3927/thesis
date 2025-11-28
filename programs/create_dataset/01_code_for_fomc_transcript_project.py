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

#Find the current working directory
print(os.getcwd())
os.chdir("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/")


# Transcript directory
transcript_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Transcripts/'
transcript_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Transcripts/*.csv'
transcript_files = glob.glob(transcript_ls)

# Attendee directory
att_road = '/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Attendees/'
att_ls = r'/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Attendees/*.csv'
att_files = glob.glob(att_ls)
#-------------------------------------------------------------------------------------------------------
#Remove some words from the transcripts that are all upper case but ARE NOT name
#Example: US GDP, FDR, CPI, LIBOR,...

#-------------------------------------------------------------------------------------------------------

#Create the variable: The number of times a person speaks in a meeting
df = pd.DataFrame()
date = []
speaker = []
number_of_times_speaker_speaks = []

for i in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[i])
    for person in np.arange(0, len(csv['Speaker'])):
        speaker_lv = csv['Speaker'][person]
        date_pick = str(csv['date'][person])
        number_of_times_speaks = csv['Speaker'].value_counts()[speaker_lv]
        date.append(date_pick)
        speaker.append(speaker_lv)
        number_of_times_speaker_speaks.append(number_of_times_speaks)
    

#DataFrame
#Create the variable: The number of times a person speaks in a meeting
df['speaker'] = speaker
df['date'] = date
df['speak_count'] = number_of_times_speaker_speaks
df = df.drop_duplicates()
df = df.reset_index()
del df['index']
#-----Take the last word of this is python
last_name_speakers = []

for ii in np.arange(0, len(df['speaker'])):
    last_name = df['speaker'][ii].split(" ")
    last_name = last_name[-1]
    last_name_speakers.append(last_name)

df['speaker_unique'] = last_name_speakers

df.to_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/search_names.csv")

list_to_delete = ['CHAIRMAN GREENSPAN MR PRELL', 'NBER MR TARULLO','MR KOCHERLAKOTA FRASER','VICE CHAIRMAN MCDONOUGH CHAIRMAN GREENSPAN','CHAIRMAN GREENSPAN MR LINDSEY','DEDO CHAIR YELLEN','PRESIDENT BOEHNE CHAIRMAN GREENSPAN','IOER MR POTTER','CHAIRMAN GREENSPAN MR KOHN', 'MR MCTEER CHAIRMAN GREENSPAN','AA MR REINHART','UCLA CHAIR YELLEN','MS MINEHAN CHAIRMAN GREENSPAN','CHAIRMAN GREENSPAN MR JORDAN','CBIAS MR LACKER','MR KELLEY CHAIRMAN GREENSPAN','DNA MR LACKER','MR TRUMAN CHAIRMAN GREENSPAN','MR BULLARD FRB','EMU MR LINDSEY','CHAIRMAN GREENSPAN MR KELLEY','MS YELLEN CHAIRMAN GREENSPAN','MR LINDSEY CHAIRMAN GREENSPAN','CHAIRMAN GREENSPAN VICE CHAIRMAN MCDONOUGH','RPIX CHAIRMAN GREENSPAN','CHAIRMAN GREENSPAN MR MOSKOW','CHAIRMAN GREENSPAN MR PARRY','MR KOHN MR FISHER','MS MINEHAN MR PRELL','MR LINDSEY MR TRUMAN','HAIRMAN GREENSPAN','CHAIRMAN GREENSPAN MR TRUMAN','CHAIRMAN GREENSPAN SPEAKER','CHAIRMAN GREENSPAN MS MINEHAN','IG MS GEORGE','MR STERN MS JOHNSON','MS PHILLIPS CHAIRMAN GREENSPAN','MR GIBSON SIV','VICE CHAIRMAN GREENSPAN','US MR REIFSCHNEIDER','CHAIRMAN GREENSPAN MR FISHER','CHAIRMAN GREENSPAN MR PREL', 'MR FISHER MR MOSKOW','MR NELSON CLO', 'HAIR YELLEN','HAIRMAN BERNANKE','MR BROADDUS CHAIRMAN GREENSPAN',  'US MR REINHART','CHAIRMAN GREENSPAN SEVERAL','MR PARRY CHAIRMAN GREENSPAN','CHAIRMAN GREENSPAN MR HOENIG',  'MR PARRY MR PRELL', 'MR TRUMAN MS MINEHAN', 'CHAIRMAN GREENSPAN MS PHILLIPS', 'NIPA MR LINDSEY', 'ON RRP', 'PCE CPI', 'ABS CDO', 'END OF MEETING', 'BMW SPEAKER', 'RIP QE', 'GE AAA', 'US TAF', 'US GDP', "HAIR YELLEN", "HAIRMAN BERNANKE", "HAIRMAN GREENSPAN"]

df = df[df.speaker.isin(list_to_delete) == False]


df = df.reset_index()
del df['index']
#----Fix if a speaker_unique is double - look in stata for this
df['speaker'] = df['speaker'].str.replace("CHAIMAN", "CHAIRMAN")
df['speaker'] = df['speaker'].str.replace("CHARMAN", "CHAIRMAN")
df['speaker'] = df['speaker'].str.replace("CHARIMAN", "CHAIRMAN")
df['speaker'] = df['speaker'].str.replace("CHARIMAN", "CHAIRMAN")
df['speaker'] = df['speaker'].str.replace("GREENPAN", "GREENSPAN")
df['speaker'] = df['speaker'].str.replace("GREESPAN", "GREENSPAN")
df['speaker'] = df['speaker'].str.replace("MR YELLEN", "MS YELLEN")
df['speaker'] = df['speaker'].str.replace("DAVID WILCOX", "MR WILCOX")
df['speaker'] = df['speaker'].str.replace("MESSRS", "MR")
df['speaker'] = df['speaker'].str.replace("GRENSPAN", "GREENSPAN")
df['speaker'] = df['speaker'].str.replace("MCDONUGH", "MCDONOUGH")
df['speaker'] = df['speaker'].str.replace("MOSCOW", "MOSKOW")
df['speaker'] = df['speaker'].str.replace("KOCHERLOKTA", "KOCHERLAKOTA")
df['speaker'] = df['speaker'].str.replace("KOCOHERLAKOTA", "KOCHERLAKOTA")
df['speaker'] = df['speaker'].str.replace("MS MINEHAN AND OTHERS", "MS MINEHAN")
df['speaker'] = df['speaker'].str.replace("SEVERAL MR KELLEY", "MR KELLEY")
df['speaker'] = df['speaker'].str.replace("VICE CHAIR DUDLEY", "VICE CHAIRMAN DUDLEY")

df = df.drop_duplicates()
df = df.reset_index()
del df['index']
#-----
df.to_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/search_names.csv")

#list(set(df['speaker']))

df['date'] = df['date'].astype(int)
len(list(set(df['date'])))
#df = pd.read_csv("/fomc_transcript/data/processed/sets/search_names.csv")
#df['date'] =df['date'].astype(int)
#-------------------------------------------------------------------------------------------------------
#
#   XXXX --- End of Variable Creation
#   XXXX --- Begin new Variable Creation
#
#
#-------------------------------------------------------------------------------------------------------
#Create the variable: Word count for that meeting for that person
df_var2 = pd.DataFrame()

date = []
speaker = []
length_of_all_speak_times = []


for i in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[i])
    for person in np.arange(0, len(csv['Speaker'])):
        date_pick = str(csv['date'][person])
        speaker_lv = csv['Speaker'][person]
        rslt_df = csv[csv['Speaker'] == speaker_lv]
        combined_talks = ' '.join(rslt_df['transcript_text'])
        #Clean the combined_talks
        combined_talks = combined_talks.split()
        combined_talks = " ".join([word for word in combined_talks if not word.isupper()])
        #
        length_of_all_speaks = len(combined_talks)
        date.append(date_pick)
        speaker.append(speaker_lv)
        length_of_all_speak_times.append(length_of_all_speaks)
    

#DataFrame
#Create the variable: The number of times a person speaks in a meeting
df_var2['speaker'] = speaker
df_var2['date'] = date
df_var2['speak_length'] = length_of_all_speak_times
df_var2 = df_var2.drop_duplicates()
df_var2['date'] =df_var2['date'].astype(int)

#   ================= MERGE ====================#
total = pd.merge(df, df_var2, on = ['date', 'speaker'], how = 'left')
len(list(set(total['date'])))

#   ================= END MERGE====================#
#-------------------------------------------------------------------------------------------------------
#
#   XXXX --- End of Variable Creation
#   XXXX --- Begin new Variable Creation
#
#
#-------------------------------------------------------------------------------------------------------
#Create the variable: [AVERAGE] Word count for the perosn who speaks immediately afterwards
df_var3 = []


for i in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[i])
    pure_text = ' '.join(csv['transcript_text'])
    pure_text = pure_text.strip()
    indices_of_con_cap_words = re.findall('[A-Z][A-Z]+(?=\s[A-Z])(?:\s[A-Z][A-Z]+)+', pure_text)
    indices_positions = [(m.start(0), m.end(0)) for m in re.finditer('[A-Z][A-Z]+(?=\s[A-Z])(?:\s[A-Z][A-Z]+)+', pure_text)]
    int_df = pd.DataFrame()
    int_df['Speaker'] = indices_of_con_cap_words
    int_df['speaker_location'] = indices_positions
    next_speaker_index_position = []
    for int_df_datum in np.arange(0, len(int_df['speaker_location'])):
        if int_df_datum+1 in np.arange(0, len(int_df['speaker_location'])):
            next_speaker = int_df['speaker_location'][int_df_datum+1]
        else:
            next_speaker = int_df['speaker_location'][int_df_datum]
        next_speaker_index_position.append(next_speaker)
    int_df['next_speaker_location'] = next_speaker_index_position    
   #Now go to each next speakr index position and count the length of that text
    nsp_text_length = []
    int_df_position = []
    for nsp in np.arange(0, len(int_df['next_speaker_location'])):
        if nsp+1 in np.arange(0, len(int_df['next_speaker_location'])):
             start = int(int_df['next_speaker_location'][nsp][0])
             end = int(int_df['next_speaker_location'][nsp+1][0])
             just_for_list = (start, end)
             nsp_text = pure_text[start:end]
             #Clean the combined_talks
             nsp_text = nsp_text.split()
             nsp_text = " ".join([word for word in nsp_text if not word.isupper()])
             #
             nsp_length = len(nsp_text)
             nsp_text_length.append(nsp_length)
             int_df_position.append(just_for_list)
        else:
            start = int(int_df['next_speaker_location'][nsp][0])
            end = -1
            just_for_list = (start, end)
            nsp_text = pure_text[start:end]
            #Clean the combined_talks
            nsp_text = nsp_text.split()
            nsp_text = " ".join([word for word in nsp_text if not word.isupper()])
            #
            nsp_length = len(nsp_text)
            nsp_text_length.append(nsp_length)
            int_df_position.append(just_for_list)
    #df_for_merge_with_next_info = pd.DataFrame()
    #df_for_merge_with_next_info['next_speaker_location'] = int_df_position
    #df_for_merge_with_next_info['next_speaker_text_length'] = nsp_text_length
    int_df['next_speaker_text_length'] = nsp_text_length
    
    length_next_speaker = []
    date_for_int_Df = []
    for first_speaker in np.arange(0, len(int_df['Speaker'])):
        if first_speaker+1 in np.arange(0, len(int_df['Speaker'])):
            date_pick = str(csv['date'][first_speaker])
            date_for_int_Df.append(date_pick)
            speaker_lv = int_df['Speaker'][first_speaker+1]
            rslt_df = int_df[int_df['Speaker'] == speaker_lv]
            #
            length_of_all_speaks = sum(rslt_df['next_speaker_text_length'])
            length_next_speaker.append(length_of_all_speaks)
        else:
            length_next_speaker.append(0)
            date_pick = str(csv['date'][first_speaker])
            date_for_int_Df.append(date_pick)
    int_df['entire_length_of_speaker'] = length_next_speaker
    int_df['date'] = date_for_int_Df
    df_var3.append(int_df)


df_var_3 = pd.concat(df_var3)
df_var_3 = df_var_3.rename(columns= {"Speaker":"speaker"})
df_var_3['date'] =df_var_3['date'].astype(int)
#   ================= MERGE ====================#
#data_avg_next_speaker = df_var_3[['date', 'speaker', 'next_speaker_text_length']]
#data_avg_next_speaker = data_avg_next_speaker.groupby(['date', 'speaker']).mean()
#data_avg_next_speaker.columns = ['avg_next_speaker_length']

#total_with_3 = pd.merge(total, data_avg_next_speaker, on = ['date', 'speaker'])
total_reserve = total
total = pd.merge(df_var_3, total, on = ['date', 'speaker'], how = 'outer')
len(list(set(total['date'])))

#total = pd.merge(total_with_3, total, on = ['date', 'speaker'])
#   ================= END MERGE====================#
#-------------------------------------------------------------------------------------------------------
#
#   XXXX --- End of Variable Creation
#   XXXX --- Begin new Variable Creation
#
#
#-------------------------------------------------------------------------------------------------------
#Create the variable: Gender of the person who speaks next

df_var4 = []
   
for i in np.arange(0, len(transcript_files)):
    csv = pd.read_csv(transcript_files[i])
    date_for_int_DF = []
    pure_text = ' '.join(csv['transcript_text'])
    pure_text = pure_text.strip()
    indices_of_con_cap_words = re.findall('[A-Z][A-Z]+(?=\s[A-Z])(?:\s[A-Z][A-Z]+)+', pure_text)
    indices_positions = [(m.start(0), m.end(0)) for m in re.finditer('[A-Z][A-Z]+(?=\s[A-Z])(?:\s[A-Z][A-Z]+)+', pure_text)]
    int_df = pd.DataFrame()
    int_df['First Speaker'] = indices_of_con_cap_words
    int_df['speaker_location'] = indices_positions
    next_speaker_index_position = []
    for int_df_datum in np.arange(0, len(int_df['speaker_location'])):
        if int_df_datum+1 in np.arange(0, len(int_df['speaker_location'])):
            next_speaker = int_df['speaker_location'][int_df_datum+1]
        else:
            next_speaker = int_df['speaker_location'][int_df_datum]
        next_speaker_index_position.append(next_speaker)
    int_df['next_speaker_location'] = next_speaker_index_position    
   #Now go to each next speakr index position and count the length of that text
    nsp_name = []
    int_df_position = []
    for nsp in np.arange(0, len(int_df['next_speaker_location'])):
        if nsp+1 in np.arange(0, len(int_df['next_speaker_location'])):
             start = int(int_df['next_speaker_location'][nsp][0])
             end = int(int_df['next_speaker_location'][nsp][1])
             just_for_list = (start, end)
             nsp_text = pure_text[start:end]
             nsp_name.append(nsp_text)
             int_df_position.append(just_for_list)
             date_pick = str(csv['date'][nsp])
             date_for_int_DF.append(date_pick)
             
        else:
            start = int(int_df['next_speaker_location'][nsp][0])
            end = int(int_df['next_speaker_location'][nsp][1])
            just_for_list = (start, end)
            nsp_text = pure_text[start:end]
            nsp_text = pure_text[start:end]
            nsp_name.append(nsp_text)
            int_df_position.append(just_for_list)
            date_pick = str(csv['date'][nsp])
            date_for_int_DF.append(date_pick)
            
    #df_for_merge_with_next_info = pd.DataFrame()
    #df_for_merge_with_next_info['next_speaker_location'] = int_df_position
    #df_for_merge_with_next_info['next_speaker_text_length'] = nsp_text_length
    int_df['next_speaker_name'] = nsp_name
    int_df['date'] = date_for_int_DF
    #Match Gender
    date_to_match_from_int = int_df['date'][1]
    for t in np.arange(0, len(att_files)):
        att_date_to_match = att_files[t][94:102]
        if date_to_match_from_int == att_date_to_match:
            transcript_csv = pd.read_csv(att_files[t])
            transcript_csv = transcript_csv.rename(columns= {"Clean Names":"next_speaker_name"})
            transcript_csv['date'] = transcript_csv['date'].astype(str)
            transcript_csv['next_speaker_name'] = transcript_csv['next_speaker_name'].apply(str.lower)
            int_df['next_speaker_name'] = int_df['next_speaker_name'].apply(str.lower)
            int_df['next_speaker_name'] = int_df['next_speaker_name'].str.replace('mr', '')
            int_df['next_speaker_name'] = int_df['next_speaker_name'].str.replace('ms', '')
            int_df['next_speaker_name'] = int_df['next_speaker_name'].str.replace('governor', '')
            int_df['next_speaker_name'] = int_df['next_speaker_name'].str.replace('president', '')
            int_df['next_speaker_name'] = int_df['next_speaker_name'].str.replace('chairman', '')
            int_df['next_speaker_name'] = int_df['next_speaker_name'].str.replace('chair', '')
            int_df['next_speaker_name'] = int_df['next_speaker_name'].str.replace('vice', '')
            int_df['next_speaker_name'] = int_df['next_speaker_name'].str.strip()
            int_total_int_transcript = pd.merge(int_df, transcript_csv, on = ['date','next_speaker_name'])
            df_var4.append(int_total_int_transcript)
        else:
            continue
   
    

df_var_4 = pd.concat(df_var4)
df_var_4 = df_var_4.rename(columns= {"First Speaker":"speaker", "Greeting":"Next Speaker Gender"})   
df_var_4['date'] =df_var_4['date'].astype(int)

transcript_files_left = ["/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Transcripts/20001115_t.csv",
                         "/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Transcripts/20080805_t.csv",
                         "/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Transcripts/20130320_t.csv"]

#   ================= MERGE ====================#
total = pd.merge(total, df_var_4, on = ['date', 'speaker', 'speaker_location', 'next_speaker_location'])
#   ================= END MERGE====================#  
total['date'] = total['date'].astype(str)   
len(list(set(total['date'])))

#-------------------------------------------------------------------------------------------------------
#
#   XXXX --- End of Variable Creation
#   XXXX --- Begin new Variable Creation
#
#
#-------------------------------------------------------------------------------------------------------
#Create the variable: Number of times others laugh after/during the time this person is talking
text_for_transcript = []
date_for_transcript = []
for row in np.arange(0, len(total)):
    date_to_match_from_total = total['date'][row]
    for t in np.arange(0, len(transcript_files)):
        transcript_date_to_match = transcript_files[t][96:104]
        if date_to_match_from_total == transcript_date_to_match:
            transcript_csv = pd.read_csv(transcript_files[t])
            pure_text = ' '.join(transcript_csv['transcript_text'])
            pure_text = pure_text.strip()
            text_for_transcript.append(pure_text)
            date_for_transcript.append(date_to_match_from_total)
        else:
            continue
  
bf_df_var_5 = pd.DataFrame()
bf_df_var_5['date']  = date_for_transcript
bf_df_var_5['transcript_text'] = text_for_transcript
text_for_transcript = ""   
  
bf_df_var_5 = bf_df_var_5.drop_duplicates()    

#bf_df_var_5['date'] =bf_df_var_5['date'].astype(int)
total = pd.merge(total,bf_df_var_5, on = ['date'])    
text_for_transcript = []   
   
#Create the variable: Number of times others laugh after/during the time this person is talking 

laughter_first_speaker = []
date_picked_cons = []
speaker_location = []
first_speaker_text_list = []

for text in np.arange(0, len(total['transcript_text'])):
    start = total['speaker_location'][text][0]
    start_end_first_for_list = total['speaker_location'][text][1]
    end = total['next_speaker_location'][text][0]
    first_speaker_text = total['transcript_text'][text][start:end]
    first_speaker_text_list.append(first_speaker_text)
    laughter_count = first_speaker_text.count("[Laughter]") + first_speaker_text.count("[laughter]")
    laughter_first_speaker.append(laughter_count)
    date_picked = total['date'][text]
    date_picked_cons.append(date_picked)
    first_speaker_loc = (start, start_end_first_for_list)
    speaker_location.append(first_speaker_loc)


df_var_5 = pd.DataFrame()
df_var_5['date'] = date_picked_cons
df_var_5['speaker_location'] = speaker_location
df_var_5['laughter_first_speaker'] = laughter_first_speaker
df_var_5['text_between_speakers'] = first_speaker_text_list
df_var_5 = df_var_5.drop_duplicates()

bf_df_var_5['date'] =bf_df_var_5['date'].astype(int)

total = pd.merge(total, df_var_5, on = ['date', 'speaker_location'])
#   ================= END MERGE====================#  
    
#-------------------------------------------------------------------------------------------------------
#
#   XXXX --- End of Variable Creation
#   XXXX --- Begin new Variable Creation
#
#
#-------------------------------------------------------------------------------------------------------
#Create the variable: Number of times a person is interrupted
#If the text between speakers ends with a dash, then it is considered a disruption.

#total = total.drop(['transcript_text'], axis = 1)
total['text_between_speakers'] = total['text_between_speakers'].str.strip()

#According to Berle et al 2023 - looked for an abrupt ending to a person's speech delineated by
#a "-" followed immediately by another person's speech
disrupted_list = []
for check_disrupt in np.arange(0, len(total['text_between_speakers'])):
    if total['text_between_speakers'][check_disrupt].endswith(('-', '—')) == False:
           disrupted_list.append(0)
    else:
        disrupted_list.append(1)
        
total['disrupted_1_yes_0_no'] = disrupted_list        
   
#-------------------------------------------------------------------------------------------------------
#
#   XXXX --- End of Variable Creation
#   XXXX --- Begin new Variable Creation
#
#
#-------------------------------------------------------------------------------------------------------  
#-------------------------------------------------------------------------------------------------------  
#Create the variable: Number of times a person interrupts somebody else.
#May not have enough variation in this, but we could look.

df_var_7 = total[['date', 'next_speaker_name', 'disrupted_1_yes_0_no']]  
df_var_7 = df_var_7.groupby(['date', 'next_speaker_name']).sum()
df_var_7.columns = ['number_of_times_all_disrupted_per_date']

total = pd.merge(total, df_var_7, on = ['date', 'next_speaker_name'])
#-------------------------------------------------------------------------------------------------------
#
#   XXXX --- End of Variable Creation
#   XXXX --- Begin new Variable Creation
#
#
#-------------------------------------------------------------------------------------------------------  
#Create the variable: If the individual introduces new topics (i.e, are they the first to speak on a new topic)
topic_dictionary = pd.read_excel("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/external/hedge_topic/topic_fixed.xlsx")

topics_discussed = []
for chunk in np.arange(0, len(total['text_between_speakers'])):
    topics_discussed_int = []
    for topics in np.arange(0, len(topic_dictionary['Terms'])):
        t_test = topic_dictionary['Terms'][topics].split(',')
        if any(word in total['text_between_speakers'][chunk] for word in t_test) == True:
            topics_discussed_int.append(topic_dictionary['Variable'][topics])
        else:
            topics_discussed_int.append("")
    topics_discussed.append(topics_discussed_int)

#topics_to_be_collapsed = pd.DataFrame(np.reshape(topics_discussed,(20973, 15)))

tuple_topics = []
for c in np.arange(0, len(topics_discussed)):
    a = topics_discussed[c]
    lists_for_topics = []
    for aa in np.arange(0, len(a)):
        if len(a[aa]) == 0:
            pass
        else:
            lists_for_topics.append(a[aa])
    tuple_topics.append(tuple(lists_for_topics))
            
df_var_8 = pd.DataFrame()
df_var_8['topics'] = tuple_topics
total['topics'] = tuple_topics       

#---Merge---#
#-------------------------------------------------------------------------------------------------------  
#Create the variable: Number of times a person uses hedging language (Tan and Lee 2016)
#I have no idea why the line calling in the tan and lee data errors, it is called correctly
#When you re-run the line, it pulls it - maybe a time out?
tan_and_lee = pd.read_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/external/hedge_topic/hedges.csv")

hedging_list = []
for check_hedge in np.arange(0, len(total['text_between_speakers'])):
    for hedge in np.arange(0, len(tan_and_lee['hedges'])):
        if tan_and_lee['hedges'][hedge] in total['text_between_speakers'][check_hedge]:
            hedging_list.append(1)
        else:
            hedging_list.append(0)
    
   
hedges_to_be_collapsed = pd.DataFrame(np.reshape(hedging_list,(61078, 255)))


tuple_hedges = []
for d in np.arange(0, len(hedges_to_be_collapsed)):
    a = sum(hedges_to_be_collapsed.loc[d])
    tuple_hedges.append(a)
            

total['hedges'] = tuple_hedges



#-----------------------------------ANYTHING TO GO BACK TO?---------------
#-----------------------------------ANYTHING TO GO BACK TO?---------------
#Yes, go back to topics to create a column that indicates 1 for topic introduction

#total = pd.read_csv("/fomc_transcript/data/processed/sets/all.csv")
total_for_topic_dummy = total[['speaker', 'date', 'topics']]
topics_dummy = total_for_topic_dummy['topics'].tolist()
#Create a list of lists from a list of tuples 
list_of_lists = []
for tup in topics_dummy:
    list_of_lists.append(list(tup))

list_of_strings = []
for list_list in list_of_lists:
    string_s = ' '.join(list_list)
    list_of_strings.append(string_s)


total_for_topic_dummy['topics'] = list_of_strings

first_values = total_for_topic_dummy.groupby(["speaker", "date"]).first()


total_for_topic = pd.merge(first_values, total_for_topic_dummy, on = ['speaker', 'date'], how = "inner")

list_of_first = []
for y in np.arange(0, len(total_for_topic)):
    if len(total_for_topic['topics_y'][y]) > 0:
        list_of_first.append(1)
    else:
        list_of_first.append(0)


total['first_to_introduce_topic'] = list_of_first 

total.drop(['speaker_location', 'next_speaker_location', 'topics', 'transcript_text'], axis = 1, inplace = True)
#Yes, go back to topics to create a column that indicates 1 for topic introduction
'''
total_for_topic_dummy = total[['speaker', 'date', 'topics']]
topics_dummy = total_for_topic_dummy['topics'].tolist()
#Create a list of lists from a list of tuples
list_of_lists = []
for tup in topics_dummy:
    list_of_lists.append(list(tup))

list_of_strings = []
for list_list in list_of_lists:
    string_s = ' '.join(list_list)
    list_of_strings.append(string_s)
    
'''
len(list(set(total['date'])))
    
total.to_csv('/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/all.csv')
total_reserve.to_csv('/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/sets/total_reserved.csv')
    
