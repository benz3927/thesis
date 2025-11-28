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


        df_names.to_csv("fomc_transcript/data/processed/Attendees/" + html_files[number_files][40:48] + "_att.csv" )


#==========================================================END ATTENDEES=============================================#

#=======================================================================BEGIN VOTERS=============================================#
voter_rows = pd.DataFrame()


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

    

        df_voters = df_voters[df_voters["Voters"] != ""]
        df_voters = df_voters[df_voters["Voters"] != ' ']
        df_voters = df_voters[df_voters["Voters"] != '  ']
        df_voters = df_voters[df_voters["Voters"] != '   ']
        df_voters = df_voters.dropna()
        df_voters = df_voters[['Voters']]
        yes_no_rows = df_voters[df_voters['Voters'].str.contains("Yes|yes|no|No")].index.values.tolist()
        
        
        for i in np.arange(0,len(yes_no_rows)):
                eye = yes_no_rows[i]
                voter_rows['Votes'] = df_voters.loc[eye]
            
                for i in np.arange(0, len(voter_rows)):
                    voter_rowss = []
                    if len(voter_rows['Votes'][i].split()) >1:
                        voters_split = voter_rows['Votes'][i].split()
                        voter_rowss = list(voters_split)
                    else:
                        voter_rowss.append(voter_rows['Votes'][i])
        
                for i in np.arange(0,len(yes_no_rows)):
                    to_drop = yes_no_rows[i]
                    df_voters = df_voters.drop([to_drop])
            #Remove upper case words
            #Remove 
# =============================================================================
#             for i in np.arange(0,len(df_voters)):
#                 if df_voters['Voters'][i].str[2:3].str.isupper()
# =============================================================================
                    df_voters['Votes'] = voter_rowss
                    df_voters.to_csv("fomc_transcript/data/processed/Voters/" + html_files[number_files][40:48] + "_voters.csv" )

#=======================================================================END VOTERS=============================================#



    
  #=====================================================  
nltk_results = nltk.ne_chunk(pos_tag(word_tokenize(pure_text)))
for nltk_result in nltk_results:
    if type(nltk_result) == Tree:
        name = ''
        for nltk_result_leaf in nltk_result.leaves():
            name += nltk_result_leaf[0] + ' '
        print ('Type: ', nltk_result.label(), 'Name: ', name)    
    
    
    
    # Identify relevant div that contains content 
    main_content = soup.find("div", {"class" : "col-sm-12 col-lg-8 offset-lg-1"})
    
    # Identify content_children of the main_content 
    content_children = main_content.findChildren(recursive = False)
    content_children = [i for i in content_children if i.text.strip()!=''] # Remove elements with no text    
    
    # Identify bold/strong/italic elements and identify the date
    strong_bold_it_elements = main_content.findAll("strong") + main_content.findAll("b") + main_content.findAll("em") # Combined
    date = [i.text.strip() for i in strong_bold_it_elements if re.match(r"(January|February|March|April|May|June|July|August|September|October|(N)?ovember|December)", i.text.strip())]
    date = date[0]
    date = " ".join(date.split())
    date_text = date
    if file_i == "2002-04-mi-rawdata.html": # Weird date where last number is not bold!
        date_text = "<strong>April 24, 200</strong>2"
        date = "April 24, 2002"
    if file_i == "1995-11-mi-rawdata.html": # Weird date where first letter is not bold!
        date_text = "N<strong>ovember 1, 1995</strong>"
        date = "November 1, 1995"
    
    # Preparation statement (if it exists) - want to remove this later
    # Ex. "Prepared at the Federal Reserve Bank of Cleveland and based on
    # information collected on or before November 22, 2013. This document
    # summarizes comments received from business and other contacts outside
    # the Federal Reserve and is not a commentary on the views of Federal
    # Reserve officials."
    preparation_statement = [i.text.strip() for i in strong_bold_it_elements if re.search(r"prepared (at|by) the federal reserve", i.text.strip().lower())]
    if len(preparation_statement) >0:
        preparation_statement = preparation_statement[0]
        preparation_statement = " ".join(preparation_statement.split())
    
    # The h1 tag denotes the bank info, grab that 
    bank_element = main_content.find("h1")
    bank_element_text = bank_element.text.strip()
    bank = bank_element.text.replace("Beige Book Report:", "").strip()
    bank = bank.replace("Beige Book:", "").strip()
    
    # Now that we have identified date and bank information, we can use string
    # cleaning, rather than web scraping, of the main_content to identify headers
    # and text. This is a pretty brute-force method, but it is more robust
    # than iterating through the content_children and making a bunch of if/else
    # statements for all of the edge cases (and edge cases of edge casess)    
    
    # Convert main_content to string (this looks messy, but do not fret, we will
    # clean it up in the next steps) 
    main_content_str = str(main_content).strip() # Make entire html code one string
    main_content_str = " ".join(main_content_str.split()) # Remove the pesky extra white spaces between words
    
    # First order of business, replace anything with bold tag, <b></b>, as 
    # strong, <strong></strong>. They are the same and we are cleaning strings
    # based on <strong> and </strong>.
    main_content_str = main_content_str.replace("<b>", "<strong>").strip()
    main_content_str = main_content_str.replace("</b>", "</strong>").strip()
    
    # Remove bank name, date, and preparation statement (if it exists)
    main_content_str = main_content_str.replace(f"{date_text}", "").strip()
    main_content_str = main_content_str.replace(f"<h1>{bank_element_text}</h1>", "").strip()
    if len(preparation_statement) >0:
        main_content_str = main_content_str.replace(f"{preparation_statement}", "").strip()
    
    # Remove div tags
    main_content_str = re.sub('<div?(.*?)>', '', main_content_str).strip()
    main_content_str = re.sub('<\/div>', '', main_content_str).strip()
    
    # Remove link back to Beige Book Archive
    main_content_str = main_content_str.replace('<a href="/region-and-community/regional-economic-indicators/beige-book-archive">‹ Back to Archive Search</a>', "").strip()
    
    # Remove horizontal line
    main_content_str = re.sub('<hr?(.*?)>', '', main_content_str).strip()
    
    # Remove paragraph tags
    main_content_str = main_content_str.replace("<p>", "").strip()
    main_content_str = main_content_str.replace("</p>", "").strip()
    
    # Remove breaks
    main_content_str = main_content_str.replace("<br/>", "").strip()
    
    # Remove "\t" 
    main_content_str = main_content_str.replace("\t", "").strip()
    
    # Remove repeated "\n"
    main_content_str = main_content_str.replace("\n", "").strip()
    
    # Bold and italics count as just bold!
    main_content_str = main_content_str.replace("<strong><em>", "<strong>").strip()
    main_content_str = main_content_str.replace("</em></strong>", "</strong>").strip()
    
    # Remove anything between <em></em> tags (italics)
    main_content_str = re.sub(r'<em>?(.*?)<\/em>', '', main_content_str).strip()
    
    # Remove anything between <table...></table> tags (tables)
    main_content_str = re.sub('<table ?(.*?)<\/table>', '', main_content_str).strip()
    
    # Remove anything between <font...></font> tags (font changes)
    main_content_str = re.sub('<font ?(.*?)<\/font>', '', main_content_str).strip()
    
    # Remove "<span>" and "</span>" - just not necessary
    main_content_str = re.sub('<span>', '', main_content_str).strip()
    main_content_str = re.sub('</span>', '', main_content_str).strip()
    
    # Remove "Highlights by Federal Reserve District
    main_content_str = re.sub('Highlights by Federal Reserve District', '', main_content_str).strip()
    
    # Report-specific cleaning: Use if/else statements with the row_url specify 
    # report-specific cleaning.
    if file_i == "2015-04-at-rawdata.html":
        main_content_str = main_content_str.replace("<strong>across</strong>", "across"). strip()
    
    # Clean up <strong> and </strong> tags - this is absolutely vital to get 
    # right as it sets the stage for string split in the for-loop that extract
    # header and text info...if something goes wrong, it's probably because of
    # a problem here!
    main_content_str = " ".join(main_content_str.split()) # Make sure there are no pesky multiple spaces
    main_content_str = main_content_str.replace("<strong> </strong>", ""). strip() # Remove instances where strong tags have no text
    main_content_str = main_content_str.replace("<strong></strong>", ""). strip() # Remove instances where strong tags have no text
    main_content_str = main_content_str.replace("<strong>.</strong>", "."). strip() # Replace bold period with regular period
    main_content_str = main_content_str.replace("</strong> <strong>", ""). strip() # Remove space between </strong> and <strong>
    main_content_str = main_content_str.replace("</strong><strong>", ""). strip() # Remove "</strong><strong>"
    main_content_str = main_content_str.replace("</strong>br&gt;", "</strong>"). strip() # Remove some weird characters in the text
    
    # Add a line break before bold text (header)
    main_content_str = main_content_str.replace("<strong>", "\n<strong>"). strip() # Remove line break between strong open and closing tags
    
   
    # Now split the main content on new line (\n). We will iterate through these
    # elements in the upcoming for-loop.
    main_content_list = main_content_str.strip().split("\n") # SPLIT ON LINE BREAK
    main_content_list = list(filter(lambda a: a != '', main_content_list)) # Remove empty strings
    
    ###########################################################################
    ################################ FOR LOOP #################################
    ###########################################################################
    
    beige_book_list = [] # Initiate empty list to be filled
    header = "NA" # Default header is NA (means no header)
    for index, line in enumerate(main_content_list):
        #print(index, line)
        if "</strong>" in line: # Identify if line contains strong end tag
            split_line = line.strip().split("</strong>") # Split on strong end tag
            
            header = split_line[0].replace("<strong>", "").strip() # Header is everything before strong end tag
            header = header.replace("&gt;", "").strip() # Weird characters - just something weird in the text, remove it
            if header == "ervices": # This happens once because the S is not bolded in the website 
                header = "Services"
            if header == "Housin": # This happens once because the S is not bolded in the website 
                header = "Housing"
            
            text_ls = split_line[1:] # Text is everything after strong end tag
            text_ls = [i.strip() for i in text_ls] # Remove leading/following white space 
            text = ' '.join(text_ls) # Concatinate to make current_text string
            beige_book_list.append({'Index':index, 'header': header,'text': text}) # Append it to the list

        else: # If there is no strong end tag, there is no header info, just grab the text
            text = line # text = line
            beige_book_list.append({'Index':index, 'header': header,'text': text}) # Append it to the list
            
    # Convert to DF
    df_beige_book = pd.DataFrame(beige_book_list)
    
    # Drop if the text begins with "This report was prepared at the Federal Reserve Bank of"
    df_beige_book = df_beige_book[~df_beige_book.text.str.startswith("This report was prepared at the Federal Reserve Bank of")]

    # Add date and bank info to DF
    df_beige_book['date'] = date
    df_beige_book["exact_date"] = pd.to_datetime(df_beige_book['date'])
    df_beige_book["pub_date"] = df_beige_book['exact_date'].apply(lambda x: x.strftime('%Y-%m-01'))
    df_beige_book['bank'] = bank
    df_beige_book['html'] = file_i  
       
    # Clean headers to make them a bit more uniform for analysis
    def remove_punctuations(text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')
        return text
    df_beige_book["header"] = df_beige_book['header'].apply(remove_punctuations) # Remove puncuation
    df_beige_book["header"] = df_beige_book['header'].str.lower() # Lower case headers
    df_beige_book["header"] = df_beige_book['header'].str.title() # Title case headers
    df_beige_book["header"] = df_beige_book['header'].str.strip() # Remove leading/following white spaces
    df_beige_book['header'] = np.where(df_beige_book['header'] == "Na", "NA", df_beige_book['header']) # Replace "Na" with "NA" 
    
    # Append Beige Book DF to the combined DF
    df_combined = df_combined.append(df_beige_book)
    
    # Phew! And this is just ONE Beige Book... looping through each row to grab
    # them all. :D 
print("Completed")

df_combined.to_csv("/if/prod-ifs/production/beige_book_nlp/data/final_data_with_headers.csv")
df_combined.to_csv("/rsma/shiny/if/collaboration/beige_books/data/final_data_with_headers.csv")

