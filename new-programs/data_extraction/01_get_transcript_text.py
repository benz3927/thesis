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
# Input: htmls
# Output: Transcript extraction

import os
from bs4 import BeautifulSoup, NavigableString, Tag
import requests
import pandas as pd
import re
import string
import numpy as np
import glob
import requests
import warnings
warnings.filterwarnings("ignore")

# HTML directory – the bash script is set to run this from the folder "new_programs"
htmls_road = "../htmls/"
htmls_ls = r"../htmls/*.html"
html_files = glob.glob(htmls_ls)

print('\n')
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
        
        #" ".join(s.lower() if s == (#'IBM', 'GE', 'CPFF', 'AMLF', 'PDCF', 'TIPS',
                                    #'US', 'UK', 'GDP', 'EM', 'MSCI', 'EDO', 'ABCP', 'SIV',
                                    #'SBA', 'COFFEE', 'BREAK', 'END', 'OF', 'MEETING', ' UPS',
                                    #'FSA', 'GDI', 'RIP', 'QE', 'SIPC', 'BNY', 'JPM', 'LTRO',
                                    #'ECB', 'ACF', 'MPC', 'RBC', 'HSBC', 'NOW', 'HOPE', 'FAQ',
                                    #'FDI', 'FDR', 'GSE', 'CPFF', 'AMLF', 'TPG', 'KKR', 'ECI',
                                    #'GSE', 'GM', 'GMAC', 'GC', 'BIS', 'OECD', 'CPI', 'CMBS',
                                    #'BOE', 'BOJ', 'ECB', 'LSAP', 'LIBOR', 'AEIOU', 'OLA',
                                    #'II', 'III', 'CLO', 'FOMC', 'CDO', 'TSLF', 'PDCF', 'CPFF',
                                    #'CP', 'CD', 'CEO', 'TALF', 'EME', 'ABCP', 'CCAR', 'CLAR',
                                    #'RRP', 'QIS', 'REITS', 'CUSIP', 'PAYGO', 'EDO', 'FR', 'SPF',
                                    #'PCE', 'TFP', 'AAA', 'GE', 'BBB', 'CNBC', 'FOIA', 'LSAP',
                                    #'TIC', 'FRB','SOMA', 'MBS', 'QM', 'QRM', 'IOER', 'ON', 'RP',
                                    #'FTC', 'MBS', 'OAS', 'ABX', 'CDS', 'NAPM', 'NFIB', 'MPC',
                                    #'BIS', 'SIV', 'PDCF', 'TSLF', 'IOER', 'LSAP', 'SPF', 'CPI',
                                    #'TAF', 'EU', 'IMP') else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'IBM' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'GE' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'CPFF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'AMLF' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'PDCF' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'TIPS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'EM' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'MSCI' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'EDO' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'ABCP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'SIV' else s for s in transcript.split())


        transcript = " ".join(s.lower() if s == 'US' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'SBA' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'COFFEE' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'BREAK' else s for s in transcript.split())


        transcript = " ".join(s.lower() if s == 'UPS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'FSA' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == "GDI" else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'RIP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'QE' else s for s in transcript.split())

         
        transcript = " ".join(s.lower() if s == "SIPC" else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'BNY' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'JPM' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'LTRO' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'ECB' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == "ACF" else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == "MPC" else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'RBC' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'HSBC' else s for s in transcript.split())

     
        transcript = " ".join(s.lower() if s == 'NOW' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'HOPE' else s for s in transcript.split())


        transcript = " ".join(s.lower() if s == 'FAQ' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'FDI' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'GSE' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'RMBS' else s for s in transcript.split())

         
        transcript = " ".join(s.lower() if s == 'CPFF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'AMLF' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'TPG' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'KKR' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'ECI' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'GSE' else s for s in transcript.split())
     
     
        transcript = " ".join(s.lower() if s == 'GM' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'GMAC' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'GC'  else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'BIS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'OECD' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'CPI' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CMBS' else s for s in transcript.split())
     
        transcript = " ".join(s.lower() if s == 'BOE' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'BOJ' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'ECB' else s for s in transcript.split())

        transcript = " ".join(s.lower() if s == 'LSAP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'LIBOR' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'AEIOU' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'OLA' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'II' else s for s in transcript.split())       
        transcript = " ".join(s.lower() if s == 'CLO' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'III' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'FOMC' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CDO' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'TSLF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'PDCF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CPFF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CD' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CEO' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'TALF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'EME' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'ABCP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'ECB' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'QE' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CCAR' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CLAR' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'RRP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'II' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'QIS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'REITS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CUSIP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'PAYGO' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'EDO' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'FR' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'SPF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'PCE' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'FOMC' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'TFP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'AAA' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'GE' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'GDP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'BBB' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CNBC' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'FOIA' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'LSAP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'TIC' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'FRB' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'SOMA' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'MBS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'QM' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'QRM' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'IOER' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'ON' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'RP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'FTC' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'MBS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'OAS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'ABX' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CDS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'NAPM' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'NFIB' else s for s in transcript.split())   
        transcript = " ".join(s.lower() if s == 'UK' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'MPC' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'BIS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'SIV' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'FDR' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'END' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'OF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'MEETING' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'PDCF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'TSLF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'IOER' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'LSAP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'SPF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CPI' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'US' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'TAF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CPI' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'GDP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'EU' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'IMF' else s for s in transcript.split())
        
        transcript = " ".join(s.lower() if s == 'SEP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'US' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'TAF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'SPEAKER' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'SEC' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'OIS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'WTO' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'LTCM' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'PCE' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CPI' else s for s in transcript.split())
        
        transcript = " ".join(s.lower() if s == 'ABS' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'US' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'RIP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'QE' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'LCR' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'NAIRU' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'TAF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'GDP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'BFI' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'MEP' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'END' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'OF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'MEETING' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'OF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'MEETING' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'ESF' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'DOJ' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'FRB' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'ICE' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'CLO' else s for s in transcript.split())
        transcript = " ".join(s.lower() if s == 'IOER' else s for s in transcript.split())

        
        
        
        transcript = transcript.replace(" end of meeting", "")
        transcript = transcript.replace(" esf", "")
        transcript = transcript.replace(" ioer", "")
        transcript = transcript.replace(" US GDP ", "")
        transcript = transcript.replace(" ABS CDO ", "")
        transcript = transcript.replace(" CRB ", "")
        transcript = transcript.replace(" END OF MEETING ", "")
        transcript = transcript.replace(" clo", "")
        transcript = transcript.replace(" frb", "")
        transcript = transcript.replace(" ice", "")
        transcript = transcript.replace(" doj", "")
        transcript = transcript.replace(" coffee break", "")
        transcript = transcript.replace(" pce cpi ", " ")
        transcript = transcript.replace(" esf ", " ")
        transcript = transcript.replace(" rip qe ", " ")
        transcript = transcript.replace(" ig ", "")
        transcript = transcript.replace(" crb ", "")
        transcript = transcript.replace(" aa ", " ")
        transcript = transcript.replace(" aaa ", " ")
        transcript = transcript.replace(" ioer ", " ")
        transcript = transcript.replace(" cbias ", " ")
        
        transcript = transcript.replace(" emu ", " ")
        transcript = transcript.replace(" abs ", " ")
        transcript = transcript.replace(" cbo ", " ")
        transcript = transcript.replace(" cdo ", " ")
        transcript = transcript.replace(" us gdp ", " ")
        transcript = transcript.replace(" dedu ", " ")
        transcript = transcript.replace(" rrp ", " ")
        
        
        pure_text = transcript
        #I think it would go here
        
        #
        indices_of_con_cap_words = re.findall(r'[A-Z][A-Z]+(?=\s[A-Z])(?:\s[A-Z][A-Z]+)+', pure_text)
        indices_positions = [(m.start(0), m.end(0)) for m in re.finditer(r'[A-Z][A-Z]+(?=\s[A-Z])(?:\s[A-Z][A-Z]+)+', pure_text)]
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
        #
        
        #
        df_transcript["date"] = str(html_files[number_files][13:21])

        # df_transcript.to_csv("/Users/ds3228/OneDrive - Yale University/Desktop/FRB/fomc_transcript/data/processed/Transcripts/" + html_files[number_files][79:87] + "_t.csv" )
        transcript_csv_location = "../data/new-processed/Transcripts/"
        df_transcript.to_csv(transcript_csv_location + html_files[number_files][13:21] + "_t.csv" )
        
