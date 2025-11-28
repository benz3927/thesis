#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:10:41 2023

@author: m1dcs04
"""
import os
from datetime import datetime, date
from bs4 import BeautifulSoup, NavigableString, Tag
from selenium import webdriver
import requests
import pandas as pd
from tqdm import tqdm
import re
import string
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.common.exceptions import TimeoutException
import glob
import pdftotree


# Parent path
parent_path = 'fomc_transcript/'
pdf_path = 'fomc_transcript/pdfs/'
html_path = 'fomc_transcript/htmls/'


# PDF directory
pdfs_ls = r'fomc_transcript/pdfs/*.pdf'

os.chdir('fomc_transcript/htmls/')
files = glob.glob(pdfs_ls)

for i in np.arange(0,len(files),1):
    html_road = html_path + files[i][41:50] + ".html"
    pdftotree.parse(files[i], html_path = html_road)
    

# Conference Call directory
ccall_ls = r'fomc_transcript/conferencecall/*.pdf'

os.chdir('fomc_transcript/conferencecall/htmls/')
ccall_path = 'fomc_transcript/conferencecall/htmls/'
files = glob.glob(ccall_ls)

for i in np.arange(0,len(files),1):
    html_road = ccall_path + files[i][56:68] + ".html"
    pdftotree.parse(files[i], html_path = html_road)
    
