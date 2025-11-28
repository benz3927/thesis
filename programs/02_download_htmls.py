#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created April 26, 2023

Author: Dylan Saez
Heavily based on Biege Book Cleaning from Isabel Kitschelt

Description: This script downloads each FOMC transcript
from the Board's website. It is publicly available data.
The scripts saves it as a local HTML. This prevents having to ping
our website everytime we download the historical data.

Board's Archive
https://www.federalreserve.gov/monetarypolicy/materials/

Trasncripts are published with a five-year lag.

Latest FOMC transcript (as of April 26, 2023) December 13, 2017
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





DOWNLOAD_PATH = r"/pdfs"


LOG_PATH = r"/pdfs/log/transcript_scrape_status.txt"

def options_init():
    options = webdriver.FirefoxOptions()
    options.set_preference("network.proxy.type", 1);
    options.set_preference("network.proxy.http", "proxy-t.frb.gov");
    options.set_preference("network.proxy.http_port", 8080);
    options.set_preference("network.proxy.ssl", "proxy-t.frb.gov");
    options.set_preference("network.proxy.ssl_port", 8080); 
    options.headless = False # Operating in headless mode
    options.add_argument("--width=1920")
    options.add_argument("--height=1080")
    options.set_preference('browser.download.folderList', 2) # custom location
    options.set_preference('browser.download.manager.showWhenStarting', False)
    options.set_preference("browser.download.dir", f"{DOWNLOAD_PATH}")
    options.set_preference('browser.helperApps.neverAsk.saveToDisk', "application/pdf")
    options.set_preference("pdfjs.disabled", True)
    options.set_preference("plugin.scan.Acrobat", "99.0")
    options.set_preference("plugin.scan.plid.all", False)
    options.set_preference("plugin.disable_full_page_plugin_for_types", "application/pdf")
    return options


def profile_init():
    fp = webdriver.FirefoxProfile()
    
    
os.chdir("fomc_transcript/pdfs")






#-----Collect PDFS
urls_df = pd.read_csv("metadata/full_url_data.csv")

for i in np.arange(0, len(urls_df)): 
    row_url = urls_df["url"][i]
    driver.get(row_url)
        
    
        
print("Local HTMLS updated")



    
#-----------------




