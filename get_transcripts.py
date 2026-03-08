#!/usr/bin/env python3
"""
Download FOMC transcripts for 2019 and 2020
"""

import requests
import os
import time

PDF_DIR = 'data/pdfs'
os.makedirs(PDF_DIR, exist_ok=True)

# FOMC meeting dates for 2019 and 2020
MEETINGS = [
    # 2019
    '20190130', '20190320', '20190501', '20190619',
    '20190731', '20190918', '20191030', '20191211',
    # 2020
    '20200129', '20200315', '20200429', '20200610',
    '20200729', '20200916', '20201105', '20201216',
]

print("="*70)
print("DOWNLOADING 2019-2020 FOMC TRANSCRIPTS")
print("="*70)

for date in MEETINGS:
    url = f"https://www.federalreserve.gov/monetarypolicy/files/FOMC{date}meeting.pdf"
    output_file = f"{PDF_DIR}/FOMC{date}meeting.pdf"
    
    if os.path.exists(output_file):
        print(f"  ✓ Already exists: {date}")
        continue
    
    print(f"  Downloading: {date}...", end=" ")
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Saved ({len(response.content)//1024} KB)")
        else:
            print(f"✗ HTTP {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    time.sleep(1)

print("\n✅ Done! Now run your PDF parsing script on the new files.")