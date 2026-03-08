#!/usr/bin/env python3
import os
import subprocess
import glob

PDF_DIR = 'data/pdfs'
HTML_DIR = 'htmls'

pdf_files = glob.glob(f'{PDF_DIR}/FOMC2019*.pdf') + glob.glob(f'{PDF_DIR}/FOMC2020*.pdf')
print(f"Found {len(pdf_files)} PDFs to convert")

for pdf_file in pdf_files:
    filename = os.path.basename(pdf_file)
    base_name = filename.replace('.pdf', '')
    html_output = f"{HTML_DIR}/{base_name}.html"
    
    if os.path.exists(html_output):
        print(f"  Already exists: {filename}")
        continue
    
    print(f"  Converting: {filename}...", end=" ")
    try:
        subprocess.run(['pdftohtml', '-i', '-noframes', pdf_file, f"{HTML_DIR}/{base_name}"], 
                      capture_output=True, check=True)
        print("done")
    except Exception as e:
        print(f"error: {e}")

print("Done!")