# Thesis

---
## Package Installation
`pip install -r requirements.txt`

## Data Extraction

**1. `01_get_transcript_text.py`**  
• *Desc:* HTML to .csv of transcript  
• *Input:* htmls  
• *Output:* Transcript extraction  

**2. `02_get_attendance_and_voters.py`**  **// there are some issues with this that i haven't been able to work out just yet**  
• *Desc:* HTML to .csv of attendance and voters  
• *Input:* htmls  
• *Output:* Attendance and Voter extraction  

## Data Processing

**3. `03_code_for_fomc_transcript_project.py`**  
• *Desc:* Creates initial variables for this project (divided into chunks)  
• *Input:* Extracted transcripts, extracted attendees  
• *Output:* `search_names.csv`, `all.csv`, `total_reserved.csv`  

**4. `04_drive.R`**  
• *Desc:* Creates sentiment variable using SentimentAnalysis package  
• *Input:* `all.csv`  
• *Output:* `after_drive.csv`  

**5. `05_after_drive_pre_collapse.py`**  
• *Desc:* Clean the `after_drive.csv` and take an average of the sentiment variable  
• *Input:* `after_drive.csv`  
• *Output:* `raw_pre_collapse.csv`  

**6. `06_owen_collapse.py`**  
• *Desc:* Collapse certain variables, create a dummy variable for Mr and Mrs/Ms, create number of times a person introduces a new topic, interrupted, interrupt someone else, uses hedging language  
• *Input:* Transcript and Attendance raw files, `raw_pre_collapse.csv`, `after_drive.csv`  
• *Output:* `06_collapsed.csv`  

**7. `07_voter_dissent.py`**  
• *Desc:* Introduce a percentage of vote dissent and the voter who dissented per meeting  
• *Input:* `voter_dummy.csv`, `06_collapsed.csv`  
• *Output:* `07_collapsed_with_cote.csv`  

**8. `08_buckets.py`**  
• *Desc:* Get the percentage of each bucket for attendance  
• *Input:* `attendance_voter_breakdown.csv`  
• *Output:* `buckets.csv`, `buckets_dummy.csv`  

**9. `09_owen_together_with_meeting_stats.py`**  
• *Desc:* Combine speaker characteristics with meeting characteristics  
• *Input:* `07_collapsed_with_vote.csv`, `attendance_voter_breakdown.csv`  
• *Output:* `final.csv`  

**10. `10_additions_to_dataset.R`**  
• *Desc:* Bring in financial crisis dates, economic crisis dates, chair names, political ideologies, add Baker, Bloom, and Davis, add a unique identifier, and miscellaneous variables  
• *Input:* `final.csv`  
• *Output:* `with_u_is_data_10162025.csv`  
