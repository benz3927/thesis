printf "Running 01_get_transcript_text.py...\n"
python3 data_extraction/01_get_transcript_text.py 

# printf "Running 02_get_attendance_and_voters.py...\n"
# python3 data_extraction/02_get_attendance_and_voters.py
printf "Skipping 02_get_attendance_and_voters.py for now..."

python3 data_processing/03_code_for_fomc_transcript_project.py
